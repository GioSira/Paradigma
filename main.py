import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncContextManager, AsyncGenerator, Callable, Optional
 
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from openrouter.errors import OpenRouterError
from pinecone.errors import PineconeError
 
from src.api.routes import router
from src.api.settings import Settings
from src.llm.model_loader import ModelLoader
from src.llm.errors import GenerationError
from src.orchestrator.llm_orchestrator import IndexingError, RagService
from src.vectordb.pinecone_db import PineconeDB


# Interfaccia web: una pagina sola, servita dallo stesso server (niente CORS).
INDEX_HTML = Path(__file__).parent / "static" / "index.html"

RagFactory = Callable[[Settings], AsyncContextManager[RagService]]

logger = logging.getLogger(__name__)


@asynccontextmanager
async def build_rag(settings: Settings) -> AsyncGenerator[RagService, None]:
    """
    Crea i servizi. Se setup() fallisce (chiave sbagliata, indice con
    un'altra dimensione...) il server non parte
    """

    logger.info("Starting up system...")

    models = ModelLoader()
    vectordb = PineconeDB()

    logger.info("LLM and DB online, starting pipeline...")

    try:

        # Per nome: il quinto parametro di RagService e' chunker, non top_k.
        rag = RagService(vectordb,
                         models,
                         settings.index_name,
                         settings.system_prompt,
                         top_k=settings.top_k,
                         max_tokens=settings.max_tokens,
                         top_n=settings.top_n)

        dimension = await rag.setup()

        logger.info("indice %r pronto, dimensione %d", settings.index_name, dimension)
        logger.info("Pipeline online")

        yield rag

    finally:

        logger.info("Closing DB and LLM connection...")
        await vectordb.aclose()
        await models.aclose()
        logger.info("Connection closed")


def create_app(settings: Optional[Settings] = None, rag_factory: RagFactory = build_rag) -> FastAPI:

    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        async with rag_factory(settings) as rag:
            app.state.rag = rag
            app.state.settings = settings
            yield

    app = FastAPI(title="Paradigma RAG", version="1.0", lifespan=lifespan)
    app.include_router(router)


    @app.get("/", include_in_schema=False)
    async def home():
        return FileResponse(INDEX_HTML, media_type="text/html")

    @app.get("/health", tags=["servizio"])
    async def health():
        return {"status": "ok"}

    @app.exception_handler(ValueError)
    async def invalid_input(request: Request, exc: ValueError):
        # RagService e DocumentChunk sollevano ValueError per input non validi
        # (es. end_idx <= start_idx, testo di soli spazi).
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(IndexingError)
    async def indexing_failed(request: Request, exc: IndexingError):
        logger.warning("indicizzazione non riuscita: %s", exc)
        return JSONResponse(status_code=503, headers={"Retry-After": "5"},
                            content={"detail": "indicizzazione non riuscita, riprova tra poco"})

    async def upstream_failed(request: Request, exc: Exception):
        # Il dettaglio va nel log, non al client: potrebbe contenere dati interni.
        logger.exception("servizio esterno non disponibile")
        return JSONResponse(status_code=502,
                            content={"detail": "servizio esterno non disponibile, riprova piu' tardi"})
 
    for error in (OpenRouterError, PineconeError, GenerationError, httpx.HTTPError):
        app.add_exception_handler(error, upstream_failed)
 
    return app




# L'app vera si crea solo quando uvicorn la chiede: i test importano
# create_app senza leggere il .env. Creata una volta sola.
_app: Optional[FastAPI] = None
 
 
def __getattr__(name: str):
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(name)
 
 
if __name__ == "__main__":
    import uvicorn
 
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
