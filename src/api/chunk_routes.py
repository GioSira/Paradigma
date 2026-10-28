import json
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Path, Request, UploadFile
from pydantic import ValidationError

from src.api.schemas import ErrorOut, ChunkOuts
from src.orchestrator.chunker_orchestrator import ChunkerService
from src.orchestrator.llm_orchestrator import RagService


ERRORS = {
    413: {"model": ErrorOut, "description": "File troppo grande"},
    422: {"model": ErrorOut, "description": "JSON non valido o chunk non validi"},
    502: {"model": ErrorOut, "description": "OpenRouter o Pinecone non rispondono"},
    503: {"model": ErrorOut, "description": "Indicizzazione non riuscita: riprovare"},
}


MAX_BYTES = 10 * 1024 * 1024    # 10 MB


# solo caratteri sicuri.
Namespace = Path(..., pattern=r"^[A-Za-z0-9_-]{1,64}$",
                 description="Spazio dei documenti, per esempio l'id di un corso")


router = APIRouter(prefix="/namespaces/{namespace}", tags=["chunk"])


def get_chunk(request: Request) -> ChunkerService:
    return request.app.state.chunk

def get_rag(request: Request) -> RagService:
    return request.app.state.rag


async def _read_chunks(namespace: str, file: UploadFile, chunkservice: ChunkerService, number_to_show: Optional[int] = None):

    contenuto = await file.read(MAX_BYTES + 1)
    if len(contenuto) > MAX_BYTES:
        raise HTTPException(413, f"file troppo grande: massimo {MAX_BYTES // (1024 * 1024)} MB")
    
    try:
        data = json.loads(contenuto)
    except json.JSONDecodeError as error:
        raise HTTPException(422, f"JSON non valido alla riga {error.lineno}, colonna {error.colno}")
    except UnicodeDecodeError as error:
        raise HTTPException(422, f"JSON non in formato UTF-8: {error}")

    try:
        chunks = await chunkservice.read_chunks_file(namespace, data, number_to_show)
    except (ValidationError, ValueError, KeyError, TypeError) as error:
        raise HTTPException(422, f"chunk non validi: {error}")

    if not chunks:
        raise HTTPException(422, "il file non contiene chunk")

    return chunks


@router.post("/chunk-file", response_model=ChunkOuts, responses=ERRORS)
async def importa_file(file: UploadFile = File(...), namespace: str = Namespace,
                       chunkservice: ChunkerService = Depends(get_chunk),
                       rag: RagService = Depends(get_rag)):
    
    """Legge un file JSON di chunk e li indicizza nel namespace (indice ibrido)."""

    chunks = await _read_chunks(namespace, file, chunkservice)
    indexed = await rag.index_legal_chunks(namespace, chunks)

    return ChunkOuts(chunks=chunks, indexed=indexed)


@router.post("/preview-file/{number_to_show}", response_model=ChunkOuts, responses=ERRORS)
async def importa_file(number_to_show: int = Path(..., ge=1, le=200), file: UploadFile = File(...), namespace: str = Namespace,
                       chunkservice: ChunkerService = Depends(get_chunk)):
    
    """Legge un file JSON di chunk e li indicizza nel namespace (indice ibrido)."""

    to_show = await _read_chunks(namespace, file, chunkservice, number_to_show)

    return ChunkOuts(chunks=to_show, indexed=0)