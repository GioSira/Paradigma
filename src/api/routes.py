from fastapi import APIRouter, Depends, HTTPException, Path, Request, Response, status
from src.orchestrator.llm_orchestrator import RagService

from uuid import uuid5, UUID
from typing import List

from src.api.schemas import ErrorOut, ChunksIn, IndexedDocument, AnswerOut, QuestionIn
from src.api.settings import Settings

from src.chunks.document_chunk import DocumentChunk


ERRORS = {
    413: {"model": ErrorOut, "description": "Troppi chunk o testo troppo lungo"},
    422: {"model": ErrorOut, "description": "Input non valido"},
    502: {"model": ErrorOut, "description": "OpenRouter o Pinecone non rispondono"},
    503: {"model": ErrorOut, "description": "Indicizzazione non riuscita: riprovare"},
}


# solo caratteri sicuri.
Namespace = Path(..., pattern=r"^[A-Za-z0-9_-]{1,64}$",
                 description="Spazio dei documenti, per esempio l'id di un corso")


router = APIRouter(prefix="/namespaces/{namespace}", tags=["rag"])



def get_rag(request: Request) -> RagService:
    return request.app.state.rag

def get_settings(request: Request) -> Settings:
    return request.app.state.settings
 
 
def _check_size(text: str, limit: int, what: str) -> None:
    if len(text) > limit:
        raise HTTPException(413, f"{what} troppo lungo: {len(text)} caratteri, massimo {limit}")


@router.put("/documents/{document_id}/chunks", response_model=IndexedDocument, responses=ERRORS)
async def insert_chunks(body: ChunksIn, document_id: UUID, namespace: str = Namespace,
                        rag: RagService = Depends(get_rag), settings: Settings = Depends(get_settings)):

    _check_size("".join(c.text for c in body.chunks), settings.max_document_chars, "testo dei chunk")
    chunks: List[DocumentChunk] = [DocumentChunk(chunk_id=uuid5(document_id, f"{c.start_idx}:{c.end_idx}"), 
                                                 document_id=document_id, 
                                                 last_modified=c.last_modified, 
                                                 start_idx=c.start_idx, 
                                                 end_idx=c.end_idx, 
                                                 text=c.text) for c in body.chunks]

    num_chunks = await rag.index_chunks(namespace, chunks)
    return IndexedDocument(document_id=document_id, chunks=num_chunks)     


@router.post("/ask", response_model=AnswerOut, responses=ERRORS)
async def ask(body: QuestionIn, namespace: str = Namespace,
              rag: RagService = Depends(get_rag),
              settings: Settings = Depends(get_settings)):

    _check_size(body.question, settings.max_question_chars, "domanda")
    answer = await rag.ask(namespace, body.question, document_id=body.document_id)
    return AnswerOut.from_domain(answer)
