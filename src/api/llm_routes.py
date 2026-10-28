from fastapi import APIRouter, Depends, HTTPException, Path, Request, Response, status
from src.orchestrator.llm_orchestrator import RagService

from uuid import uuid5, UUID
from typing import List

from src.api.schemas import ErrorOut, ChunksIn, IndexedDocument, AnswerOut, QuestionIn, CandidateIn, CandidateOut, GenerationIn
from src.api.settings import Settings

from src.chunks.document_chunk import DocumentChunk
from src.llm.conversation import Message


ERRORS = {
    413: {"model": ErrorOut, "description": "Troppi chunk o testo troppo lungo"},
    422: {"model": ErrorOut, "description": "Input non valido"},
    404: {"model": ErrorOut, "description": "metodo non esistente"},
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

    num_chunks = await rag.index_chunks_vectors(namespace, chunks)
    return IndexedDocument(document_id=document_id, chunks=num_chunks)


@router.post("/rewrite/question", response_model=str, responses=ERRORS)
async def rewrite_question(body: QuestionIn, rag: RagService = Depends(get_rag), settings: Settings = Depends(get_settings)):
    history = [Message(role=turn.role, content=turn.content) for turn in body.history]
    return await rag.rewrite_question(settings.condense_prompt, body.question, history)


@router.post("/search/vectors", response_model=CandidateOut, responses=ERRORS)
async def get_candidates_vectors(body: QuestionIn, namespace: str = Namespace, 
                                 rag: RagService = Depends(get_rag), settings: Settings = Depends(get_settings)):
    
    candidates = await rag.search_chunks_embedding(namespace, body.question, body.document_id, settings.top_k)
    return CandidateOut(question=body.question, document_id=body.document_id, candidates=candidates, method="vectors")


@router.post("/search/hybrid", response_model=CandidateOut, responses=ERRORS)
async def get_candidates_hybrid(body: QuestionIn, namespace: str = Namespace, 
                                rag: RagService = Depends(get_rag), settings: Settings = Depends(get_settings)):
    
    candidates = await rag.search_chunks_hybrid(namespace, body.question, body.document_id, settings.top_k)
    return CandidateOut(question=body.question, document_id=body.document_id, candidates=candidates, method="hybrid")


@router.post("/search/rerank", response_model=CandidateOut, responses=ERRORS)
async def rerank_candidates(body: CandidateIn, rag: RagService = Depends(get_rag), settings: Settings = Depends(get_settings)):

    candidates = await rag.evaluate_candidates(body.question, body.candidates, settings.top_n, body.min_score)
    return CandidateOut(question=body.question, document_id=None, candidates=candidates, method="rerank")


@router.post("/generate", response_model=AnswerOut, responses=ERRORS)
async def generate(body: GenerationIn, rag: RagService = Depends(get_rag), settings: Settings = Depends(get_settings)):

    history = [Message(role=turn.role, content=turn.content) for turn in body.history]
    answer = await rag.generate(settings.system_prompt, body.question, body.candidates, history)

    return AnswerOut.from_domain(answer)


@router.post("/ask", response_model=AnswerOut, responses=ERRORS)
async def ask(body: QuestionIn, namespace: str = Namespace,
              rag: RagService = Depends(get_rag),
              settings: Settings = Depends(get_settings)):

    _check_size(body.question, settings.max_question_chars, "domanda")

    history = [Message(role=turn.role, content=turn.content) for turn in body.history]

    answer = await rag.ask(namespace, 
                           body.question, 
                           settings.system_prompt,
                           settings.condense_prompt,
                           document_id=body.document_id, 
                           history=history)
    
    return AnswerOut.from_domain(answer)
