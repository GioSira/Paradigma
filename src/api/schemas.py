from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional, Literal, Tuple
from uuid import UUID

from src.orchestrator.answer import Answer
from src.chunks.document_chunk import DocumentChunk
from src.chunks.legal_chunk import LegalChunk
from datetime import datetime


# ========================================================= IN SCHEMAS =========================================================

class ChunkIn(BaseModel):

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1)
    last_modified: Optional[datetime] = None
    start_idx: int = Field(..., ge=0)
    end_idx: int = Field(..., ge=0)


class ChunksIn(BaseModel):
 
    model_config = ConfigDict(extra="forbid")
 
    chunks: List[ChunkIn] = Field(..., min_length=1)


class IndexedDocument(BaseModel):
    document_id: UUID
    chunks: int


class TurnIn(BaseModel):

    """Un messaggio gia' scambiato nella conversazione."""
    model_config = ConfigDict(extra="forbid")
 
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=8_000)


class QuestionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
 
    question: str = Field(..., min_length=1)
    document_id: Optional[UUID] = None   # per cercare in un solo documento
    history: List[TurnIn] = Field(default_factory=list, max_length=50)


class CandidateIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    candidates: List[Tuple[DocumentChunk, float]] = Field(..., max_length=100)
    min_score: Optional[float] = None


class GenerationIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    candidates: List[Tuple[DocumentChunk, float]] = Field(..., max_length=100)
    history: List[TurnIn] = Field(default_factory=list, max_length=50)

    


# ========================================================= OUT SCHEMAS =========================================================


class ChunkOuts(BaseModel):
    chunks: List[LegalChunk]
    indexed: int 


class SourceOut(BaseModel):
    number: int
    document_id: UUID
    chunk_id: UUID
    text: str
    score: float


class AnswerOut(BaseModel):
    answer: str
    sources: List[SourceOut]
 
    @classmethod
    def from_domain(cls, answer: Answer) -> "AnswerOut":
        return cls(answer=answer.text, sources=[
            SourceOut(number=s.number, document_id=s.document_id, chunk_id=s.chunk_id,
                      text=s.text, score=s.score)
            for s in answer.sources
        ])


class CandidateOut(BaseModel):
 
    question: str
    document_id: Optional[UUID] = None
    candidates: List[Tuple[DocumentChunk, float]]
    method: Literal["vectors", "hybrid", "rerank"]
 
 
class ErrorOut(BaseModel):
    detail: str
