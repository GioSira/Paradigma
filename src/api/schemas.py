from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional
from uuid import UUID

from src.orchestrator.answer import Answer
from datetime import datetime


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


class SourceOut(BaseModel):
    number: int
    document_id: UUID
    chunk_id: UUID
    text: str
    score: float


class QuestionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
 
    question: str = Field(..., min_length=1)
    document_id: Optional[UUID] = None   # per cercare in un solo documento
 

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
 
 
class ErrorOut(BaseModel):
    detail: str