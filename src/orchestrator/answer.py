from uuid import UUID
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class Source:
    number: int            # il numero [n] citato nella risposta
    document_id: UUID
    chunk_id: UUID
    text: str
    score: float           # punteggio del rerank
 
 
@dataclass(frozen=True)
class Answer:
    text: str
    sources: List[Source]