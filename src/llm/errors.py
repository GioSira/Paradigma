
from typing import Optional

class GenerationError(RuntimeError):
    """Il modello ha risposto senza testo. finish_reason dice perche':
    "length" = token finiti (spesso tutti spesi nel ragionamento),
    "content_filter" = risposta bloccata, None/"stop" = risposta vuota."""
 
    def __init__(self, message: str, finish_reason: Optional[str] = None):
        super().__init__(message)
        self.finish_reason = finish_reason