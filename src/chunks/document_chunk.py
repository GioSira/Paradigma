from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from datetime import datetime
from typing import Optional, ClassVar
from src.collection.document import Document
import ujson


class DocumentChunk(BaseModel):

    # evita field vuoti
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    # id del chunk
    chunk_id: str =  Field(..., min_length=1, description="mandatory chunk ID")
    # id del documento originale
    document_id: str = Field(..., min_length=1, description="mandatory document ID")
    last_modified: Optional[datetime]
    start_idx: int = Field(..., ge=0)
    end_idx: int = Field(..., ge=0)
    text: str = Field(..., min_length=1)

    DATE_FORMAT: ClassVar[str] = "[{d:02d}/{m:02d}/{y:04d} - {hh:02d}:{mm:02d}:{ss:02d}]"


    # --------------------- VALIDATORS -------------------------------

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value):
        # "   " supera min_length ma non produce un embedding utile.
        if not value.strip():
            raise ValueError("Text cannot be empty")
        return value
 
    @model_validator(mode="after")
    def span_must_be_forward(self):
        if self.end_idx <= self.start_idx:
            raise ValueError(
                f"span not valid: end_idx ({self.end_idx}) must be greater than "
                f"start_idx ({self.start_idx})"
            )
        return self

    def _assign(self, field, value):
        """Assegna ripristinando lo stato se la validazione fallisce.

        Con validate_assignment pydantic esegue i validator di campo PRIMA di
        scrivere, ma model_validator(mode="after") DOPO: senza rollback un set
        rifiutato lascerebbe l'oggetto con lo span gia' corrotto.
        """
        previous = getattr(self, field)
        try:
            setattr(self, field, value)
        except Exception:
            self.__dict__[field] = previous
            raise
    
    # ----------------------------------------------------------------


    def get_chunk_id(self):
        """Chiave primaria per il vector db."""
        return self.chunk_id

    def get_document_id(self):
        return self.document_id
 
    def set_document_id(self, document_id):
        self.document_id = document_id
 
    def get_start_index(self):
        return self.start_idx
 
    def set_start_index(self, start_index):
        self._assign("start_idx", start_index)

    def get_end_index(self):
        return self.end_idx
 
    def set_end_index(self, end_index):
        self._assign("end_idx", end_index)
 
    def format_last_modified(self):
        if self.last_modified is None:
            return ""
        
        date = self.last_modified
        return self.DATE_FORMAT.format(d=date.day, m=date.month, y=date.year, 
                                       hh=date.hour, mm=date.minute, ss=date.second)
 
    def set_last_modified(self, last_modified):
        self.last_modified = last_modified

    def get_last_modified(self):
        return self.last_modified
 
    def get_chunk_text(self):
        return self.text
 
    def set_chunk_text(self, text):
        self._assign("text", text)

    @classmethod
    def build_chunck_id(self) -> str:
        return f"{self.document_id}/{self.chunk_id}/{str(self.start_idx)}-{str(self.end_idx)}"

    def belong_to(self, document: Document) -> bool:
        return self.document_id == document.document_id

    def is_stale(self, document:Document) -> bool:
        if self.last_modified is None and document.last_modified is None:
            return False

        return self.last_modified != document.last_modified

    def overlaps(self, other) -> bool:
        if self.document_id != other.document_id:
            return False
 
        return (self.start_idx < other.end_idx and other.start_idx < self.end_idx)


    def get_chunk_length(self):
        return self.end_idx - self.start_idx
     
    def _identity(self):
        return (self.document_id, self.chunk_id)

    def __eq__(self, other):
        if not isinstance(other, DocumentChunk):
            return NotImplemented
 
        return self._identity() == other._identity()    
 
    def __hash__(self):
        return hash(self._identity())

    def _json_payload(self):
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "start_index": self.start_idx,
            "end_index": self.end_idx,
            "last_modified": (
                self.last_modified.isoformat()
                if self.last_modified is not None
                else None
            ),
            "text": self.text,
        }

    def to_json(self):
        return ujson.dumps(self._json_payload())
