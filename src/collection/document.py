from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import Optional, ClassVar
import ujson

class Document(BaseModel):

    # evita field vuoti
    model_config = ConfigDict(validate_assignment=True)

    doc_id: str = Field(..., min_length=1, description="mandatory session ID")
    name: str
    last_modified: Optional[datetime]
    text: Optional[str]

    DATE_FORMAT: ClassVar[str] = "[{d:02d}/{m:02d}/{y:04d} - {hh:02d}:{mm:02d}:{ss:02d}]"

    @property
    def document_id(self):
        return self.doc_id

    def set_document_id(self, new_id):
        self.doc_id = new_id

        
    def get_document_name(self):
        return self.name

    def set_document_name(self, name):
        self.name = name


    def get_last_modified(self):
        return self.last_modified

    def format_last_modified(self):
        if self.last_modified is None:
            return ""
        
        date = self.last_modified
        return self.DATE_FORMAT.format(d=date.day, m=date.month, y=date.year, 
                                       hh=date.hour, mm=date.minute, ss=date.second)

    def set_last_modified(self, last_doc_modified):
        self.last_modified = last_doc_modified


    def get_document_text(self):
        return self.text

    def set_document_text(self, doc_text):
        self.text = doc_text


    def _identity(self):
        return (self.document_id, self.name)

    
    def __eq__(self, other):

        if not isinstance(other, Document):
            return NotImplemented

        return self._identity() == other._identity()


    def __hash__(self):
        return hash(self._identity())


    def _json_payload(self):
        """Punto di estensione: le sottoclassi aggiungono qui i loro campi."""
        return {
            "id": self.doc_id,
            "name": self.name,
            "last_modified": (
                self.last_modified.isoformat()
                if self.last_modified is not None
                else None
            ),
            "text": self.text,
        }

 
    def to_json(self):
        return ujson.dumps(self._json_payload())