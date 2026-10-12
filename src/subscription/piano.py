from typing import Mapping, Any
from pydantic import BaseModel, Field, ConfigDict
from uuid import UUID
from datetime import datetime

import ujson


class Piano(BaseModel):

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: UUID
    nome: str = Field(..., min_length=1)
    descrizione: str = Field(..., min_length=1)
    durata: int = Field(..., ge=0)
    costo: int = Field(..., ge=0)


    # ----------------------------- GETTER/SETTER ----------------------------------

    def set_id(self, new_id: UUID):
        self.id = new_id

    def get_id(self):
        return self.id


    def set_nome(self, new_name: str):
        self.nome = new_name

    def get_nome(self):
        return self.nome


    def set_descrizione(self, new_descrizione: str):
        self.nome = new_descrizione
    
    def get_descrizione(self):
        return self.nome


    def set_costo(self, new_costo: int):
        self.nome = new_costo
    
    def get_costo(self):
        return self.costo


    def set_durata(self, new_durata: int):
        self.durata = new_durata

    def get_durata(self):
        return self.durata


    # ----------------------------- JSON ----------------------------------
    
    def to_json(self):
        return ujson.dumps(self._json_payload())

    def _json_payload(self):

        return {
            "id": str(self.id),
            "nome": self.get_nome(),
            "costo": self.get_costo(),
            "durata": self.get_durata(),
            "descrizione": self.get_descrizione()
        }
    
    @classmethod
    def from_db(cls, record: Mapping[str, Any]) -> "Piano":
        return cls(**record)
