from typing import Mapping, Any
from pydantic import BaseModel, Field, AwareDatetime, model_validator, ConfigDict
from uuid import UUID
from datetime import datetime

import ujson



class Abbonamento(BaseModel):

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: UUID
    nome: str = Field(..., min_length=1)
    student_id: UUID
    inizio: AwareDatetime
    fine: AwareDatetime
    costo: int = Field(..., ge=0)


    # ----------------------------- GETTER/SETTER ----------------------------------

    @model_validator(mode="after")
    def fine_dopo_inizio(self):

        if self.fine <= self.inizio:
            raise ValueError("fine deve essere dopo inizio")

        return self
    

    # ----------------------------- GETTER/SETTER ----------------------------------

    def set_id(self, new_id: UUID):
        self.id = new_id

    def get_id(self):
        return self.id


    def set_nome(self, new_name: str):
        self.nome = new_name

    def get_nome(self):
        return self.nome


    def set_student_id(self, student_id: UUID):

        self.student_id = student_id

    def get_student_id(self) -> UUID:
        return self.student_id


    def set_inizio(self, new_inizio: datetime):
        self.inizio = new_inizio

    def get_inizio(self):
        return self.inizio


    def set_fine(self, new_fine: datetime):

        assert new_fine > self.inizio, ValueError("fine must be greather than inizio")

        self.fine = new_fine

    def get_fine(self):

        return self.fine


    def set_costo(self, new_costo: int):

        assert new_costo >= 0, ValueError("costo cannot be negative")

        self.costo = new_costo


    def get_costo(self):
        return self.costo


    # ----------------------------- UTILS ----------------------------------

    @staticmethod
    def _format_date(date: datetime):
        return "{d:02d}/{m:02d}/{y:04d}".format(d=date.day, m=date.month, y=date.year)


    def format_inizio(self):
        return self._format_date(self.inizio)

    def format_fine(self):
        return self._format_date(self.fine)

    # ----------------------------- JSON ----------------------------------

    def to_json(self):
        return ujson.dumps(self._json_payload())

    def _json_payload(self):

        return {
            "id": str(self.id),
            "student_id": str(self.student_id),
            "nome": self.get_nome(),
            "costo": self.get_costo(),
            "inizio": self.inizio.isoformat(),
            "fine": self.fine.isoformat()
        }


    @classmethod
    def from_db(cls, record: Mapping[str, Any]) -> "Abbonamento":
        return cls(**record)
