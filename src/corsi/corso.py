from pydantic import BaseModel, Field, field_validator, AwareDatetime
from typing import Optional, List, Any
from src.roles.tutor import Tutor
from uuid import UUID
from datetime import datetime, timezone


class Corso(BaseModel):

    id: UUID
    title: str
    docente: List[Tutor] = Field(..., min_length=1)
    descrizione: str
    data_inizio: Optional[AwareDatetime]
    data_fine: Optional[AwareDatetime]



    # ----------------------------- GETTER/SETTER ----------------------------------


    def get_id(self):
        return self.id

    def set_id(self, new_id: UUID):
        self.id = new_id


    def set_title(self, new_title: str):
        self.title = new_title

    def get_title(self):
        return self.title


    def add_docente(self, tutor: Tutor):
        if tutor not in self.docente:
            self.docente.append(tutor)

    def add_docenti(self, tutors: List[Tutor]):
        for tutor in tutors:
            self.add_docente(tutor)

    def remove_docente(self, tutor: Tutor):
        self.docente.remove(tutor)

    def remove_docenti(self, tutors: List[Tutor]):

        if len(self.docente) == 1:
            return

        for tutor in tutors:
            self.remove_docente(tutor)
            if len(self.docente) == 1:
                break

    def set_new_docenti(self, tutors: List[Tutor]):

        assert len(tutors) > 0, ValueError("tutors list must contain more than one element")

        self.docente = tutors


    def set_descrizione(self, new_descrizione):
        self.descrizione = new_descrizione

    def get_descriziione(self):
        return self.descrizione


    def set_data_inizio(self, new_data: datetime):
        self.data_inizio = new_data

    def get_data_inizio(self):
        return self.data_inizio


    def set_data_fine(self, new_data: datetime):
        
        assert new_data > self.data_inizio

        self.data_fine = new_data


    @staticmethod
    def _format_data(data: datetime):
        return "{d:02d}/{m:02d}/{y:04d}".format(d=data.day, m=data.month, y=data.year)

    def format_data_inizio(self):
        return self._format_data(self.data_inizio)

    def format_data_fine(self):
        return self._format_data(self.data_fine)


    # ----------------------------- GETTER/SETTER ----------------------------------

    def _json_payload(self):
        return {
            "id": str(self.id),
            "title": self.title,
            "descrizione": self.descrizione,
            "docenti": [d.to_json() for d in self.docente],
            "data_inizio": self.data_inizio.isoformat(),
            "data_fine": self.data_fine.isoformat()
        }


    def to_json(self):
        return self._json_payload()
    