from typing import List, Optional
from pydantic import BaseModel, Field, AwareDatetime
from uuid import UUID

class Abbonamento(BaseModel):

    id: UUID
    nome: str = Field(..., min_length=1)
    durata: int = Field(..., ge=0)
    inizio: AwareDatetime
    fine: AwareDatetime
    costo: int = Field(..., ge=0)


    # ----------------------------- GETTER/SETTER ----------------------------------

    