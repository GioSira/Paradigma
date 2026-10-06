from datetime import datetime
from uuid import UUID
from typing import List, Optional

from src.abbonamenti.abbonamento import Abbonamento

from src.sqldb.tables import AbbonamentoRow, StudentRow

from sqlalchemy import func, insert, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession


_ABBONAMENTI = AbbonamentoRow.__table__

_STUDENTS = StudentRow.__table__

_ABBONAMENTI_JOIN_STUDENT = _ABBONAMENTI.join(_STUDENTS, _ABBONAMENTI.c.student_id == _STUDENTS.c.user_id)



class AbbonamentoRepository:

    def __init__(self, session:AsyncSession):
    
        self._session = session


    # ================================= READ ==============================

    async def get_by_id(self, abbonamento_id: UUID) -> Optional[Abbonamento]:

        query = (
            select(_ABBONAMENTI).where(_ABBONAMENTI.c.id == abbonamento_id)
        )

        result = await self._fetch_one(*query)
        return result


    async def get_active_subscription(self, student_id: UUID, at: Optional[datetime] = None) -> Optional[Abbonamento]:

        moment = at if at else func.now()

        query = select(_ABBONAMENTI).where(_ABBONAMENTI.c.student_id == student_id,
                                           _ABBONAMENTI.c.inizio <= moment,
                                           moment < _ABBONAMENTI.c.fine)

        return await self._fetch_one(query)


    async def get_history(self, student_id: UUID):

        query = (
            select(_ABBONAMENTI).where(_ABBONAMENTI.c.student_id == student_id).order_by(*self._order_desc())
        )

        return await self._fetch_all(*query)


    # ================================= WRITE ==============================

    



    # ================================= UTILS ==============================

    def _order(self):

        return (
            _ABBONAMENTI.c.data_fine, _ABBONAMENTI.c.data_inizio, _ABBONAMENTI.c.nome, _ABBONAMENTI.c.id
        )

    def _order_desc(self):

        return (
            _ABBONAMENTI.c.data_fine.desc(), _ABBONAMENTI.c.data_inizio.desc(), _ABBONAMENTI.c.nome.desc(), _ABBONAMENTI.c.id.desc()
        )

    async def _fetch_one(self, query):

        row = (await self._session.execute(query)).mappings().one_or_none()

        if row:
            return Abbonamento.from_db(row)

        return None


    async def _fetch_all(self, query):

        results = (await self._session.execute(query)).mappings().all()

        return [Abbonamento.from_db(result) for result in results]


    async def _exist_student(self, student_id: UUID):

        query = (
            select(_STUDENTS).where(_STUDENTS.c.user_id == student_id)
        )

        return (await self._session.execute(*query)).first() is not None

    