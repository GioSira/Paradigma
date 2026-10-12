from datetime import datetime
from uuid import UUID
from typing import List, Optional, Any

from src.subscription.abbonamento import Abbonamento

from src.sqldb.tables import AbbonamentoRow, StudentRow
from src.sqldb.errors import _constraint_name, IntegrityError, AbbonamentoAlreadyPresent

from sqlalchemy import func, insert, select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession


_ABBONAMENTI = AbbonamentoRow.__table__

_STUDENTS = StudentRow.__table__

_ABBONAMENTO_PRIMARY_KEY = "pk_abbonamento"



class AbbonamentoRepository:

    def __init__(self, session:AsyncSession):
    
        self._session = session


    # ================================= READ ==============================

    async def get_by_id(self, abbonamento_id: UUID) -> Optional[Abbonamento]:

        query = select(_ABBONAMENTI).where(_ABBONAMENTI.c.id == abbonamento_id)

        return await self._fetch_one(query)


    async def get_active_subscription(self, student_id: UUID, at: Optional[datetime] = None) -> Optional[Abbonamento]:

        moment = at if at else func.now()

        query = select(_ABBONAMENTI).where(
            _ABBONAMENTI.c.student_id == student_id,
            _ABBONAMENTI.c.inizio <= moment, 
            moment < _ABBONAMENTI.c.fine
        )

        return await self._fetch_one(query)


    async def get_history(self, student_id: UUID):

        query = select(_ABBONAMENTI).where(_ABBONAMENTI.c.student_id == student_id).order_by(*self._order_desc())

        return await self._fetch_all(query)


    # ================================= WRITE ==============================

    async def add(self, subscription: Abbonamento):


        statements = (
            insert(_ABBONAMENTI).values(**self._to_row(subscription)),
        )

        self._write_all(statements, subscription)

    async def update(self, subscription: Abbonamento):

        statement = (
            update(_ABBONAMENTI).where(_ABBONAMENTI.c.id == subscription.get_id()).values(**self._to_row(subscription)),
        )

        self._write_all(statement, subscription)

    async def delete(self, subscription_id: UUID) -> bool:

        statement = (
            delete(_ABBONAMENTI).where(_ABBONAMENTI.c.id == subscription_id),
        )

        result = await self._session.execute(statement)

        return result.rowcount == 1



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

        result = (await self._session.execute(query)).mappings().one_or_none()
        return Abbonamento.from_db(dict(result))


    async def _fetch_all(self, query):

        results = (await self._session.execute(query)).mappings().all()
        return [Abbonamento.from_db(dict(result)) for result in results]


    async def _exist_student(self, student_id: UUID):

        query = select(_STUDENTS).where(_STUDENTS.c.user_id == student_id)

        return (await self._session.execute(query)).first() is not None


    async def _write_all(self, statement: List[Any], subscription: Abbonamento):

        try:
            async with self._session.begin_nested():
                return [await self._session.execute(s) for s in statement]
        except IntegrityError as error:
            domain_error = self._domain_error(error, subscription) 
            if domain_error is None: raise
            raise domain_error from error

    @staticmethod
    def _to_row(abbonamento: Abbonamento, for_update=False):

        d = {
            "nome": abbonamento.get_nome(),
            "inizio": abbonamento.get_inizio(),
            "fine": abbonamento.get_fine(),
            "costo": abbonamento.get_costo(),
            "student_id": abbonamento.get_student_id()
        }

        if not for_update:
            d["id"] = abbonamento.get_id()


    @staticmethod
    def _domain_error(error: IntegrityError, abbonamento: Abbonamento) -> Optional[Exception]:

        constraint = _constraint_name(error)
        
        if constraint == _ABBONAMENTO_PRIMARY_KEY:
            return AbbonamentoAlreadyPresent(abbonamento.get_id())
        
        return None
