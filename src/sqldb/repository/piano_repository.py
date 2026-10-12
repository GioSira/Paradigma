from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, insert, delete, update, desc, select

from src.sqldb.tables import PianoRow
from src.sqldb.errors import _constraint_name, IntegrityError, PianoAlreadyPresent
from src.subscription.piano import Piano

from typing import List, Optional
from uuid import UUID


_PIANI = PianoRow.__table__

_PIANO_PRIMARY_KEY = "pk_piano"


class PianoRepository:

    def __init__(self, session: AsyncSession):
        self._session = session

    # ================================= READ ===============================

    async def get_all(self):

        query = select(_PIANI).order_by(_PIANI.c.durata, _PIANI.c.costo, _PIANI.c.nome, _PIANI.c.id)
        return await self._fetch_all(query)

    async def get_by_id(self, piano_id: UUID):

        query = select(_PIANI).where(_PIANI.c.id == piano_id)
        return await self._fetch_all(query)

    async def get_by_name(self, piano_name: str):

        query = select(_PIANI).where(_PIANI.c.nome == piano_name)
        return await self._fetch_all(query)

    async def get_by_durata(self, durata: int):

        query = select(_PIANI).where(_PIANI.c.durata == durata)
        return await self._fetch_all(query)

    async def get_durata_range(self, min_start: int, max_end: int):

        query = select(_PIANI).where(_PIANI.c.durata >= min_start, _PIANI.c.durata <= max_end).order_by(_PIANI.c.durata, _PIANI.c.costo, _PIANI.c.id)
        return await self._fetch_all(query)

    async def get_costo_range(self, min_price: int, max_price: int):

        if min_price < 0 or max_price < 0:
            raise ValueError("values cannot be negative")

        if max_price < min_price:
            raise ValueError("max price cannot be less than min price")

        query = select(_PIANI).where(_PIANI.c.costo >= min_price, _PIANI.c.durata <= max_price).order_by(_PIANI.c.costo, _PIANI.c.durata, _PIANI.c.id)
        return await self._fetch_all(query)

    # ================================= WRITE ==============================

    async def add(self, piano: Piano):

        statement = insert(_PIANI).values(**self._to_row(piano))

        self._write(statement, piano)


    async def update(self, piano: Piano):

        statement = update(_PIANI).where(_PIANI.c.id == piano.get_id()).values(**self._to_row(piano, for_update=True))

        self._write(statement, piano)


    async def delete(self, piano_id: UUID):

        statement = delete(_PIANI).where(_PIANI.c.id == piano_id)

        self._write(statement)


    # ================================= UTILS ==============================

    async def _fetch_one(self, query):

        result = (await self._session.execute(query)).one_or_none()
        return Piano.from_db(dict(result))


    async def _fetch_all(self, query):

        results = (await self._session.execute(query)).mappings().all()
        return [Piano.from_db(dict(result)) for result in results]


    async def _write(self, statement):

        try:
            async with self._session.begin_nested():
                self._session.execute(statement)
        except IntegrityError as error:
            domain_error = self._domain_error(error)
            if domain_error is None:
                raise
            raise domain_error from error

    @staticmethod
    def _domain_error(error: IntegrityError, piano: Piano) -> Optional[Exception]:

        constraint = _constraint_name(error)
        
        if constraint == _PIANO_PRIMARY_KEY:
            return PianoAlreadyPresent(piano.get_id())
        
        return None


    @staticmethod
    def _to_row(piano: Piano, for_update=False):

        d = {
            "nome": piano.get_nome(),
            "descrizione": piano.get_descrizione(),
            "durata": piano.get_durata(),
            "costo": piano.get_costo()
        }

        if not for_update:
            d["id"] = piano.get_id()

        return d
