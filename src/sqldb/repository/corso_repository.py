from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update, delete, literal, true
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.sqldb.tables import CorsoRow, CORSO_DOCENTE, TutorRow
from src.corsi.corso import Corso
from src.sqldb.errors import UserNotFound, IntegrityError, _constraint_name, CorsoAlreadyExists, CorsoNotFound, UltimoDocente

from uuid import UUID
from typing import Optional, List, Iterable

_CORSI = CorsoRow.__table__
_TUTORS = TutorRow.__table__
_CORSI_PRIMARY_KEY = "pk_corso"


class CorsoRepository:
    """Corsi e loro docenti. Regola: un corso ha sempre almeno un docente."""

    def __init__(self, session: AsyncSession):

        self._session = session

    # ================================= READ ===============================

    async def get_by_id(self, corso_id: UUID) -> Optional[Corso]:

        query = select(_CORSI).where(_CORSI.c.id == corso_id)
        return await self._fetch_one(query)

    async def get_by_tutor(self, tutor_id: UUID) -> List[Corso]:

        query = (
            select(_CORSI)
            .join(CORSO_DOCENTE, CORSO_DOCENTE.c.corso_id == _CORSI.c.id)
            .where(CORSO_DOCENTE.c.tutor_id == tutor_id)
            .order_by(*self._order())
        )
        return await self._fetch_all(query)

    async def get_docenti_ids(self, corso_id: UUID) -> List[UUID]:

        query = (
            select(CORSO_DOCENTE.c.tutor_id)
            .where(CORSO_DOCENTE.c.corso_id == corso_id)
            .order_by(CORSO_DOCENTE.c.tutor_id)
        )
        return list((await self._session.execute(query)).scalars())

    # ================================= WRITE ==============================

    async def add(self, corso: Corso, tutor_ids: Optional[Iterable[UUID]] = None) -> None:
        """Salva il corso con i suoi docenti, tutto o niente.

        ValueError se non ci sono docenti, UserNotFound se uno non e' un tutor,
        CorsoAlreadyExists se l'id e' gia' usato."""

        corso_id = corso.get_id()
        tutor_ids = list(dict.fromkeys(tutor_ids or []))    # niente doppioni, ordine conservato

        if not tutor_ids:
            raise ValueError("Un corso deve avere almeno un docente")

        try:
            # SAVEPOINT: se un docente non esiste, sparisce anche il corso.
            async with self._session.begin_nested():

                await self._session.execute(insert(_CORSI).values(**self._to_row(corso)))

                result = await self._session.execute(
                    insert(CORSO_DOCENTE).from_select(
                        ["corso_id", "tutor_id"],
                        select(literal(corso_id), _TUTORS.c.user_id)
                        .where(_TUTORS.c.user_id.in_(tutor_ids))
                    )
                )

                if result.rowcount != len(tutor_ids):
                    raise UserNotFound(await self._first_missing_tutor(tutor_ids))

        except IntegrityError as error:
            if _constraint_name(error) == _CORSI_PRIMARY_KEY:
                raise CorsoAlreadyExists(corso_id) from error
            raise


    async def update(self, corso: Corso) -> None:
        """Aggiorna i dati del corso; i docenti non cambiano."""

        corso_id = corso.get_id()
        statement = update(_CORSI).where(_CORSI.c.id == corso_id).values(**self._to_row(corso, for_update=True))

        async with self._session.begin_nested():
            result = await self._session.execute(statement)

        if result.rowcount == 0:
            raise CorsoNotFound(corso_id)


    async def delete(self, corso_id: UUID) -> bool:

        # ON DELETE CASCADE: spariscono anche docenti e iscrizioni del corso.
        statement = delete(_CORSI).where(_CORSI.c.id == corso_id)
        result = await self._session.execute(statement)

        return result.rowcount == 1


    async def add_docente(self, corso_id: UUID, tutor_id: UUID) -> bool:
        """True se il docente e' nuovo, False se c'era gia'."""

        statement = (
            pg_insert(CORSO_DOCENTE)
            .from_select(
                ["corso_id", "tutor_id"],
                # Al massimo una riga: entrambe filtrate sulla chiave primaria.
                select(_CORSI.c.id, _TUTORS.c.user_id)
                .select_from(_CORSI.join(_TUTORS, true()))
                .where(_CORSI.c.id == corso_id, _TUTORS.c.user_id == tutor_id),
            )
            .on_conflict_do_nothing()
        )

        result = await self._session.execute(statement)
        if result.rowcount == 1:
            return True

        if tutor_id in await self.get_docenti_ids(corso_id):
            return False

        if await self.get_by_id(corso_id) is None:
            raise CorsoNotFound(corso_id)

        raise UserNotFound(tutor_id)


    async def remove_docente(self, corso_id: UUID, tutor_id: UUID) -> bool:
        """True se tolto, False se non insegnava nel corso.
        UltimoDocente se e' l'unico rimasto."""

        # 1. Lock sul corso PRIMA di contare i docenti: due rimozioni in
        #    parallelo si mettono in fila, e la seconda conta i docenti dopo
        #    che la prima ha finito. Senza, entrambe vedrebbero 2 docenti e il
        #    corso resterebbe senza nessuno.
        locked = await self._session.execute(
            select(_CORSI.c.id).where(_CORSI.c.id == corso_id).with_for_update()
        )
        if locked.first() is None:
            raise CorsoNotFound(corso_id)

        docenti = await self.get_docenti_ids(corso_id)

        # 2. Prima "insegna qui?", poi "e' l'ultimo?": togliere chi non
        #    insegna non e' un errore, anche se il docente e' uno solo.
        if tutor_id not in docenti:
            return False

        if len(docenti) == 1:
            raise UltimoDocente(corso_id)

        await self._session.execute(delete(CORSO_DOCENTE).where(
            CORSO_DOCENTE.c.corso_id == corso_id, CORSO_DOCENTE.c.tutor_id == tutor_id)
        )

        return True

    # ================================= UTILS ==============================

    async def _first_missing_tutor(self, tutor_ids: List[UUID]) -> UUID:

        query = select(_TUTORS.c.user_id).where(_TUTORS.c.user_id.in_(tutor_ids))
        found = set((await self._session.execute(query)).scalars())

        return next(tutor_id for tutor_id in tutor_ids if tutor_id not in found)

    async def _fetch_one(self, query) -> Optional[Corso]:

        row = (await self._session.execute(query)).mappings().one_or_none()
        return Corso.from_db(dict(row)) if row else None

    async def _fetch_all(self, query) -> List[Corso]:

        rows = (await self._session.execute(query)).mappings().all()
        return [Corso.from_db(dict(row)) for row in rows]

    @staticmethod
    def _to_row(corso: Corso, for_update=False):
        d = {
            "title": corso.get_title(),
            "descrizione": corso.get_descriziione(),
            "data_inizio": corso.get_data_inizio(),
            "data_fine": corso.get_data_fine()
        }

        if not for_update:
            d["id"] = corso.get_id()

        return d

    @staticmethod
    def _order():
        return (_CORSI.c.data_inizio, _CORSI.c.title, _CORSI.c.id)