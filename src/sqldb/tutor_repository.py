from src.sqldb.tables import TutorRow, UserRow
from src.sqldb.user_repository import UserRepository, _constraint_name
from src.roles.tutor import Tutor
from src.roles.user import User

from typing import List
from uuid import UUID

from sqlalchemy import delete, func, insert, select, update, and_, exists, literal
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from src.sqldb.errors import EmailAlreadyRegistered, UserAlreadyExists, UserNotFound




_TUTORS = TutorRow.__table__
_USERS = UserRow.__table__

_USER_JOIN_TUTOR = _USERS.join(_TUTORS, _USERS.c.user_id == _TUTORS.c.user_id)

_TUTOR_PRIMARY_KEY = "pk_tutor"




class TutorRepository(UserRepository):

    def __init__(self, session):
        super().__init__(session)


    # ================================= GENERAL FUNCTION ==============================

    def _select(self):

        return (
            select(_USERS, _TUTORS.c.title, _TUTORS.c.bio, _TUTORS.c.subjects).select_from(_USER_JOIN_TUTOR)
        )


    def _from_db(self, row: dict) -> Tutor:
        return Tutor.from_db(row)


    # ================================= SEARCH ==============================

    async def get_all(self):

        query = self._select().order_by(*self._order())
        results = await self._fetch_all(query)

        return results


    async def get_by_id(self, tutor_id: UUID):

        query = self._select().where(_TUTORS.c.user_id == tutor_id)
        result = await self._fetch_one(query)

        return result


    async def get_by_title(self, tutor_title: str):

        query = self._select().where(_TUTORS.c.title == tutor_title).order_by(*self._order())
        result = await self._fetch_all(query)

        return result


    async def get_by_subject(self, subject: str):

        return await self.get_by_subjects([subject])


    async def get_by_subjects(self, subjects: List[str], match_all: bool = False):

        # match_all: True per cercare i tutor che le insegnano tutte, False almeno una

        if not subjects or len(subjects) == 0:
            return []

        wanted = list(dict.fromkeys(subjects))
        if match_all:
            condition = and_(*(self._teaches_any([subject]) for subject in wanted))
        else:
            condition = self._teaches_any(wanted)
        query = self._select().where(condition).order_by(*self._order())

        return await self._fetch_all(query)


    @staticmethod
    def _teaches_any(subjects: List[str]):

        subject = func.unnest(_TUTORS.c.subjects).column_valued("subject")
        return exists(
            select(1).where(func.lower(subject).in_(
                [func.lower(literal(s)) for s in subjects]
            ))
        )


    # ================================= WRITE ==============================

    async def add(self, tutor: Tutor):

        tutor_id = tutor.get_user_id()

        statements = (
            insert(_USERS).values(**self._to_row(tutor)),
            insert(_TUTORS).values(user_id=tutor_id, **self._tutor_to_row(tutor))
        )

        await self._write_all(tutor, *statements)


    async def update(self, tutor: Tutor):

        tutor_id = tutor.get_user_id()

        user_values = self._to_row(tutor, for_update=True)
        tutor_values = self._tutor_to_row(tutor)
        statements = (
            update(_USERS).values(**user_values).where(_USERS.c.user_id == tutor_id,
                                                       _USERS.c.user_id.in_(select(_TUTORS.c.user_id))),
            update(_TUTORS).values(**tutor_values).where(_TUTORS.c.user_id == tutor_id)
        )

        results = await self._write_all(
            tutor,
            *statements
        )

        for r in results:
            if r.rowcount == 0:
                raise UserNotFound(tutor_id)



    async def delete(self, tutor_id: UUID) -> bool:

        statement = delete(_USERS).where(_USERS.c.user_id == tutor_id, _USERS.c.user_id.in_(select(_TUTORS.c.user_id)))

        result = await self._session.execute(statement)

        return result.rowcount == 1




    # ================================= UTILS ==============================

    @staticmethod
    def _tutor_to_row(tutor: Tutor):

        d = {
            "bio":tutor.get_bio(),
            "subjects": tutor.get_subjects(),
            "title": tutor.get_title()
        }

        return d


    @staticmethod
    def _domain_error(error: IntegrityError, user: Tutor):

        if _constraint_name(error) == _TUTOR_PRIMARY_KEY:
            return UserAlreadyExists(user.get_user_id())

        return UserRepository._domain_error(error, user)
        