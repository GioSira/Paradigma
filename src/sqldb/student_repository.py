from src.sqldb.tables import StudentRow, CorsoRow, UserRow, ISCRIZIONE
from src.sqldb.user_repository import UserRepository, _constraint_name

from src.roles.student import Student
from src.corsi.corso import Corso

from typing import List, Optional
from uuid import UUID

from sqlalchemy import delete, func, insert, select, update, and_, exists, literal, true
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert
from src.sqldb.errors import UserAlreadyExists, UserNotFound, CorsoNotFound


_USERS = UserRow.__table__
_STUDENTS = StudentRow.__table__

_CORSI = CorsoRow.__table__


_USERS_JOIN_STUDENTS = _USERS.join(_STUDENTS, _USERS.c.user_id == _STUDENTS.c.user_id)

# "questo utente e' uno studente": protegge update e delete sugli altri utenti.
_IS_STUDENT = _USERS.c.user_id.in_(select(_STUDENTS.c.user_id))


_STUDENT_PRIMARY_KEY = "pk_student"




class StudentRepository(UserRepository):

    def __init__(self, session):
        super().__init__(session)


    # ================================= GENERAL FUNCTION ==============================
    
    def _select(self):

        return (
            select(_USERS).select_from(_USERS_JOIN_STUDENTS)
        )


    def _corsi_order(self):

        return (
            (_CORSI.c.data_inizio, _CORSI.c.title, _CORSI.c.id) 
        )
    
    
    def _from_db(self, row: dict) -> Student:
        return Student.from_db(row)


    # ================================= WRITE ==============================

    async def add(self, student: Student):

        student_id = student.get_user_id()

        statements = (
            insert(_USERS).values(**self._to_row(student)),
            insert(_STUDENTS).values(user_id=student_id)
        )

        await self._write_all(student, *statements)


    async def update(self, student: Student):

        student_id = student.get_user_id()

        user_values = self._to_row(student, for_update=True)
        student_values = self._student_to_row(student)

        statements = (
            update(_USERS).values(**user_values).where(_USERS.c.user_id == student_id, _IS_STUDENT),
            update(_STUDENTS).values(**student_values).where(_STUDENTS.c.user_id == student_id)
        )

        results = await self._write_all(
            student,
            *statements
        )

        for r in results:
            if r.rowcount == 0:
                raise UserNotFound(student_id)



    async def delete(self, student_id: UUID) -> bool:

        statement = delete(_USERS).where(_USERS.c.user_id == student_id, _IS_STUDENT)

        result = await self._session.execute(statement)

        return result.rowcount == 1

    # ================================= READ ==============================

    async def get_all(self):
    
        query = self._select().order_by(*self._order())
        results = await self._fetch_all(query)

        return results
    
    
    async def get_by_id(self, student_id: UUID):
    
        query = self._select().where(_STUDENTS.c.user_id == student_id)
        result = await self._fetch_one(query)

        return result


    async def get_all_corsi(self, student_id: UUID):

        query = (
            select(_CORSI)
            .join(ISCRIZIONE, ISCRIZIONE.c.corso_id == _CORSI.c.id)
            .where(ISCRIZIONE.c.student_id == student_id)
            .order_by(*self._corsi_order())
        )

        rows = (await self._session.execute(*query)).mappings().all()

        return [Corso.from_db(row) for row in rows]


    async def get_enrolled_students(self, corso_id: UUID):

        query = (
            self._select()
            .join(ISCRIZIONE, ISCRIZIONE.c.student_id == _STUDENTS.c.user_id)
            .where(ISCRIZIONE.c.corso_id == corso_id)
            .order_by(*self._order())
        )

        return await self._fetch_all(*query)


    # ================================= ISCRIZIONI ==============================

    async def enroll(self, student_id: UUID, corso_id: UUID) -> bool:

        statement = (
            pg_insert(ISCRIZIONE).from_select(
                ["student_id", "corso_id"],
                select(_STUDENTS.c.user_id, _CORSI.c.id)
                .select_from(_STUDENTS.join(_CORSI, true())) # scan su tutte le coppie che hanno gli id cercati
                .where(_STUDENTS.c.user_id == student_id, _CORSI.c.id == corso_id)
            )
            .on_conflict_do_nothing()
        )

        result = await self._session.execute(*statement)
        if result == 1:
            return True

        if await self._is_enrolled(student_id, corso_id):
            return False

        if await self.get_by_id(student_id) is None:
            raise UserNotFound(student_id)

        raise CorsoNotFound(corso_id)


    async def disenroll(self, student_id: UUID, corso_id: UUID):

        statement = (
            delete(ISCRIZIONE).where(ISCRIZIONE.c.student_id == student_id, ISCRIZIONE.c.corso_id == corso_id)
        )

        result = await self._session.execute(*statement)

        return result.rowcount == 1


    async def _is_enrolled(self, student_id: UUID, corso_id: UUID):

        query = (
            self._select()
            .join(ISCRIZIONE, ISCRIZIONE.c.student_id == student_id)
            .where(ISCRIZIONE.c.corso_id == corso_id)
            .order_by(*self._corsi_order())
        )

        result = await self._session.execute(*query)

        return result.first() is not None


    # ================================= UTILS ==============================

    @staticmethod
    def _domain_error(error: IntegrityError, user: Student):

        if _constraint_name(error) == _STUDENT_PRIMARY_KEY:
            return UserAlreadyExists(user.get_user_id())

        return UserRepository._domain_error(error, user) 


    