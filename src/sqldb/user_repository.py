from typing import Any, Dict, Optional, List
from uuid import UUID
 
from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
 
from src.sqldb.tables import USERS_EMAIL_UNIQUE, UserRow
from src.sqldb.errors import EmailAlreadyRegistered, UserAlreadyExists, UserNotFound
from src.roles.user import User
 
 
_USERS = UserRow.__table__
_USERS_PRIMARY_KEY = "pk_users"




def _constraint_name(error: IntegrityError):

    original = error.orig
    # asyncpg: l'eccezione originale e' la causa di quella adattata da SQLAlchemy.
    name = getattr(getattr(original, "__cause__", None), "constraint_name", None)
    
    if name is None:
        # psycopg (versione sincrona): il nome sta nella diagnostica.
        name = getattr(getattr(original, "diag", None), "constraint_name", None)

    return name


class UserRepository:

    def __init__(self, session:AsyncSession):

        self._session = session


    # ================================= GENERAL FUNCTION ==============================

    def _select(self):

        return select(_USERS)


    def _order(self):

        return (
            _USERS.c.surname, _USERS.c.name, _USERS.c.user_id
        )


    # ============================ LETTURA ========================

    async def get_by_id(self, user_id:UUID) -> Optional[User]:

        query = self._select().where(_USERS.c.user_id == user_id)
        return await self._fetch_one(query)


    async def get_by_email(self, user_email: str) -> Optional[User]:

        query = self._select().where(func.lower(_USERS.c.email) == user_email.lower())
        return await self._fetch_one(query)


    async def get_by_name(self, user_name: str) -> List[User]:

        query = self._select().where(_USERS.c.name == user_name).order_by(*self._order())
        return await self._fetch_all(query)


    async def get_by_surname(self, user_surname: str) -> List[User]:
    
        query = self._select().where(_USERS.c.surname == user_surname).order_by(*self._order())
        return await self._fetch_all(query)


    
    # ============================ SCRITTURA ========================

    async def add(self, user: User) -> None:

        statement = insert(_USERS).values(**self._to_row(user))
        await self._write(statement, user)


    async def update(self, user: User) -> None:

        values = self._to_row(user, for_update=True)
        statement = update(_USERS).where(_USERS.c.user_id == user.get_user_id()).values(**values)

        result = await self._write(statement, user)

        if result.rowcount == 0:
            raise UserNotFound(user.get_user_id())


    async def delete(self, user_id: UUID) -> bool:

        statement = delete(_USERS).where(_USERS.c.user_id == user_id)
        result = await self._session.execute(statement)

        return result.rowcount == 1



    # ============================ UTILS ========================


    def _from_db(self, row: dict) -> User:
        """Converte una riga nell'oggetto di dominio: le sottoclassi lo ridefiniscono."""
        return User.from_db(row)


    async def _fetch_one(self, query) -> Optional[User]:

        row = (await self._session.execute(query)).mappings().one_or_none()

        return self._from_db(dict(row)) if row else None


    async def _fetch_all(self, query) -> List[User]:

        rows = (await self._session.execute(query)).mappings().all()
        return [self._from_db(dict(row)) for row in rows]



    async def _write(self, query, user: User):

        result = await self._write_all(user, query)
        return result[0]



    async def _write_all(self, user: User, *quieries) -> list:

        try:
            async with self._session.begin_nested():
                return [await self._session.execute(query) for query in quieries]
        except IntegrityError as error:
            domain_error = self._domain_error(error, user)
            if domain_error is None:
                raise
            raise domain_error from error


    @staticmethod
    def _domain_error(error: IntegrityError, user: User) -> Optional[Exception]:

        constraint = _constraint_name(error)

        if constraint == USERS_EMAIL_UNIQUE:
            return EmailAlreadyRegistered(user.get_email())
        
        if constraint == _USERS_PRIMARY_KEY:
            return UserAlreadyExists(user.get_user_id())
        
        return None


    @staticmethod
    def _to_row(user: User, for_update=False) -> dict:

        user_dict = {
            "name": user.get_name(),
            "surname": user.get_surname(),
            "email": user.get_email(),
            "password_hash": user.get_password_hash(),
        }

        if not for_update:
            user_dict["user_id"] = user.get_user_id()
            user_dict["created_at"] = user.get_created_at()

        return user_dict
    