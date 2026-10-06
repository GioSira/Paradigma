"""Test di integrazione di UserRepository su PostgreSQL reale.

Richiedono TEST_DATABASE_URL (nell'ambiente o nel .env), per esempio
    postgresql+asyncpg://postgres:postgres@localhost:5432/paradigma_test
Il database deve esistere; le tabelle vengono create dai test. Senza la
variabile i test vengono saltati.

Ogni test gira in una transazione annullata alla fine: il database resta vuoto.
NON puntare TEST_DATABASE_URL a un database con dati veri: all'avvio le
tabelle vengono ricreate.
"""
import asyncio
import os
import string
import uuid
from datetime import datetime, timezone

import pytest
from argon2 import PasswordHasher
from dotenv import load_dotenv
from hypothesis import Phase, given, settings, strategies as st
from sqlalchemy import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

import src.roles.user as user_module
from src.roles.user import User
from src.sqldb.base import Base, make_engine
from src.sqldb.errors import EmailAlreadyRegistered, UserAlreadyExists, UserNotFound
from src.sqldb.user_repository import UserRepository
from src.sqldb.tables import UserRow

load_dotenv()
DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DATABASE_URL, reason="TEST_DATABASE_URL non impostata"),
]

# argon2 veloce, come in test_user.py: i costi di produzione renderebbero lenti i test.
FAST_HASHER = PasswordHasher(time_cost=1, memory_cost=8, parallelism=1)
# Ogni esempio apre una connessione al database: niente shrinking, che su un
# test fallito ripeterebbe decine di esempi a vuoto (lo stesso vale per Pinecone).
DB = settings(max_examples=10, deadline=None,
              phases=[Phase.explicit, Phase.reuse, Phase.generate])
PASSWORD = "una-password-lunga-e-sicura"
NOW = datetime(2024, 3, 9, 14, 5, 6, tzinfo=timezone.utc)

names = st.text(min_size=1, max_size=50).filter(lambda s: "\x00" not in s)  # PostgreSQL rifiuta NUL
emails = st.builds(
    "{}@{}.{}".format,
    st.text(alphabet=string.ascii_lowercase + string.digits, min_size=1, max_size=20),
    st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=15),
    st.sampled_from(["it", "com", "org"]),
)
passwords = st.text(min_size=15, max_size=64)
dates = st.datetimes(min_value=datetime(1970, 1, 1), max_value=datetime(2100, 1, 1),
                     timezones=st.just(timezone.utc))


@pytest.fixture(scope="session", autouse=True)
def fast_password_hashing():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(user_module, "PH", FAST_HASHER)
        yield


@pytest.fixture(scope="session", autouse=True)
def schema():
    async def recreate():
        engine = make_engine(DATABASE_URL)
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.drop_all)
            await connection.run_sync(Base.metadata.create_all)
        await engine.dispose()

    asyncio.run(recreate())


def in_transaction(test):
    """Esegue test(repository, session) in una transazione annullata alla fine.

    Un engine per chiamata: le connessioni asyncpg appartengono all'event loop
    che le ha create, e ogni asyncio.run ne crea uno nuovo.
    """
    async def run():
        engine = make_engine(DATABASE_URL)
        try:
            async with engine.connect() as connection:
                transaction = await connection.begin()
                session = AsyncSession(bind=connection, expire_on_commit=False)
                try:
                    return await test(UserRepository(session), session)
                finally:
                    await session.close()
                    await transaction.rollback()
        finally:
            await engine.dispose()

    return asyncio.run(run())


def make_user(user_id=None, name="Mario", surname="Rossi", email=None,
              password=PASSWORD, created_at=NOW):
    if user_id is None:
        user_id = uuid.uuid4()
    if email is None:
        email = f"{user_id.hex}@example.com"   # email unica per default
    return User.create(user_id=user_id, name=name, surname=surname, email=email,
                       password=password, created_at=created_at)


def ids(users):
    return sorted(user.get_user_id() for user in users)


# ============================================================ lettura

class TestAddAndGetById:

    @DB
    @given(name=names, surname=names, email=emails, password=passwords, created_at=dates)
    def test_round_trip_keeps_every_field(self, name, surname, email, password, created_at):
        user = make_user(name=name, surname=surname, email=email, password=password,
                         created_at=created_at)

        async def test(repository, session):
            await repository.add(user)
            return await repository.get_by_id(user.get_user_id())

        loaded = in_transaction(test)

        assert loaded == user
        assert (loaded.get_name(), loaded.get_surname(), loaded.get_email()) == (name, surname, email)
        assert loaded.get_created_at() == created_at
        assert loaded.get_created_at().tzinfo is not None
        # l'hash torna identico: nessun doppio hash, il login funziona
        assert loaded.get_password_hash() == user.get_password_hash()
        assert loaded.verify_password(password)

    def test_missing_user_is_none(self):
        async def test(repository, session):
            return await repository.get_by_id(uuid.uuid4())

        assert in_transaction(test) is None


class TestGetByEmail:

    @DB
    @given(email=emails)
    def test_lookup_ignores_case(self, email):
        user = make_user(email=email)

        async def test(repository, session):
            await repository.add(user)
            return await repository.get_by_email(email.upper())

        assert in_transaction(test) == user

    def test_missing_email_is_none(self):
        async def test(repository, session):
            return await repository.get_by_email("nessuno@example.com")

        assert in_transaction(test) is None


class TestGetByNameAndSurname:

    @DB
    @given(name=names, other=names, count=st.integers(min_value=1, max_value=4))
    def test_get_by_name_returns_every_match_and_only_those(self, name, other, count):
        matching = [make_user(name=name) for _ in range(count)]
        different = make_user(name=other)

        async def test(repository, session):
            for user in matching + [different]:
                await repository.add(user)
            return await repository.get_by_name(name)

        found = in_transaction(test)

        expected = matching + ([different] if other == name else [])
        assert ids(found) == ids(expected)
        assert all(isinstance(user, User) for user in found)

    @DB
    @given(surname=names, other=names, count=st.integers(min_value=1, max_value=4))
    def test_get_by_surname_returns_every_match_and_only_those(self, surname, other, count):
        matching = [make_user(surname=surname) for _ in range(count)]
        different = make_user(surname=other)

        async def test(repository, session):
            for user in matching + [different]:
                await repository.add(user)
            return await repository.get_by_surname(surname)

        found = in_transaction(test)

        expected = matching + ([different] if other == surname else [])
        assert ids(found) == ids(expected)

    def test_no_match_is_an_empty_list(self):
        async def test(repository, session):
            return (await repository.get_by_name("Nessuno"),
                    await repository.get_by_surname("Nessuno"))

        assert in_transaction(test) == ([], [])


# ================================================================ add

class TestAddConflicts:

    @DB
    @given(email=emails)
    def test_same_email_in_another_case_is_rejected(self, email):
        # pydantic normalizza il dominio in minuscolo, la parte locale no:
        # "Mario@EXAMPLE.COM" diventa "MARIO@example.com".
        duplicate = make_user(email=email.upper())

        async def test(repository, session):
            await repository.add(make_user(email=email))
            with pytest.raises(EmailAlreadyRegistered) as raised:
                await repository.add(duplicate)
            return raised.value

        error = in_transaction(test)

        assert error.email == duplicate.get_email()
        assert duplicate.get_email() not in str(error)  # niente dati personali nel messaggio

    def test_same_id_is_rejected(self):
        user_id = uuid.uuid4()

        async def test(repository, session):
            await repository.add(make_user(user_id=user_id, email="a@example.com"))
            with pytest.raises(UserAlreadyExists):
                await repository.add(make_user(user_id=user_id, email="b@example.com"))

        in_transaction(test)

    def test_failed_write_does_not_break_the_transaction(self):
        # Il SAVEPOINT annulla solo la scrittura fallita: le altre restano.
        first, duplicate, third = (make_user(email="uno@example.com"),
                                   make_user(email="uno@example.com"),
                                   make_user(email="tre@example.com"))

        async def test(repository, session):
            await repository.add(first)
            with pytest.raises(EmailAlreadyRegistered):
                await repository.add(duplicate)
            await repository.add(third)
            return [await repository.get_by_id(u.get_user_id()) for u in (first, duplicate, third)]

        assert in_transaction(test) == [first, None, third]


# ============================================================= update

class TestUpdate:

    @DB
    @given(name=names, email=emails, password=passwords)
    def test_changes_are_saved(self, name, email, password):
        user = make_user()

        async def test(repository, session):
            await repository.add(user)
            user.set_name(name)
            user.set_email(email)
            user.set_password(password)
            await repository.update(user)
            return await repository.get_by_id(user.get_user_id())

        loaded = in_transaction(test)

        assert (loaded.get_name(), loaded.get_email()) == (name, email)
        assert loaded.verify_password(password)
        assert loaded.get_created_at() == NOW

    def test_missing_user_raises(self):
        async def test(repository, session):
            await repository.update(make_user())

        with pytest.raises(UserNotFound):
            in_transaction(test)

    def test_email_of_another_user_is_rejected(self):
        mario, luigi = make_user(email="mario@example.com"), make_user(email="luigi@example.com")

        async def test(repository, session):
            await repository.add(mario)
            await repository.add(luigi)
            luigi.set_email("MARIO@example.com")
            with pytest.raises(EmailAlreadyRegistered):
                await repository.update(luigi)
            return await repository.get_by_id(luigi.get_user_id())

        assert in_transaction(test).get_email() == "luigi@example.com"


# ============================================================= delete

class TestDelete:

    def test_delete_is_idempotent(self):
        user = make_user()

        async def test(repository, session):
            await repository.add(user)
            first = await repository.delete(user.get_user_id())
            second = await repository.delete(user.get_user_id())
            return first, second, await repository.get_by_id(user.get_user_id())

        assert in_transaction(test) == (True, False, None)


# ======================================================= vincoli nel DB

class TestDatabaseConstraints:
    """Le regole valgono anche per chi scrive senza passare dal repository."""

    @pytest.mark.parametrize("column, value", [
        ("password_hash", "password-in-chiaro"),
        ("name", ""),
        ("surname", ""),
    ])
    def test_invalid_rows_are_rejected(self, column, value):
        row = {"user_id": uuid.uuid4(), "name": "Mario", "surname": "Rossi",
               "email": "mario@example.com", "password_hash": FAST_HASHER.hash(PASSWORD),
               "created_at": NOW}
        row[column] = value

        async def test(repository, session):
            await session.execute(insert(UserRow.__table__).values(**row))

        with pytest.raises(IntegrityError):
            in_transaction(test)
