"""Configurazione comune dei test.

Profili di Hypothesis e fixture condivise dai test sul database
(test_*_repository.py).

Il database di test (TEST_DATABASE_URL) viene ricreato UNA volta per sessione
di pytest, non una volta per file.

Due sessioni di pytest sullo stesso database (per esempio VS Code che lancia
i test mentre li lanci anche dal terminale) si cancellerebbero le tabelle a
vicenda: e' il "deadlock detected" su drop_all. Per evitarlo ogni sessione
prende un advisory lock di PostgreSQL e lo tiene fino alla fine: la seconda
sessione aspetta che la prima finisca, invece di rompere i suoi test.
"""

import asyncio
import os

import pytest
from argon2 import PasswordHasher
from dotenv import load_dotenv
from hypothesis import HealthCheck, settings
from sqlalchemy import text

import src.roles.user as user_module
from src.sqldb.base import Base, make_engine

load_dotenv()


@pytest.fixture(scope="session", autouse=True)
def set_test_env():
    pass


# Profili di hypothesis, scelti con la variabile HYPOTHESIS_PROFILE:
#   dev (default)  veloce, per lo sviluppo
#   ci             piu' esempi e seme fisso: una failure in CI si riproduce
#                  identica rilanciando, invece di dipendere dal caso
# I test che fissano le proprie impostazioni con @settings(...) (quelli su
# Pinecone e sul database) non sono toccati dai profili.
settings.register_profile("dev", max_examples=100)
settings.register_profile(
    "ci",
    max_examples=500,
    derandomize=True,
    deadline=None,
    print_blob=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


# ============================================================ database di test

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

# Numero qualsiasi, uguale per tutte le sessioni: identifica "chi usa il database di test".
_TEST_DATABASE_LOCK = 7_391_004

# argon2 veloce: i costi di produzione renderebbero lenti i test.
FAST_HASHER = PasswordHasher(time_cost=1, memory_cost=8, parallelism=1)


@pytest.fixture(scope="session")
def fast_password_hashing():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(user_module, "PH", FAST_HASHER)
        yield


@pytest.fixture(scope="session")
def database(fast_password_hashing):
    """Lock esclusivo sul database di test + tabelle ricreate da zero.

    Il lock vive su una connessione dedicata, in un event loop dedicato: le
    connessioni asyncpg appartengono al loop che le ha create, e i test ne
    creano uno nuovo a ogni asyncio.run.
    """
    if not DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL non impostata")

    loop = asyncio.new_event_loop()
    engine = make_engine(DATABASE_URL)

    async def acquire():
        connection = await engine.connect()
        # Session-level: resta anche dopo il commit, fino a unlock o chiusura.
        await connection.execute(text("SELECT pg_advisory_lock(:key)"), {"key": _TEST_DATABASE_LOCK})
        await connection.commit()        # niente transazione aperta mentre girano i test

        async with engine.begin() as schema:
            await schema.run_sync(Base.metadata.drop_all)
            await schema.run_sync(Base.metadata.create_all)
        return connection

    async def release(connection):
        try:
            await connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": _TEST_DATABASE_LOCK})
            await connection.commit()
        finally:
            await connection.close()
            await engine.dispose()

    lock_connection = loop.run_until_complete(acquire())
    try:
        yield DATABASE_URL
    finally:
        loop.run_until_complete(release(lock_connection))
        loop.close()