from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.engine import URL, make_url

from typing import Union


# Nomi dei vincoli deterministici: Alembic li genera uguali su ogni macchina,
# e il repository puo' riconoscere quale vincolo e' scattato dal suo nome.
NAMING_CONVENTION = {
    "pk": "pk_%(table_name)s",                                                # CHIAVE PRIMARIA
    "uq": "uq_%(table_name)s_%(column_0_name)s",                              # VALORE UNICO
    "ix": "ix_%(table_name)s_%(column_0_name)s",                              # INDICE
    "ck": "ck_%(table_name)s_%(constraint_name)s",                            # CHECK (NOT EMPTY)
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",      # FOREIGN KEY
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


# sslmode di libpq (stringhe di Neon, Supabase, Heroku...) -> parametro ssl di asyncpg.
_SSLMODE_TO_ASYNCPG = {
    "disable": "disable", "allow": "allow", "prefer": "prefer",
    "require": "require", "verify-ca": "verify-ca", "verify-full": "verify-full",
}
# Parametri di libpq che asyncpg non conosce e rifiuterebbe.
_LIBPQ_ONLY = ("sslmode", "channel_binding")


def to_asyncpg_url(url: Union[str, URL]) -> URL:
    """Rende utilizzabile con asyncpg un URL PostgreSQL copiato da un provider.
 
    - postgres:// o postgresql:// (driver psycopg2 implicito) -> postgresql+asyncpg://
    - ?sslmode=require -> ?ssl=require; channel_binding viene rimosso
    """
    url = make_url(url)
    if url.get_backend_name() not in ("postgresql", "postgres"):
        raise ValueError(f"atteso un URL PostgreSQL, ricevuto {url.get_backend_name()!r}")
 
    query = dict(url.query)
    sslmode = query.get("sslmode")
    for key in _LIBPQ_ONLY:
        query.pop(key, None)
    if sslmode is not None and "ssl" not in query:
        query["ssl"] = _SSLMODE_TO_ASYNCPG.get(sslmode, sslmode)
 
    return url.set(drivername="postgresql+asyncpg", query=query)
 
 
def make_engine(url: str, **options) -> AsyncEngine:
    """Engine asincrono. url nella forma postgresql+asyncpg://utente:password@host:porta/db.
 
    pool_pre_ping: una connessione chiusa dal server (riavvio, timeout) viene
    scartata prima dell'uso invece di far fallire la richiesta.
    """
    return create_async_engine(to_asyncpg_url(url), pool_pre_ping=True, **options)
 
 
def make_session_factory(engine: AsyncEngine) -> "async_sessionmaker[AsyncSession]":
    """expire_on_commit=False: dopo il commit gli oggetti restano leggibili
    senza una nuova query, necessario con le sessioni asincrone."""
    return async_sessionmaker(engine, expire_on_commit=False)
