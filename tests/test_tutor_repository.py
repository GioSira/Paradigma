"""Test di integrazione di TutorRepository su PostgreSQL reale.

Richiedono TEST_DATABASE_URL (nell'ambiente o nel .env), per esempio
    postgresql+asyncpg://postgres:postgres@localhost:5432/paradigma_test
Il database deve esistere; le tabelle vengono create dai test. Senza la
variabile i test vengono saltati.

Ogni test gira in una transazione annullata alla fine: il database resta vuoto.
NON puntare TEST_DATABASE_URL a un database con dati veri: all'avvio le
tabelle vengono ricreate.

Gli input sono volutamente ostili: apici e commenti SQL, graffe e virgole
(la sintassi dei letterali array di PostgreSQL), caratteri jolly di LIKE,
lettere che si scrivono in due modi (e + accento combinante, Ohm e Omega),
testo da destra a sinistra, emoji, stringhe lunghe. In piu' una macchina a
stati esegue sequenze casuali di add/update/delete/ricerche e dopo ogni passo
confronta il database con un modello in memoria.
"""
import asyncio
import os
import string
import uuid
from datetime import datetime, timezone

import pytest
from argon2 import PasswordHasher
from dotenv import load_dotenv
from hypothesis import HealthCheck, Phase, given, settings, strategies as st
from hypothesis.stateful import (Bundle, RuleBasedStateMachine, invariant, multiple,
                                 rule)
from sqlalchemy import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

import src.roles.user as user_module
from src.roles.tutor import Tutor
from src.roles.user import User
from src.sqldb.base import Base, make_engine
from src.sqldb.errors import EmailAlreadyRegistered, UserAlreadyExists, UserNotFound
from src.sqldb.tables import TutorRow, UserRow
from src.sqldb.tutor_repository import TutorRepository
from src.sqldb.user_repository import UserRepository

load_dotenv()
DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DATABASE_URL, reason="TEST_DATABASE_URL non impostata"),
]

# argon2 veloce, come in test_user_repository.py.
FAST_HASHER = PasswordHasher(time_cost=1, memory_cost=8, parallelism=1)
# Ogni esempio apre una connessione al database: niente shrinking.
DB = settings(max_examples=10, deadline=None,
              phases=[Phase.explicit, Phase.reuse, Phase.generate])
PASSWORD = "una-password-lunga-e-sicura"
NOW = datetime(2024, 3, 9, 14, 5, 6, tzinfo=timezone.utc)
OTHER_DATE = datetime(1999, 12, 31, 23, 59, 59, tzinfo=timezone.utc)


# ============================================================ strategie

# Stringhe che rompono le query costruite a mano, i letterali array
# ('{a,b}' e' un array di due elementi se finisce nel SQL come testo), LIKE e
# i confronti che normalizzano Unicode o maiuscole senza dirlo.
TRICKY = [
    "'", '"', "\\", "\\\\", "`", ";", "--",
    "' OR '1'='1", "'); DROP TABLE tutor; --", "$$", "$1",
    "{", "}", "{}", "{a,b}", "a,b", '{"a","b"}', "{NULL}", "NULL", "null", "None",
    "%", "_", "%%", "mate%", "_atematica", "*", "[a-z]",
    " ", "  ", "\t", "\n", "\r\n", " matematica", "matematica ",
    "é", "é",            # stessa lettera, composta e scomposta
    "Ω", "Ω",        # Ohm e Omega: uguali dopo NFKC, diversi per ==
    "ß", "SS", "İ", "ı", "ﬁ", "Ǆ",
    "😀", "👩‍🏫", "‮maerts", "​", "﻿",
    "x" * 1000,
]

SAFE_CHARS = st.characters(codec="utf-8", exclude_characters="\x00")  # PostgreSQL rifiuta NUL


def nasty(max_size=40):
    return st.one_of(st.sampled_from(TRICKY),
                     st.text(alphabet=SAFE_CHARS, min_size=1, max_size=max_size))


names = nasty()
titles = nasty()
bios = st.none() | nasty(max_size=500)
# Tutor scarta le materie fatte solo di spazi e rifiuta una lista che resta
# vuota: qui si testa il repository, quindi ogni materia ha almeno un carattere
# visibile. I doppioni (anche "Matematica"/"matematica") restano: li toglie
# Tutor, e i test confrontano sempre con tutor.get_subjects(), gia' normalizzato.
subjects_text = nasty().filter(lambda s: s.strip() != "")
subject_lists = st.lists(subjects_text, min_size=1, max_size=8)
emails = st.builds(
    "{}@{}.{}".format,
    st.text(alphabet=string.ascii_lowercase + string.digits, min_size=1, max_size=20),
    st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=15),
    st.sampled_from(["it", "com", "org"]),
)
passwords = st.text(min_size=15, max_size=64)
dates = st.datetimes(min_value=datetime(1970, 1, 1), max_value=datetime(2100, 1, 1),
                     timezones=st.just(timezone.utc))


# ============================================================ fixture

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
    """Esegue test(tutors, users, session) in una transazione annullata alla fine."""
    async def run():
        engine = make_engine(DATABASE_URL)
        try:
            async with engine.connect() as connection:
                transaction = await connection.begin()
                session = AsyncSession(bind=connection, expire_on_commit=False)
                try:
                    return await test(TutorRepository(session), UserRepository(session), session)
                finally:
                    await session.close()
                    await transaction.rollback()
        finally:
            await engine.dispose()

    return asyncio.run(run())


# ============================================================ helper

def make_tutor(user_id=None, name="Mario", surname="Rossi", email=None, password=PASSWORD,
               created_at=NOW, bio=None, title="Prof.", subjects=("matematica",)):
    
    if user_id is None:
        user_id = uuid.uuid4()
    
    if email is None:
        email = f"{user_id.hex}@example.com"   # email unica per default
    
    return Tutor.create(user_id=user_id, name=name, surname=surname, email=email,
                        password=password, created_at=created_at,
                        bio=bio, tutor_title=title, subjects=list(subjects))


def make_student(user_id=None, name="Paolo", surname="Verdi", email=None):
    if user_id is None:
        user_id = uuid.uuid4()
    if email is None:
        email = f"{user_id.hex}@example.com"
    return User.create(user_id=user_id, name=name, surname=surname, email=email,
                       password=PASSWORD, created_at=NOW)


def user_fields(user):
    return (user.get_user_id(), user.get_name(), user.get_surname(), user.get_email(),
            user.get_password_hash(), user.get_created_at())


def tutor_fields(tutor):
    return user_fields(tutor) + (tutor.get_bio(), tutor.get_title(), list(tutor.get_subjects()))


def ids(users):
    return sorted(user.get_user_id() for user in users)


async def add_all(repository, users):
    for user in users:
        await repository.add(user)


# ============================================================ add e lettura

class TestRoundTrip:

    @DB
    @given(name=names, surname=names, email=emails, password=passwords, created_at=dates,
           bio=bios, title=titles, subjects=subject_lists)
    def test_every_field_survives_byte_for_byte(self, name, surname, email, password,
                                                created_at, bio, title, subjects):
        tutor = make_tutor(name=name, surname=surname, email=email, password=password,
                           created_at=created_at, bio=bio, title=title, subjects=subjects)

        async def test(tutors, users, session):
            await tutors.add(tutor)
            return (await tutors.get_by_id(tutor.get_user_id()),
                    await tutors.get_by_email(email.upper()),
                    await users.get_by_id(tutor.get_user_id()))

        by_id, by_email, as_user = in_transaction(test)

        assert isinstance(by_id, Tutor)
        # Nessuna normalizzazione: spazi, accenti scomposti, ordine e duplicati
        # delle materie tornano identici.
        assert tutor_fields(by_id) == tutor_fields(tutor)
        assert tutor_fields(by_email) == tutor_fields(tutor)
        assert by_id.get_created_at().tzinfo is not None
        assert by_id.verify_password(password)
        # Il tutor e' anche un utente qualsiasi per UserRepository.
        assert user_fields(as_user) == user_fields(tutor)

    def test_missing_tutor_is_none(self):
        async def test(tutors, users, session):
            return (await tutors.get_by_id(uuid.uuid4()),
                    await tutors.get_by_email("nessuno@example.com"))

        assert in_transaction(test) == (None, None)


class TestAddConflicts:

    @DB
    @given(email=emails)
    def test_same_email_in_another_case_is_rejected(self, email):
        first, duplicate = make_tutor(email=email), make_tutor(email=email.upper())

        async def test(tutors, users, session):
            await tutors.add(first)
            with pytest.raises(EmailAlreadyRegistered) as raised:
                await tutors.add(duplicate)
            return raised.value, await users.get_by_id(duplicate.get_user_id())

        error, leftover = in_transaction(test)

        assert error.email == duplicate.get_email()
        assert duplicate.get_email() not in str(error)  # niente dati personali nel messaggio
        assert leftover is None

    def test_same_id_is_rejected(self):
        user_id = uuid.uuid4()

        async def test(tutors, users, session):
            await tutors.add(make_tutor(user_id=user_id))
            with pytest.raises(UserAlreadyExists):
                await tutors.add(make_tutor(user_id=user_id, email="altra@example.com"))

        in_transaction(test)

    def test_failed_write_does_not_break_the_transaction(self):
        first, duplicate, third = (make_tutor(email="uno@example.com"),
                                   make_tutor(email="UNO@example.com"),
                                   make_tutor(email="tre@example.com"))

        async def test(tutors, users, session):
            await tutors.add(first)
            with pytest.raises(EmailAlreadyRegistered):
                await tutors.add(duplicate)
            await tutors.add(third)
            return [await tutors.get_by_id(t.get_user_id()) for t in (first, duplicate, third)]

        assert in_transaction(test) == [first, None, third]


# ============================================================ ricerche

class TestSearchesAreExact:
    """Le ricerche confrontano con ==: niente LIKE, niente normalizzazione."""

    @DB
    @given(titles_used=st.lists(titles, min_size=1, max_size=5), data=st.data())
    def test_get_by_title_matches_only_the_same_string(self, titles_used, data):
        catalog = [make_tutor(title=title) for title in titles_used]
        # Cerca anche titoli mai usati: "%" non deve trovare tutto.
        query = data.draw(st.sampled_from(titles_used) | titles, label="query")

        async def test(tutors, users, session):
            await add_all(tutors, catalog)
            return await tutors.get_by_title(query)

        found = in_transaction(test)

        assert ids(found) == ids(t for t in catalog if t.get_title() == query)

    def test_wildcards_and_case_are_taken_literally(self):
        # Caso fisso: con una ricerca LIKE/ILIKE "%" troverebbe tutti e
        # "PROF." troverebbe "Prof." e "prof.".
        catalog = [make_tutor(title=title) for title in ["Prof.", "prof.", "%", "_", "%%"]]
        queries = ["%", "_", "%%", "Prof.", "PROF.", "Pro%"]

        async def test(tutors, users, session):
            await add_all(tutors, catalog)
            return {query: ids(await tutors.get_by_title(query)) for query in queries}

        found = in_transaction(test)

        assert found == {query: ids(t for t in catalog if t.get_title() == query)
                         for query in queries}

    @DB
    @given(name=names, other=names, surname=names)
    def test_get_by_name_and_surname_ignore_students(self, name, other, surname):
        tutor = make_tutor(name=name, surname=surname)
        different = make_tutor(name=other, surname=other)
        student = make_student(name=name, surname=surname)

        async def test(tutors, users, session):
            await add_all(tutors, [tutor, different])
            await users.add(student)
            return await tutors.get_by_name(name), await tutors.get_by_surname(surname)

        by_name, by_surname = in_transaction(test)

        assert ids(by_name) == ids([tutor] + ([different] if other == name else []))
        assert ids(by_surname) == ids([tutor] + ([different] if other == surname else []))
        assert all(isinstance(t, Tutor) for t in by_name + by_surname)


class TestSubjects:

    @DB
    @given(pool=st.lists(subjects_text, min_size=1, max_size=6, unique=True), data=st.data())
    def test_search_agrees_with_python_sets(self, pool, data):
        # Materie prese da un insieme piccolo: tante sovrapposizioni tra tutor.
        some_subjects = st.lists(st.sampled_from(pool), min_size=1, max_size=4)
        catalog = [make_tutor(subjects=subjects)
                   for subjects in data.draw(st.lists(some_subjects, min_size=1, max_size=6),
                                             label="catalog")]
        query = data.draw(st.lists(st.sampled_from(pool) | nasty(), max_size=4), label="query")
        match_all = data.draw(st.booleans(), label="match_all")

        async def test(tutors, users, session):
            await add_all(tutors, catalog)
            return await tutors.get_by_subjects(query, match_all=match_all)

        found = in_transaction(test)

        wanted = set(query)
        if not wanted:
            expected = []
        elif match_all:
            expected = [t for t in catalog if wanted <= set(t.get_subjects())]
        else:
            expected = [t for t in catalog if wanted & set(t.get_subjects())]
        assert ids(found) == ids(expected)

    @DB
    @given(subjects=subject_lists, data=st.data())
    def test_get_by_subject_finds_each_of_its_subjects(self, subjects, data):
        tutor = make_tutor(subjects=subjects)
        # Dalle materie salvate, non da quelle passate: Tutor toglie i doppioni.
        subject = data.draw(st.sampled_from(tutor.get_subjects()), label="subject")

        async def test(tutors, users, session):
            await tutors.add(tutor)
            return (await tutors.get_by_subject(subject),
                    await tutors.get_by_subjects([subject, subject], match_all=True))

        single, duplicated_query = in_transaction(test)

        assert ids(single) == ids(duplicated_query) == [tutor.get_user_id()]

    def test_lookalike_subjects_are_different_subjects(self):
        composed, decomposed = make_tutor(subjects=["é"]), make_tutor(subjects=["é"])

        async def test(tutors, users, session):
            await add_all(tutors, [composed, decomposed])
            return (await tutors.get_by_subject("é"), await tutors.get_by_subject("Matematica"),
                    await tutors.get_by_subject("%"))

        accented, wrong_case, wildcard = in_transaction(test)

        assert ids(accented) == [composed.get_user_id()]
        assert wrong_case == [] and wildcard == []

    def test_empty_query_is_an_empty_list(self):
        async def test(tutors, users, session):
            await tutors.add(make_tutor())
            return (await tutors.get_by_subjects([]),
                    await tutors.get_by_subjects([], match_all=True))

        assert in_transaction(test) == ([], [])


class TestOrdering:

    @DB
    @given(people=st.lists(st.tuples(st.sampled_from(["anna", "bruno", "carla"]),
                                     st.sampled_from(["neri", "rossi", "verdi"])),
                           min_size=1, max_size=8))
    def test_results_are_ordered_by_surname_name_id(self, people):
        # Solo minuscole ASCII: l'ordine non dipende dalla collation del database.
        # I doppioni obbligano a usare anche user_id per spareggiare.
        catalog = [make_tutor(name=name, surname=surname, subjects=["storia"])
                   for name, surname in people]

        async def test(tutors, users, session):
            await add_all(tutors, catalog)
            return [await tutors.get_all(), await tutors.get_by_subject("storia"),
                    await tutors.get_by_title("Prof.")]

        expected = sorted(catalog, key=lambda t: (t.get_surname(), t.get_name(), t.get_user_id()))

        for found in in_transaction(test):
            assert [t.get_user_id() for t in found] == [t.get_user_id() for t in expected]


# ============================================================ update

class TestUpdate:

    @DB
    @given(name=names, email=emails, password=passwords, bio=bios, title=titles,
           subjects=subject_lists)
    def test_changes_are_saved_but_created_at_is_not(self, name, email, password, bio,
                                                     title, subjects):
        tutor = make_tutor()
        new = make_tutor(user_id=tutor.get_user_id(), name=name, email=email, password=password,
                         created_at=OTHER_DATE, bio=bio, title=title, subjects=subjects)

        async def test(tutors, users, session):
            await tutors.add(tutor)
            await tutors.update(new)
            return await tutors.get_by_id(tutor.get_user_id())

        loaded = in_transaction(test)

        assert tutor_fields(loaded)[1:] == (tutor_fields(new)[1:5] + (NOW,) + tutor_fields(new)[6:])
        assert loaded.verify_password(password)

    def test_missing_tutor_raises(self):
        async def test(tutors, users, session):
            await tutors.update(make_tutor())

        with pytest.raises(UserNotFound):
            in_transaction(test)

    def test_email_of_another_user_is_rejected_and_nothing_changes(self):
        student = make_student(email="preso@example.com")
        tutor = make_tutor(email="mio@example.com", title="Dott.")
        new = make_tutor(user_id=tutor.get_user_id(), email="PRESO@example.com", title="Prof.")

        async def test(tutors, users, session):
            await users.add(student)
            await tutors.add(tutor)
            with pytest.raises(EmailAlreadyRegistered):
                await tutors.update(new)
            return await tutors.get_by_id(tutor.get_user_id())

        assert tutor_fields(in_transaction(test)) == tutor_fields(tutor)


# ============================================================ delete

class TestDelete:

    def test_delete_is_idempotent_and_cascades(self):
        tutor, other = make_tutor(), make_tutor()

        async def test(tutors, users, session):
            await add_all(tutors, [tutor, other])
            first = await tutors.delete(tutor.get_user_id())
            second = await tutors.delete(tutor.get_user_id())
            orphans = (await session.execute(
                TutorRow.__table__.select().where(TutorRow.user_id == tutor.get_user_id())
            )).all()
            return (first, second, await users.get_by_id(tutor.get_user_id()), orphans,
                    await tutors.get_all())

        first, second, as_user, orphans, remaining = in_transaction(test)

        assert (first, second, as_user, orphans) == (True, False, None, [])
        assert remaining == [other]

    def test_deleted_id_and_email_can_be_reused(self):
        
        tutor = make_tutor()

        async def test(tutors, users, session):
            await tutors.add(tutor)
            await tutors.delete(tutor.get_user_id())
            await tutors.add(tutor)
            return await tutors.get_by_id(tutor.get_user_id())

        assert in_transaction(test) == tutor


# ============================================================ studenti

class TestStudentsAreOffLimits:
    """Un id o un'email di un utente non tutor: TutorRepository non deve toccarlo."""

    @DB
    @given(name=names, surname=names, email=emails, title=titles, subjects=subject_lists)
    def test_tutor_operations_never_modify_a_student(self, name, surname, email, title,
                                                     subjects):
        student = make_student(name=name, surname=surname, email=email)
        impostor = make_tutor(user_id=student.get_user_id(), name="Impostore",
                              email=f"x{email}", title=title, subjects=subjects)
        same_email = make_tutor(email=email.upper())

        async def test(tutors, users, session):
            await users.add(student)
            with pytest.raises(UserNotFound):
                await tutors.update(impostor)
            deleted = await tutors.delete(student.get_user_id())
            with pytest.raises(UserAlreadyExists):
                await tutors.add(impostor)
            with pytest.raises(EmailAlreadyRegistered):
                await tutors.add(same_email)
            return (deleted, await users.get_by_id(student.get_user_id()),
                    await tutors.get_by_id(student.get_user_id()),
                    await tutors.get_by_email(email), await tutors.get_all())

        deleted, still_there, as_tutor, by_email, every_tutor = in_transaction(test)

        assert deleted is False
        assert user_fields(still_there) == user_fields(student)
        assert (as_tutor, by_email, every_tutor) == (None, None, [])


# ============================================================ atomicita'

class TestAtomicity:
    """Se la scrittura su tutor fallisce, anche quella su users va annullata.

    Il guasto viene iniettato in _tutor_to_row (materie vuote, vietate da
    subjects_not_empty): cosi' il test non dipende dalle validazioni di Tutor.
    """

    @pytest.fixture
    def broken_tutor_row(self, monkeypatch):
        original = TutorRepository._tutor_to_row
        monkeypatch.setattr(TutorRepository, "_tutor_to_row",
                            staticmethod(lambda tutor: {**original(tutor), "subjects": []}))

    def test_failed_add_leaves_no_user_behind(self, broken_tutor_row):
        tutor = make_tutor()

        async def test(tutors, users, session):
            with pytest.raises(IntegrityError):
                await tutors.add(tutor)
            return await users.get_by_id(tutor.get_user_id())

        assert in_transaction(test) is None

    def test_failed_update_keeps_the_old_user_fields(self, monkeypatch):
        tutor = make_tutor(name="Mario")
        renamed = make_tutor(user_id=tutor.get_user_id(), name="Luigi")

        async def test(tutors, users, session):
            await tutors.add(tutor)
            original = TutorRepository._tutor_to_row
            monkeypatch.setattr(TutorRepository, "_tutor_to_row",
                                staticmethod(lambda t: {**original(t), "subjects": []}))
            with pytest.raises(IntegrityError):
                await tutors.update(renamed)
            monkeypatch.undo()
            return await tutors.get_by_id(tutor.get_user_id())

        assert tutor_fields(in_transaction(test)) == tutor_fields(tutor)


# ============================================================ vincoli nel DB

def _insert_user_and_tutor(**tutor_overrides):
    user_id = uuid.uuid4()
    user = {"user_id": user_id, "name": "Mario", "surname": "Rossi",
            "email": f"{user_id.hex}@example.com", "password_hash": FAST_HASHER.hash(PASSWORD),
            "created_at": NOW}
    tutor = {"user_id": user_id, "bio": None, "title": "Prof.", "subjects": ["matematica"]}
    tutor.update(tutor_overrides)

    async def test(tutors, users, session):
        await session.execute(insert(UserRow.__table__).values(**user))
        await session.execute(insert(TutorRow.__table__).values(**tutor))

    in_transaction(test)


class TestDatabaseConstraints:
    """Le regole valgono anche per chi scrive senza passare dal repository."""

    @pytest.mark.parametrize("column, value", [
        ("subjects", []),
        ("subjects", None),
        ("title", None),
        ("user_id", "senza-utente"),   # nessuna riga in users: viola la foreign key
    ])
    def test_invalid_rows_are_rejected(self, column, value):
        if value == "senza-utente":
            value = uuid.uuid4()
        with pytest.raises(IntegrityError):
            _insert_user_and_tutor(**{column: value})

    # Buchi nello schema: oggi il database li accetta. strict=True: quando
    # aggiungi il vincolo il test passa, pytest lo segnala e togli lo xfail.
    @pytest.mark.xfail(strict=True, reason="manca un CHECK: subjects accetta elementi NULL")
    def test_null_subject_is_rejected(self):
        with pytest.raises(IntegrityError):
            _insert_user_and_tutor(subjects=[None])

    @pytest.mark.xfail(strict=True, reason="manca un CHECK: subjects accetta stringhe vuote")
    def test_empty_subject_is_rejected(self):
        with pytest.raises(IntegrityError):
            _insert_user_and_tutor(subjects=["matematica", ""])

    @pytest.mark.xfail(strict=True, reason="manca un CHECK: title vuoto (name e surname lo hanno)")
    def test_empty_title_is_rejected(self):
        with pytest.raises(IntegrityError):
            _insert_user_and_tutor(title="")


# ============================================================ macchina a stati

# Insiemi piccoli e pieni di quasi-doppioni: le collisioni (email uguali a meno
# delle maiuscole, materie e titoli simili) diventano frequenti.
EMAIL_POOL = ["anna@example.com", "ANNA@example.com", "Anna@Example.COM",
              "luca@example.it", "LUCA@EXAMPLE.IT", "zoe@example.org", "bea@example.org"]
NAME_POOL = ["Anna", "anna", "ANNA", "Ω", "Ω", "'; --", "é", "é"]
TITLE_POOL = ["Prof.", "prof.", "Dott.", "%", "_", "'", " Prof."]
SUBJECT_POOL = ["matematica", "Matematica", "fisica", "{a,b}", "a,b", "NULL", "%", "'",
                "é", "é", "😀"]

tutor_args = st.fixed_dictionaries({
    "name": st.sampled_from(NAME_POOL),
    "surname": st.sampled_from(NAME_POOL),
    "email": st.sampled_from(EMAIL_POOL),
    "password": st.sampled_from(["password-numero-uno", "password-numero-due"]),
    "bio": st.none() | st.sampled_from(TRICKY),
    "title": st.sampled_from(TITLE_POOL),
    "subjects": st.lists(st.sampled_from(SUBJECT_POOL), min_size=1, max_size=4),
})


class TutorRepositoryMachine(RuleBasedStateMachine):
    """Sequenze casuali di operazioni, confrontate con un modello in memoria.

    Tutta la sequenza gira in un'unica transazione (annullata alla fine) su un
    unico event loop, perche' le connessioni asyncpg sono legate al loop.
    """

    tutor_ids = Bundle("tutor_ids")
    student_ids = Bundle("student_ids")

    def __init__(self):
        super().__init__()
        self.tutors_model = {}      # user_id -> Tutor atteso
        self.created_at = {}        # user_id -> created_at dell'add (update non lo cambia)
        self.students = {}          # user_id -> User
        self.loop = asyncio.Runner()
        self.loop.run(self._open())

    async def _open(self):
        self.engine = make_engine(DATABASE_URL)
        self.connection = await self.engine.connect()
        self.transaction = await self.connection.begin()
        self.session = AsyncSession(bind=self.connection, expire_on_commit=False)
        self.tutors = TutorRepository(self.session)
        self.users = UserRepository(self.session)

    def teardown(self):
        async def close():
            await self.session.close()
            await self.transaction.rollback()
            await self.connection.close()
            await self.engine.dispose()

        try:
            self.loop.run(close())
        finally:
            self.loop.close()

    def run(self, coroutine):
        return self.loop.run(coroutine)

    def email_owner(self, email):
        everyone = {**self.tutors_model, **self.students}
        return next((uid for uid, user in everyone.items()
                     if user.get_email().lower() == email.lower()), None)

    # ------------------------------------------------------------ scritture

    @rule(target=tutor_ids, args=tutor_args)
    def add_tutor(self, args):
        tutor = make_tutor(**args)
        if self.email_owner(tutor.get_email()) is not None:
            with pytest.raises(EmailAlreadyRegistered):
                self.run(self.tutors.add(tutor))
            return multiple()

        self.run(self.tutors.add(tutor))
        self.tutors_model[tutor.get_user_id()] = tutor
        self.created_at[tutor.get_user_id()] = tutor.get_created_at()
        return tutor.get_user_id()

    @rule(target=student_ids, email=st.sampled_from(EMAIL_POOL), name=st.sampled_from(NAME_POOL))
    def add_student(self, email, name):
        student = make_student(email=email, name=name, surname=name)
        if self.email_owner(student.get_email()) is not None:
            with pytest.raises(EmailAlreadyRegistered):
                self.run(self.users.add(student))
            return multiple()

        self.run(self.users.add(student))
        self.students[student.get_user_id()] = student
        return student.get_user_id()

    @rule(user_id=tutor_ids | student_ids)
    def add_tutor_with_a_used_id(self, user_id):
        # Email nuova: l'unico conflitto possibile e' sull'id.
        tutor = make_tutor(user_id=user_id)
        if user_id in self.tutors_model or user_id in self.students:
            with pytest.raises(UserAlreadyExists):
                self.run(self.tutors.add(tutor))
        else:   # tutor cancellato: l'id torna libero
            self.run(self.tutors.add(tutor))
            self.tutors_model[user_id] = tutor
            self.created_at[user_id] = tutor.get_created_at()

    @rule(user_id=tutor_ids, args=tutor_args)
    def update_tutor(self, user_id, args):
        new = make_tutor(user_id=user_id, created_at=OTHER_DATE, **args)
        owner = self.email_owner(new.get_email())

        if user_id not in self.tutors_model:
            with pytest.raises(UserNotFound):
                self.run(self.tutors.update(new))
        elif owner not in (None, user_id):
            with pytest.raises(EmailAlreadyRegistered):
                self.run(self.tutors.update(new))
        else:
            self.run(self.tutors.update(new))
            self.tutors_model[user_id] = new

    @rule(user_id=student_ids, args=tutor_args)
    def update_student_through_tutors(self, user_id, args):
        with pytest.raises(UserNotFound):
            self.run(self.tutors.update(make_tutor(user_id=user_id, **args)))

    @rule(user_id=tutor_ids | student_ids)
    def delete(self, user_id):
        expected = user_id in self.tutors_model
        assert self.run(self.tutors.delete(user_id)) is expected
        self.tutors_model.pop(user_id, None)

    # ------------------------------------------------------------ ricerche

    def check(self, found, expected):
        assert all(isinstance(t, Tutor) for t in found)
        assert ids(found) == ids(expected)

    @rule(query=st.lists(st.sampled_from(SUBJECT_POOL), max_size=3), match_all=st.booleans())
    def search_subjects(self, query, match_all):
        wanted = set(query)
        if not wanted:
            expected = []
        elif match_all:
            expected = [t for t in self.tutors_model.values() if wanted <= set(t.get_subjects())]
        else:
            expected = [t for t in self.tutors_model.values() if wanted & set(t.get_subjects())]
        self.check(self.run(self.tutors.get_by_subjects(query, match_all=match_all)), expected)

    @rule(title=st.sampled_from(TITLE_POOL))
    def search_title(self, title):
        self.check(self.run(self.tutors.get_by_title(title)),
                   [t for t in self.tutors_model.values() if t.get_title() == title])

    @rule(name=st.sampled_from(NAME_POOL))
    def search_name_and_surname(self, name):
        self.check(self.run(self.tutors.get_by_name(name)),
                   [t for t in self.tutors_model.values() if t.get_name() == name])
        self.check(self.run(self.tutors.get_by_surname(name)),
                   [t for t in self.tutors_model.values() if t.get_surname() == name])

    @rule(email=st.sampled_from(EMAIL_POOL))
    def search_email(self, email):
        found = self.run(self.tutors.get_by_email(email))
        owner = self.email_owner(email)
        expected = self.tutors_model.get(owner)
        assert (found is None) == (expected is None)
        if found is not None:
            assert found.get_user_id() == owner

    # ------------------------------------------------------------ invarianti

    @invariant()
    def database_matches_model(self):
        stored = {t.get_user_id(): t for t in self.run(self.tutors.get_all())}

        assert stored.keys() == self.tutors_model.keys()
        for user_id, tutor in stored.items():
            expected = self.tutors_model[user_id]
            assert tutor_fields(tutor)[:5] == tutor_fields(expected)[:5]
            assert tutor_fields(tutor)[6:] == tutor_fields(expected)[6:]
            assert tutor.get_created_at() == self.created_at[user_id]

    @invariant()
    def students_are_untouched(self):
        for user_id, student in self.students.items():
            assert user_fields(self.run(self.users.get_by_id(user_id))) == user_fields(student)
            assert self.run(self.tutors.get_by_id(user_id)) is None


TutorRepositoryMachine.TestCase.settings = settings(
    max_examples=30, stateful_step_count=30, deadline=None,
    # Qui lo shrinking resta: riduce una sequenza fallita ai pochi passi che contano.
    suppress_health_check=[HealthCheck.too_slow],
)
TestTutorRepositoryMachine = TutorRepositoryMachine.TestCase
