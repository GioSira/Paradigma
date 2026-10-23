"""Test di integrazione di CorsoRepository su PostgreSQL reale.

Stesse regole degli altri test_*_repository.py: TEST_DATABASE_URL punta a un
database DI TEST, le tabelle le ricrea la fixture "database" di conftest.py e
ogni test gira in una transazione annullata alla fine.

Fanno eccezione i test sulla concorrenza (TestConcurrency): due sessioni
devono vedere gli stessi dati, quindi i dati vengono salvati davvero e
cancellati alla fine del test.

La regola che conta di piu': un corso ha SEMPRE almeno un docente.
"""
import asyncio
import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from dotenv import load_dotenv
from hypothesis import Phase, given, settings, strategies as st
from sqlalchemy import delete, func, insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.corsi.corso import Corso
from src.roles.student import Student
from src.roles.tutor import Tutor
from src.sqldb.base import make_engine
from src.sqldb.errors import CorsoAlreadyExists, CorsoNotFound, UltimoDocente, UserNotFound
from src.sqldb.repository.corso_repository import CorsoRepository
from src.sqldb.repository.student_repository import StudentRepository
from src.sqldb.repository.tutor_repository import TutorRepository
from src.sqldb.tables import CORSO_DOCENTE, ISCRIZIONE, CorsoRow, UserRow

load_dotenv()
DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DATABASE_URL, reason="TEST_DATABASE_URL non impostata"),
    pytest.mark.usefixtures("database"),     # conftest.py: lock + tabelle ricreate una volta
]

DB = settings(max_examples=10, deadline=None,
              phases=[Phase.explicit, Phase.reuse, Phase.generate])
PASSWORD = "una-password-lunga-e-sicura"
NOW = datetime(2024, 3, 9, 14, 5, 6, tzinfo=timezone.utc)

_CORSI = CorsoRow.__table__
_USERS = UserRow.__table__

# Testi ostili: apici, commenti SQL, jolly di LIKE, unicode scomposto, emoji.
TRICKY = ["'", "' OR '1'='1", "'); DROP TABLE corso; --", "%", "_", "{a,b}", "NULL",
          "é", "é", "😀", "‮maerts", "x" * 1000]
SAFE_CHARS = st.characters(codec="utf-8", exclude_characters="\x00")   # PostgreSQL rifiuta NUL
titles = st.sampled_from(TRICKY) | st.text(alphabet=SAFE_CHARS, min_size=1, max_size=60)
descriptions = st.sampled_from(TRICKY + [""]) | st.text(alphabet=SAFE_CHARS, max_size=300)
starts = st.datetimes(min_value=datetime(1990, 1, 1), max_value=datetime(2090, 1, 1),
                      timezones=st.just(timezone.utc))
durations = st.timedeltas(min_value=timedelta(seconds=1), max_value=timedelta(days=800))


def in_transaction(test):
    """Esegue test(corsi, tutors, session) in una transazione annullata alla fine."""
    async def run():
        engine = make_engine(DATABASE_URL)
        try:
            async with engine.connect() as connection:
                transaction = await connection.begin()
                session = AsyncSession(bind=connection, expire_on_commit=False)
                try:
                    return await test(CorsoRepository(session), TutorRepository(session), session)
                finally:
                    await session.close()
                    await transaction.rollback()
        finally:
            await engine.dispose()

    return asyncio.run(run())


# ============================================================ helper

def make_corso(corso_id=None, title="Basi di dati", descrizione="Il modello relazionale",
               start=NOW, length=timedelta(days=90)):
    return Corso(id=corso_id or uuid.uuid4(), title=title, descrizione=descrizione,
                 data_inizio=start, data_fine=start + length)


def make_tutor(surname="Rossi"):
    user_id = uuid.uuid4()
    return Tutor.create(user_id=user_id, name="Mario", surname=surname,
                        email=f"{user_id.hex}@example.com", password=PASSWORD,
                        tutor_title="Prof.", subjects=["informatica"], created_at=NOW)


def make_student():
    user_id = uuid.uuid4()
    return Student.create(user_id=user_id, name="Giulia", surname="Ferri",
                          email=f"{user_id.hex}@example.com", password=PASSWORD)


async def add_tutors(tutors, n):
    added = [make_tutor() for _ in range(n)]
    for tutor in added:
        await tutors.add(tutor)
    return [t.get_user_id() for t in added]


async def count(session, table, **where):
    query = select(func.count()).select_from(table)
    for column, value in where.items():
        query = query.where(table.c[column] == value)
    return (await session.execute(query)).scalar_one()


def corso_fields(corso):
    return (corso.get_id(), corso.get_title(), corso.get_descriziione(),
            corso.get_data_inizio(), corso.get_data_fine())


# ============================================================ add e lettura

class TestAddAndGet:

    @DB
    @given(title=titles, descrizione=descriptions, start=starts, length=durations,
           n_tutors=st.integers(min_value=1, max_value=4))
    def test_round_trip_keeps_every_field(self, title, descrizione, start, length, n_tutors):
        corso = make_corso(title=title, descrizione=descrizione, start=start, length=length)

        async def test(corsi, tutors, session):
            tutor_ids = await add_tutors(tutors, n_tutors)
            await corsi.add(corso, tutor_ids)
            return await corsi.get_by_id(corso.get_id()), await corsi.get_docenti_ids(corso.get_id()), tutor_ids

        loaded, docenti, tutor_ids = in_transaction(test)

        assert isinstance(loaded, Corso)
        assert corso_fields(loaded) == corso_fields(corso), "nessuna normalizzazione"
        assert loaded.get_data_inizio().tzinfo is not None
        assert docenti == sorted(tutor_ids)

    def test_missing_corso_is_none(self):
        async def test(corsi, tutors, session):
            return await corsi.get_by_id(uuid.uuid4()), await corsi.get_docenti_ids(uuid.uuid4())

        assert in_transaction(test) == (None, [])

    def test_duplicate_tutor_ids_count_once(self):
        corso = make_corso()

        async def test(corsi, tutors, session):
            (tutor_id,) = await add_tutors(tutors, 1)
            await corsi.add(corso, [tutor_id, tutor_id, tutor_id])
            return await corsi.get_docenti_ids(corso.get_id()), tutor_id

        docenti, tutor_id = in_transaction(test)
        assert docenti == [tutor_id]

    def test_tutor_ids_can_be_any_iterable(self):
        corso = make_corso()

        async def test(corsi, tutors, session):
            tutor_ids = await add_tutors(tutors, 2)
            await corsi.add(corso, (t for t in tutor_ids))      # generatore
            return len(await corsi.get_docenti_ids(corso.get_id()))

        assert in_transaction(test) == 2


class TestAddRejections:

    @pytest.mark.parametrize("tutor_ids", [None, [], ()])
    def test_a_corso_needs_at_least_one_docente(self, tutor_ids):
        corso = make_corso()

        async def test(corsi, tutors, session):
            with pytest.raises(ValueError):
                await corsi.add(corso, tutor_ids)
            return await count(session, _CORSI)

        assert in_transaction(test) == 0

    def test_missing_tutor_writes_nothing(self):
        corso = make_corso()
        missing = uuid.uuid4()

        async def test(corsi, tutors, session):
            tutor_ids = await add_tutors(tutors, 2)
            with pytest.raises(UserNotFound) as raised:
                await corsi.add(corso, [tutor_ids[0], missing, tutor_ids[1]])
            return (raised.value.user_id, await count(session, _CORSI),
                    await count(session, CORSO_DOCENTE))

        assert in_transaction(test) == (missing, 0, 0)

    def test_a_student_is_not_a_tutor(self):
        corso = make_corso()
        student = make_student()

        async def test(corsi, tutors, session):
            await StudentRepository(session).add(student)
            with pytest.raises(UserNotFound):
                await corsi.add(corso, [student.get_user_id()])
            return await count(session, _CORSI)

        assert in_transaction(test) == 0

    def test_same_id_is_rejected_and_the_original_is_kept(self):
        corso = make_corso(title="Originale")
        copy = make_corso(corso_id=corso.get_id(), title="Copia")

        async def test(corsi, tutors, session):
            tutor_ids = await add_tutors(tutors, 1)
            await corsi.add(corso, tutor_ids)
            with pytest.raises(CorsoAlreadyExists) as raised:
                await corsi.add(copy, tutor_ids)
            return raised.value.corso_id, (await corsi.get_by_id(corso.get_id())).get_title()

        assert in_transaction(test) == (corso.get_id(), "Originale")

    def test_failed_add_leaves_the_transaction_usable(self):
        first, second = make_corso(), make_corso()

        async def test(corsi, tutors, session):
            tutor_ids = await add_tutors(tutors, 1)
            with pytest.raises(UserNotFound):
                await corsi.add(first, [uuid.uuid4()])
            await corsi.add(second, tutor_ids)
            return await corsi.get_by_id(first.get_id()), await corsi.get_by_id(second.get_id())

        missing, found = in_transaction(test)
        assert missing is None and found is not None


# ============================================================ ricerche

class TestGetByTutor:

    @DB
    @given(assignments=st.lists(st.sets(st.integers(min_value=0, max_value=2), min_size=1),
                                min_size=1, max_size=5),
           offsets=st.lists(st.integers(min_value=0, max_value=2), min_size=5, max_size=5))
    def test_each_tutor_sees_exactly_his_corsi_in_order(self, assignments, offsets):
        """assignments[i] = quali dei 3 tutor insegnano nel corso i."""
        async def test(corsi, tutors, session):
            tutor_ids = await add_tutors(tutors, 3)
            expected = {t: [] for t in tutor_ids}
            for i, teachers in enumerate(assignments):
                # date e titoli ripetuti apposta: l'ordine deve spareggiare fino all'id
                corso = make_corso(title=f"Corso {i % 2}", start=NOW + timedelta(days=offsets[i]))
                await corsi.add(corso, [tutor_ids[k] for k in teachers])
                for k in teachers:
                    expected[tutor_ids[k]].append(corso)
            found = {t: await corsi.get_by_tutor(t) for t in tutor_ids}
            return found, expected

        found, expected = in_transaction(test)

        key = lambda c: (c.get_data_inizio(), c.get_title(), c.get_id())
        for tutor_id, corsi_attesi in expected.items():
            assert [c.get_id() for c in found[tutor_id]] == [c.get_id() for c in sorted(corsi_attesi, key=key)]

    def test_unknown_tutor_has_no_corsi(self):
        async def test(corsi, tutors, session):
            return await corsi.get_by_tutor(uuid.uuid4())

        assert in_transaction(test) == []


# ============================================================ update

class TestUpdate:

    @DB
    @given(title=titles, descrizione=descriptions, start=starts, length=durations)
    def test_changes_are_saved_and_docenti_untouched(self, title, descrizione, start, length):
        corso = make_corso()
        new = make_corso(corso_id=corso.get_id(), title=title, descrizione=descrizione,
                         start=start, length=length)

        async def test(corsi, tutors, session):
            tutor_ids = await add_tutors(tutors, 2)
            await corsi.add(corso, tutor_ids)
            await corsi.update(new)
            return (await corsi.get_by_id(corso.get_id()), await corsi.get_docenti_ids(corso.get_id()),
                    sorted(tutor_ids))

        loaded, docenti, tutor_ids = in_transaction(test)

        assert corso_fields(loaded) == corso_fields(new)
        assert docenti == tutor_ids

    def test_missing_corso_raises(self):
        async def test(corsi, tutors, session):
            await corsi.update(make_corso())

        with pytest.raises(CorsoNotFound):
            in_transaction(test)

    def test_invalid_dates_are_rejected_and_nothing_changes(self):
        corso = make_corso()
        broken = make_corso(corso_id=corso.get_id(), title="Rotto", length=timedelta(0))

        async def test(corsi, tutors, session):
            await corsi.add(corso, await add_tutors(tutors, 1))
            with pytest.raises(IntegrityError):
                await corsi.update(broken)
            return (await corsi.get_by_id(corso.get_id())).get_title()

        assert in_transaction(test) == corso.get_title()


# ============================================================ delete

class TestDelete:

    def test_delete_is_idempotent_and_cascades(self):
        corso = make_corso()
        student = make_student()

        async def test(corsi, tutors, session):
            tutor_ids = await add_tutors(tutors, 2)
            await corsi.add(corso, tutor_ids)
            students = StudentRepository(session)
            await students.add(student)
            await students.enroll(student.get_user_id(), corso.get_id())

            first = await corsi.delete(corso.get_id())
            second = await corsi.delete(corso.get_id())
            return (first, second,
                    await count(session, CORSO_DOCENTE, corso_id=corso.get_id()),
                    await count(session, ISCRIZIONE, corso_id=corso.get_id()),
                    await count(session, _USERS))

        # via docenti e iscrizioni; i 2 tutor e lo studente restano
        assert in_transaction(test) == (True, False, 0, 0, 3)

    def test_other_corsi_are_untouched(self):
        a, b = make_corso(), make_corso()

        async def test(corsi, tutors, session):
            tutor_ids = await add_tutors(tutors, 1)
            await corsi.add(a, tutor_ids)
            await corsi.add(b, tutor_ids)
            await corsi.delete(a.get_id())
            return await corsi.get_by_tutor(tutor_ids[0])

        assert [c.get_id() for c in in_transaction(test)] == [b.get_id()]


# ============================================================ docenti

class TestAddProfessor:

    def test_add_is_idempotent(self):
        corso = make_corso()

        async def test(corsi, tutors, session):
            first_id, second_id = await add_tutors(tutors, 2)
            await corsi.add(corso, [first_id])
            added = await corsi.add_docente(corso.get_id(), second_id)
            again = await corsi.add_docente(corso.get_id(), second_id)
            return added, again, await corsi.get_docenti_ids(corso.get_id()), sorted([first_id, second_id])

        added, again, docenti, expected = in_transaction(test)
        assert (added, again) == (True, False) and docenti == expected

    def test_missing_corso(self):
        async def test(corsi, tutors, session):
            (tutor_id,) = await add_tutors(tutors, 1)
            with pytest.raises(CorsoNotFound):
                await corsi.add_docente(uuid.uuid4(), tutor_id)

        in_transaction(test)

    @pytest.mark.parametrize("who", ["nessuno", "studente"])
    def test_only_existing_tutors(self, who):
        corso = make_corso()
        student = make_student()

        async def test(corsi, tutors, session):
            await corsi.add(corso, await add_tutors(tutors, 1))
            await StudentRepository(session).add(student)
            target = uuid.uuid4() if who == "nessuno" else student.get_user_id()
            with pytest.raises(UserNotFound):
                await corsi.add_docente(corso.get_id(), target)
            return len(await corsi.get_docenti_ids(corso.get_id()))

        assert in_transaction(test) == 1


class TestRemoveDocente:

    def test_remove_one_of_two(self):
        corso = make_corso()

        async def test(corsi, tutors, session):
            first_id, second_id = await add_tutors(tutors, 2)
            await corsi.add(corso, [first_id, second_id])
            removed = await corsi.remove_docente(corso.get_id(), first_id)
            return removed, await corsi.get_docenti_ids(corso.get_id()), second_id

        removed, docenti, second_id = in_transaction(test)
        assert removed is True and docenti == [second_id]

    def test_the_last_docente_cannot_be_removed(self):
        corso = make_corso()

        async def test(corsi, tutors, session):
            (tutor_id,) = await add_tutors(tutors, 1)
            await corsi.add(corso, [tutor_id])
            with pytest.raises(UltimoDocente) as raised:
                await corsi.remove_docente(corso.get_id(), tutor_id)
            return raised.value.corso_id, await corsi.get_docenti_ids(corso.get_id()), tutor_id

        corso_id, docenti, tutor_id = in_transaction(test)
        assert corso_id == corso.get_id() and docenti == [tutor_id]

    def test_removing_someone_who_does_not_teach_is_false(self):
        """Anche se il corso ha un solo docente: non si sta togliendo lui."""
        corso = make_corso()

        async def test(corsi, tutors, session):
            teacher, outsider = await add_tutors(tutors, 2)
            await corsi.add(corso, [teacher])
            return await corsi.remove_docente(corso.get_id(), outsider)

        assert in_transaction(test) is False

    def test_missing_corso(self):
        async def test(corsi, tutors, session):
            with pytest.raises(CorsoNotFound):
                await corsi.remove_docente(uuid.uuid4(), uuid.uuid4())

        in_transaction(test)

    @DB
    @given(n=st.integers(min_value=1, max_value=5), order=st.permutations(range(5)))
    def test_removing_everyone_always_leaves_one(self, n, order):
        corso = make_corso()

        async def test(corsi, tutors, session):
            tutor_ids = await add_tutors(tutors, n)
            await corsi.add(corso, tutor_ids)
            refused = 0
            for i in order:
                if i >= n:
                    continue
                try:
                    await corsi.remove_docente(corso.get_id(), tutor_ids[i])
                except UltimoDocente:
                    refused += 1
            return refused, len(await corsi.get_docenti_ids(corso.get_id()))

        refused, remaining = in_transaction(test)
        assert remaining == 1 and refused == 1


# ============================================================ concorrenza

class TestConcurrency:
    """Due richieste che tolgono due docenti diversi dello stesso corso,
    nello stesso momento. Se entrambe controllano "ci sono 2 docenti" prima che
    l'altra cancelli, il corso resta senza docenti. Il lock sul corso deve
    arrivare PRIMA del conteggio."""

    def test_two_parallel_removals_leave_one_docente(self):
        async def run():
            engine = make_engine(DATABASE_URL)
            corso = make_corso()
            try:
                async with AsyncSession(engine, expire_on_commit=False) as session:
                    tutor_ids = await add_tutors(TutorRepository(session), 2)
                    await CorsoRepository(session).add(corso, tutor_ids)
                    await session.commit()

                async def remove(tutor_id):
                    async with AsyncSession(engine, expire_on_commit=False) as session:
                        try:
                            result = await CorsoRepository(session).remove_docente(corso.get_id(), tutor_id)
                            await session.commit()
                            return result
                        except UltimoDocente:
                            await session.rollback()
                            return "rifiutato"

                outcomes = await asyncio.gather(*(remove(t) for t in tutor_ids))

                async with AsyncSession(engine) as session:
                    remaining = await CorsoRepository(session).get_docenti_ids(corso.get_id())
                return sorted(map(str, outcomes)), len(remaining)
            finally:
                async with engine.begin() as connection:     # pulizia: questi dati sono salvati davvero
                    await connection.execute(delete(_CORSI).where(_CORSI.c.id == corso.get_id()))
                    await connection.execute(delete(_USERS))
                await engine.dispose()

        outcomes, remaining = asyncio.run(run())

        assert remaining == 1
        assert outcomes == ["True", "rifiutato"]


# ======================================================= vincoli nel DB

class TestDatabaseConstraints:
    """Le regole valgono anche per chi scrive senza passare dal repository."""

    @pytest.mark.parametrize("column, value", [
        ("title", ""),
        ("data_fine", NOW),                        # uguale all'inizio
        ("data_fine", NOW - timedelta(days=1)),
        ("title", None),
        ("descrizione", None),
    ])
    def test_invalid_corso_rows_are_rejected(self, column, value):
        row = dict(id=uuid.uuid4(), title="Corso", descrizione="d",
                   data_inizio=NOW, data_fine=NOW + timedelta(days=1))
        row[column] = value

        async def test(corsi, tutors, session):
            await session.execute(insert(_CORSI).values(**row))

        with pytest.raises(IntegrityError):
            in_transaction(test)

    def test_docente_must_be_a_tutor(self):
        corso = make_corso()
        student = make_student()

        async def test(corsi, tutors, session):
            await corsi.add(corso, await add_tutors(tutors, 1))
            await StudentRepository(session).add(student)
            await session.execute(insert(CORSO_DOCENTE).values(corso_id=corso.get_id(),
                                                               tutor_id=student.get_user_id()))

        with pytest.raises(IntegrityError):
            in_transaction(test)

    # Buco nello schema: cancellare un tutor lo toglie dai corsi (ON DELETE
    # CASCADE) anche se era l'ultimo docente. strict=True: quando il database
    # lo impedira', pytest lo segnalera' e lo xfail andra' tolto.
    @pytest.mark.xfail(strict=True, reason="cancellare l'ultimo docente lascia il corso senza docenti")
    def test_deleting_the_last_tutor_does_not_orphan_the_corso(self):
        corso = make_corso()

        async def test(corsi, tutors, session):
            (tutor_id,) = await add_tutors(tutors, 1)
            await corsi.add(corso, [tutor_id])
            try:
                await tutors.delete(tutor_id)
            except IntegrityError:
                pass
            return len(await corsi.get_docenti_ids(corso.get_id()))

        assert in_transaction(test) >= 1