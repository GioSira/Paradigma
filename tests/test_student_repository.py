"""Test di integrazione di StudentRepository su PostgreSQL reale.

Stesse regole di test_user_repository.py: richiedono TEST_DATABASE_URL (un
database DI TEST: le tabelle vengono ricreate all'avvio) e ogni test gira in
una transazione annullata alla fine.

Oltre alle operazioni sugli studenti verificano il confine con gli altri
utenti: lo StudentRepository non deve mai leggere, modificare o cancellare un
utente che non e' uno studente (per esempio un tutor).
"""
import asyncio
import os
import string
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from dotenv import load_dotenv
from hypothesis import Phase, given, settings, strategies as st
from sqlalchemy import func, insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.roles.student import Student
from src.roles.user import User
from src.sqldb.base import make_engine
from src.sqldb.errors import CorsoNotFound, EmailAlreadyRegistered, UserAlreadyExists, UserNotFound
from src.sqldb.repository.student_repository import StudentRepository
from src.sqldb.repository.user_repository import UserRepository
from src.sqldb.tables import ISCRIZIONE, CorsoRow, StudentRow, UserRow

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

_USERS = UserRow.__table__
_STUDENTS = StudentRow.__table__
_CORSI = CorsoRow.__table__

names = st.text(min_size=1, max_size=50).filter(lambda s: "\x00" not in s)
emails = st.builds(
    "{}@{}.{}".format,
    st.text(alphabet=string.ascii_lowercase + string.digits, min_size=1, max_size=20),
    st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=15),
    st.sampled_from(["it", "com", "org"]),
)
passwords = st.text(min_size=15, max_size=64)




def in_transaction(test):
    """Esegue test(students, users, session) in una transazione annullata alla fine."""
    async def run():
        engine = make_engine(DATABASE_URL)
        try:
            async with engine.connect() as connection:
                transaction = await connection.begin()
                session = AsyncSession(bind=connection, expire_on_commit=False)
                try:
                    return await test(StudentRepository(session), UserRepository(session), session)
                finally:
                    await session.close()
                    await transaction.rollback()
        finally:
            await engine.dispose()

    return asyncio.run(run())


# ============================================================ helper

def make_student(user_id=None, name="Giulia", surname="Ferri", email=None, password=PASSWORD):
    user_id = user_id or uuid.uuid4()
    email = email or f"{user_id.hex}@example.com"
    return Student.create(user_id=user_id, name=name, surname=surname, email=email,
                          password=password)


def make_user(user_id=None, email=None):
    """Un utente che NON e' uno studente (per esempio un tutor)."""
    user_id = user_id or uuid.uuid4()
    email = email or f"{user_id.hex}@example.com"
    return User.create(user_id=user_id, name="Paolo", surname="Neri", email=email,
                       password=PASSWORD, created_at=NOW)


async def add_corso(session, title="Basi di dati", start=NOW, corso_id=None):
    """I corsi si scrivono direttamente in tabella: qui si testa lo StudentRepository."""
    corso_id = corso_id or uuid.uuid4()
    await session.execute(insert(_CORSI).values(
        id=corso_id, title=title, descrizione="descrizione",
        data_inizio=start, data_fine=start + timedelta(days=90)))
    return corso_id


async def count(session, table, **where):
    query = select(func.count()).select_from(table)
    for column, value in where.items():
        query = query.where(table.c[column] == value)
    return (await session.execute(query)).scalar_one()


def ids(users):
    return sorted(user.get_user_id() for user in users)


# ============================================================ lettura

class TestAddAndGetById:

    @DB
    @given(name=names, surname=names, email=emails, password=passwords)
    def test_round_trip_keeps_every_field(self, name, surname, email, password):
        student = make_student(name=name, surname=surname, email=email, password=password)

        async def test(students, users, session):
            await students.add(student)
            return await students.get_by_id(student.get_user_id())

        loaded = in_transaction(test)

        assert isinstance(loaded, Student)
        assert loaded == student
        assert (loaded.get_name(), loaded.get_surname(), loaded.get_email()) == (name, surname, email)
        assert loaded.get_created_at() == student.get_created_at()
        assert loaded.verify_password(password)

    def test_add_writes_both_tables(self):
        student = make_student()

        async def test(students, users, session):
            await students.add(student)
            return (await count(session, _USERS, user_id=student.get_user_id()),
                    await count(session, _STUDENTS, user_id=student.get_user_id()))

        assert in_transaction(test) == (1, 1)

    def test_missing_student_is_none(self):
        async def test(students, users, session):
            return await students.get_by_id(uuid.uuid4())

        assert in_transaction(test) is None

    def test_a_user_who_is_not_a_student_is_not_found(self):
        other = make_user()

        async def test(students, users, session):
            await users.add(other)
            return (await students.get_by_id(other.get_user_id()),
                    await students.get_by_email(other.get_email()),
                    await students.get_by_name(other.get_name()),
                    await students.get_all())

        assert in_transaction(test) == (None, None, [], [])


class TestGetAll:

    def test_only_students_in_a_stable_order(self):
        roster = [make_student(surname=s, name=n) for s, n in
                  [("Rossi", "Anna"), ("Bianchi", "Luca"), ("Rossi", "Aldo"), ("Verdi", "Zoe")]]

        async def test(students, users, session):
            await users.add(make_user())                     # non deve comparire
            for student in roster:
                await students.add(student)
            return await students.get_all()

        found = in_transaction(test)

        assert all(isinstance(s, Student) for s in found)
        assert ids(found) == ids(roster)
        keys = [(s.get_surname(), s.get_name(), s.get_user_id()) for s in found]
        assert keys == sorted(keys), "ordine per cognome, nome, id"

    def test_empty(self):
        async def test(students, users, session):
            return await students.get_all()

        assert in_transaction(test) == []


class TestInheritedLookups:
    """get_by_email/name/surname vengono da UserRepository: devono restituire
    solo studenti, come Student."""

    @DB
    @given(email=emails)
    def test_email_ignores_case_and_returns_a_student(self, email):
        student = make_student(email=email)

        async def test(students, users, session):
            await students.add(student)
            return await students.get_by_email(email.upper())

        found = in_transaction(test)
        assert found == student and isinstance(found, Student)

    @DB
    @given(name=names, count_=st.integers(min_value=1, max_value=3))
    def test_name_returns_students_only(self, name, count_):
        matching = [make_student(name=name) for _ in range(count_)]
        tutor = User.create(user_id=uuid.uuid4(), name=name, surname="Neri",
                            email=f"{uuid.uuid4().hex}@example.com", password=PASSWORD)

        async def test(students, users, session):
            await users.add(tutor)
            for student in matching:
                await students.add(student)
            return await students.get_by_name(name)

        assert ids(in_transaction(test)) == ids(matching)


# ================================================================ add

class TestAddConflicts:

    def test_email_already_used_by_another_user(self):
        async def test(students, users, session):
            await users.add(make_user(email="mario@example.com"))
            with pytest.raises(EmailAlreadyRegistered):
                await students.add(make_student(email="MARIO@example.com"))

        in_transaction(test)

    def test_same_id_is_rejected(self):
        user_id = uuid.uuid4()

        async def test(students, users, session):
            await students.add(make_student(user_id=user_id, email="a@example.com"))
            with pytest.raises(UserAlreadyExists):
                await students.add(make_student(user_id=user_id, email="b@example.com"))

        in_transaction(test)

    def test_id_of_an_existing_non_student_user_is_rejected(self):
        other = make_user()

        async def test(students, users, session):
            await users.add(other)
            with pytest.raises(UserAlreadyExists):
                await students.add(make_student(user_id=other.get_user_id(), email="x@example.com"))
            return await count(session, _STUDENTS, user_id=other.get_user_id())

        assert in_transaction(test) == 0

    def test_failed_add_leaves_no_half_written_rows(self):
        """users e student si scrivono insieme o per niente."""
        first = make_student(email="uno@example.com")
        duplicate = make_student(email="uno@example.com")
        third = make_student(email="tre@example.com")

        async def test(students, users, session):
            await students.add(first)
            with pytest.raises(EmailAlreadyRegistered):
                await students.add(duplicate)
            await students.add(third)              # la transazione e' ancora usabile
            return (await count(session, _USERS, user_id=duplicate.get_user_id()),
                    await count(session, _STUDENTS, user_id=duplicate.get_user_id()),
                    ids(await students.get_all()))

        assert in_transaction(test) == (0, 0, ids([first, third]))


# ============================================================= update

class TestUpdate:

    @DB
    @given(name=names, surname=names, email=emails, password=passwords)
    def test_changes_are_saved(self, name, surname, email, password):
        student = make_student()

        async def test(students, users, session):
            await students.add(student)
            student.set_name(name)
            student.set_surname(surname)
            student.set_email(email)
            student.set_password(password)
            await students.update(student)
            return await students.get_by_id(student.get_user_id())

        loaded = in_transaction(test)

        assert (loaded.get_name(), loaded.get_surname(), loaded.get_email()) == (name, surname, email)
        assert loaded.verify_password(password)
        assert loaded.get_created_at() == student.get_created_at()

    def test_missing_student_raises(self):
        async def test(students, users, session):
            await students.update(make_student())

        with pytest.raises(UserNotFound):
            in_transaction(test)

    def test_a_non_student_user_is_not_modified(self):
        other = make_user()

        async def test(students, users, session):
            await users.add(other)
            impostor = make_student(user_id=other.get_user_id(), name="Cambiato",
                                    email=other.get_email())
            with pytest.raises(UserNotFound):
                await students.update(impostor)
            return await users.get_by_id(other.get_user_id())

        assert in_transaction(test).get_name() == "Paolo"

    def test_email_of_another_user_is_rejected(self):
        mario = make_student(email="mario@example.com")
        luigi = make_student(email="luigi@example.com")

        async def test(students, users, session):
            await students.add(mario)
            await students.add(luigi)
            luigi.set_email("MARIO@example.com")
            with pytest.raises(EmailAlreadyRegistered):
                await students.update(luigi)
            return await students.get_by_id(luigi.get_user_id())

        assert in_transaction(test).get_email() == "luigi@example.com"


# ============================================================= delete

class TestDelete:

    def test_delete_is_idempotent_and_removes_both_rows(self):
        student = make_student()

        async def test(students, users, session):
            await students.add(student)
            first = await students.delete(student.get_user_id())
            second = await students.delete(student.get_user_id())
            return (first, second,
                    await count(session, _USERS, user_id=student.get_user_id()),
                    await count(session, _STUDENTS, user_id=student.get_user_id()))

        assert in_transaction(test) == (True, False, 0, 0)

    def test_a_non_student_user_is_not_deleted(self):
        other = make_user()

        async def test(students, users, session):
            await users.add(other)
            deleted = await students.delete(other.get_user_id())
            return deleted, await users.get_by_id(other.get_user_id())

        deleted, still_there = in_transaction(test)
        assert deleted is False and still_there == other

    def test_deleting_a_student_removes_the_enrollments(self):
        student = make_student()

        async def test(students, users, session):
            await students.add(student)
            corso = await add_corso(session)
            await students.enroll(student.get_user_id(), corso)
            await students.delete(student.get_user_id())
            return (await count(session, ISCRIZIONE, student_id=student.get_user_id()),
                    await count(session, _CORSI, id=corso))

        assert in_transaction(test) == (0, 1), "via l'iscrizione, il corso resta"


# ========================================================== iscrizioni

class TestEnroll:

    def test_enroll_is_idempotent(self):
        student = make_student()

        async def test(students, users, session):
            await students.add(student)
            corso = await add_corso(session)
            first = await students.enroll(student.get_user_id(), corso)
            second = await students.enroll(student.get_user_id(), corso)
            return first, second, await count(session, ISCRIZIONE, student_id=student.get_user_id())

        assert in_transaction(test) == (True, False, 1)

    def test_missing_student(self):
        async def test(students, users, session):
            corso = await add_corso(session)
            with pytest.raises(UserNotFound):
                await students.enroll(uuid.uuid4(), corso)
            return await count(session, ISCRIZIONE)

        assert in_transaction(test) == 0

    def test_a_non_student_user_cannot_enroll(self):
        other = make_user()

        async def test(students, users, session):
            await users.add(other)
            corso = await add_corso(session)
            with pytest.raises(UserNotFound):
                await students.enroll(other.get_user_id(), corso)

        in_transaction(test)

    def test_missing_corso(self):
        student = make_student()
        corso_id = uuid.uuid4()

        async def test(students, users, session):
            await students.add(student)
            with pytest.raises(CorsoNotFound) as raised:
                await students.enroll(student.get_user_id(), corso_id)
            return raised.value

        assert in_transaction(test).corso_id == corso_id

    def test_failed_enroll_leaves_the_transaction_usable(self):
        student = make_student()

        async def test(students, users, session):
            await students.add(student)
            corso = await add_corso(session)
            with pytest.raises(CorsoNotFound):
                await students.enroll(student.get_user_id(), uuid.uuid4())
            return await students.enroll(student.get_user_id(), corso)

        assert in_transaction(test) is True


class TestDisenroll:

    def test_disenroll_is_idempotent(self):
        student = make_student()

        async def test(students, users, session):
            await students.add(student)
            corso = await add_corso(session)
            await students.enroll(student.get_user_id(), corso)
            first = await students.disenroll(student.get_user_id(), corso)
            second = await students.disenroll(student.get_user_id(), corso)
            return first, second, await students.get_all_corsi(student.get_user_id())

        assert in_transaction(test) == (True, False, [])

    def test_only_that_enrollment_is_removed(self):
        anna, luca = make_student(), make_student()

        async def test(students, users, session):
            await students.add(anna)
            await students.add(luca)
            a, b = await add_corso(session, "A"), await add_corso(session, "B")
            for student in (anna, luca):
                for corso in (a, b):
                    await students.enroll(student.get_user_id(), corso)
            await students.disenroll(anna.get_user_id(), a)
            return ([c.get_id() for c in await students.get_all_corsi(anna.get_user_id())],
                    len(await students.get_all_corsi(luca.get_user_id())), b)

        anna_corsi, luca_count, b = in_transaction(test)
        assert anna_corsi == [b] and luca_count == 2


class TestCorsiAndStudents:

    @DB
    @given(enrolled=st.lists(st.booleans(), min_size=1, max_size=6),
           offsets=st.lists(st.integers(min_value=0, max_value=3), min_size=6, max_size=6))
    def test_get_all_corsi_returns_exactly_the_enrolled_ones_in_order(self, enrolled, offsets):
        student = make_student()

        async def test(students, users, session):
            await students.add(student)
            expected = []
            for i, take in enumerate(enrolled):
                # date ripetute apposta: a parita' di data decide il titolo, poi l'id
                start = NOW + timedelta(days=offsets[i])
                corso = await add_corso(session, title=f"Corso {i % 2}", start=start)
                if take:
                    await students.enroll(student.get_user_id(), corso)
                    expected.append((start, f"Corso {i % 2}", corso))
            found = await students.get_all_corsi(student.get_user_id())
            return found, sorted(expected)

        found, expected = in_transaction(test)

        assert [c.get_id() for c in found] == [corso for _, _, corso in expected]

    def test_get_all_corsi_of_a_student_without_courses(self):
        student = make_student()

        async def test(students, users, session):
            await students.add(student)
            await add_corso(session)
            return await students.get_all_corsi(student.get_user_id())

        assert in_transaction(test) == []

    def test_get_enrolled_students(self):
        enrolled = [make_student(surname=s) for s in ("Rossi", "Bianchi", "Verdi")]
        not_enrolled = make_student()

        async def test(students, users, session):
            corso, other = await add_corso(session), await add_corso(session, "Altro")
            for student in enrolled + [not_enrolled]:
                await students.add(student)
            for student in enrolled:
                await students.enroll(student.get_user_id(), corso)
            await students.enroll(not_enrolled.get_user_id(), other)
            return await students.get_enrolled_students(corso), await students.get_enrolled_students(uuid.uuid4())

        found, nobody = in_transaction(test)

        assert ids(found) == ids(enrolled) and all(isinstance(s, Student) for s in found)
        assert [s.get_surname() for s in found] == ["Bianchi", "Rossi", "Verdi"]
        assert nobody == []


# ======================================================= vincoli nel DB

class TestDatabaseConstraints:
    """Le regole valgono anche per chi scrive senza passare dal repository."""

    def test_student_without_user_is_rejected(self):
        async def test(students, users, session):
            await session.execute(insert(_STUDENTS).values(user_id=uuid.uuid4()))

        with pytest.raises(IntegrityError):
            in_transaction(test)

    @pytest.mark.parametrize("student_exists, corso_exists", [(False, True), (True, False)])
    def test_enrollment_needs_student_and_corso(self, student_exists, corso_exists):
        student = make_student()

        async def test(students, users, session):
            if student_exists:
                await students.add(student)
            corso = await add_corso(session) if corso_exists else uuid.uuid4()
            await session.execute(insert(ISCRIZIONE).values(student_id=student.get_user_id(),
                                                            corso_id=corso))

        with pytest.raises(IntegrityError):
            in_transaction(test)

    def test_enrollment_is_unique(self):
        student = make_student()

        async def test(students, users, session):
            await students.add(student)
            corso = await add_corso(session)
            row = dict(student_id=student.get_user_id(), corso_id=corso)
            await session.execute(insert(ISCRIZIONE).values(**row))
            await session.execute(insert(ISCRIZIONE).values(**row))

        with pytest.raises(IntegrityError):
            in_transaction(test)

    @pytest.mark.parametrize("column, value", [("title", ""), ("data_fine", NOW - timedelta(days=1))])
    def test_invalid_corso_is_rejected(self, column, value):
        row = dict(id=uuid.uuid4(), title="Corso", descrizione="d",
                   data_inizio=NOW, data_fine=NOW + timedelta(days=1))
        row[column] = value

        async def test(students, users, session):
            await session.execute(insert(_CORSI).values(**row))

        with pytest.raises(IntegrityError):
            in_transaction(test)
