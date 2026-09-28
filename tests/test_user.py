import string
import uuid
from datetime import datetime, timezone

import pytest
import ujson
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from hypothesis import given, strategies as st
from pydantic import BaseModel, ValidationError

import src.roles.user as user_module
from src.roles.user import User

# argon2 con i parametri di produzione impiega ~125 ms per hash: centinaia di
# esempi di hypothesis diventerebbero minuti, e la deadline di 200 ms farebbe
# fallire i test a caso. Nei test si usa argon2 vero con costi minimi.
FAST_HASHER = PasswordHasher(time_cost=1, memory_cost=8, parallelism=1)


@pytest.fixture(scope="session", autouse=True)
def fast_password_hashing():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(user_module, "PH", FAST_HASHER)
        yield


# ------------------------------------------------------------- strategie

user_ids = st.uuids()
names = st.text(min_size=1, max_size=50)
# pydantic normalizza il dominio in minuscolo: si generano email gia' normalizzate.
emails = st.builds(
    "{}@{}.{}".format,
    st.text(alphabet=string.ascii_lowercase + string.digits, min_size=1, max_size=20),
    st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=15),
    st.sampled_from(["it", "com", "org", "eu"]),
)
passwords = st.text(min_size=1, max_size=64)
dates = st.datetimes(min_value=datetime(1000, 1, 1), timezones=st.just(timezone.utc))
not_emails = st.text(max_size=40).filter(lambda s: "@" not in s)


def not_a_uuid(value):
    try:
        uuid.UUID(value)
        return False
    except (ValueError, AttributeError, TypeError):
        return True


invalid_ids = st.text(max_size=40).filter(not_a_uuid)

NOW = datetime(2024, 3, 9, 14, 5, 6, tzinfo=timezone.utc)


def make_user(user_id=None, name="Mario", surname="Rossi", email="mario@example.com",
              password="segreta", created_at=NOW):
    # `is None`, non `or`: un id vuoto ("") deve arrivare alla classe, non
    # essere sostituito da un UUID valido.
    if user_id is None:
        user_id = uuid.uuid4()
    return User.create(user_id=user_id, name=name, surname=surname, email=email,
                       password=password, created_at=created_at)


def db_record(password_hash=None, **overrides):
    """Una riga del database, come la restituirebbe una query."""
    record = {"user_id": str(uuid.uuid4()), "name": "Mario", "surname": "Rossi",
              "email": "mario@example.com",
              "password_hash": password_hash or FAST_HASHER.hash("segreta"),
              "created_at": NOW.isoformat()}
    record.update(overrides)
    return record


def reload_from_db(user):
    """Ricostruisce l'utente come farebbe il livello di persistenza."""
    return User.from_db({"user_id": str(user.get_user_id()), "name": user.get_name(),
                         "surname": user.get_surname(), "email": user.get_email(),
                         "password_hash": user.get_password_hash(),
                         "created_at": user.get_created_at().isoformat()})


def stored_hash(user):
    return user.get_password_hash()


def verifies(user, password):
    # Verifica indipendente dalla classe: argon2 usato direttamente.
    try:
        return PasswordHasher().verify(stored_hash(user), password)
    except VerifyMismatchError:
        return False


# ============================================================== creazione

class TestUserCreation:

    @given(user_id=user_ids, name=names, surname=names, email=emails, created_at=dates)
    def test_valid_user_stores_every_field(self, user_id, name, surname, email, created_at):
        user = make_user(user_id, name, surname, email, created_at=created_at)

        assert user.get_user_id() == user_id
        assert user.get_name() == name
        assert user.get_surname() == surname
        assert user.get_email() == email
        assert user.get_created_at() == created_at

    @given(email=not_emails)
    def test_invalid_email_is_rejected(self, email):
        with pytest.raises(ValidationError):
            make_user(email=email)

    @given(user_id=invalid_ids)
    def test_user_id_must_be_a_uuid(self, user_id):
        with pytest.raises(ValidationError):
            make_user(user_id=user_id)

    @pytest.mark.parametrize("field", ["name", "surname"])
    def test_empty_name_and_surname_are_rejected(self, field):
        with pytest.raises(ValidationError):
            make_user(**{field: ""})

    def test_undeclared_fields_are_rejected(self):
        with pytest.raises(ValidationError):
            User.from_db(db_record(ruolo="admin"))

    def test_create_defaults_to_now_in_utc(self):
        before = datetime.now(timezone.utc)

        user = User.create(user_id=uuid.uuid4(), name="Mario", surname="Rossi",
                           email="mario@example.com", password="segreta")

        assert before <= user.get_created_at() <= datetime.now(timezone.utc)


# =============================================================== password

class TestPassword:

    @given(password=passwords)
    def test_password_is_never_stored_in_clear(self, password):
        user = make_user(password=password)

        assert stored_hash(user) != password
        assert stored_hash(user).startswith("$argon2")

    @given(password=passwords)
    def test_stored_hash_verifies_the_password(self, password):
        assert verifies(make_user(password=password), password)

    @given(password=passwords, other=passwords)
    def test_wrong_password_does_not_verify(self, password, other):
        user = make_user(password=password)

        assert verifies(user, other) == (other == password)

    @given(password=passwords)
    def test_same_password_gives_different_hashes(self, password):
        # Il sale casuale impedisce di riconoscere chi usa la stessa password.
        assert stored_hash(make_user(password=password)) != stored_hash(make_user(password=password))

    @given(old=passwords, new=passwords)
    def test_set_password_replaces_the_hash(self, old, new):
        user = make_user(password=old)

        user.set_password(new)

        assert verifies(user, new)
        assert verifies(user, old) == (old == new)

    @given(password=passwords, other=passwords)
    def test_verify_password(self, password, other):
        user = make_user(password=password)

        assert user.verify_password(password) is True
        assert user.verify_password(other) == (other == password)

    @given(password=passwords)
    def test_hash_does_not_leak_in_repr(self, password):
        # repr() finisce nei log e nei traceback: niente password ne' hash.
        user = make_user(password=password)

        assert stored_hash(user) not in repr(user)
        assert stored_hash(user) not in str(user)


class TestReloadFromDatabase:

    @given(password=passwords, user_id=user_ids, name=names, email=emails, created_at=dates)
    def test_reloaded_user_can_still_log_in(self, password, user_id, name, email, created_at):
        # Il bug che i test sulla vecchia classe non potevano coprire: l'hash
        # salvato veniva ri-hashato al caricamento e il login falliva.
        original = make_user(user_id, name=name, email=email, password=password,
                             created_at=created_at)

        reloaded = reload_from_db(original)

        assert reloaded.get_password_hash() == original.get_password_hash()
        assert reloaded.verify_password(password)
        assert reloaded == original

    @given(password=passwords.filter(lambda p: not p.startswith("$argon2")))
    def test_plain_password_is_rejected_as_a_hash(self, password):
        with pytest.raises(ValidationError):
            User.from_db(db_record(password_hash=password))

    def test_corrupted_hash_does_not_verify(self):
        user = User.from_db(db_record(password_hash="$argon2id$non-un-hash-valido"))

        assert user.verify_password("qualsiasi") is False

    def test_outdated_hash_needs_rehash(self):
        stronger = PasswordHasher(time_cost=2, memory_cost=16, parallelism=1)
        user = make_user(password="segreta")
        old = User.from_db(db_record(password_hash=stronger.hash("segreta")))

        assert user.needs_rehash() is False  # hash con i parametri correnti
        assert old.needs_rehash() is True    # hash con parametri diversi

        old.set_password("segreta")
        assert old.needs_rehash() is False


class TestConstruction:

    def test_direct_construction_is_refused(self):
        with pytest.raises(TypeError, match="User.create"):
            User(**db_record())

    def test_positional_construction_is_refused(self):
        with pytest.raises(TypeError):
            User(uuid.uuid4(), "Mario", "Rossi", "mario@example.com", "segreta", NOW)

    def test_model_validate_is_refused(self):
        # Anche la strada di pydantic passa dal costruttore protetto.
        with pytest.raises(TypeError, match="User.from_db"):
            User.model_validate(db_record())

    def test_nested_models_accept_only_built_users(self):
        class Subscription(BaseModel):
            user: User

        user = make_user()

        assert Subscription(user=user).user == user
        with pytest.raises(TypeError):
            Subscription(user=db_record())


# ================================================================= setter

class TestSetters:

    @given(name=names, surname=names, email=emails, created_at=dates)
    def test_setters_update_the_fields(self, name, surname, email, created_at):
        user = make_user()

        user.set_name(name)
        user.set_surname(surname)
        user.set_email(email)
        user.set_created_at(created_at)

        assert (user.get_name(), user.get_surname(), user.get_email(), user.get_created_at()) == \
               (name, surname, email, created_at)

    @given(email=not_emails)
    def test_invalid_email_is_rejected_and_the_old_one_kept(self, email):
        user = make_user(email="mario@example.com")

        with pytest.raises(ValidationError):
            user.set_email(email)

        assert user.get_email() == "mario@example.com"

    @given(user_id=invalid_ids)
    def test_invalid_user_id_is_rejected_and_the_old_one_kept(self, user_id):
        original = uuid.uuid4()
        user = make_user(user_id=original)

        with pytest.raises(ValidationError):
            user.set_user_id(user_id)

        assert user.get_user_id() == original

    def test_empty_name_is_rejected_and_the_old_one_kept(self):
        user = make_user(name="Mario")

        with pytest.raises(ValidationError):
            user.set_name("")

        assert user.get_name() == "Mario"


# ======================================================= uguaglianza/hash

class TestEqualityAndHash:

    @given(user_id=user_ids, name_a=names, name_b=names, email_a=emails, email_b=emails)
    def test_equal_users_have_equal_hashes(self, user_id, name_a, name_b, email_a, email_b):
        # Contratto di Python: a == b implica hash(a) == hash(b), altrimenti
        # set e dict perdono o duplicano gli utenti.
        pairs = [
            (make_user(user_id, name=name_a, email=email_a), make_user(user_id, name=name_b, email=email_b)),
            (make_user(name=name_a, email=email_a), make_user(name=name_a, email=email_a)),
        ]
        for a, b in pairs:
            if a == b:
                assert hash(a) == hash(b)

    @given(name_a=names, name_b=names)
    def test_equality_is_transitive(self, name_a, name_b):
        shared_id = uuid.uuid4()
        a = make_user(shared_id, name=name_a)
        b = make_user(shared_id, name=name_b)
        c = make_user(name=name_b)

        if a == b and b == c:
            assert a == c

    @given(user_id=user_ids, other=st.one_of(st.none(), st.integers(), st.text(), st.uuids()))
    def test_comparison_with_other_types_is_false(self, user_id, other):
        assert make_user(user_id) != other

    @given(user_id=user_ids)
    def test_a_user_is_equal_to_itself(self, user_id):
        user = make_user(user_id)

        assert user == user
        assert len({user, user}) == 1


# ================================================================ date

class TestCreatedAt:

    def test_format_is_day_first(self):
        assert make_user(created_at=NOW).format_created_at() == "[09/03/2024 - 14:05:06]"

    def test_naive_datetime_is_rejected(self):
        with pytest.raises(ValidationError):
            make_user(created_at=datetime(2024, 3, 9, 14, 5, 6))

    @given(created_at=dates)
    def test_format_has_a_fixed_width(self, created_at):
        formatted = make_user(created_at=created_at).format_created_at()

        assert len(formatted) == len("[09/03/2024 - 14:05:06]")


# ================================================================ JSON

class TestJson:

    @given(user_id=user_ids, name=names, surname=names, email=emails, created_at=dates)
    def test_json_round_trip(self, user_id, name, surname, email, created_at):
        user = make_user(user_id, name, surname, email, created_at=created_at)

        payload = ujson.loads(user.to_json())

        assert payload["id"] == str(user_id)
        assert (payload["name"], payload["surname"], payload["email"]) == (name, surname, email)
        assert datetime.fromisoformat(payload["created_at"]) == created_at

    @given(password=passwords)
    def test_json_never_contains_the_password(self, password):
        # Il JSON finisce nelle risposte delle API e nei log: ne' la password
        # ne' il suo hash (che si puo' attaccare offline) devono uscire.
        user = make_user(password=password)

        payload = ujson.loads(user.to_json())

        assert "password" not in payload
        assert stored_hash(user) not in user.to_json()

    def test_json_keys_are_stable(self):
        payload = ujson.loads(make_user().to_json())

        assert set(payload) == {"id", "name", "surname", "email", "created_at"}
