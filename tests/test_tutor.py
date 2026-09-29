import string
import uuid
from datetime import datetime, timezone

import pytest
import ujson
from argon2 import PasswordHasher
from hypothesis import given, strategies as st
from pydantic import ValidationError

import src.roles.tutor as tutor_module
import src.roles.user as user_module
from src.roles.tutor import Tutor
from src.roles.user import User

# argon2 veloce, come in test_user.py. tutor.py ha un suo PasswordHasher:
# si accelerano entrambi (raising=False nel caso in cui venga tolto).
FAST_HASHER = PasswordHasher(time_cost=1, memory_cost=8, parallelism=1)


@pytest.fixture(scope="session", autouse=True)
def fast_password_hashing():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(user_module, "PH", FAST_HASHER)
        patch.setattr(tutor_module, "PH", FAST_HASHER, raising=False)
        yield


# ------------------------------------------------------------- strategie

PASSWORD = "una-password-lunga-e-sicura"
NOW = datetime(2024, 3, 9, 14, 5, 6, tzinfo=timezone.utc)

words = st.text(alphabet=string.ascii_letters + "àèéìòù'", min_size=1, max_size=20)
# Materie con spazi e maiuscole variabili: la classe deve riconoscerle come uguali.
subject_names = st.builds(lambda pre, word, post: pre + word + post,
                          st.sampled_from(["", " ", "  "]), words, st.sampled_from(["", " "]))
subject_lists = st.lists(subject_names, min_size=1, max_size=8)
titles = st.text(max_size=60)
bios = st.none() | st.text(max_size=300)


def expected_subjects(subjects):
    """La regola di Tutor.remove_duplicates: due materie sono uguali se
    coincidono ignorando spazi ai lati e maiuscole; resta la prima, com'e' scritta."""
    seen, result = set(), []
    for subject in subjects:
        key = subject.strip().casefold()
        if key and key not in seen:
            seen.add(key)
            result.append(subject)
    return result


def make_tutor(subjects=("Diritto civile",), title="Avvocata", bio=None, user_id=None):
    return Tutor.create(user_id=user_id or uuid.uuid4(), name="Anna", surname="Bianchi",
                        email="anna@example.com", password=PASSWORD, tutor_title=title,
                        subjects=list(subjects), bio=bio, created_at=NOW)


# ============================================================ creazione

class TestCreation:

    @given(title=titles, subjects=subject_lists, bio=bios)
    def test_create_stores_user_and_tutor_fields(self, title, subjects, bio):
        tutor = make_tutor(subjects, title, bio)

        assert isinstance(tutor, Tutor) and isinstance(tutor, User)
        assert (tutor.get_name(), tutor.get_surname(), tutor.get_email()) == \
               ("Anna", "Bianchi", "anna@example.com")
        assert tutor.get_created_at() == NOW
        assert tutor.verify_password(PASSWORD)
        assert tutor.get_password_hash() != PASSWORD
        assert (tutor.get_title(), tutor.get_bio()) == (title, bio)
        assert tutor.get_subjects() == expected_subjects(subjects)

    def test_bio_is_optional(self):
        tutor = Tutor.create(user_id=uuid.uuid4(), name="Anna", surname="Bianchi",
                             email="anna@example.com", password=PASSWORD,
                             tutor_title="Avvocata", subjects=["Contratti"])

        assert tutor.get_bio() is None

    @pytest.mark.parametrize("subjects", [[], [""], ["   "], ["", "  "]])
    def test_at_least_one_real_subject_is_required(self, subjects):
        with pytest.raises(ValidationError):
            make_tutor(subjects=subjects)

    def test_direct_construction_is_refused(self):
        # Il blocco sul costruttore di User vale anche per Tutor.
        with pytest.raises(TypeError):
            Tutor(user_id=uuid.uuid4(), name="Anna", surname="Bianchi",
                  email="anna@example.com", password_hash=FAST_HASHER.hash(PASSWORD),
                  created_at=NOW, title="Avvocata", bio=None, subjects=["Contratti"])

    @given(subjects=subject_lists, bio=bios)
    def test_reload_from_db_keeps_hash_and_fields(self, subjects, bio):
        original = make_tutor(subjects, bio=bio)
        record = {"user_id": original.get_user_id(), "name": "Anna", "surname": "Bianchi",
                  "email": "anna@example.com", "password_hash": original.get_password_hash(),
                  "created_at": NOW, "title": "Avvocata", "bio": bio,
                  "subjects": original.get_subjects()}

        reloaded = Tutor.from_db(record)

        assert isinstance(reloaded, Tutor)
        assert reloaded == original
        assert reloaded.get_password_hash() == original.get_password_hash()
        assert reloaded.verify_password(PASSWORD)
        assert reloaded.get_subjects() == original.get_subjects()


# ======================================================= aggiunta materie

class TestAddSubjects:

    @given(initial=subject_lists, new=subject_names)
    def test_add_subject(self, initial, new):
        tutor = make_tutor(initial)

        tutor.add_subject(new)

        assert tutor.get_subjects() == expected_subjects(initial + [new])

    @given(initial=subject_lists, new=st.lists(subject_names, min_size=1, max_size=8))
    def test_add_subjects(self, initial, new):
        tutor = make_tutor(initial)

        tutor.add_subjects(new)

        assert tutor.get_subjects() == expected_subjects(initial + new)

    @pytest.mark.parametrize("not_a_list", [[], "Contratti", None])
    def test_add_subjects_wants_a_non_empty_list(self, not_a_list):
        tutor = make_tutor(["Contratti"])

        with pytest.raises(Exception):
            tutor.add_subjects(not_a_list)

        assert tutor.get_subjects() == ["Contratti"]

    def test_same_subject_in_another_case_is_not_added_twice(self):
        tutor = make_tutor(["Contratti"])

        tutor.add_subject(" contratti ")

        assert tutor.get_subjects() == ["Contratti"]


# ====================================================== rimozione materie

class TestRemoveSubjects:

    def test_remove_ignores_case_and_spaces_like_add(self):
        tutor = make_tutor(["Diritto civile", "Contratti"])

        tutor.remove_subject(" contratti ")

        assert tutor.get_subjects() == ["Diritto civile"]

    def test_removing_a_missing_subject_changes_nothing(self):
        tutor = make_tutor(["Diritto civile", "Contratti"])

        tutor.remove_subject("Astronomia")

        assert tutor.get_subjects() == ["Diritto civile", "Contratti"]

    def test_last_subject_cannot_be_removed(self):
        # subjects e' dichiarato con min_length=1.
        tutor = make_tutor(["Contratti"])

        with pytest.raises(ValidationError):
            tutor.remove_subject("Contratti")

        assert tutor.get_subjects() == ["Contratti"]

    @given(subjects=st.lists(words, min_size=2, max_size=8, unique_by=str.casefold), data=st.data())
    def test_remove_subjects_ignores_case_and_missing_subjects(self, subjects, data):
        tutor = make_tutor(subjects)
        to_remove = data.draw(st.lists(st.sampled_from(subjects), max_size=len(subjects) - 1,
                                       unique=True))

        tutor.remove_subjects([s.upper() for s in to_remove] + ["Materia che non esiste"])

        assert tutor.get_subjects() == [s for s in subjects if s not in to_remove]

    @given(subjects=st.lists(words, min_size=1, max_size=5, unique_by=str.casefold))
    def test_removing_every_subject_is_refused(self, subjects):
        tutor = make_tutor(subjects)

        with pytest.raises(ValidationError):
            tutor.remove_subjects(subjects)

        assert tutor.get_subjects() == subjects

    def test_remove_subjects_wants_a_list(self):
        tutor = make_tutor(["Contratti", "Diritto civile"])

        with pytest.raises(Exception):
            tutor.remove_subjects("Contratti")

        assert tutor.get_subjects() == ["Contratti", "Diritto civile"]


# =========================================================== set_subjects

class TestSetSubjects:

    @given(initial=subject_lists, new=subject_lists)
    def test_set_subjects_applies_the_same_rules_as_create(self, initial, new):
        tutor = make_tutor(initial)

        tutor.set_subjects(new)

        assert tutor.get_subjects() == expected_subjects(new)

    def test_empty_list_is_rejected_and_the_old_one_kept(self):
        tutor = make_tutor(["Contratti"])

        with pytest.raises(ValidationError):
            tutor.set_subjects([])

        assert tutor.get_subjects() == ["Contratti"]

    def test_a_single_string_is_not_split_into_letters(self):
        # set_subjects si aspetta una lista: passare "Contratti" per errore non
        # deve diventare ["C", "o", "n", "t", "r", "a", "i"].
        tutor = make_tutor(["Diritto civile"])

        try:
            tutor.set_subjects("Contratti")
        except (TypeError, ValidationError):
            pass  # rifiutare la stringa va bene

        assert tutor.get_subjects() in (["Diritto civile"], ["Contratti"])


# ============================================================ altri campi

class TestSetters:

    @given(title=titles, bio=bios)
    def test_title_and_bio(self, title, bio):
        tutor = make_tutor()

        tutor.set_title(title)
        tutor.set_bio(bio)

        assert (tutor.get_title(), tutor.get_bio()) == (title, bio)


class TestFormatAndJson:

    def test_format_subjects(self):
        assert make_tutor(["Diritto civile", "Contratti"]).format_subjects() == \
               "Diritto civile; Contratti"

    @given(subjects=subject_lists, bio=bios, title=titles)
    def test_json_contains_the_tutor_fields(self, subjects, bio, title):
        tutor = make_tutor(subjects, title=title, bio=bio)

        payload = ujson.loads(tutor.to_json())

        assert payload["id"] == str(tutor.get_user_id())
        assert (payload["title"], payload["bio"]) == (title, bio)
        assert payload["subjects"] == tutor.format_subjects()
        assert "password" not in payload and "password_hash" not in payload


class TestIdentity:

    def test_tutor_and_user_with_the_same_id_are_the_same_person(self):
        user_id = uuid.uuid4()
        tutor = make_tutor(user_id=user_id)
        user = User.from_db({"user_id": user_id, "name": "Anna", "surname": "Bianchi",
                             "email": "anna@example.com",
                             "password_hash": tutor.get_password_hash(), "created_at": NOW})

        assert tutor == user and user == tutor
        assert hash(tutor) == hash(user)