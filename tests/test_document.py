from datetime import datetime

import pytest
import ujson
from hypothesis import given, strategies as st
from pydantic import ValidationError

from src.collection.document import Document

valid_ids = st.text(min_size=1, max_size=50)
invalid_ids = st.just("")
messy_strings = st.text(min_size=1, max_size=1000)
optional_strings = st.none() | messy_strings
timestamps = st.datetimes()
optional_timestamps = st.none() | timestamps


def make_doc(doc_id="id", name="name", last_modified=None, text=None):
    """Tutti i campi sono obbligatori in Pydantic v2, anche gli Optional."""
    return Document(doc_id=doc_id, name=name, last_modified=last_modified, text=text)


class TestDocumentCreation:

    @given(docum_id=valid_ids, doc_name=messy_strings)
    def test_document_creation(self, docum_id, doc_name):
        doc = make_doc(docum_id, doc_name)

        assert doc.document_id == docum_id
        assert doc.get_document_name() == doc_name
        assert doc.get_document_text() is None

    @given(
        docum_id=valid_ids,
        doc_name=messy_strings,
        modified=optional_timestamps,
        doc_text=optional_strings,
    )
    def test_all_fields_are_stored(self, docum_id, doc_name, modified, doc_text):
        doc = make_doc(docum_id, doc_name, modified, doc_text)

        assert doc.doc_id == docum_id
        assert doc.name == doc_name
        assert doc.last_modified == modified
        assert doc.text == doc_text

    @given(docum_id=invalid_ids, doc_name=messy_strings)
    def test_empty_id_is_rejected(self, docum_id, doc_name):
        with pytest.raises(ValidationError):
            make_doc(docum_id, doc_name)

    @given(doc_name=messy_strings)
    def test_optional_fields_are_still_required(self, doc_name):
        # Optional[X] senza default != campo facoltativo in Pydantic v2:
        # va passato esplicitamente None.
        with pytest.raises(ValidationError):
            Document(doc_id="id", name=doc_name)

    @given(docum_id=valid_ids)
    def test_name_is_mandatory(self, docum_id):
        with pytest.raises(ValidationError):
            Document(doc_id=docum_id, last_modified=None, text=None)


class TestAccessors:

    @given(docum_id=valid_ids, new_doc_id=valid_ids, doc_name=messy_strings)
    def test_id_change(self, docum_id, doc_name, new_doc_id):
        doc = make_doc(docum_id, doc_name)

        doc.set_document_id(new_doc_id)

        assert doc.document_id == new_doc_id

    @given(docum_id=valid_ids, doc_name=messy_strings, new_doc_name=messy_strings)
    def test_set_name(self, docum_id, doc_name, new_doc_name):
        doc = make_doc(docum_id, doc_name)

        doc.set_document_name(new_doc_name)

        assert doc.get_document_name() == new_doc_name

    @given(
        docum_id=valid_ids,
        doc_name=messy_strings,
        doc_text=messy_strings,
        new_doc_text=messy_strings,
    )
    def test_set_text(self, docum_id, doc_name, doc_text, new_doc_text):
        doc = make_doc(docum_id, doc_name, text=doc_text)

        doc.set_document_text(new_doc_text)

        assert doc.get_document_text() == new_doc_text

    @given(docum_id=valid_ids, doc_name=messy_strings, modified=timestamps)
    def test_set_last_modified(self, docum_id, doc_name, modified):
        doc = make_doc(docum_id, doc_name)

        doc.set_last_modified(modified)

        assert doc.last_modified == modified

    @given(docum_id=valid_ids, doc_name=messy_strings, doc_text=messy_strings)
    def test_setters_do_not_touch_other_fields(self, docum_id, doc_name, doc_text):
        doc = make_doc(docum_id, doc_name, text=doc_text)

        doc.set_document_id("nuovo-id")

        assert doc.get_document_name() == doc_name
        assert doc.get_document_text() == doc_text
        assert doc.last_modified is None

    @given(docum_id=valid_ids, doc_name=messy_strings)
    def test_setter_validates_empty_id(self, docum_id, doc_name):
        doc = make_doc(docum_id, doc_name)

        with pytest.raises(ValidationError):
            doc.set_document_id("")


class TestLastModifiedFormatting:

    @given(docum_id=valid_ids, doc_name=messy_strings)
    def test_missing_date_returns_none(self, docum_id, doc_name):
        doc = make_doc(docum_id, doc_name)

        assert doc.get_last_modified() == None

    #@pytest.mark.xfail(strict=True, reason="stesso bug di isoformat()")
    def test_expected_format_is_day_first(self):
        doc = make_doc(last_modified=datetime(2024, 3, 9, 14, 5, 6))

        assert doc.format_last_modified() == "[09/03/2024 - 14:05:06]"


class TestEqualityAndHash:

    @given(docum_id=valid_ids, doc_name=messy_strings, doc_text=optional_strings)
    def test_equality_is_reflexive(self, docum_id, doc_name, doc_text):
        doc = make_doc(docum_id, doc_name, text=doc_text)

        assert doc == doc

    @given(docum_id=valid_ids, doc_name=messy_strings)
    def test_same_id_and_name_are_equal(self, docum_id, doc_name):
        assert make_doc(docum_id, doc_name) == make_doc(docum_id, doc_name)

    @given(docum_id=valid_ids, other_id=valid_ids, doc_name=messy_strings)
    def test_different_id_is_not_equal(self, docum_id, other_id, doc_name):
        doc_a = make_doc(docum_id, doc_name)
        doc_b = make_doc(other_id, doc_name)

        assert (doc_a == doc_b) == (docum_id == other_id)

    @given(docum_id=valid_ids, doc_name=messy_strings, name_b=messy_strings)
    def test_different_name_is_not_equal(self, docum_id, doc_name, name_b):
        doc_a = make_doc(docum_id, doc_name)
        doc_b = make_doc(docum_id, name_b)

        assert (doc_a == doc_b) == (doc_name == name_b)

    @given(docum_id=valid_ids, doc_name=messy_strings, text_a=messy_strings, text_b=messy_strings)
    def test_text_is_ignored_by_equality(self, docum_id, doc_name, text_a, text_b):
        # Comportamento voluto? Due documenti con contenuto diverso risultano uguali.
        assert make_doc(docum_id, doc_name, text=text_a) == make_doc(docum_id, doc_name, text=text_b)

    @given(
        docum_id=valid_ids,
        doc_name=messy_strings,
        text_a=messy_strings,
        text_b=messy_strings,
    )
    def test_equal_objects_share_the_same_hash(self, docum_id, doc_name, text_a, text_b):
        doc_a = make_doc(docum_id, doc_name, text=text_a)
        doc_b = make_doc(docum_id, doc_name, text=text_b)

        assert doc_a == doc_b
        assert hash(doc_a) == hash(doc_b)

    @given(docum_id=valid_ids, doc_name=messy_strings, text_a=messy_strings, text_b=messy_strings)
    def test_set_deduplicates_equal_documents(self, docum_id, doc_name, text_a, text_b):
        doc_a = make_doc(docum_id, doc_name, text=text_a)
        doc_b = make_doc(docum_id, doc_name, text=text_b)

        assert len({doc_a, doc_b}) == 1

    @given(docum_id=valid_ids, doc_name=messy_strings, modified=timestamps)
    def test_hash_works_with_a_date(self, docum_id, doc_name, modified):
        assert isinstance(hash(make_doc(docum_id, doc_name, modified)), int)

    @given(docum_id=valid_ids, doc_name=messy_strings, other=st.one_of(st.none(), st.integers(), st.text()))
    def test_comparison_with_other_types_is_false(self, docum_id, doc_name, other):
        assert make_doc(docum_id, doc_name) != other


class TestSerialization:

    @given(docum_id=valid_ids, doc_name=messy_strings, doc_text=optional_strings)
    def test_json_round_trip_without_date(self, docum_id, doc_name, doc_text):
        doc = make_doc(docum_id, doc_name, text=doc_text)

        payload = ujson.loads(doc.to_json())

        assert payload["id"] == docum_id
        assert payload["name"] == doc_name
        assert payload["text"] == doc_text
        assert payload["last_modified"] is None

    @given(docum_id=valid_ids, doc_name=messy_strings)
    def test_json_key_is_id_not_doc_id(self, docum_id, doc_name):
        payload = ujson.loads(make_doc(docum_id, doc_name).to_json())

        assert set(payload) == {"id", "name", "last_modified", "text"}

    @given(docum_id=valid_ids, doc_name=messy_strings, modified=timestamps)
    def test_json_serializes_a_date(self, docum_id, doc_name, modified):
        payload = ujson.loads(make_doc(docum_id, doc_name, modified).to_json())

        assert payload["last_modified"] == modified.isoformat()