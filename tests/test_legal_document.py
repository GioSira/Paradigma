import pytest
import ujson
from hypothesis import given, strategies as st
from pydantic import ValidationError

from src.collection.document import Document
from src.collection.legal_document import LegalDocument

valid_ids = st.text(min_size=1, max_size=50)
messy_strings = st.text(min_size=1, max_size=1000)
optional_strings = st.none() | messy_strings
doc_types = st.text(min_size=1, max_size=50)
doc_numbers = st.none() | st.integers(min_value=-1000, max_value=100000)
timestamps = st.datetimes()


def make_legal(doc_id="id", name="name", last_modified=None, text=None,
               document_type="contratto", document_number=None):
    return LegalDocument(
        doc_id=doc_id, name=name, last_modified=last_modified, text=text,
        document_type=document_type, document_number=document_number,
    )


def make_plain(doc_id="id", name="name"):
    return Document(doc_id=doc_id, name=name, last_modified=None, text=None)


class TestLegalDocumentCreation:

    @given(docum_id=valid_ids, doc_name=messy_strings, doc_type=doc_types, number=doc_numbers)
    def test_creation_stores_every_field(self, docum_id, doc_name, doc_type, number):
        doc = make_legal(docum_id, doc_name, document_type=doc_type, document_number=number)

        assert doc.document_id == docum_id
        assert doc.get_document_name() == doc_name
        assert doc.get_document_type() == doc_type
        assert doc.get_document_number() == number


    @given(docum_id=valid_ids, doc_name=messy_strings)
    def test_document_type_is_mandatory(self, docum_id, doc_name):
        with pytest.raises(ValidationError):
            LegalDocument(doc_id=docum_id, name=doc_name, last_modified=None, text=None,
                          document_number=None)


    @given(doc_name=messy_strings, doc_type=doc_types)
    def test_parent_validation_still_applies(self, doc_name, doc_type):
        with pytest.raises(ValidationError):
            make_legal("", doc_name, document_type=doc_type)


    @given(docum_id=valid_ids, doc_name=messy_strings, doc_type=doc_types)
    def test_number_must_be_an_int(self, docum_id, doc_name, doc_type):
        with pytest.raises(ValidationError):
            make_legal(docum_id, doc_name, document_type=doc_type, document_number="sette")


    @given(docum_id=valid_ids, doc_name=messy_strings, doc_type=doc_types)
    def test_is_a_document(self, docum_id, doc_name, doc_type):
        assert isinstance(make_legal(docum_id, doc_name, document_type=doc_type), Document)


class TestLegalAccessors:

    @given(doc_type=doc_types, new_type=doc_types)
    def test_set_document_type(self, doc_type, new_type):
        doc = make_legal(document_type=doc_type)

        doc.set_document_type(new_type)

        assert doc.get_document_type() == new_type


    @given(number=doc_numbers, new_number=doc_numbers)
    def test_set_document_number(self, number, new_number):
        doc = make_legal(document_number=number)

        doc.set_document_number(new_number)

        assert doc.get_document_number() == new_number


    @given(doc_type=doc_types)
    def test_setter_validation_is_inherited(self, doc_type):
        doc = make_legal(document_type=doc_type)

        with pytest.raises(ValidationError):
            doc.set_document_id("")


    @given(modified=timestamps)
    def test_inherits_date_formatting(self, modified):
        doc = make_legal(last_modified=modified)

        assert doc.format_last_modified().startswith("[")


    def test_inherits_empty_date(self):
        assert make_legal().format_last_modified() == ""


class TestLegalEqualityAndHash:

    @given(docum_id=valid_ids, doc_name=messy_strings, doc_type=doc_types, number=doc_numbers)
    def test_is_hashable(self, docum_id, doc_name, doc_type, number):
        doc = make_legal(docum_id, doc_name, document_type=doc_type, document_number=number)

        assert isinstance(hash(doc), int)


    @given(docum_id=valid_ids, doc_name=messy_strings, doc_type=doc_types, number=doc_numbers)
    def test_identical_documents_are_equal(self, docum_id, doc_name, doc_type, number):
        args = dict(document_type=doc_type, document_number=number)

        assert make_legal(docum_id, doc_name, **args) == make_legal(docum_id, doc_name, **args)


    @given(docum_id=valid_ids, doc_name=messy_strings, doc_type=doc_types)
    def test_missing_number_does_not_break_equality(self, docum_id, doc_name, doc_type):
        # None or "" trasformava self in "" lasciando other a None: due documenti
        # senza numero risultavano diversi.
        doc_a = make_legal(docum_id, doc_name, document_type=doc_type, document_number=None)
        doc_b = make_legal(docum_id, doc_name, document_type=doc_type, document_number=None)

        assert doc_a == doc_b
        assert hash(doc_a) == hash(doc_b)


    @given(docum_id=valid_ids, doc_name=messy_strings, doc_type=doc_types)
    def test_zero_is_not_the_same_as_missing(self, docum_id, doc_name, doc_type):
        zero = make_legal(docum_id, doc_name, document_type=doc_type, document_number=0)
        missing = make_legal(docum_id, doc_name, document_type=doc_type, document_number=None)

        assert zero != missing


    @given(docum_id=valid_ids, doc_name=messy_strings, type_a=doc_types, type_b=doc_types)
    def test_type_is_part_of_the_identity(self, docum_id, doc_name, type_a, type_b):
        doc_a = make_legal(docum_id, doc_name, document_type=type_a)
        doc_b = make_legal(docum_id, doc_name, document_type=type_b)

        assert (doc_a == doc_b) == (type_a == type_b)


    @given(docum_id=valid_ids, doc_name=messy_strings, doc_type=doc_types,
           num_a=doc_numbers, num_b=doc_numbers)
    def test_number_is_part_of_the_identity(self, docum_id, doc_name, doc_type, num_a, num_b):
        doc_a = make_legal(docum_id, doc_name, document_type=doc_type, document_number=num_a)
        doc_b = make_legal(docum_id, doc_name, document_type=doc_type, document_number=num_b)

        assert (doc_a == doc_b) == (num_a == num_b)


    @given(docum_id=valid_ids, doc_name=messy_strings, doc_type=doc_types,
           num=doc_numbers, text_a=messy_strings, text_b=messy_strings)
    def test_equal_objects_share_the_same_hash(self, docum_id, doc_name, doc_type, num, text_a, text_b):
        doc_a = make_legal(docum_id, doc_name, text=text_a, document_type=doc_type, document_number=num)
        doc_b = make_legal(docum_id, doc_name, text=text_b, document_type=doc_type, document_number=num)

        assert doc_a == doc_b
        assert hash(doc_a) == hash(doc_b)
        assert len({doc_a, doc_b}) == 1

    @given(docum_id=valid_ids, doc_name=messy_strings, doc_type=doc_types,
           other=st.one_of(st.none(), st.integers(), st.text()))
    def test_comparison_with_other_types_is_false(self, docum_id, doc_name, doc_type, other):
        assert make_legal(docum_id, doc_name, document_type=doc_type) != other


    @given(docum_id=valid_ids, doc_name=messy_strings, doc_type=doc_types)
    def test_not_equal_to_a_plain_document(self, docum_id, doc_name, doc_type):
        legal = make_legal(docum_id, doc_name, document_type=doc_type)
        plain = make_plain(docum_id, doc_name)

        assert legal != plain


    @given(docum_id=valid_ids, doc_name=messy_strings, doc_type=doc_types)
    def test_comparison_with_parent_is_symmetric(self, docum_id, doc_name, doc_type):
        legal = make_legal(docum_id, doc_name, document_type=doc_type)
        plain = make_plain(docum_id, doc_name)

        assert (plain == legal) == (legal == plain)


class TestLegalSerialization:

    @given(docum_id=valid_ids, doc_name=messy_strings, doc_type=doc_types,
           number=doc_numbers, doc_text=optional_strings)
    def test_json_keeps_the_subclass_fields(self, docum_id, doc_name, doc_type, number, doc_text):
        doc = make_legal(docum_id, doc_name, text=doc_text,
                         document_type=doc_type, document_number=number)

        payload = ujson.loads(doc.to_json())

        assert payload["id"] == docum_id
        assert payload["name"] == doc_name
        assert payload["text"] == doc_text
        assert payload["document_type"] == doc_type
        assert payload["document_number"] == number

    @given(modified=timestamps, doc_type=doc_types)
    def test_json_date_stays_iso(self, modified, doc_type):
        doc = make_legal(last_modified=modified, document_type=doc_type)

        assert ujson.loads(doc.to_json())["last_modified"] == modified.isoformat()