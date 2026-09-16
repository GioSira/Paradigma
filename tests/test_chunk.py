from datetime import datetime, timedelta
 
import pytest
import ujson
from hypothesis import given, strategies as st
from pydantic import ValidationError
 
from src.collection.document import Document
from src.chunks.document_chunk import DocumentChunk


valid_ids = st.text(min_size=1, max_size=50)
chunk_texts = st.text(min_size=1, max_size=1000).filter(lambda s: s.strip() != "")
blank_texts = st.text(alphabet=" \t\n\r", min_size=0, max_size=10)
timestamps = st.datetimes()
optional_timestamps = st.none() | timestamps
 
 
@st.composite
def spans(draw, max_start=10000, max_len=1000):
    start = draw(st.integers(min_value=0, max_value=max_start))
    length = draw(st.integers(min_value=1, max_value=max_len))
    return start, start + length
 
 
def make_chunk(chunk_id, document_id, last_modified=None,
               start_idx=0, end_idx=0, text=""):
    return DocumentChunk(
        chunk_id=chunk_id, document_id=document_id, last_modified=last_modified,
        start_idx=start_idx, end_idx=end_idx, text=text,
    )
 
 
def make_document(doc_id, name, last_modified=None, text=""):
    return Document(doc_id=doc_id, name=name, last_modified=last_modified, text=text)



class TestChunkCreation:
 
    @given(chunk_id=valid_ids, doc_id=valid_ids, span=spans(),
           text=chunk_texts, modified=optional_timestamps)
    def test_creation_stores_every_field(self, chunk_id, doc_id, span, text, modified):

        start, end = span
        c = make_chunk(chunk_id, doc_id, modified, start, end, text)
 
        assert c.get_chunk_id() == chunk_id
        assert c.get_document_id() == doc_id
        assert c.get_start_index() == start
        assert c.get_end_index() == end
        assert c.get_chunk_text() == text
        assert c.get_last_modified() == modified

 
    @given(doc_id=valid_ids, span=spans(), text=chunk_texts)
    def test_empty_chunk_id_is_rejected(self, doc_id, span, text):
        with pytest.raises(ValidationError):
            make_chunk("", doc_id, start_idx=span[0], end_idx=span[1], text=text)

 
    @given(chunk=valid_ids, span=spans(), text=chunk_texts)
    def test_empty_document_id_is_rejected(self, chunk, span, text):
        with pytest.raises(ValidationError):
            make_chunk(chunk, "", start_idx=span[0], end_idx=span[1], text=text)

 
    @given(chunk=valid_ids, doc_id=valid_ids, span=spans(), text=blank_texts)
    def test_blank_text_is_rejected(self, chunk, doc_id, span, text):
        with pytest.raises(ValidationError):
            make_chunk(chunk, doc_id, start_idx=span[0], end_idx=span[1], text="")


class TestSpanValidation:
 
    @given(chunk=valid_ids, doc_id=valid_ids, span=spans(), text=chunk_texts)
    def test_inverted_span_is_rejected(self, chunk, doc_id, span, text):
        start, end = span
 
        with pytest.raises(ValidationError):
            make_chunk(chunk, doc_id, start_idx=end, end_idx=start, text=text)
 
    @given(chunk=valid_ids, doc_id=valid_ids, index=st.integers(min_value=0, max_value=1000),
           text=chunk_texts)
    def test_empty_span_is_rejected(self, chunk, doc_id, index, text):
        with pytest.raises(ValidationError):
            make_chunk(chunk, doc_id, start_idx=index, end_idx=index, text=text)
 
    @given(chunk=valid_ids, doc_id=valid_ids, text=chunk_texts,
           negative=st.integers(min_value=-1000, max_value=-1))
    def test_negative_indexes_are_rejected(self, chunk, doc_id, text, negative):
        with pytest.raises(ValidationError):
            make_chunk(chunk, doc_id, start_idx=negative, end_idx=10, text=text)
 
    @given(chunk=valid_ids, doc_id=valid_ids, span=spans(), text=chunk_texts)
    def test_length_matches_the_span(self, chunk, doc_id, span, text):
        start, end = span
        c = make_chunk(chunk, doc_id, start_idx=start, end_idx=end, text=text)
 
        assert c.get_chunk_length() == end - start
 
    @given(chunk=valid_ids, doc_id=valid_ids, span=spans(), text=chunk_texts)
    def test_setter_cannot_break_the_span(self, chunk, doc_id, span, text):
        start, end = span
        c = make_chunk(chunk, doc_id, start_idx=start, end_idx=end, text=text)
 
        with pytest.raises(ValidationError):
            c.set_end_index(start)
 
        assert c.get_end_index() == end


class TestOverlap:
 
    @given(doc_id=valid_ids, text=chunk_texts)
    def test_adjacent_chunks_do_not_overlap(self, doc_id, text):
        a = make_chunk("a", doc_id, start_idx=0, end_idx=10, text=text)
        b = make_chunk("b", doc_id, start_idx=10, end_idx=20, text=text)
 
        assert a.overlaps(b) is False
        assert b.overlaps(a) is False
 
    @given(doc_id=valid_ids, text=chunk_texts)
    def test_sliding_window_chunks_overlap(self, doc_id, text):
        a = make_chunk("a", doc_id, start_idx=0, end_idx=100, text=text)
        b = make_chunk("b", doc_id, start_idx=80, end_idx=180, text=text)
 
        assert a.overlaps(b) is True
        assert b.overlaps(a) is True
 
    @given(span_a=spans(), span_b=spans(), doc_a=valid_ids, doc_b=valid_ids, text=chunk_texts)
    def test_overlap_is_symmetric(self, span_a, span_b, doc_a, doc_b, text):
        a = make_chunk("a", doc_a, start_idx=span_a[0], end_idx=span_a[1], text=text)
        b = make_chunk("b", doc_b, start_idx=span_b[0], end_idx=span_b[1], text=text)
 
        assert a.overlaps(b) == b.overlaps(a)
 
    @given(span=spans(), doc_a=valid_ids, doc_b=valid_ids, text=chunk_texts)
    def test_different_documents_never_overlap(self, span, doc_a, doc_b, text):
        a = make_chunk("a", doc_a, start_idx=span[0], end_idx=span[1], text=text)
        b = make_chunk("b", doc_b, start_idx=span[0], end_idx=span[1], text=text)
 
        assert a.overlaps(b) == (doc_a == doc_b)



class TestDateFormatting:
 
    @given(chunk=valid_ids, doc_id=valid_ids, span=spans(), text=chunk_texts)
    def test_missing_date_returns_empty_string(self, chunk, doc_id, span, text):
        c = make_chunk(chunk, doc_id, None, span[0], span[1], text)
 
        assert c.get_last_modified() == None

    @given(chunk_id=valid_ids, doc_id=valid_ids, span=spans(), text=chunk_texts)
    def test_date_is_day_first(self, chunk_id, doc_id, span, text):
        c = make_chunk(chunk_id, doc_id, datetime(2024, 3, 9, 14, 5, 6), span[0], span[1], text)
 
        assert c.format_last_modified() == "[09/03/2024 - 14:05:06]"
 
    @given(chunk=valid_ids, doc_id=valid_ids, span=spans(), text=chunk_texts, modified=timestamps)
    def test_any_date_formats_to_a_fixed_width(self, chunk, doc_id, span, text, modified):
        c = make_chunk(chunk, doc_id, modified, span[0], span[1], text)
 
        assert len(c.format_last_modified()) == len("[09/03/2024 - 14:05:06]")
 


class TestStaleness:
 
    @given(chunk_id=valid_ids, doc_name=chunk_texts, doc_id=valid_ids, modified=timestamps)
    def test_fresh_chunk_is_not_stale(self, chunk_id, doc_id, doc_name, modified):
        document = make_document(doc_id, doc_name, last_modified=modified)
        c = make_chunk(chunk_id, doc_id, modified, 0, 10, text=doc_name)
        assert c.is_stale(document) == False
 
    @given(chunk_id=valid_ids, doc_name=chunk_texts, doc_id=valid_ids, modified=timestamps)
    def test_chunk_is_stale_when_the_document_changes(self, chunk_id, doc_id, doc_name, modified):
        document = make_document(doc_id, doc_name, last_modified=modified + timedelta(seconds=1))
        c = make_chunk(chunk_id, doc_id, modified, 0, 10, text=doc_name)
 
        assert c.is_stale(document) is True
 
    @given(chunk_id=valid_ids, doc_name=chunk_texts, doc_id=valid_ids, modified=timestamps)
    def test_missing_date_on_one_side_counts_as_stale(self, chunk_id, doc_id, doc_name, modified):
        document = make_document(doc_id, doc_name)
        c = make_chunk(chunk_id, doc_id, modified, 0, 10, text=doc_name)
 
        assert c.is_stale(document) is True
 
    @given(chunk_id=valid_ids, doc_name=chunk_texts, doc_id=valid_ids)
    def test_missing_date_on_both_sides(self, chunk_id, doc_id, doc_name):
        document = make_document(doc_id, doc_name)
        c = make_chunk(chunk_id, doc_id, last_modified=None, start_idx=0, end_idx=10, text=doc_name)
 
        assert c.is_stale(document) is False



class TestChunkIdentity:
 
    @given(chunk=valid_ids, doc_id=valid_ids, span=spans(), text_a=chunk_texts, text_b=chunk_texts)
    def test_same_ids_are_the_same_chunk(self, chunk, doc_id, span, text_a, text_b):
        # Il testo non fa identita': un chunk re-embeddato resta lo stesso vettore.
        start, end = span
        a = make_chunk(chunk, doc_id, start_idx=start, end_idx=end, text=text_a)
        b = make_chunk(chunk, doc_id, start_idx=start, end_idx=end, text=text_b)
 
        assert a == b
        assert hash(a) == hash(b)
        assert len({a, b}) == 1
 
    @given(id_a=valid_ids, id_b=valid_ids, doc_id=valid_ids, span=spans(), text=chunk_texts)
    def test_chunk_id_separates_chunks(self, id_a, id_b, doc_id, span, text):
        a = make_chunk(id_a, doc_id, start_idx=span[0], end_idx=span[1], text=text)
        b = make_chunk(id_b, doc_id, start_idx=span[0], end_idx=span[1], text=text)
 
        assert (a == b) == (id_a == id_b)
 
    @given(chunk=valid_ids, doc_a=valid_ids, doc_b=valid_ids, span=spans(), text=chunk_texts)
    def test_same_chunk_id_in_different_documents_is_not_equal(self, chunk, doc_a, doc_b, span, text):
        a = make_chunk(chunk, doc_a, start_idx=span[0], end_idx=span[1], text=text)
        b = make_chunk(chunk, doc_b, start_idx=span[0], end_idx=span[1], text=text)
 
        assert (a == b) == (doc_a == doc_b)
 
    @given(doc_id=valid_ids, count=st.integers(min_value=1, max_value=30), text=chunk_texts)
    def test_chunks_of_one_document_stay_distinct(self, doc_id, count, text):
        chunks = {
            make_chunk(f"c{i}", doc_id, start_idx=i * 10, end_idx=i * 10 + 10, text=text)
            for i in range(count)
        }
 
        assert len(chunks) == count
 
    @given(chunk=valid_ids, doc_id=valid_ids, span=spans(), text=chunk_texts,
           other=st.one_of(st.none(), st.integers(), st.text()))
    def test_comparison_with_other_types_is_false(self, chunk, doc_id, span, text, other):
        c = make_chunk(chunk, doc_id, start_idx=span[0], end_idx=span[1], text=text)
 
        assert c != other
 
    @given(chunk=valid_ids, doc_id=valid_ids, doc_name=chunk_texts, span=spans(), text=chunk_texts)
    def test_not_equal_to_a_document(self, chunk, doc_id, doc_name, span, text):
        c = make_chunk(chunk, doc_id, start_idx=span[0], end_idx=span[1], text=text)
        document = make_document(doc_id, doc_name)
 
        assert c != document
        assert document != c


class TestChunkSerialization:
 
    @given(chunk=valid_ids, doc_id=valid_ids, span=spans(), text=chunk_texts)
    def test_json_round_trip_without_date(self, chunk, doc_id, span, text):
        c = make_chunk(chunk, doc_id, None, span[0], span[1], text)
 
        payload = ujson.loads(c.to_json())
 
        assert payload["chunk_id"] == chunk
        assert payload["document_id"] == doc_id
        assert payload["start_index"] == span[0]
        assert payload["end_index"] == span[1]
        assert payload["text"] == text
        assert payload["last_modified"] is None
 
    @given(chunk=valid_ids, doc_id=valid_ids, span=spans(), text=chunk_texts, modified=timestamps)
    def test_json_date_is_iso_not_display_format(self, chunk, doc_id, span, text, modified):
        c = make_chunk(chunk, doc_id, modified, span[0], span[1], text)
 
        assert ujson.loads(c.to_json())["last_modified"] == modified.isoformat()
 
    @given(chunk=valid_ids, doc_id=valid_ids, span=spans(), text=chunk_texts)
    def test_json_keys_are_stable(self, chunk, doc_id, span, text):
        payload = ujson.loads(make_chunk(chunk, doc_id, None, span[0], span[1], text).to_json())
 
        assert set(payload) == {
            "chunk_id", "document_id", "start_index", "end_index", "last_modified", "text",
        }