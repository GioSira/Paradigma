"""Test di integrazione di PineconeDB su Pinecone reale.

Richiedono PINECONE_API_KEY (nell'ambiente o nel file .env). Senza chiave
vengono saltati.

Indici usati (creati alla prima esecuzione e riusati nelle successive):
    PINECONE_TEST_INDEX          indice a embedding integrato  (default: pinecone-db-test)
    PINECONE_TEST_VECTOR_INDEX   indice a vettori, dimensione 8 (default: pinecone-db-test-vectors)
Con PINECONE_TEST_CLEANUP=1 vengono cancellati a fine sessione.

Ogni test scrive in un namespace nuovo, cancellato alla fine, quindi i test
non si influenzano tra loro e l'indice resta vuoto.

Esempi:
    pytest tests/test_pinecone_db.py               # tutti
    pytest tests/test_pinecone_db.py -m "not slow" # senza creare indici temporanei
"""


import os
import string
import time
import uuid
from contextlib import contextmanager
from datetime import datetime

import pytest
from dotenv import load_dotenv
from hypothesis import HealthCheck, Phase, example, given, settings, strategies as st
from pinecone import ServerlessSpec

# Stesso percorso usato da pinecone_db: importando da un percorso diverso
# (es. src.chunks...) Python carica la classe due volte e i chunk non
# risultano mai uguali, perche' __eq__ fa isinstance su due classi distinte.
from src.chunks.document_chunk import DocumentChunk
from src.db.pinecone_db import IndexMode, PineconeDB

load_dotenv()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get("PINECONE_API_KEY"),
                       reason="PINECONE_API_KEY non impostata"),
]

TEST_INDEX = os.environ.get("PINECONE_TEST_INDEX", "pinecone-db-test")
VECTOR_INDEX = os.environ.get("PINECONE_TEST_VECTOR_INDEX", "pinecone-db-test-vectors")
VECTOR_DIMENSION = 8
RERANK_MODEL = os.environ.get("PINECONE_TEST_RERANK_MODEL", "bge-reranker-v2-m3")
# Pinecone e' eventualmente consistente: un record appena scritto diventa
# visibile alle ricerche dopo qualche secondo.
CONSISTENCY_TIMEOUT = float(os.environ.get("PINECONE_TEST_TIMEOUT", "120"))



# Ogni esempio di hypothesis fa chiamate di rete: pochi esempi, nessuna
# deadline e niente shrinking, che ripeterebbe decine di chiamate a vuoto.

LIVE = settings(
    max_examples=3,
    deadline=None,
    phases=[Phase.explicit, Phase.reuse, Phase.generate],
    suppress_health_check=[HealthCheck.too_slow],
)

PURE = settings(max_examples=100, deadline=None)


# ------------------------------------------------------------- strategie -------------------------------------------------------------

TEXT_ALPHABET = string.ascii_letters + string.digits + " .,;:'àèéìòù"

texts = st.text(alphabet=TEXT_ALPHABET, min_size=1, max_size=150).filter(
    lambda s: s.strip() != ""
)
text_lists = st.lists(texts, min_size=1, max_size=6)
# Gli id finiscono negli id dei record: solo caratteri ASCII sicuri.
document_ids = st.from_regex(r"[a-z][a-z0-9-]{0,20}", fullmatch=True)
dates = st.none() | st.datetimes(min_value=datetime(2000, 1, 1),
                                 max_value=datetime(2100, 1, 1))
unit_floats = st.floats(min_value=-1, max_value=1, allow_nan=False, width=32)
vectors = st.lists(unit_floats, min_size=VECTOR_DIMENSION, max_size=VECTOR_DIMENSION).filter(
    lambda values: any(abs(v) > 1e-3 for v in values)  # la cosine vuole un vettore non nullo
)


# --------------------------------------------------------------- helper ---------------------------------------------------------------

def make_chunks(document_id, chunk_texts, last_modified=None):
    chunks, start = [], 0
    for position, text in enumerate(chunk_texts):
        chunks.append(DocumentChunk(
            chunk_id=f"{document_id}#{position}", document_id=document_id,
            last_modified=last_modified, start_idx=start, end_idx=start + len(text), text=text,
        ))
        start += len(text)
    return chunks


def hit_ids(response):
    return [PineconeDB._read(hit, "id", "_id") for hit in response.result.hits]


def hit_scores(response):
    return [float(PineconeDB._read(hit, "score", "_score")) for hit in response.result.hits]


def is_descending(scores):
    return all(a >= b - 1e-6 for a, b in zip(scores, scores[1:]))


def wait_for(read, ready, timeout=CONSISTENCY_TIMEOUT, interval=2.0):
    """Ripete read() finche' ready(valore) e' vero; restituisce l'ultimo valore."""
    deadline = time.monotonic() + timeout
    last_error = value = None
    while True:
        try:
            value = read()
            if ready(value):
                return value
        except Exception as error:  # namespace non ancora creato, ecc.
            last_error = error
        if time.monotonic() > deadline:
            raise AssertionError(f"condizione non raggiunta in {timeout}s; "
                                 f"ultimo valore={value!r}, ultimo errore={last_error!r}")
        time.sleep(interval)


def wait_searchable(db, index_name, namespace, count):
    """Aspetta che una ricerca veda `count` record nel namespace."""
    return wait_for(
        lambda: db.search_document(index_name, namespace, "documento", top_k=max(count, 1)),
        lambda response: response is not None and len(response.result.hits) == count,
    )


def fetched_ids(db, index_name, namespace, ids):
    index = db._client.Index(name=index_name)
    return set(index.fetch(ids=list(ids), namespace=namespace).vectors)


@contextmanager
def fresh_namespace(db, index_name):
    """Namespace nuovo per ogni esempio, cancellato alla fine."""
    namespace = f"t-{uuid.uuid4().hex[:12]}"
    try:
        yield namespace
    finally:
        try:
            db._client.Index(name=index_name).delete(delete_all=True, namespace=namespace)
        except Exception:
            pass  # namespace mai creato


def temporary_index_name():
    return f"pdb-tmp-{uuid.uuid4().hex[:10]}"


# -------------------------------------------------------------- fixture

@pytest.fixture(scope="session")
def db():
    return PineconeDB()


@pytest.fixture(scope="session")
def index_name(db):
    db.create_index(TEST_INDEX)  # attende che l'indice sia pronto
    yield TEST_INDEX
    if os.environ.get("PINECONE_TEST_CLEANUP") == "1":
        db._client.delete_index(TEST_INDEX)


@pytest.fixture(scope="session")
def vector_index(db):
    # La classe non crea indici a vettori (BM25_HYB non e' implementato):
    # l'indice di prova viene creato direttamente con il client.
    if not db._client.has_index(VECTOR_INDEX):
        db._client.create_index(
            name=VECTOR_INDEX, dimension=VECTOR_DIMENSION, metric="cosine",
            spec=ServerlessSpec(cloud=db._cloud, region=db._region),
        )
    yield VECTOR_INDEX
    if os.environ.get("PINECONE_TEST_CLEANUP") == "1":
        db._client.delete_index(VECTOR_INDEX)


# ================================================================ indici

class TestCreateIndex:

    @pytest.mark.slow
    def test_new_index_is_created_once(self, db):
        name = temporary_index_name()
        try:
            assert db.create_index(name) is True
            assert db._client.has_index(name)
            assert db.create_index(name) is False
        finally:
            if db._client.has_index(name):
                db._client.delete_index(name)

    def test_existing_index_is_not_recreated(self, db, index_name):
        assert db.create_index(index_name) is False

    def test_unsupported_modes_create_nothing(self, db):
        name = temporary_index_name()

        with pytest.raises(NotImplementedError):
            db.create_index(name, IndexMode.BM25_HYB)
        with pytest.raises(ValueError):
            db.create_index(name, "LLM_ONLY")

        assert not db._client.has_index(name)


# ============================================================= scrittura

class TestInsertDocuments:

    @LIVE
    @given(count=st.integers(min_value=1, max_value=150))
    @example(count=PineconeDB.MAX_RECORDS_PER_BATCH + 1)  # attraversa il limite del batch
    def test_all_records_are_written(self, db, index_name, count):
        records = [{"_id": f"r{i}", db._field_value: f"documento numero {i}"}
                   for i in range(count)]

        with fresh_namespace(db, index_name) as namespace:
            assert db.insert_documents(index_name, namespace, records) is True

            ids = [record["_id"] for record in records]
            wait_for(lambda: fetched_ids(db, index_name, namespace, ids),
                     lambda found: found == set(ids))

    @LIVE
    @given(chunk_texts=text_lists)
    def test_text_is_stored_in_the_embedded_field(self, db, index_name, chunk_texts):
        records = [{"_id": f"r{i}", db._field_value: text} for i, text in enumerate(chunk_texts)]

        with fresh_namespace(db, index_name) as namespace:
            assert db.insert_documents(index_name, namespace, records) is True

            response = wait_searchable(db, index_name, namespace, len(records))
            stored = sorted(PineconeDB._read(hit, "fields", "_fields")[db._field_value]
                            for hit in response.result.hits)
            assert stored == sorted(chunk_texts)

    def test_invalid_input_returns_false(self, db, index_name):
        missing = temporary_index_name()

        with fresh_namespace(db, index_name) as namespace:
            assert db.insert_documents(index_name, namespace, []) is False
            assert db.insert_documents(index_name, namespace,
                                       [{db._field_value: "senza id"}]) is False
            assert db.insert_documents(missing, namespace, [{"_id": "a", db._field_value: "x"}],
                                       create_if_missing=False) is False

        assert not db._client.has_index(missing)


class TestInsertDocument:

    @LIVE
    @given(text=texts)
    def test_single_record_is_written(self, db, index_name, text):
        with fresh_namespace(db, index_name) as namespace:
            assert db.insert_document(index_name, namespace,
                                      {"_id": "unico", db._field_value: text}) is True

            wait_for(lambda: fetched_ids(db, index_name, namespace, ["unico"]),
                     lambda found: found == {"unico"})


class TestInsertVectors:

    @LIVE
    @given(values=st.lists(vectors, min_size=1, max_size=5))
    def test_vectors_are_written(self, db, vector_index, values):
        payload = [{"id": f"v{i}", "values": vector} for i, vector in enumerate(values)]

        with fresh_namespace(db, vector_index) as namespace:
            assert db.insert_vectors(vector_index, namespace, payload) is True

            ids = [item["id"] for item in payload]
            wait_for(lambda: fetched_ids(db, vector_index, namespace, ids),
                     lambda found: found == set(ids))

    def test_missing_index_is_not_created(self, db):
        missing = temporary_index_name()

        assert db.insert_vectors(missing, "ns", [{"id": "a", "values": [0.1] * 8}]) is False
        assert not db._client.has_index(missing)

    def test_wrong_dimension_is_rejected_by_pinecone(self, db, vector_index):
        with fresh_namespace(db, vector_index) as namespace:
            wrong = [{"id": "a", "values": [0.1] * (VECTOR_DIMENSION + 1)}]

            assert db.insert_vectors(vector_index, namespace, wrong) is False


class TestInsertChunks:

    @LIVE
    @given(document_id=document_ids, chunk_texts=text_lists, last_modified=dates)
    def test_every_chunk_is_written(self, db, index_name, document_id, chunk_texts,
                                    last_modified):
        chunks = make_chunks(document_id, chunk_texts, last_modified)

        with fresh_namespace(db, index_name) as namespace:
            assert db.insert_chunks(index_name, namespace, chunks) is True

            ids = [chunk.chunk_id for chunk in chunks]
            wait_for(lambda: fetched_ids(db, index_name, namespace, ids),
                     lambda found: found == set(ids))

    @LIVE
    @given(document_id=document_ids, first=text_lists, replacement=texts)
    def test_duplicates_are_written_once_and_last_wins(self, db, index_name, document_id,
                                                        first, replacement):
        original = make_chunks(document_id, first)
        rewritten = [chunk.model_copy(update={"text": replacement}) for chunk in original]

        with fresh_namespace(db, index_name) as namespace:
            assert db.insert_chunks(index_name, namespace, original + rewritten) is True

            response = wait_searchable(db, index_name, namespace, len(original))
            stored = {PineconeDB._read(hit, "fields", "_fields")[db._field_value]
                      for hit in response.result.hits}
            assert stored == {replacement}

    def test_no_chunks_returns_false(self, db, index_name):
        with fresh_namespace(db, index_name) as namespace:
            assert db.insert_chunks(index_name, namespace, []) is False


# =============================================================== ricerca ===============================================================

class TestSearchDocument:

    @LIVE
    @given(document_id=document_ids, chunk_texts=text_lists, data=st.data())
    def test_results_respect_top_k_and_are_ranked(self, db, index_name, document_id,
                                                  chunk_texts, data):
        chunks = make_chunks(document_id, chunk_texts)
        top_k = data.draw(st.integers(min_value=1, max_value=len(chunks)), label="top_k")

        with fresh_namespace(db, index_name) as namespace:
            db.insert_chunks(index_name, namespace, chunks)
            wait_searchable(db, index_name, namespace, len(chunks))

            response = db.search_document(index_name, namespace, chunk_texts[0], top_k=top_k)

            assert len(response.result.hits) == top_k
            assert set(hit_ids(response)) <= {chunk.chunk_id for chunk in chunks}
            assert is_descending(hit_scores(response))

    @LIVE
    @given(texts_a=text_lists, texts_b=text_lists)
    def test_filter_restricts_to_one_document(self, db, index_name, texts_a, texts_b):
        chunks_a, chunks_b = make_chunks("doc-a", texts_a), make_chunks("doc-b", texts_b)
        total = len(chunks_a) + len(chunks_b)

        with fresh_namespace(db, index_name) as namespace:
            db.insert_chunks(index_name, namespace, chunks_a + chunks_b)
            wait_searchable(db, index_name, namespace, total)

            response = db.search_document(index_name, namespace, "documento", top_k=total,
                                          filter={"document_id": {"$eq": "doc-a"}})

            assert sorted(hit_ids(response)) == sorted(c.chunk_id for c in chunks_a)

    def test_missing_index_returns_none_without_creating_it(self, db):
        missing = temporary_index_name()

        assert db.search_document(missing, "ns", "query") is None
        assert not db._client.has_index(missing)


class TestSearchByIndex:

    @LIVE
    @given(document_id=document_ids, chunk_texts=st.lists(texts, min_size=2, max_size=6))
    def test_record_is_its_own_best_match(self, db, index_name, document_id, chunk_texts):
        chunks = make_chunks(document_id, chunk_texts)
        target = chunks[0]

        with fresh_namespace(db, index_name) as namespace:
            db.insert_chunks(index_name, namespace, chunks)
            wait_searchable(db, index_name, namespace, len(chunks))

            response = db.search_by_index(index_name, namespace, target.chunk_id,
                                          top_k=len(chunks))

            scores = {match.id: match.score for match in response.matches}
            assert len(scores) == len(chunks)
            # Testi uguali danno lo stesso vettore: si confronta lo score, non la posizione.
            assert scores[target.chunk_id] == pytest.approx(max(scores.values()), abs=1e-4)
            assert response.matches[0].metadata["document_id"] == document_id

    def test_missing_index_returns_none_without_creating_it(self, db):
        missing = temporary_index_name()

        assert db.search_by_index(missing, "ns", "id") is None
        assert not db._client.has_index(missing)


class TestSearchDocumentAndRerank:

    @LIVE
    @given(document_id=document_ids, chunk_texts=st.lists(texts, min_size=2, max_size=6))
    def test_default_top_n_is_half_top_k(self, db, index_name, document_id, chunk_texts):
        chunks = make_chunks(document_id, chunk_texts)

        with fresh_namespace(db, index_name) as namespace:
            db.insert_chunks(index_name, namespace, chunks)
            wait_searchable(db, index_name, namespace, len(chunks))

            response = db.search_document_and_rerank(index_name, namespace, chunk_texts[0],
                                                     top_k=len(chunks), rerank_model=RERANK_MODEL)

            assert len(response.result.hits) == max(1, len(chunks) // 2)
            assert set(hit_ids(response)) <= {chunk.chunk_id for chunk in chunks}
            assert is_descending(hit_scores(response))

    @LIVE
    @given(texts_a=st.lists(texts, min_size=2, max_size=5), texts_b=text_lists, data=st.data())
    def test_explicit_top_n_and_filter(self, db, index_name, texts_a, texts_b, data):
        chunks_a, chunks_b = make_chunks("doc-a", texts_a), make_chunks("doc-b", texts_b)
        top_n = data.draw(st.integers(min_value=1, max_value=len(chunks_a)), label="top_n")

        with fresh_namespace(db, index_name) as namespace:
            db.insert_chunks(index_name, namespace, chunks_a + chunks_b)
            wait_searchable(db, index_name, namespace, len(chunks_a) + len(chunks_b))

            response = db.search_document_and_rerank(
                index_name, namespace, texts_a[0], top_k=len(chunks_a), top_n=top_n,
                filter={"document_id": {"$eq": "doc-a"}}, rerank_model=RERANK_MODEL,
            )

            assert len(response.result.hits) == top_n
            assert set(hit_ids(response)) <= {chunk.chunk_id for chunk in chunks_a}

    def test_top_n_out_of_range_is_rejected(self, db, index_name):
        with pytest.raises(ValueError):
            db.search_document_and_rerank(index_name, "ns", "q", top_k=4, top_n=5,
                                          rerank_model=RERANK_MODEL)


class TestRerankResults:

    @LIVE
    @given(documents=st.lists(texts, min_size=1, max_size=8), top_n=st.integers(1, 8))
    def test_returns_top_n_distinct_documents_ranked(self, db, documents, top_n):
        payload = [{db._field_value: text} for text in documents]

        result = db.rerank_results(payload, documents[0], rerank_model=RERANK_MODEL,
                                   top_n=top_n)

        positions = [item.index for item in result.data]
        assert len(positions) == min(top_n, len(documents))
        assert len(set(positions)) == len(positions)
        assert all(0 <= p < len(documents) for p in positions)
        assert is_descending([item.score for item in result.data])

    def test_rank_fields_can_be_a_list(self, db):
        payload = [{"body": "primo testo"}, {"body": "secondo testo"}]

        result = db.rerank_results(payload, "primo", rerank_model=RERANK_MODEL, top_n=2,
                                   rank_fields=["body"])

        assert sorted(item.index for item in result.data) == [0, 1]

    def test_relevant_document_ranks_first(self, db):
        documents = [
            "La ricetta della carbonara prevede guanciale, uova e pecorino.",
            "Il contratto di locazione scade il 31 dicembre e si rinnova tacitamente.",
            "La partita di calcio e' finita due a uno dopo i tempi supplementari.",
        ]
        payload = [{db._field_value: text} for text in documents]

        result = db.rerank_results(payload, "Quando scade il contratto d'affitto?",
                                   rerank_model=RERANK_MODEL, top_n=3)

        assert result.data[0].index == 1


class TestSearchChunks:

    @LIVE
    @given(document_id=document_ids, chunk_texts=text_lists, last_modified=dates)
    def test_round_trip_without_rerank(self, db, index_name, document_id, chunk_texts,
                                       last_modified):
        chunks = make_chunks(document_id, chunk_texts, last_modified)
        by_id = {chunk.chunk_id: chunk for chunk in chunks}

        with fresh_namespace(db, index_name) as namespace:
            db.insert_chunks(index_name, namespace, chunks)
            wait_searchable(db, index_name, namespace, len(chunks))

            results = db.search_chunks(index_name, namespace, chunk_texts[0], top_k=len(chunks))

            assert len(results) == len(chunks)
            for chunk, score in results:
                original = by_id[chunk.chunk_id]
                assert chunk == original
                assert (chunk.text, chunk.start_idx, chunk.end_idx, chunk.last_modified) == \
                       (original.text, original.start_idx, original.end_idx,
                        original.last_modified)
                assert isinstance(score, float)

    @LIVE
    @given(document_id=document_ids, chunk_texts=st.lists(texts, min_size=2, max_size=6))
    def test_rerank_returns_scored_chunks(self, db, index_name, document_id, chunk_texts):
        chunks = make_chunks(document_id, chunk_texts)

        with fresh_namespace(db, index_name) as namespace:
            db.insert_chunks(index_name, namespace, chunks)
            wait_searchable(db, index_name, namespace, len(chunks))

            results = db.search_chunks(index_name, namespace, chunk_texts[0], top_k=len(chunks),
                                       rerank_model=RERANK_MODEL, top_n=1)

            assert len(results) == 1
            chunk, score = results[0]
            assert chunk in chunks
            assert isinstance(score, float)

    def test_missing_index_returns_no_chunks(self, db):
        missing = temporary_index_name()

        assert db.search_chunks(missing, "ns", "q") == []
        assert not db._client.has_index(missing)


# ================================================ funzioni senza rete ================================================
# ========================= Logica pura: nessuna chiamata a Pinecone, quindi niente da simulare. ======================

class TestRerankTopN:

    @PURE
    @given(top_k=st.integers(min_value=1, max_value=10_000))
    def test_default_is_half_top_k_and_never_zero(self, top_k):
        assert PineconeDB._rerank_top_n(top_k, None) == max(1, top_k // 2)

    @PURE
    @given(top_k=st.integers(min_value=1, max_value=10_000), data=st.data())
    def test_out_of_range_is_rejected(self, top_k, data):
        top_n = data.draw(st.integers(max_value=0) | st.integers(min_value=top_k + 1))

        with pytest.raises(ValueError):
            PineconeDB._rerank_top_n(top_k, top_n)


class TestChunkToRecord:

    @PURE
    @given(document_id=document_ids, text=texts, last_modified=dates)
    def test_record_shape(self, db, document_id, text, last_modified):
        chunk = make_chunks(document_id, [text], last_modified)[0]

        expected = {"_id": chunk.chunk_id, db._field_value: text, "document_id": document_id,
                    "start_index": 0, "end_index": len(text)}
        if last_modified is not None:
            expected["last_modified"] = last_modified.isoformat()

        assert db._chunk_to_record(chunk) == expected
