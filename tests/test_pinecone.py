"""Test di integrazione di PineconeDB (asincrono) su Pinecone reale.

Richiedono PINECONE_API_KEY (nell'ambiente o nel file .env). Senza chiave
vengono saltati. I test senza rete sono in test_pinecone.py.

Indice usato (creato alla prima esecuzione e riusato nelle successive):
    PINECONE_TEST_VECTOR_INDEX   indice a vettori, dimensione 8 (default: pinecone-db-test-vectors)
Con PINECONE_TEST_CLEANUP=1 viene cancellato a fine sessione. I test marcati
slow creano e cancellano indici temporanei (anche con embedding integrato).

I vettori sono scritti a mano (assi di uno spazio a 8 dimensioni): si prova
Pinecone, non un modello di embedding. Ogni test scrive in un namespace
nuovo, cancellato alla fine.

Pinecone e' eventualmente consistente: un record appena scritto diventa
visibile alle ricerche dopo qualche secondo, per questo le letture ritentano.

Esempi:
    pytest tests/test_pinecone_live.py
    pytest tests/test_pinecone_live.py -m "not slow"
"""
import asyncio
import os
import string
import time
import uuid
from datetime import datetime

import pytest
from dotenv import load_dotenv
from hypothesis import HealthCheck, Phase, given, settings, strategies as st

from src.chunks.document_chunk import DocumentChunk
from src.collection.document import Document
from src.vectordb.errors import DimensionMismatchError
from src.vectordb.index_mode import IndexMode
from src.vectordb.pinecone_db import PineconeDB

load_dotenv()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get("PINECONE_API_KEY"),
                       reason="PINECONE_API_KEY non impostata"),
]

VECTOR_INDEX = os.environ.get("PINECONE_TEST_VECTOR_INDEX", "pinecone-db-test-vectors")
DIMENSION = 8
RERANK_MODEL = os.environ.get("PINECONE_TEST_RERANK_MODEL", "bge-reranker-v2-m3")
EMBED_MODEL = os.environ.get("PINECONE_TEST_EMBED_MODEL", "llama-text-embed-v2")
CONSISTENCY_TIMEOUT = float(os.environ.get("PINECONE_TEST_TIMEOUT", "120"))

# Ogni esempio fa chiamate di rete: pochi esempi, niente deadline, niente shrinking.
LIVE = settings(max_examples=3, deadline=None,
                phases=[Phase.explicit, Phase.reuse, Phase.generate],
                suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])

# Un solo event loop per tutto il modulo: il client tiene connessioni HTTP
# legate al loop che le ha aperte (con asyncio.run a ogni chiamata si
# otterrebbe "Event loop is closed").
_LOOP = asyncio.new_event_loop()


def run(coroutine):
    return _LOOP.run_until_complete(coroutine)


def teardown_module():
    _LOOP.close()


# ============================================================ strategie e helper

texts = st.text(alphabet=string.ascii_letters + string.digits + " .,;:'àèéìòù",
                min_size=1, max_size=150).filter(lambda s: s.strip())
text_lists = st.lists(texts, min_size=1, max_size=DIMENSION)
dates = st.none() | st.datetimes(min_value=datetime(2000, 1, 1), max_value=datetime(2100, 1, 1))


def axis(i):
    """Vettore unitario lungo l'asse i: vettori diversi, distanze note."""
    return [1.0 if j == i % DIMENSION else 0.0 for j in range(DIMENSION)]


def make_chunks(document_id, chunk_texts, last_modified=None):
    chunks, start = [], 0
    for text in chunk_texts:
        chunks.append(DocumentChunk(chunk_id=uuid.uuid4(), document_id=document_id,
                                    last_modified=last_modified, start_idx=start,
                                    end_idx=start + len(text), text=text))
        start += len(text)
    return chunks


def wait_for(read, ready, timeout=CONSISTENCY_TIMEOUT, interval=2.0):
    """Ripete read() (una coroutine) finche' ready(valore) e' vero."""
    deadline = time.monotonic() + timeout
    value = None
    while True:
        value = run(read())
        if ready(value):
            return value
        if time.monotonic() > deadline:
            raise AssertionError(f"condizione non raggiunta in {timeout}s; ultimo valore={value!r}")
        time.sleep(interval)


def wait_searchable(db, namespace, count, vector=None):
    """Aspetta che una ricerca veda esattamente `count` chunk nel namespace."""
    return wait_for(lambda: db.search_chunks(VECTOR_INDEX, namespace, vector or axis(0),
                                             top_k=max(count, 1) + 5),
                    lambda results: len(results) == count)


def temporary_index_name():
    return f"pdb-tmp-{uuid.uuid4().hex[:10]}"


# ============================================================ fixture

@pytest.fixture(scope="module")
def db():
    vectordb = PineconeDB(model=EMBED_MODEL)
    run(vectordb.ensure_index(VECTOR_INDEX, DIMENSION))   # attende che sia pronto
    yield vectordb
    if os.environ.get("PINECONE_TEST_CLEANUP") == "1":
        run(vectordb._client.delete_index(VECTOR_INDEX))
    run(vectordb.aclose())


class _Namespaces:
    """Namespace nuovi per ogni esempio, tutti cancellati alla fine del test."""

    def __init__(self, db):
        self.db, self.names = db, []

    def new(self):
        name = f"t-{uuid.uuid4().hex[:12]}"
        self.names.append(name)
        return name

    def cleanup(self):
        async def delete_all():
            index = await self.db._index(VECTOR_INDEX)
            for name in self.names:
                try:
                    await index.delete(delete_all=True, namespace=name)
                except Exception:
                    pass   # namespace mai creato
        run(delete_all())


@pytest.fixture
def namespaces(db):
    created = _Namespaces(db)
    yield created
    created.cleanup()


# ============================================================ indici

class TestIndexes:

    def test_ensure_index_is_idempotent_and_checks_the_dimension(self, db):
        assert run(db.ensure_index(VECTOR_INDEX, DIMENSION)) is False
        with pytest.raises(DimensionMismatchError):
            run(db.ensure_index(VECTOR_INDEX, DIMENSION + 1))

    @pytest.mark.slow
    @pytest.mark.parametrize("mode", [IndexMode.LLM, IndexMode.PINECONE])
    def test_new_index_is_created_once(self, db, mode):
        name = temporary_index_name()
        try:
            assert run(db.create_index(name, mode, dimension=DIMENSION)) is True
            assert run(db._client.has_index(name))
            assert run(db.create_index(name, mode, dimension=DIMENSION)) is False
        finally:
            if run(db._client.has_index(name)):
                run(db._client.delete_index(name))

    def test_unsupported_modes_create_nothing(self, db):
        name = temporary_index_name()

        with pytest.raises(NotImplementedError):
            run(db.create_index(name, IndexMode.BM25_HYB))
        with pytest.raises(ValueError):
            run(db.create_index(name, "LLM"))
        assert not run(db._client.has_index(name))


# ============================================================ scrittura

class TestInsertChunks:

    @LIVE
    @given(document_id=st.uuids(), chunk_texts=text_lists, last_modified=dates)
    def test_every_chunk_comes_back_unchanged(self, db, namespaces, document_id, chunk_texts,
                                              last_modified):
        namespace = namespaces.new()
        chunks = make_chunks(document_id, chunk_texts, last_modified)

        assert run(db.insert_chunks(VECTOR_INDEX, namespace, chunks,
                                    [axis(i) for i in range(len(chunks))])) is True

        results = wait_searchable(db, namespace, len(chunks))
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        assert {chunk.chunk_id for chunk, _ in results} == set(by_id)
        for chunk, score in results:
            assert chunk.model_dump() == by_id[chunk.chunk_id].model_dump()
            assert isinstance(score, float)

    @LIVE
    @given(document_id=st.uuids(), first=text_lists, replacement=texts)
    def test_duplicates_are_written_once_and_last_wins(self, db, namespaces, document_id,
                                                       first, replacement):
        namespace = namespaces.new()
        original = make_chunks(document_id, first)
        rewritten = [chunk.model_copy(update={"text": replacement}) for chunk in original]
        vectors = [axis(i) for i in range(len(original))] * 2

        assert run(db.insert_chunks(VECTOR_INDEX, namespace, original + rewritten, vectors)) is True

        results = wait_searchable(db, namespace, len(original))
        assert {chunk.get_chunk_text() for chunk, _ in results} == {replacement}

    def test_invalid_writes_return_false(self, db, namespaces):
        namespace = namespaces.new()
        missing = temporary_index_name()
        chunk = make_chunks(uuid.uuid4(), ["testo"])

        assert run(db.insert_chunks(VECTOR_INDEX, namespace, [], [])) is False
        assert run(db.insert_chunks(VECTOR_INDEX, namespace, chunk, [[0.1] * (DIMENSION + 1)])) is False
        assert run(db.insert_chunks(missing, namespace, chunk, [axis(0)])) is False
        assert not run(db._client.has_index(missing))


# ============================================================ ricerca

class TestSearchChunks:

    def test_nearest_chunk_comes_first(self, db, namespaces):
        namespace = namespaces.new()
        chunks = make_chunks(uuid.uuid4(), ["zero", "uno", "due"])
        run(db.insert_chunks(VECTOR_INDEX, namespace, chunks, [axis(0), axis(1), axis(2)]))
        wait_searchable(db, namespace, 3)

        results = run(db.search_chunks(VECTOR_INDEX, namespace, axis(1), top_k=3))

        assert results[0][0] == chunks[1]
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_filters_namespaces_and_missing_index(self, db, namespaces):
        namespace = namespaces.new()
        doc_a, doc_b = uuid.uuid4(), uuid.uuid4()
        chunks_a, chunks_b = make_chunks(doc_a, ["a1", "a2"]), make_chunks(doc_b, ["b1"])
        run(db.insert_chunks(VECTOR_INDEX, namespace, chunks_a + chunks_b, [axis(0), axis(1), axis(2)]))
        wait_searchable(db, namespace, 3)
        document_a = Document(doc_id=doc_a, name="A", last_modified=None, text=None)

        only_a = run(db.search_chunks_of_document(VECTOR_INDEX, namespace, axis(2), document_a, top_k=5))
        a_from_start = run(db.search_chunks_of_document(VECTOR_INDEX, namespace, axis(0), document_a,
                                                        top_k=5, filter={"start_index": {"$eq": 0}}))

        assert {c for c, _ in only_a} == set(chunks_a)
        assert [c for c, _ in a_from_start] == [chunks_a[0]]
        assert run(db.search_chunks(VECTOR_INDEX, f"{namespace}-vuoto", axis(0))) == []
        assert run(db.search_chunks(temporary_index_name(), namespace, axis(0))) == []


class TestRerank:

    def test_reranker_reads_the_stored_text(self, db, namespaces):
        # Tutti i vettori uguali: la ricerca per vettore non sa distinguere,
        # l'ordine lo decide solo il reranker leggendo il testo dei chunk.
        namespace = namespaces.new()
        chunks = make_chunks(uuid.uuid4(), [
            "La ricetta della carbonara prevede guanciale, uova e pecorino.",
            "Il contratto di locazione scade il 31 dicembre e si rinnova tacitamente.",
            "La partita di calcio e' finita due a uno dopo i tempi supplementari.",
        ])
        run(db.insert_chunks(VECTOR_INDEX, namespace, chunks, [axis(0)] * 3))
        wait_searchable(db, namespace, 3)

        results = run(db.search_chunks(VECTOR_INDEX, namespace, axis(0), top_k=3,
                                       rerank_model=RERANK_MODEL, top_n=2,
                                       rerank_query="Quando scade il contratto d'affitto?"))

        assert len(results) == 2
        assert results[0][0] == chunks[1]
        assert results[0][1] >= results[1][1]

    def test_rerank_results_ranks_the_relevant_document_first(self, db):
        documents = [{"chunk_text": "La ricetta della carbonara prevede guanciale e uova."},
                     {"chunk_text": "Il contratto di locazione scade il 31 dicembre."},
                     {"chunk_text": "La partita e' finita due a uno."}]

        result = run(db.rerank_results(documents, "Quando scade il contratto d'affitto?",
                                       rerank_model=RERANK_MODEL, top_n=3))

        assert result.data[0].index == 1


class TestSearchByIndexAndDelete:

    def test_a_record_is_its_own_best_match(self, db, namespaces):
        namespace = namespaces.new()
        chunks = make_chunks(uuid.uuid4(), ["zero", "uno", "due"])
        run(db.insert_chunks(VECTOR_INDEX, namespace, chunks, [axis(0), axis(1), axis(2)]))
        wait_searchable(db, namespace, 3)

        response = run(db.search_by_index(VECTOR_INDEX, namespace, str(chunks[2].chunk_id), top_k=3))

        assert response.matches[0].id == str(chunks[2].chunk_id)
        assert response.matches[0].metadata["text"] == "due"

    def test_delete_removes_only_that_document(self, db, namespaces):
        namespace = namespaces.new()
        doc_a, doc_b = uuid.uuid4(), uuid.uuid4()
        chunks = make_chunks(doc_a, ["a1", "a2"]) + make_chunks(doc_b, ["b1"])
        run(db.insert_chunks(VECTOR_INDEX, namespace, chunks, [axis(0), axis(1), axis(2)]))
        wait_searchable(db, namespace, 3)

        assert run(db.delete_chunks_of_document(VECTOR_INDEX, namespace, doc_a)) is True

        remaining = wait_searchable(db, namespace, 1)
        assert remaining[0][0].get_document_id() == doc_b
        assert run(db.delete_chunks_of_document(VECTOR_INDEX, f"{namespace}-vuoto", doc_a)) is False
