"""Test di RagService (src/orchestrator/llm_orchestrator.py) senza rete.

PineconeDB e ModelLoader sono sostituiti da finti con le STESSE firme e gli
stessi valori di ritorno delle classi vere:
- FakeVectorDB costruisce i record con _chunk_to_record vero: un vettore nel
  formato sbagliato fallisce qui come fallirebbe con Pinecone;
- FakeModels.embed_text_batch restituisce coppie (testo, vettore) e
  FakeModels.rerank vuole dei Document, come ModelLoader.
Gli embedding finti contano le parole: testi con parole in comune sono vicini.
"""
import asyncio
import hashlib
import math
import re
import uuid

import pytest

from src.chunks.document_chunk import DocumentChunk
#from src.chunks.text_chunker import TextChunker
from src.collection.document import Document
from src.orchestrator.answer import Answer
from src.orchestrator.llm_orchestrator import NO_ANSWER, IndexingError, RagService
from src.vectordb.errors import DimensionMismatchError
from src.vectordb.utils import _chunk_to_record, hit_to_chunk

SYSTEM_PROMPT = "Sei un tutor. Rispondi solo con il contesto e cita i brani con [n]."


# ============================================================ finti

def words(text):
    return re.findall(r"\w+", text.lower())


def bag_of_words(text, dimension=64):
    vector = [0.0] * dimension
    for word in words(text):
        vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % dimension] += 1.0
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


class FakeVectorDB:
    """PineconeDB in memoria, con le firme di src/vectordb/pinecone_db.py."""

    def __init__(self):
        self.indexes = {}           # nome -> dimensione
        self.records = {}           # namespace -> id -> record (come quelli inviati a Pinecone)
        self.calls = []
        self.fail_inserts = 0

    async def ensure_index(self, index_name, dimension):
        self.calls.append(("ensure_index", index_name, dimension))
        if index_name in self.indexes and self.indexes[index_name] != dimension:
            raise DimensionMismatchError("dimensione diversa")
        created = index_name not in self.indexes
        self.indexes[index_name] = dimension
        return created

    async def insert_chunks(self, index_name, namespace, chunks, vectors):
        self.calls.append(("insert_chunks", namespace, len(chunks)))
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunk ma {len(vectors)} vettori")
        records = [_chunk_to_record(c, v) for c, v in zip(chunks, vectors)]  # come la classe vera
        if not records or self.fail_inserts:
            self.fail_inserts = max(0, self.fail_inserts - 1)
            return False
        self.records.setdefault(namespace, {}).update({r["id"]: r for r in records})
        return True

    async def delete_chunks_of_document(self, index_name, namespace, document_id):
        self.calls.append(("delete", namespace, document_id))
        stored = self.records.get(namespace)
        if stored is None:
            return False
        for record_id in [i for i, r in stored.items()
                          if r["metadata"]["document_id"] == str(document_id)]:
            del stored[record_id]
        return True

    async def search_chunks(self, index_name, namespace, query, top_k=10, filter=None,
                            rerank_model=None, rank_field=None, top_n=None, rerank_query=None):
        self.calls.append(("search_chunks", namespace, top_k, filter))
        if rerank_model is not None and not rerank_query:
            raise ValueError("rerank senza rerank_query")
        wanted = (filter or {}).get("document_id", {}).get("$eq")
        scored = []
        for record in self.records.get(namespace, {}).values():
            if wanted is not None and record["metadata"]["document_id"] != wanted:
                continue
            score = sum(a * b for a, b in zip(query, record["values"]))
            scored.append((hit_to_chunk(record["id"], record["metadata"]), score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def calls_named(self, name):
        return [call for call in self.calls if call[0] == name]


class FakeModels:
    """ModelLoader finto, con le firme di src/llm/model_loader.py."""

    def __init__(self, answer="Il contratto scade il 31 dicembre [1]."):
        self.answer = answer
        self.calls = []
        self.fail_embeddings = False

    async def embed_text(self, text):
        self.calls.append(("embed_text", text))
        if not text or not text.strip():
            raise ValueError("il testo da trasformare in embedding e' vuoto")
        return bag_of_words(text)

    async def embed_text_batch(self, text_batch):
        self.calls.append(("embed_text_batch", len(text_batch)))
        if self.fail_embeddings:
            raise RuntimeError("OpenRouter non risponde")
        return [(text, bag_of_words(text)) for text in text_batch]   # coppie, come ModelLoader

    async def rerank_texts(self, query, texts, top_n=3):
        self.calls.append(("rerank_texts", len(texts), top_n))
        q = set(words(query))
        scored = [(i, len(q & set(words(t))) / len(q)) for i, t in enumerate(texts)]
        return sorted(scored, key=lambda item: item[1], reverse=True)[:min(top_n, len(texts))]

    async def rerank(self, query, documents, top_n=3):
        # Come ModelLoader.rerank: vuole dei Document, non delle stringhe.
        texts = [d.get_document_text() or "" for d in documents]
        return [(documents[i], s) for i, s in await self.rerank_texts(query, texts, top_n)]

    async def generate(self, system_prompt, query, context, max_tokens=1000, top_p=0.95):
        self.calls.append(("generate", system_prompt, query, list(context), max_tokens))
        return self.answer

    def calls_named(self, name):
        return [call for call in self.calls if call[0] == name]


# ============================================================ helper

def run(coroutine):
    return asyncio.run(coroutine)


def document(text, name="doc"):
    return Document(doc_id=uuid.uuid4(), name=name, last_modified=None, text=text)


def service(vectordb=None, models=None, **kwargs):
    #kwargs.setdefault("chunker", TextChunker(max_chars=200, overlap=20))
    return RagService(vectordb or FakeVectorDB(), models or FakeModels(), "corsi",
                      system_prompt=SYSTEM_PROMPT, **kwargs)


CONTRATTO = ("Il contratto di locazione scade il 31 dicembre. "
             "Il rinnovo e' tacito per altri quattro anni.")
RICETTA = "La carbonara si prepara con guanciale, uova e pecorino romano."


# ============================================================ costruttore e setup

class TestConstructorAndSetup:

    @pytest.mark.parametrize("kwargs", [{"top_k": 3, "top_n": 4}, {"top_n": 0},
                                        {"embed_batch_size": 0}])
    def test_invalid_parameters_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            service(**kwargs)

    def test_setup_creates_the_index_with_the_model_dimension(self):
        db = FakeVectorDB()

        assert run(service(db).setup()) == 64
        assert db.indexes == {"corsi": 64}

        db.indexes["corsi"] = 1024   # creato con un altro modello
        with pytest.raises(DimensionMismatchError):
            run(service(db).setup())


# ============================================================ indicizzazione

class TestIndexDocument:

    def test_indexed_document_is_searchable(self):
        db = FakeVectorDB()
        rag = service(db)

        chunks = run(rag.index_document("corso", document(CONTRATTO)))

        assert chunks == len(db.records["corso"]) >= 1
        answer = run(rag.ask("corso", "Quando scade il contratto?"))
        assert "contratto" in answer.sources[0].text

    def test_reindexing_leaves_no_trace_of_the_old_version(self):
        db = FakeVectorDB()
        rag = service(db)
        doc = document("Il vecchio orario di ricevimento e' il lunedi.")
        run(rag.index_document("corso", doc))

        doc.set_document_text("Il nuovo orario di ricevimento e' il giovedi.")
        run(rag.index_document("corso", doc))

        texts = [r["metadata"]["text"] for r in db.records["corso"].values()]
        assert texts and all("giovedi" in t for t in texts)

    def test_embedding_failure_keeps_the_old_version(self):
        db, models = FakeVectorDB(), FakeModels()
        rag = service(db, models)
        doc = document("Il ricevimento e' il lunedi alle dieci.")
        run(rag.index_document("corso", doc))

        models.fail_embeddings = True
        doc.set_document_text("Il ricevimento e' il giovedi alle undici.")
        with pytest.raises(RuntimeError):
            run(rag.index_document("corso", doc))

        # Gli embedding si calcolano prima di cancellare: la versione vecchia resta.
        assert len(db.calls_named("delete")) == 1
        assert all("lunedi" in r["metadata"]["text"] for r in db.records["corso"].values())

    def test_failed_insert_raises_and_retry_works(self):
        db = FakeVectorDB()
        db.fail_inserts = 1
        rag = service(db)
        doc = document(CONTRATTO)

        with pytest.raises(IndexingError):
            run(rag.index_document("corso", doc))
        assert run(rag.index_document("corso", doc)) == len(db.records["corso"])

    def test_embeddings_are_requested_in_batches(self):
        models = FakeModels()
        rag = service(models=models, 
                      #chunker=TextChunker(max_chars=40, overlap=0),
                      embed_batch_size=3)
        doc = document(" ".join(f"Frase numero {i} del documento." for i in range(10)))

        chunks = run(rag.index_document("corso", doc))

        sizes = [size for _, size in models.calls_named("embed_text_batch")]
        assert sum(sizes) == chunks and max(sizes) <= 3

    @pytest.mark.parametrize("doc", [None, document("   ")])
    def test_missing_or_empty_document_is_rejected_before_any_call(self, doc):
        db, models = FakeVectorDB(), FakeModels()

        with pytest.raises(ValueError):
            run(service(db, models).index_document("corso", doc))
        assert db.calls == [] and models.calls == []


class TestIndexChunks:

    def test_one_delete_per_document_and_vectors_aligned(self):
        db = FakeVectorDB()
        rag = service(db)
        doc_a, doc_b = uuid.uuid4(), uuid.uuid4()
        chunks = [DocumentChunk(chunk_id=uuid.uuid4(), document_id=doc, last_modified=None,
                                start_idx=0, end_idx=len(text), text=text)
                  for doc, text in [(doc_a, CONTRATTO), (doc_a, "Secondo brano di A."),
                                    (doc_b, RICETTA)]]

        assert run(rag.index_chunks("corso", chunks)) == 3

        assert sorted(str(call[2]) for call in db.calls_named("delete")) == sorted([str(doc_a), str(doc_b)])
        for chunk in chunks:
            record = db.records["corso"][str(chunk.chunk_id)]
            assert record["values"] == bag_of_words(chunk.get_chunk_text())

    def test_no_chunks_means_no_calls(self):
        db = FakeVectorDB()

        assert run(service(db).index_chunks("corso", [])) == 0
        assert db.calls == []


class TestRemoveDocument:

    def test_removed_document_is_no_longer_found(self):
        rag = service()
        doc = document(CONTRATTO)
        run(rag.index_document("corso", doc))

        assert run(rag.remove_document("corso", doc.document_id)) is True
        assert run(rag.ask("corso", "contratto")).sources == []


# ============================================================ domande

class TestAsk:

    def test_answer_uses_numbered_sources_from_the_best_chunks(self):
        models = FakeModels()
        rag = service(models=models)
        run(rag.index_document("corso", document(CONTRATTO)))
        run(rag.index_document("corso", document(RICETTA)))

        answer = run(rag.ask("corso", "Quando scade il contratto di locazione?"))

        assert isinstance(answer, Answer) and answer.text == models.answer
        assert "contratto" in answer.sources[0].text
        _, system, query, context, _ = models.calls_named("generate")[0]
        assert (system, query) == (SYSTEM_PROMPT, "Quando scade il contratto di locazione?")
        # Il modello vede i brani numerati come le fonti: [n] nella risposta = sources[n-1].
        assert context == [f"[{s.number}] {s.text}" for s in answer.sources]
        assert [s.number for s in answer.sources] == list(range(1, len(answer.sources) + 1))

    def test_vector_search_is_wide_and_rerank_narrows_it(self):
        db, models = FakeVectorDB(), FakeModels()
        rag = service(db, models, top_k=6, top_n=2)
        for i in range(8):
            run(rag.index_document("corso", document(f"Capitolo {i}: il contratto parte {i}.")))

        answer = run(rag.ask("corso", "contratto"))

        assert db.calls_named("search_chunks")[-1][2] == 6        # top_k dalla ricerca
        assert models.calls_named("rerank_texts") == [("rerank_texts", 6, 2)]
        assert len(answer.sources) == 2                            # top_n al modello

    def test_no_candidates_means_no_model_call(self):
        models = FakeModels()
        rag = service(models=models)

        answer = run(rag.ask("corso-vuoto", "Quanto costa il treno?"))

        assert (answer.text, answer.sources) == (NO_ANSWER, [])
        assert models.calls_named("generate") == []

    def test_min_score_can_leave_nothing_and_then_the_model_is_not_called(self):
        models = FakeModels()
        rag = service(models=models, min_score=0.6)
        run(rag.index_document("corso", document(RICETTA)))

        answer = run(rag.ask("corso", "carbonara treno aereo"))   # 1 parola su 3: 0.33

        assert (answer.text, answer.sources) == (NO_ANSWER, [])
        assert models.calls_named("generate") == []

    def test_document_id_restricts_the_search(self):
        rag = service()
        contratto, lavoro = document(CONTRATTO), document("Il contratto di lavoro e' annuale.")
        run(rag.index_document("corso", contratto))
        run(rag.index_document("corso", lavoro))

        everything = run(rag.ask("corso", "contratto"))
        only_one = run(rag.ask("corso", "contratto", document_id=lavoro.document_id))

        assert {s.document_id for s in everything.sources} == {contratto.document_id,
                                                              lavoro.document_id}
        assert {s.document_id for s in only_one.sources} == {lavoro.document_id}

    def test_namespaces_are_isolated(self):
        rag = service()
        run(rag.index_document("corso-a", document(CONTRATTO)))

        assert run(rag.ask("corso-b", "contratto")).sources == []

    @pytest.mark.parametrize("namespace, question", [("corso", ""), ("corso", "   \n"),
                                                     ("", "domanda")])
    def test_invalid_input_is_rejected_before_any_call(self, namespace, question):
        db, models = FakeVectorDB(), FakeModels()

        with pytest.raises(ValueError):
            run(service(db, models).ask(namespace, question))
        assert db.calls == [] and models.calls == []

    def test_question_is_stripped_before_embedding(self):
        models = FakeModels()
        rag = service(models=models)
        run(rag.index_document("corso", document(CONTRATTO)))

        run(rag.ask("corso", "  contratto  \n"))

        assert ("embed_text", "contratto") in models.calls
