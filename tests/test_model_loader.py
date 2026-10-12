"""Test di integrazione di ModelLoader su OpenRouter reale.

Richiedono OPENROUTER_API_KEY (nell'ambiente o nel .env); senza vengono
saltati. Usano i modelli predefiniti di ModelLoader, che sono gratuiti:
contano comunque nei limiti di richieste al minuto del tuo account.

Le risposte dei modelli cambiano da una chiamata all'altra, quindi non si
confrontano testi esatti ma proprieta' che un modello che funziona rispetta
sempre: frasi con lo stesso significato hanno embedding vicini, il documento
pertinente viene messo primo, la risposta usa il contesto passato.

OPENROUTER_SERVER_URL (facoltativa) manda le richieste a un altro indirizzo,
per esempio un proxy aziendale.

Esempi:
    pytest tests/test_model_loader_live.py
    pytest tests/test_model_loader_live.py -k Generate
"""
import asyncio
import math
import os
import uuid

import pytest
from dotenv import load_dotenv
from openrouter import OpenRouter

from src.collection.document import Document
from src.llm.model_loader import ModelLoader

load_dotenv()
API_KEY = os.environ.get("OPENROUTER_API_KEY")
SERVER_URL = os.environ.get("OPENROUTER_SERVER_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not API_KEY, reason="OPENROUTER_API_KEY non impostata"),
]


# ============================================================ fixture e helper

# Un solo event loop per tutto il modulo. Il client HTTP riusa le connessioni
# (keep-alive) e ognuna appartiene al loop che l'ha aperta: con asyncio.run
# a ogni chiamata, dalla seconda in poi si otterrebbe "Event loop is closed".
_LOOP = asyncio.new_event_loop()


def run(coroutine):
    return _LOOP.run_until_complete(coroutine)


@pytest.fixture(scope="module")
def loader():
    client = OpenRouter(api_key=API_KEY, server_url=SERVER_URL) if SERVER_URL else None
    # I modelli gratuiti possono essere lenti o momentaneamente occupati:
    # timeout e tentativi larghi.
    models = ModelLoader(timeout_seconds=90, retry_seconds=120, client=client)
    yield models
    run(models.aclose())   # chiude le connessioni finche' il loop e' vivo


def teardown_module():
    _LOOP.close()


def document(text, name):
    return Document(doc_id=uuid.uuid4(), name=name, last_modified=None, text=text)


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def is_vector(value):
    return (isinstance(value, list) and len(value) > 0
            and all(isinstance(x, float) for x in value)
            and any(x != 0 for x in value))


CONTRACT = "Il contratto di locazione scade il 31 dicembre e si rinnova tacitamente per altri quattro anni."
RECIPE = "La carbonara si prepara con guanciale, uova, pecorino romano e pepe nero."
FOOTBALL = "La partita di calcio e' finita due a uno dopo i tempi supplementari."
WEATHER = "Domani sono previste piogge sparse al nord e sole al sud."


# ============================================================ embed_text

class TestEmbedText:

    def test_returns_a_vector_of_stable_dimension(self, loader):
        first = run(loader.embed_text("Il gatto dorme sul divano."))
        second = run(loader.embed_text("Una frase completamente diversa, molto piu' lunga della prima, "
                                       "per vedere che la dimensione non dipende dal testo."))

        assert is_vector(first) and is_vector(second)
        assert len(first) == len(second)

    def test_similar_meaning_is_closer_than_unrelated_meaning(self, loader):
        cat = run(loader.embed_text("Il gatto dorme sul divano."))
        feline = run(loader.embed_text("Un felino riposa sul sofa'."))
        finance = run(loader.embed_text("La banca centrale ha alzato i tassi di interesse."))

        assert cosine(cat, feline) > cosine(cat, finance)


# ============================================================ embed_documents

class TestEmbedDocuments:

    def test_one_vector_per_document_in_the_same_order(self, loader):
        docs = [document(CONTRACT, "contratto"), document(RECIPE, "ricetta"),
                document(FOOTBALL, "calcio")]

        result = run(loader.embed_documents(docs))

        assert [doc_id for doc_id, _ in result] == [doc.document_id for doc in docs]
        dimensions = {len(vector) for _, vector in result}
        assert len(dimensions) == 1 and all(is_vector(vector) for _, vector in result)

    def test_each_vector_belongs_to_its_document(self, loader):
        # Se l'ordine si mescolasse, il vettore del contratto finirebbe sulla
        # ricetta: confronto ogni vettore con l'embedding del suo testo da solo.
        docs = [document(CONTRACT, "contratto"), document(RECIPE, "ricetta")]

        batch = dict(run(loader.embed_documents(docs)))
        alone = {doc.document_id: run(loader.embed_text(doc.get_document_text())) for doc in docs}

        for doc in docs:
            own = cosine(batch[doc.document_id], alone[doc.document_id])
            others = [cosine(batch[doc.document_id], alone[other.document_id])
                      for other in docs if other is not doc]
            assert own > 0.99 and own > max(others)


# ============================================================ rerank

class TestRerank:

    def test_relevant_document_comes_first(self, loader):
        docs = [document(RECIPE, "ricetta"), document(CONTRACT, "contratto"),
                document(FOOTBALL, "calcio")]

        result = run(loader.rerank("Quando scade il contratto d'affitto?", docs, top_n=3))

        assert result[0][0].name == "contratto"

    def test_top_n_and_scores_are_respected(self, loader):
        docs = [document(RECIPE, "ricetta"), document(CONTRACT, "contratto"),
                document(FOOTBALL, "calcio"), document(WEATHER, "meteo")]

        result = run(loader.rerank("Che tempo fara' domani?", docs, top_n=2))

        assert len(result) == 2
        assert all(doc in docs for doc, _ in result)                  # i Document originali
        assert len({doc.document_id for doc, _ in result}) == 2       # nessun doppione
        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)
        assert result[0][0].name == "meteo"


# ============================================================ generate

class TestGenerate:

    def test_answer_uses_the_context(self, loader):
        # Un dato inventato: il modello puo' conoscerlo solo dal contesto.
        context = ["Il codice della cassaforte dell'aula 7 e' 4721.",
                   "L'aula 7 si trova al secondo piano."]

        answer = run(loader.generate(
            system_prompt="Rispondi in italiano usando solo le informazioni del contesto.",
            query="Qual e' il codice della cassaforte dell'aula 7?",
            context=context,
        ))

        assert isinstance(answer, str) and "4721" in answer

    def test_system_prompt_is_followed(self, loader):
        answer = run(loader.generate(
            system_prompt=("Rispondi esclusivamente con la parola SI oppure con la parola NO, "
                           "in maiuscolo, senza punteggiatura e senza altro testo."),
            query="Roma e' la capitale d'Italia?",
            context=[],
        ))

        assert answer.strip().strip(".!").upper() in {"SI", "SÌ"}
