from src.llm.model_loader import ModelLoader
from src.llm.conversation import Chat, Message
from src.llm.jev_loader import JevLoader
from src.llm.errors import GenerationError

from src.chunks.document_chunk import DocumentChunk
from src.chunks.legal_chunk import LegalChunk

from src.vectordb.pinecone_db import PineconeDB, IndexMode

from src.orchestrator.answer import Answer, Source
from src.orchestrator.errors import IndexingError, DeleteError

from typing import Dict, List, Optional, Tuple, Sequence
from uuid import UUID

import logging
logger = logging.getLogger(__name__)


# Testo qualsiasi per scoprire la dimensione dei vettori del modello.
_PROBE = "dimensione"

NO_ANSWER = "Non so come rispondere a questa domanda: nei documenti non ho trovato informazioni."

# Le due modalita' che RagService sa gestire (PINECONE ha l'embedding integrato).
SUPPORTED_MODES = (IndexMode.LLM, IndexMode.BM25_HYB)


class RagService:
    """Ricerca nei documenti + risposta del modello.

    Un RagService lavora su UN indice, di UN tipo, scelto con `mode`:

    - IndexMode.LLM: solo vettori. Si cerca solo per significato.
    - IndexMode.BM25_HYB: vettori + testo con BM25. Si puo' cercare in modo
      ibrido oppure solo per vettori (la parte densa dello stesso indice).

    Il tipo di un indice non cambia dopo la creazione: per avere entrambi i
    casi servono due RagService con due nomi di indice diversi.

    Gli embedding li calcola ModelLoader; Pinecone li conserva e cerca i
    vicini. Documenti e domande passano dallo stesso modello: e' l'unico modo
    in cui i vettori sono confrontabili.
    """

    def __init__(self,
                 vectordb: PineconeDB,
                 llm_model: ModelLoader,
                 jev_model: JevLoader,
                 index_name: str,
                 mode: IndexMode = IndexMode.BM25_HYB,
                 top_k: int = 20,
                 top_n: int = 5,
                 min_score: Optional[float] = None,
                 max_tokens: int = 1000,
                 embed_batch_size: int = 64):

        if mode not in SUPPORTED_MODES:
            raise ValueError(f"mode {mode!r} non supportato: usa IndexMode.LLM o IndexMode.BM25_HYB")
        
        if not 1 <= top_n <= top_k:
            raise ValueError("serve 1 <= top_n <= top_k")
        
        if embed_batch_size < 1:
            raise ValueError("embed_batch_size deve essere almeno 1")

        self._vectordb = vectordb
        self._models = llm_model
        self._jev = jev_model
        self._index = index_name
        self._mode = mode

        self._top_k = top_k
        self._top_n = top_n
        self._min_score = min_score
        self._max_tokens = max_tokens
        self._embed_batch_size = embed_batch_size


    @property
    def mode(self) -> IndexMode:
        return self._mode


    async def setup(self) -> int:
        """Da chiamare all'avvio: crea l'indice del tipo `mode` con la
        dimensione del modello di embedding, o verifica quello esistente."""

        dimension = len(await self._models.embed_text(_PROBE))
        await self._vectordb.ensure_index(self._index, dimension, self._mode)
        logger.info("Indice %r pronto: %s, dimensione %d", self._index, self._mode.name, dimension)
        return dimension


    # ============================== UTILS ==============================

    async def embed_chunks(self, chunks: Sequence[DocumentChunk]) -> List[List[float]]:

        if not chunks:
            return []

        vectors = await self._embed_batch([chunk.get_chunk_text() for chunk in chunks])
        logger.info("Calcolati %d vettori", len(vectors))
        return vectors


    async def _embed_batch(self, texts: Sequence[str]) -> List[List[float]]:

        vectors: List[List[float]] = []

        for start in range(0, len(texts), self._embed_batch_size):
            batch = list(texts[start:start + self._embed_batch_size])
            vectors += [vector for _, vector in await self._models.embed_text_batch(batch)]

        return vectors


    async def delete_chunks_of(self, namespace: str, document_id: UUID) -> bool:
        try:
            await self._vectordb.delete_chunks_of_document(self._index, namespace, document_id, self._mode)
            logger.info("Cancellati i chunk di %s", document_id)
            return True
        except Exception:
            # La causa finisce nel log: DeleteError da sola non la direbbe.
            logger.exception("Cancellazione dei chunk di %s non riuscita", document_id)
            return False


    # ============================== INDICIZZAZIONE ==============================

    async def index_chunks(self, namespace: str, chunks: Sequence[DocumentChunk]) -> int:
        """Salva i chunk nell'indice, nel modo dell'indice (`mode`).

        Per ogni documento presente nei chunk, prima cancella i chunk vecchi:
        ricaricare un documento lo sostituisce invece di duplicarlo.
        Vale per DocumentChunk e per le sottoclassi (LegalChunk)."""

        if not chunks:
            return 0

        # 1. Embedding prima di tutto: se OpenRouter fallisce, niente e' stato cancellato.
        vectors = await self.embed_chunks(chunks)

        # 2. Una sola cancellazione per documento, non una per chunk.
        document_ids: List[UUID] = list(dict.fromkeys(chunk.get_document_id() for chunk in chunks))
        for document_id in document_ids:
            if not await self.delete_chunks_of(namespace, document_id):
                raise DeleteError(document_id, namespace)

        # 3. Scrittura. PineconeDB risponde False invece di sollevare: lo traduco qui.
        if not await self._vectordb.insert_chunks(self._index, namespace, list(chunks), vectors, self._mode):
            raise IndexingError(document_ids)

        logger.info("Scritti %d chunk (%s) in %r", len(chunks), self._mode.name, namespace)
        return len(chunks)


    async def index_chunks_vectors(self, namespace: str, chunks: Sequence[DocumentChunk]) -> int:
        return await self.index_chunks(namespace, chunks)


    async def index_legal_chunks(self, namespace: str, chunks: Sequence[LegalChunk]) -> int:
        return await self.index_chunks(namespace, chunks)


    # ============================== JEV EVAL =============================

    async def _evaluate_query(self, question: str, answer: str, threshold: float = 0.5) -> bool:

        response = await self._jev.noul(
            state=answer,
            instruction=f"Does the text answer the question: \"{question}\"?"
        )
        logger.info("JEV -> %s", response)

        return response.confidence > threshold


    async def _evaluate_candidates(self, question: str,
                                   candidates: List[Tuple[DocumentChunk, float]]) -> List[Tuple[DocumentChunk, float]]:

        valid_candidates = []

        for candidate, score in candidates:

            response = await self._jev.choice(
                state=candidate.get_chunk_text().strip(),
                instruction=f"Does it answer the question?\nQuestion: {question}\n",
                criteria={
                    "yes": "The text answers the question completely or partially.",
                    "no": "The text does not answer the question, neither completely nor partially."
                }
            )

            if response.choice == "yes":
                valid_candidates.append((candidate, score))

        return valid_candidates


    # ============================== DOMANDE ==============================

    async def rewrite_question(self, prompt: str, question: str, history: Optional[Sequence[Message]] = None) -> str:

        chat = Chat.of(history)

        search_question = question
        try:
            search_question = await self._models.expanse_question(question, prompt, chat)
            logger.info("Domanda riscritta: %s", search_question)
        except GenerationError:
            logger.info("Riscrittura non riuscita: provo a concatenare")
            user_message = chat.last_user_message()
            if user_message:
                search_question = f"{user_message}\n{question}"

        return search_question


    def _search_mode(self, method: Optional[IndexMode]) -> IndexMode:
        """Che ricerca fare. None = quella dell'indice.

        Un indice ibrido sa cercare anche solo per vettori; un indice a soli
        vettori non ha il testo per BM25. ValueError (-> 422) se non si puo'."""

        method = method or self._mode

        if method not in SUPPORTED_MODES:
            raise ValueError(f"ricerca {method!r} non supportata: usa IndexMode.LLM o IndexMode.BM25_HYB")
        if method is IndexMode.BM25_HYB and self._mode is IndexMode.LLM:
            raise ValueError(f"l'indice {self._index!r} ha solo vettori: la ricerca ibrida non e' disponibile")

        return method


    async def search_chunks(self,
                            namespace: str,
                            question: str,
                            document_id: Optional[UUID] = None,
                            top_k: Optional[int] = None,
                            method: Optional[IndexMode] = None) -> List[Tuple[DocumentChunk, float]]:
        """Cerca i chunk piu' vicini alla domanda. `method` None = il modo dell'indice."""

        method = self._search_mode(method)
        top_k = top_k or self._top_k

        query_vector = await self._models.embed_text(question)
        doc_filter = {"document_id": {"$eq": str(document_id)}} if document_id is not None else None

        candidates = await self._vectordb.search_chunks(self._index,
                                                        namespace,
                                                        query_vector,
                                                        index_mode=method,
                                                        top_k=top_k,
                                                        filter=doc_filter)

        logger.info("Ricerca %s: %d candidati su %d", method.name, len(candidates), top_k)
        return candidates


    # Nomi usati dalle route /search/vectors e /search/hybrid.
    async def search_chunks_embedding(self, namespace: str, question: str,
                                      document_id: Optional[UUID] = None,
                                      top_k: Optional[int] = None) -> List[Tuple[DocumentChunk, float]]:
        return await self.search_chunks(namespace, question, document_id, top_k, IndexMode.LLM)

    async def search_chunks_hybrid(self, namespace: str, question: str,
                                   document_id: Optional[UUID] = None,
                                   top_k: Optional[int] = None) -> List[Tuple[DocumentChunk, float]]:
        return await self.search_chunks(namespace, question, document_id, top_k, IndexMode.BM25_HYB)


    async def evaluate_candidates(self,
                                  question: str,
                                  candidates: List[Tuple[DocumentChunk, float]],
                                  top_n: Optional[int] = None,
                                  min_score: Optional[float] = None) -> List[Tuple[DocumentChunk, float]]:

        if not candidates:
            return []

        top_n = top_n or self._top_n

        # 1. JEV toglie i brani che non c'entrano...
        jev_candidates = await self._evaluate_candidates(question, candidates)
        logger.info("JEV: %d candidati validi su %d", len(jev_candidates), len(candidates))

        # ...ma se li scarta tutti probabilmente sbaglia: si tengono quelli di prima.
        candidates = jev_candidates or candidates

        # 2. Rerank
        candidates = await self._models.rerank(question, [chunk for chunk, _ in candidates], top_n=top_n)

        logger.info("Rerank: %d candidati su %d", len(candidates), top_n)
        for chunk, score in candidates:
            logger.info("[%s] %r", score, chunk.get_chunk_text())

        if min_score is not None:
            candidates = [(chunk, score) for chunk, score in candidates if score >= min_score]

        return candidates


    async def generate(self, prompt: str, question: str,
                       candidates: List[Tuple[DocumentChunk, float]],
                       history: Optional[Sequence[Message]] = None) -> Answer:

        sources = [
            Source(number=n,
                   document_id=chunk.get_document_id(),
                   chunk_id=chunk.get_chunk_id(),
                   text=chunk.get_chunk_text(),
                   score=score)
            for n, (chunk, score) in enumerate(candidates, start=1)
        ]

        context = [f"[{source.number}] {source.text}" for source in sources]

        answer = await self._models.generate(question, context, Chat.of(history), prompt)
        logger.info("Risposta generata: %r", answer)

        return Answer(text=answer, sources=sources)


    async def ask(self,
                  namespace: str,
                  question: str,
                  system_prompt: str,
                  condense_prompt: str,
                  document_id: Optional[UUID] = None,
                  history: Optional[Sequence[Message]] = None,
                  method: Optional[IndexMode] = None) -> Answer:
        """Domanda completa: riscrittura -> ricerca -> JEV e rerank -> risposta.
        `method` None = la ricerca del tipo dell'indice."""

        question = (question or "").strip()
        if not namespace or not question:
            raise ValueError("servono un namespace e una domanda non vuota")

        # 1. Ricerca
        search_question = await self.rewrite_question(condense_prompt, question, history)
        candidates = await self.search_chunks(namespace, search_question, document_id, self._top_k, method)

        # Nessun dato trovato: non rispondo
        if not candidates:
            return Answer(text=NO_ANSWER, sources=[])

        # 2. Filtro e riordino
        candidates = await self.evaluate_candidates(search_question, candidates, self._top_n, self._min_score)

        # Nessun dato abbastanza pertinente
        if not candidates:
            return Answer(text=NO_ANSWER, sources=[])

        # 3. Generazione
        return await self.generate(system_prompt, search_question, candidates, history)