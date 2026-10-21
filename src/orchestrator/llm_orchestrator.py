from src.llm.model_loader import ModelLoader
from src.llm.conversation import Chat, Message
from src.llm.jev_loader import JevLoader
from src.llm.errors import GenerationError

from src.chunks.document_chunk import DocumentChunk
#from src.chunks.text_chunker import TextChunker
from src.collection.document import Document
from src.vectordb.pinecone_db import PineconeDB
from src.orchestrator.answer import Answer, Source

from typing import Dict, List, Optional, Sequence, Tuple
from uuid import UUID

import logging
logger = logging.getLogger(__name__)


# Testo qualsiasi per scoprire la dimensione dei vettori del modello.
_PROBE = "dimensione"

NO_ANSWER = "Non so come rispondere a questa domanda: nei documenti non ho trovato informazioni."


class IndexingError(RuntimeError):
    """Pinecone non ha salvato i chunk: ripetere l'indicizzazione e' sicuro."""

    def __init__(self, document_ids: Sequence[UUID]):
        super().__init__(f"indicizzazione non riuscita per {len(document_ids)} documento/i")
        self.document_ids = list(document_ids)


class RagService:
    """Ricerca nei documenti + risposta del modello.

    Gli embedding li calcola ModelLoader; Pinecone li conserva e cerca i
    vicini. Documenti e domande passano dallo stesso modello: e' l'unico modo
    in cui i vettori sono confrontabili.
    """

    def __init__(self,
                 vectordb: PineconeDB,
                 llm_model: ModelLoader,
                 jev_model: JevLoader,
                 index_name: str,
                 #chunker: Optional[TextChunker] = None,
                 top_k: int = 20,
                 top_n: int = 5,
                 min_score: Optional[float] = None,
                 max_tokens: int = 1000,
                 embed_batch_size: int = 64):


        if not 1 <= top_n <= top_k:
            raise ValueError("serve 1 <= top_n <= top_k")
        if embed_batch_size < 1:
            raise ValueError("embed_batch_size deve essere almeno 1")

        self._vectordb = vectordb
        self._models = llm_model
        self._index = index_name
        self._jev = jev_model

        #self._chunker = chunker or TextChunker()
        self._top_k = top_k
        self._top_n = top_n
        self._min_score = min_score
        self._max_tokens = max_tokens
        self._embed_batch_size = embed_batch_size


    async def setup(self) -> int:
        """Da chiamare all'avvio: crea l'indice con la dimensione del modello
        di embedding, o verifica che quello esistente coincida. Sostituisce
        create_index, che userebbe una dimensione fissa."""

        dimension = len(await self._models.embed_text(_PROBE))
        await self._vectordb.ensure_index(self._index, dimension)
        return dimension


    # ============================== INDICIZZAZIONE ==============================

    async def index_chunks(self, namespace: str, chunks: List[DocumentChunk]) -> int:

        if not chunks:
            return 0

        # 1. Embedding prima di tutto: se OpenRouter fallisce, niente e' stato cancellato.
        vectors = await self._embed_batch([chunk.get_chunk_text() for chunk in chunks])
        logger.info("ritornati vettori batch")

        # 2. Una sola cancellazione per documento, non una per chunk.
        document_ids: Dict[str, UUID] = {str(c.get_document_id()): c.get_document_id() for c in chunks}
        for document_id in document_ids.values():
            await self._vectordb.delete_chunks_of_document(self._index, namespace, document_id)
            logger.info(f"Cancellati vettori per {document_id}")

        # 3. Scrittura. PineconeDB risponde False invece di sollevare: lo traduco qui.
        if not await self._vectordb.insert_chunks(self._index, namespace, chunks, vectors):
            raise IndexingError(list(document_ids.values()))
        logger.info("Scritti vettori su DB Pinecone")

        return len(chunks)


    async def index_document(self, namespace: str, document: Document) -> int:

        if not document:
            raise ValueError("document cannot be None")

        chunks = self._chunker.split(document)

        if not chunks:
            raise ValueError(f"il documento {document.document_id} non ha testo da indicizzare")

        return await self.index_chunks(namespace, chunks)


    async def remove_document(self, namespace: str, document_id: UUID) -> bool:

        return await self._vectordb.delete_chunks_of_document(self._index, namespace, document_id)


    async def _embed_batch(self, texts: Sequence[str]) -> List[List[float]]:

        vectors: List[List[float]] = []

        for start in range(0, len(texts), self._embed_batch_size):
            batch = list(texts[start:start + self._embed_batch_size])
            vectors += [vector for _, vector in await self._models.embed_text_batch(batch)]

        return vectors


    # ============================== JEV EVAL =============================

    async def _evaluate_query(self, question: str, answer: str, threshold: float = 0.5) -> bool:

        response = await self._jev.noul(
            state=answer,
            instruction=f"Does the text answer the question: \"{question}\"?"
        )
        logging.info("JEV -> " + response)

        return response.confidence > threshold


    async def _evaluate_candidates(self, question: str, candidates: List[Tuple[DocumentChunk, float]]):

        valid_candidates = []

        for (candidate, score) in candidates:
            
            response = await self._jev.choice(
                state=candidate.get_chunk_text().strip(),
                instruction=f"Does it answer the question?\nQuestion: {question}\n",
                criteria={
                    "yes": "The text answers the question completely or partialy.",
                    "no": "The text does not anwer the question, neither completely nor partially."
                }
            )

            if response.choice == "yes":
                valid_candidates.append((candidate, score))

        return valid_candidates


    # ============================== DOMANDE ==============================

    async def ask(self, namespace: str, question: str, 
                  system_prompt: str,
                  condense_prompt: str,
                  document_id: Optional[UUID] = None, 
                  history: Optional[Sequence[Message]] = None) -> Answer:

        question = (question or "").strip()
        if not namespace or not question:
            raise ValueError("servono un namespace e una domanda non vuota")

        # 1. Ricerca
        
        chat = Chat.of(history)

        search_question = question
        try:
            search_question = await self._models.expanse_question(question, condense_prompt, chat)
            logger.info(f"Domanda cambiata. Nuova domanda espansa: {search_question}")
        except GenerationError:
            logger.info("Generation error: provo a concatenare")
            user_message = chat.last_user_message()
            if user_message:
                search_question = f"{user_message}\n{question}"
                logger.info("Query concatenata con ultimo messaggio dell'utente")


        query_vector = await self._models.embed_text(search_question)
        doc_filter = {"document_id": {"$eq": str(document_id)}} if document_id is not None else None
        candidates: List[Tuple[DocumentChunk, float]] = await self._vectordb.search_chunks(self._index, 
                                                                                           namespace, 
                                                                                           query_vector,
                                                                                           top_k=self._top_k, 
                                                                                           filter=doc_filter)
        
        #logger.info("Ritornati i seguenti candidati:\n\n" + "\n".join([f"[{score}] {sent.text}" for (sent, score) in candidates]))
        logger.info(f"Ritornati {len(candidates)} candidati possibili su {self._top_k}")

        # nessun dato trovato, non rispondo
        if not candidates or len(candidates) == 0:
            return Answer(text=NO_ANSWER, sources=[])

        # 1.5 JEV filtering
        logger.info("Procedo a valutarli con JEV")
        jev_candidates = await self._evaluate_candidates(search_question, candidates)
        logger.info(f"Ritornati {len(jev_candidates)} candidati validi su {self._top_k}")

        # check nel caso jev sbagli
        candidates = jev_candidates if len(jev_candidates) > 0 else candidates

        # 2. Rerank solo se jev ha dato piu' di top_n candidati
        candidates: List[Tuple[DocumentChunk, float]] = await self._models.rerank(search_question,
                                        [chunk for chunk, _ in candidates], 
                                        top_n=self._top_n)
        #logger.info("Ritornati i seguehnti candidati dopo rerank:\n\n" + "\n".join([f"[{score}] {chunk.text}" for chunk, score in ranked]))
        logger.info(f"Ritornati {len(candidates)} candidati su {self._top_n}")
    
        if self._min_score is not None:
            candidates = [(chunk, score) for chunk, score in candidates if score >= self._min_score]

        # Nessun dato abbastanza pertinente
        if not candidates or len(candidates) == 0:
            return Answer(text=NO_ANSWER, sources=[])

        # 3. Generazione
        sources = [
            Source(number=n,
                   document_id=chunk.get_document_id(),
                   chunk_id=chunk.get_chunk_id(),
                   text=chunk.get_chunk_text(),
                   score=score)
            for n, (chunk, score) in enumerate(candidates, start=1)
        ]

        context = [f"[{source.number}] {source.text}" for source in sources]

        #while True:
        answer = await self._models.generate(question, context, chat, system_prompt)
        logger.info(f"Generata risposta: {answer}")

        #    is_valid = self._evaluate_query(search_question, answer)
        #    logger.info("La risposta e': " + "valida" if is_valid else "invalida. Rigenero.")

        #    if is_valid: break

        return Answer(text=answer, sources=sources)
