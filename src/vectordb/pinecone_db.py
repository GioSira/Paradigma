import logging
import asyncio
from pinecone import PineconeAsyncio, ServerlessSpec, DenseVectorQuery, TextQuery
from pinecone.exceptions import NotFoundException
from pinecone.schema_builder import SchemaBuilder

from uuid import UUID

from typing import List, Dict, Any, Sequence, Optional, Mapping, Tuple, Union

from dotenv import load_dotenv
from src.chunks.document_chunk import DocumentChunk
from src.chunks.legal_chunk import LegalChunk
from src.collection.document import Document
from src.vectordb.errors import IndexNotFoundError, DimensionMismatchError
from src.vectordb.utils import _batches, _chunk_to_record, _create_args, _read, hit_to_chunk, _match_fields
from src.vectordb.hybrid_utils import _chunk_to_record_bm25, _reciprocal_rank_fusion, hit_to_chunk_hybrid
from src.vectordb.index_mode import IndexMode


logger = logging.getLogger(__name__)


# Metadato in cui _chunk_to_record salva il testo del chunk: il rerank legge da qui.
CHUNK_TEXT_FIELD = "text"


class PineconeDB:

    # Limite di Pinecone per upsert_records su indici con embedding integrato.
    MAX_RECORDS_PER_BATCH = 16

    # Batch per upsert() di vettori gia' calcolati (limite server: 2 MB a richiesta).
    VECTOR_BATCH_SIZE = 100

    def __init__(self, cloud: str ='aws', region: str ='us-east-1',
                 model: Optional[str] = None,
                 field_value: str = "chunk_text",
                 metric: str = "cosine",
                 emb_size: int = 300,
                 client: Optional[PineconeAsyncio] = None):
        """
        model / field_value: solo per gli indici con embedding integrato
            (IndexMode.PINECONE): modello di Pinecone e campo con il testo.
        metric / emb_size: per gli indici a vettori (IndexMode.LLM); emb_size
            e' la dimensione predefinita se create_index non ne riceve una.
        """
        self._cloud = cloud
        self._region = region
        # ServerlessSpec accetta solo argomenti con nome.
        self._specs = ServerlessSpec(cloud=cloud, region=region)
        self._model = model
        self._field_value = field_value
        self._client = client or self._connect()
        self._indexes: Dict[str, Any] = {}
        self._metric = metric
        self._emb_size = emb_size
        self._lock = asyncio.Lock()


    # ============================== ASYNCIO LIFETIME ==============================

    async def aclose(self) -> None:
        for index in self._indexes.values():
            await index.close()

        self._indexes.clear()
        await self._client.close()


    async def __aenter__(self) -> "PineconeDB":
        return self


    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.aclose()

    # ============================== UTILS ==============================

    @staticmethod
    def _connect():

        load_dotenv() # read .env file
        return PineconeAsyncio()


    @property
    def field_map(self) -> Dict[str, str]:
        return {"text": self._field_value}


    def get_indexes(self):
        return self._indexes


    # ============================== INDEXES ==============================

    async def _index(self, index_name: str):

        # Il lock evita che due richieste insieme aprano due client per lo stesso indice.
        async with self._lock:
            index = self._indexes.get(index_name)
            if index is None:
                if not await self._client.has_index(index_name):
                    raise IndexNotFoundError(f"l'indice {index_name!r} non esiste")

                host = (await self._client.describe_index(index_name)).host
                index = self._client.IndexAsyncio(host=host)
                self._indexes[index_name] = index

            return index


    async def create_index(self, 
                           index_name: str,
                           index_mode: IndexMode = IndexMode.BM25_HYB,
                           dimension: Optional[int] = None) -> bool:

        """Crea l'indice se non esiste. True se e' stato creato ora."""

        if not isinstance(index_mode, IndexMode):
            raise ValueError(f"Modalita' {index_mode!r} non supportata. Usa un valore di IndexMode.")

        if index_mode is IndexMode.BM25_HYB:
            return await self._create_index_bm25_and_vectors(index_name, dimension or self._emb_size)

        if index_mode is IndexMode.PINECONE:
            return await self._create_index_for_model(index_name)

        # IndexMode.LLM: indice a vettori, gli embedding arrivano da fuori.
        if await self._client.has_index(index_name):
            return False

        await self._client.create_index(name=index_name, spec=self._specs,
                                        dimension=dimension or self._emb_size,
                                        metric=self._metric)
        return True


    async def _create_index_for_model(self, index_name: str) -> bool:

        if await self._client.has_index(index_name):
            return False

        # Il modello serve solo per crearlo, non per riconoscere un indice esistente.
        if not self._model:
            raise ValueError("per un indice con embedding integrato serve model (es. 'llama-text-embed-v2')")

        await self._client.create_index_for_model(
            name=index_name,
            cloud=self._cloud,
            region=self._region,
            embed={
                "model": self._model,
                "field_map": self.field_map
            }
        )

        return True


    async def _create_index_bm25_and_vectors(self, index_name: str, dimension: int, language: str = "it"):

        if await self._client.has_index(index_name):
            return False

        schema = SchemaBuilder()\
                    .add_string_field(name="text", language=language, stemming=True, stop_words=True, full_text_search={})\
                    .add_string_field(name="document_name", language=language, stemming=True, stop_words=True)\
                    .add_string_field(name="document_title", language=language, stemming=True, stop_words=True)\
                    .add_dense_vector_field(name="embedding", dimension=dimension, metric=self._metric)\
                    .build()

        await self._client.indexes.create(
            name=index_name,
            schema=schema,
            deployment={
                "deployment_type": "managed", 
                "cloud": self._cloud, 
                "region": self._region
            }
        )

        return True
        

    async def ensure_index(self, index_name: str, dimension: int, mode: IndexMode = IndexMode.BM25_HYB) -> bool:

        """Crea l'indice se manca (True se creato ora). Se esiste gia' con una
        dimensione diversa solleva DimensionMismatchError: cambiare modello di
        embedding vuol dire ricreare l'indice e reindicizzare tutto."""
        
        if dimension < 1:
            raise ValueError("dimension deve essere almeno 1")

        if await self._client.has_index(index_name):
            existing = getattr(await self._client.describe_index(index_name), "dimension", None)
            if existing is not None and existing != dimension:
                raise DimensionMismatchError(
                    f"l'indice {index_name!r} ha dimensione {existing}, il modello {dimension}")
            return False

        return await self.create_index(index_name, mode, dimension=dimension)


    # ============================== WRITING ==============================


    async def insert_vectors(self, index_name: str, namespace: str, vectors: List[dict]) -> bool:

        if not vectors or len(vectors) == 0:
            return False

        try:

            index = await self._index(index_name)

            response = await index.upsert(namespace=namespace, vectors=vectors, batch_size=self.VECTOR_BATCH_SIZE, show_progress=False)

            if getattr(response, "has_errors", False):
                return False

            return True

        except Exception as e:
            return False


    async def insert_documents(self, index_name: str, namespace: str, documents: List[dict]) -> bool:

        if not documents or len(documents) == 0:
            return False

        index = await self._index(index_name)

        try:

            for batch in _batches(documents, size=self.MAX_RECORDS_PER_BATCH):
                await index.documents.upsert(namespace=namespace, documents=list(batch))

            return True

        except Exception as e:
            first = documents[0]
            logger.exception("Upsert su %r/%r non riuscito. Primo record: %s", 
                             index_name, namespace, 
                             {k: (type(v).__name__, len(v) if hasattr(v, "__len__") else v) 
                              for k, v in first.items()})
            return False


    async def insert_document(self, index_name: str, namespace: str, document: dict) -> bool:

        return await self.insert_documents(index_name, namespace, [document])


    async def insert_chunks(self, index_name: str, namespace: str,
                      chunks: Union[Sequence[DocumentChunk], Sequence[LegalChunk]],
                      vectors: Sequence[Sequence[float]],
                      index_mode: IndexMode = IndexMode.BM25_HYB) -> bool:

        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunk ma {len(vectors)} vettori")

        if index_mode == IndexMode.LLM:

            # A parita' di id vince l'ultimo: nessun upsert doppio.
            records = {str(chunk.get_chunk_id()): _chunk_to_record(chunk, vector)
                    for chunk, vector in zip(chunks, vectors)}

            if not records: return False
            return await self.insert_vectors(index_name, namespace, list(records.values()))

        if index_mode == IndexMode.BM25_HYB:

            records = {str(chunk.get_chunk_id()): _chunk_to_record_bm25(chunk, vector)
                       for chunk, vector in zip(chunks, vectors)}

            if not records: return False
            return await self.insert_documents(index_name, namespace, list(records.values()))

        raise ValueError(f"mode {index_mode} non supportato: usa {IndexMode.LLM} o {IndexMode.BM25_HYB}")


    async def delete_chunks_of_document(self, index_name: str, namespace: str, document_id: UUID, mode: IndexMode) -> bool:

        doc_filter = {"document_id": {"$eq": str(document_id)}}
        try:
            index = await self._index(index_name)
            if mode == IndexMode.BM25_HYB:
                # indice con schema: i documenti si cancellano dall'API documents
                await index.documents.delete(namespace=namespace, filter=doc_filter)
            else:
                await index.delete(filter=doc_filter, namespace=namespace)

        except (IndexNotFoundError, NotFoundException):
            return False

        return True


    # ============================== SEARCHING ==============================


    async def search_document(self, index_name: str, namespace: str, query: Sequence[float],
                              top_k:int = 10, filter: Optional[Mapping[str, Any]] = None,
                              fields: Optional[List[str]] = None):

        try:
            index = await self._index(index_name)

            results = await index.search(
                vector=list(query),
                **_create_args(namespace=namespace, top_k=top_k, fields=fields, filter=filter)
            )

        except (IndexNotFoundError, NotFoundException) as idxErr:
            return []

        return results


    async def search_by_index(self, index_name: str, namespace: str, query_id: str, top_k:int = 5,
                        include_metadata:bool = True, include_values:bool = False):

        try:

            index = await self._index(index_name)

            results = await index.query(
                namespace=namespace,
                id=query_id,
                top_k=top_k,
                include_metadata=include_metadata,
                include_values=include_values
            )

        except (IndexNotFoundError, NotFoundException) as idxErr:
            return []

        return results



    async def search_document_and_rerank(self, index_name: str, namespace: str, query: Sequence[float], 
                                         top_k:int = 5,
                                         top_n: Optional[int] = None, 
                                         fields:List[str] = None, 
                                         filter: Optional[Mapping[str, Any]] = None,
                                         rank_fields: Union[List[str], str, None] = None,
                                         rerank_model:str = "cohere-rerank-4-fast",
                                         rerank_query: Optional[str] = None):

        # top_n e rerank_query vengono controllati prima di contattare Pinecone.
        args = _create_args(namespace=namespace,
                            rerank_query=rerank_query,
                            top_k=top_k,
                            top_n=top_n,
                            rerank_model=rerank_model,
                            fields=fields,
                            rank_fields=rank_fields or CHUNK_TEXT_FIELD,
                            filter=filter)

        try:
            index = await self._index(index_name)
            results = await index.search(vector=list(query), **args)

        except (IndexNotFoundError, NotFoundException) as idxErr:
            return []

        return results


    async def rerank_results(self, documents: List[dict], 
                             query: str, 
                             rerank_model:str = "cohere-rerank-4-fast",
                             top_n:int = 5, 
                             rank_fields: Union[str, List[str], None] = None):

        rank_fields = rank_fields or self._field_value
        if isinstance(rank_fields, str):
            rank_fields = [rank_fields]

        results = await self._client.inference.rerank(
            model=rerank_model,
            query=query,
            documents=documents,
            top_n=top_n,
            rank_fields=rank_fields,
            return_documents=True,
            parameters={
                "truncate": "END"
            }
        )

        return results


    @staticmethod
    def _hits_to_chunks(response) -> List[Tuple[DocumentChunk, float]]:
        """index.search restituisce response.result.hits, ognuno con id, score e
        fields (i metadati salvati con il vettore)."""

        if not response:
            return []

        return [(hit_to_chunk(_read(hit, "id", "_id"), _read(hit, "fields", "_fields")),
                 float(_read(hit, "score", "_score")))
                for hit in response.result.hits]

    @staticmethod
    def _hits_to_chunks_hybrid(fused) -> List[Tuple[LegalChunk, float]]:
        """`fused`: lista di coppie (match, score_rrf) da _reciprocal_rank_fusion."""

        if not fused:
            return []

        chunks = []
        for hit, score in fused:

            try:
                fields = _read(hit, "fields", "_fields")
            except KeyError:
                fields = None

            if not isinstance(fields, Mapping):
                fields = _match_fields(hit)

            chunk_id = str(_read(hit, "id", "_id"))
            chunks.append((hit_to_chunk_hybrid(chunk_id, fields), float(score or 0.0)))

        return chunks


    async def search_chunks(
        self, 
        index_name: str, 
        namespace: str, 
        query: Sequence[float],
        top_k: int = 10, 
        index_mode: IndexMode = IndexMode.BM25_HYB,
        query_text: Optional[str] = None, 
        filter: Optional[Mapping[str, Any]] = None,
        rerank_model: Optional[str] = None,
        rank_field: Union[str, List[str], None] = None,
        top_n: Optional[int] = None,
        rerank_query: Optional[str] = None
    ):

        if index_mode == IndexMode.LLM:
            return await self.search_chunks_vector(index_name, namespace, query, top_k,
                                                   filter, rerank_model, rank_field, top_n,
                                                   rerank_query)

        if index_mode == IndexMode.BM25_HYB:
            return await self.search_chunks_hybrid(index_name, namespace, query, query_text, top_k, filter)

        raise ValueError(f"mode {index_mode} non supportato: usa {IndexMode.LLM} o {IndexMode.BM25_HYB}")


    async def search_chunks_hybrid(self, index_name: str, 
                                   namespace: str, 
                                   query_vector: Sequence[float],
                                   query_text: str,
                                   top_k: int = 10,
                                   filter: Optional[Dict[str, Any]] = None,
                                   rrf_k: int = 60) -> List[Tuple[DocumentChunk, float]]:
 
        if top_k < 1:
            raise ValueError("top_k deve essere almeno 1")
  
        index = await self._index(index_name)
        # include_fields: tutti i campi che hit_to_chunk legge, non solo il testo.
        common = dict(namespace=namespace, top_k=top_k, filter=filter, include_fields=["*"])
 
        searches = [index.documents.search(
            score_by=[DenseVectorQuery(field="embedding", values=[float(x) for x in query_vector])],
            **common)]
 
        if query_text and query_text.strip():
            searches.append(index.documents.search(
                score_by=[TextQuery(query=query_text, fields=["text", "document_name", "document_title"])],
                **common))
 
        # Le due richieste partono insieme: la latenza e' quella della piu' lenta.
        responses = await asyncio.gather(*searches)
 
        fused = _reciprocal_rank_fusion([response.matches for response in responses], k=rrf_k)
 
        return self._hits_to_chunks_hybrid(fused[:top_k])
 
        

    async def search_chunks_vector(self, index_name: str, namespace: str, query: Sequence[float],
                            top_k: int = 10, filter: Optional[Mapping[str, Any]] = None,
                            rerank_model: Optional[str] = None,
                            rank_field: Union[str, List[str], None] = None,
                            top_n: Optional[int] = None,
                            rerank_query: Optional[str] = None) -> List[Tuple[DocumentChunk, float]]:
        
        """(chunk, punteggio) dal piu' rilevante. Con rerank_model il punteggio
        e' quello del rerank (serve rerank_query, la domanda in testo),
        altrimenti la similarita' tra vettori."""

        if rerank_model is None:
            response = await self.search_document(index_name, namespace, query, top_k=top_k, filter=filter)
        else:
            response = await self.search_document_and_rerank(index_name, namespace, query, top_k=top_k,
                                                            filter=filter, rerank_model=rerank_model,
                                                            top_n=top_n, rank_fields=rank_field,
                                                            rerank_query=rerank_query)

        return self._hits_to_chunks(response)


    async def search_chunks_of_document(self, index_name: str, namespace: str,
                                  query: Sequence[float], document: Document, **kwargs) -> List[Tuple[DocumentChunk, float]]:

        extra = kwargs.pop("filter", None)
        doc_filter = {"document_id": {"$eq": str(document.document_id)}}
        if extra:
            doc_filter = {"$and": [doc_filter, dict(extra)]}

        return await self.search_chunks(index_name, namespace, query, filter=doc_filter, **kwargs)
