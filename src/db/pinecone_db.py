from pinecone import Pinecone
from dotenv import load_dotenv
from typing import List, Dict, Any, Sequence, Iterator, Iterable, Optional, Mapping, Tuple, Union
from datetime import datetime
from src.chunks.document_chunk import DocumentChunk
from enum import Enum, auto


class IndexMode(Enum):

    BM25_HYB = auto()
    LLM_ONLY = auto()


class IndexNotFoundError(LookupError):
    """L'indice richiesto non esiste su Pinecone."""


class PineconeDB:

    # Limite di Pinecone per upsert_records su indici con embedding integrato.
    MAX_RECORDS_PER_BATCH = 96

    # Batch per upsert() di vettori gia' calcolati (limite server: 2 MB a richiesta).
    VECTOR_BATCH_SIZE = 100

    def __init__(self, cloud: str ='aws', region: str ='us-east-1', 
                 model: str ='llama-text-embed-v2', field_value: str = "chunk_text"):

        self._cloud = cloud
        self._region = region
        self._model = model
        self._field_value = field_value
        self._client = self._connect()
        self._indexes: Dict[str, Any] = {}


    @staticmethod
    def _connect():

        load_dotenv() # read .env file
        return Pinecone()


    @property
    def field_map(self) -> Dict[str, str]:
        return {"text": self._field_value}


    def get_indexes(self):
        return self._indexes


    # ---------------------- SCRITTURA ----------------------------

    def _index(self, index_name: str, mode:IndexMode = IndexMode.LLM_ONLY, create_if_missing: bool = True):

        index = self._indexes.get(index_name)
        if index is not None:
            return index

        if not self._client.has_index(index_name):
            if not create_if_missing:
                raise IndexNotFoundError(f"L'indice {index_name!r} non esiste")

            self.create_index(index_name, mode)

        index = self._client.Index(name=index_name)
        self._indexes[index_name] = index
 
        return index


    def create_index(self, index_name: str, mode: IndexMode = IndexMode.LLM_ONLY) -> bool:
        """Crea l'indice se non esiste. True se e' stato creato ora."""

        if mode is IndexMode.LLM_ONLY:
            return self._create_index_for_model(index_name)
 
        if mode is IndexMode.BM25_HYB:
            return self._create_index_bm25_or_vectors(index_name)
 
        raise ValueError(f"Modalita' {mode!r} non supportata. Usa un valore di IndexMode.")


    def _create_index_for_model(self, index_name: str) -> bool:

        if self._client.has_index(index_name):
            return False

        self._client.create_index_for_model(
            name=index_name,
            cloud=self._cloud,
            region=self._region,
            embed={
                "model": self._model,
                "field_map": self.field_map
            }
        )

        return True


    def _create_index_bm25_or_vectors(self, index_name):

        raise NotImplementedError("La modalita' ibrida BM25 non e' ancora implementata")


    # ------------------------------- LETTURA ----------------------------

    @staticmethod
    def _batches(items: Sequence[Any], size: int = 16) -> Iterator[Sequence[Any]]:

        for start in range(0, len(items), size):
            yield items[start:start + size]
    

    def insert_vectors(self, index_name: str, namespace: str, vectors: List[dict]) -> bool:

        if not vectors or len(vectors) == 0:
            return False

        try:

            index = self._index(index_name, create_if_missing=False)

            response = index.upsert(namespace=namespace, vectors=vectors, batch_size=self.VECTOR_BATCH_SIZE, show_progress=False)

            if getattr(response, "has_errors", False):
                return False

            return True

        except Exception as e:
            return False


    def insert_documents(self, index_name: str, namespace: str, 
                         documents: List[dict], create_if_missing:bool =True) -> bool:

        if not documents or len(documents) == 0:
            return False

        try:

            index = self._index(index_name, create_if_missing=create_if_missing)

            for batch in self._batches(documents, size=self.MAX_RECORDS_PER_BATCH):
                index.upsert_records(namespace=namespace, records=list(batch))

            return True

        except Exception as e:
            return False


    def insert_document(self, index_name: str, namespace: str, document: dict) -> bool:

        return self.insert_documents(index_name, namespace, [document])


    def insert_chunks(self, index_name: str, namespace: str, chunks: Iterable[DocumentChunk],
                      create_if_missing: bool = True) -> bool:
        
        """Inserisce i chunk. A parita' di identita' vince l'ultimo: nessun upsert doppio."""
        unique = {chunk: chunk for chunk in chunks}
        records = [self._chunk_to_record(chunk) for chunk in unique.values()]
 
        return self.insert_documents(index_name, namespace, records, create_if_missing)

 
    def _chunk_to_record(self, chunk: DocumentChunk) -> Dict[str, Any]:

        payload = chunk._json_payload()
        record = {"_id": payload.pop("chunk_id"), self._field_value: payload.pop("text")}
        # Pinecone non accetta metadati null: i campi assenti vengono omessi.
        record.update({key: value for key, value in payload.items() if value is not None})
 
        return record
    


    # ------------------------ SEARCH IN DB -----------------------

    @staticmethod
    def _create_args(namespace: str, query: str, top_k: int, top_n: Optional[int] = None, 
                    fields: Optional[List[str]] = None, filter: Optional[Mapping[str, Any]] = None,
                    rerank_model: Optional[str] = None, rank_fields: Union[str, List[str], None] = "chunk_text"):

        if top_k < 1:
            raise ValueError(f"top_k deve essere almeno 1, ricevuto {top_k}")

        kwargs = {}

        kwargs["namespace"] = namespace

        search_query = {"inputs": {"text": query}, "top_k": top_k}
        if filter:
            search_query["filter"] = dict(filter)

        kwargs["query"] = search_query

        if fields:
            kwargs["fields"] = list(fields)

        if rerank_model:
            kwargs["rerank"] = {
                "model": rerank_model,
                "top_n": PineconeDB._rerank_top_n(top_k, top_n),
                #"rank_fields": ["chunk_text"],
                "rank_fields": [rank_fields] if isinstance(rank_fields, str) else rank_fields
            }

        return kwargs


    @staticmethod
    def _rerank_top_n(top_k: int, top_n: Optional[int]) -> int:

        if top_n is None:
            return max(1, top_k // 2)

        if not 1 <= top_n <= top_k:
            raise ValueError(f"top_n must be between 1 and top_k. Found top_n at {str(top_n)}.")

        return top_n
    

    def search_document(self, index_name: str, namespace: str, query: str, top_k:int = 5, 
                        filter: Optional[Mapping[str, Any]] = None, fields: Optional[List[str]] = None, 
                        index_mode: IndexMode = IndexMode.LLM_ONLY):

        try:
            index = self._index(index_name, mode=index_mode, create_if_missing=False)
        except IndexNotFoundError as idxErr:
            return None

        results = index.search(
            **self._create_args(namespace=namespace, query=query, top_k=top_k, fields=fields, filter=filter)
        )

        return results


    def search_by_index(self, index_name: str, namespace: str, query_id: str, top_k:int = 5, 
                        include_metadata:bool = True, include_values:bool = False, index_mode: IndexMode = IndexMode.LLM_ONLY):
        
        try:
            index = self._index(index_name, mode=index_mode, create_if_missing=False)
        except IndexNotFoundError as idxErr:
            return None

        results = index.query(
            namespace=namespace,
            id=query_id, 
            top_k=top_k,
            include_metadata=include_metadata,
            include_values=include_values
        )

        return results

    

    def search_document_and_rerank(self, index_name: str, namespace: str, query: str, top_k:int = 5, 
                                   top_n: Optional[int] = None, fields:List[str] = None, filter: Optional[Mapping[str, Any]] = None,
                                   rank_fields: Union[List[str], str, None] = None,
                                   rerank_model:str = "cohere-rerank-4-fast", index_mode: IndexMode = IndexMode.LLM_ONLY):

        try:
            index = self._index(index_name, mode=index_mode, create_if_missing=False)
        except IndexNotFoundError as idxErr:
            return None

        results = index.search(
            **self._create_args(namespace=namespace, 
                                query=query, 
                                top_k=top_k, 
                                top_n=top_n,
                                rerank_model=rerank_model,
                                fields=fields, 
                                rank_fields= rank_fields or self._field_value,
                                filter=filter
            )
        )

        return results


    def rerank_results(self, documents: List[dict], query: str, rerank_model:str = "cohere-rerank-4-fast", 
                       top_n:int = 5, rank_fields: Union[str, List[str], None] = None):

        rank_fields = rank_fields or self._field_value
        if isinstance(rank_fields, str):
            rank_fields = [rank_fields]

        results = self._client.inference.rerank(
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
    def _read(hit: Any, name: str, legacy_name: str) -> Any:
        """Legge un campo di un risultato: pinecone 10 usa hit["id"], la 7.x hit["_id"]."""
        for key in (name, legacy_name):
            try:
                return hit[key]
            except (KeyError, AttributeError, TypeError):
                continue
    
        raise KeyError(f"il risultato non ha il campo {name!r}")


    def hit_to_chunk(self, hit: Any) -> DocumentChunk:

        fields = self._read(hit, "fields", "_fields")
        last_modified = fields.get("last_modified")
    
        return DocumentChunk(
            chunk_id= self._read(hit, "id", "_id"),
            document_id=fields["document_id"],
            # Pinecone salva i numeri dei metadati come float.
            start_idx=int(fields["start_index"]),
            end_idx=int(fields["end_index"]),
            last_modified=datetime.fromisoformat(last_modified) if last_modified else None,
            text=fields[self._field_value],
        )

            
    def search_chunks(self, index_name: str, namespace: str, query: str, top_k: int = 5,
                    filter: Optional[Mapping[str, Any]] = None,
                    rerank_model: Optional[str] = None, rank_field: Union[str, List[str], None] = None,
                    top_n: Optional[int] = None) -> List[Tuple[DocumentChunk, float]]:

        if rerank_model is None:
            # cerca solo i chunk senza ordinarli
            response = self.search_document(index_name, namespace, query, top_k=top_k,
                                            filter=filter)

            if response:
                return [(self.hit_to_chunk(hit), 0.0) for hit in response.result.hits]
            
        else:
            # Come search_document, ma restituisce (chunk, score) in ordine di rilevanza.
            response = self.search_document_and_rerank(index_name, namespace, query, top_k=top_k,
                                                       filter=filter, rerank_model=rerank_model, top_n=top_n, rank_fields=rank_field)

            if response:
                return [(self.hit_to_chunk(hit), float(self._read(hit, "score", "_score"))) for hit in response.result.hits]

        return []

        
if __name__ == "__main__":

    pc = PineconeDB()

    index_name = "developer-quickstart-py"

    value = pc.create_index(index_name)

    print(value)