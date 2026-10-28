from typing import Sequence, Dict, Any, Iterator, Optional, List, Mapping, Union
from src.chunks.document_chunk import DocumentChunk
from src.chunks.legal_chunk import LegalChunk

from uuid import UUID
from datetime import datetime

_MATCH_META = {"_id", "id", "_score", "score"}



def _chunk_to_record(chunk: DocumentChunk, vector: Sequence[float]) -> Dict[str, Any]:

    payload = chunk._json_payload()
    chunk_id = payload.pop("chunk_id")
    # Pinecone non accetta metadati null: i campi assenti vengono omessi.
    metadata = {key: value for key, value in payload.items() if value is not None}
    return {"id": chunk_id, "values": [float(x) for x in vector], "metadata": metadata}


def _batches(items: Sequence[Any], size: int = 16) -> Iterator[Sequence[Any]]:

    for start in range(0, len(items), size):
        yield items[start:start + size]


def _rerank_top_n(top_k: int, top_n: Optional[int]) -> int:

    if top_n is None:
        return max(1, top_k // 2)

    if not 1 <= top_n <= top_k:
        raise ValueError(f"top_n must be between 1 and top_k. Found top_n at {str(top_n)}.")

    return top_n


def _read(hit: Any, name: str, legacy_name: str) -> Any:
    """Legge un campo di un risultato: pinecone 10 usa hit["id"], la 7.x hit["_id"]."""
    for key in (name, legacy_name):
        try:
            return hit[key]
        except (KeyError, AttributeError, TypeError):
            value = getattr(hit, key, object())
            if value is not object():
                return value
            continue

    raise KeyError(f"il risultato non ha il campo {name!r}")


def _match_fields(match: Any) -> Dict[str, Any]:

    data = _match_to_dict(match)
    nested = data.get("fields") or data.get("_fields")

    if isinstance(nested, Mapping):
        return dict(nested)
    
    return {k: v for k, v in data.items() if k not in _MATCH_META}


def _match_to_dict(match: Any) -> Dict[str, Any]:
    if isinstance(match, Mapping):
        return dict(match)
    
    for method in ("to_dict", "model_dump", "dict"):
        fn = getattr(match, method, None)
        if callable(fn):
            try:
                data = fn()
                if isinstance(data, Mapping):
                    return dict(data)
            except Exception:
                pass

    if hasattr(match, "__dict__"):
        return {k: v for k, v in vars(match).items() if not k.startswith("__")}
    
    return {k: getattr(match, k) for k in dir(match)
            if not k.startswith("__") and not callable(getattr(match, k, None))}


def _create_args(namespace: str, top_k: int, top_n: Optional[int] = None,
                 document_id: Optional[UUID] = None,
                 fields: Optional[List[str]] = None,
                 filter: Optional[Mapping[str, Any]] = None,
                 rerank_model: Optional[str] = None,
                 rank_fields: Union[str, List[str], None] = "text",
                 rerank_query: Optional[str] = None) -> Dict[str, Any]:
    
    """Argomenti per index.search. Il vettore lo aggiunge chi chiama.

    rerank_query: la domanda in testo. Con una ricerca per vettore e'
    obbligatoria se c'e' il rerank: il reranker confronta testo con testo, e
    dal vettore non puo' ricavare la domanda.
    """

    if top_k < 1:
        raise ValueError(f"top_k deve essere almeno 1, ricevuto {top_k}")

    kwargs: Dict[str, Any] = {"namespace": namespace, "top_k": top_k}

    # Filtro dell'utente e filtro per documento si combinano con $and.
    conditions = [dict(filter)] if filter else []
    if document_id is not None:
        conditions.append({"document_id": {"$eq": str(document_id)}})
    if len(conditions) == 1:
        kwargs["filter"] = conditions[0]
    elif conditions:
        kwargs["filter"] = {"$and": conditions}

    if fields:
        kwargs["fields"] = list(fields)

    if rerank_model:
        if not rerank_query or not rerank_query.strip():
            raise ValueError("con una ricerca per vettore il rerank ha bisogno della domanda "
                             "in testo: passa rerank_query")
        kwargs["rerank"] = {
            "model": rerank_model,
            "top_n": _rerank_top_n(top_k, top_n),
            "rank_fields": [rank_fields] if isinstance(rank_fields, str) else list(rank_fields),
            "query": rerank_query,
        }

    return kwargs


def hit_to_chunk(chunk_id: str, metadata: Mapping[str, Any]) -> DocumentChunk:

    last_modified = metadata.get("last_modified")

    return DocumentChunk(
        chunk_id=UUID(chunk_id),
        document_id=UUID(metadata["document_id"]),
        # Pinecone restituisce i numeri dei metadati come float.
        start_idx=int(metadata["start_index"]),
        end_idx=int(metadata["end_index"]),
        last_modified=datetime.fromisoformat(last_modified) if last_modified else None,
        text=metadata["text"],
    )
