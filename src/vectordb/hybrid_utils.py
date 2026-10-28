from typing import Any, Dict, Hashable, List, Sequence, Tuple, TypeVar, Mapping
from src.chunks.legal_chunk import LegalChunk
from src.vectordb.utils import _read

from uuid import UUID
from datetime import datetime
import json
import math

T = TypeVar("T")

_RESERVED_FIELDS = {"_id", "id", "embedding", "text", "document_title", "document_name"}


_CHUNK_FIELDS = {"document_id", "text", "document_name", "document_title", "chunk_type",
                 "article", "article_title", "paragraph", "letter",
                 "start_index", "end_index", "last_modified", "embedding"}


def _hit_key(item: Any) -> str:
    return str(_read(item, "id", "_id"))


def _reciprocal_rank_fusion(rankings: Sequence[Sequence[T]], key = _hit_key, k: int = 60) -> List[Tuple[T, float]]:
    """Fonde piu' classifiche in una: ogni elemento prende 1/(k + posizione)
    da ogni classifica in cui compare, e i contributi si sommano.
 
    Conta solo la posizione, non il punteggio: coseno e BM25 hanno scale
    diverse e non si possono sommare. Un brano che sta in alto in entrambe le
    classifiche batte uno che e' primo in una sola.
 
    k (60 e' il valore classico) smorza il peso delle prime posizioni.
    A parita' di punteggio l'ordine e' deciso dalla chiave: il risultato e'
    sempre lo stesso a parita' di input.
    """
    if k <= 0:
        raise ValueError("k deve essere positivo")
 
    scores: Dict[Hashable, float] = {}
    items: Dict[Hashable, T] = {}
 
    for ranking in rankings:
        
        seen = set()
        
        for position, item in enumerate(ranking, start=1):
            item_key = key(item)
            
            if item_key in seen:          # doppione nella stessa classifica: conta una volta
                continue
            seen.add(item_key)

            scores[item_key] = scores.get(item_key, 0.0) + 1.0 / (k + position)
            items.setdefault(item_key, item)
 
    order = sorted(scores, key=lambda item_key: (-scores[item_key], str(item_key)))
    return [(items[item_key], scores[item_key]) for item_key in order]


def _chunk_to_record_bm25(chunk: "LegalChunk", vector: Sequence[float]) -> Dict[str, Any]:
    payload = chunk._json_payload()

    chunk_id = payload.pop("chunk_id", "")
    chunk_text = payload.pop("text", "")
    document_name = payload.pop("document_name", "")
    document_title = payload.pop("document_title", "")

    raw_metadata = payload.pop("metadata", {})
    if isinstance(raw_metadata, str):
        try:
            raw_metadata = json.loads(raw_metadata)
        except Exception:
            raw_metadata = {"raw_string": raw_metadata}

    if not isinstance(raw_metadata, dict):
        raw_metadata = {}

    # Il resto del payload confluisce nei metadati
    for key, value in payload.items():
        if value is not None:
            raw_metadata[key] = value

    record: Dict[str, Any] = {
        "_id": str(chunk_id),
        "embedding": [float(x) for x in vector],
        "text": str(chunk_text),
        "document_title": str(document_title),
        "document_name": str(document_name),
    }

    groups = raw_metadata.pop("linked_predicate_offences", [])
    record["linked_predicate_articles"] = sorted({a for g in groups for a in (json.loads(g) if isinstance(g, str) else g)})

    flat_metadata = _sanitize_metadata(raw_metadata)

    # Metadati come campi di primo livello (NON sotto "metadata")
    for key, value in flat_metadata.items():
        field = f"meta_{key}" if key in _RESERVED_FIELDS else key
        record[field] = value

    return record


def _sanitize_metadata(metadata: dict) -> dict:
    clean: Dict[str, Any] = {}

    def _rec(prefix: str, val: Any) -> None:
        
        if val is None:
            return
        
        if isinstance(val, str) and val.strip()[:1] in ("{", "["):
            try:
                _rec(prefix, json.loads(val))
                return
            except Exception:
                pass

        if isinstance(val, bool) or isinstance(val, (int, str)):
            clean[prefix] = val

        elif isinstance(val, float):
            if math.isfinite(val):          # NaN/inf non sono accettati
                clean[prefix] = val

        elif isinstance(val, list):
            clean[prefix] = [
                json.dumps(i, ensure_ascii=False) if isinstance(i, (dict, list)) else str(i)
                for i in val if i is not None
            ]

        elif isinstance(val, dict):
            for k, v in val.items():
                _rec(f"{prefix}_{k}" if prefix else str(k), v)

        else:
            clean[prefix] = str(val)

    for k, v in metadata.items():
        _rec(str(k), v)

    return clean


def hit_to_chunk_hybrid(chunk_id: str, fields: Mapping[str, Any]) -> LegalChunk:

    fields = fields or {}
    last_modified = fields.get("last_modified")

    return LegalChunk(
        chunk_id=UUID(str(chunk_id)),
        document_id=UUID(str(fields["document_id"])),   
        text=str(fields.get("text", "")),
        document_name=str(fields.get("document_name", "")),
        document_title=str(fields.get("document_title", "")),
        chunk_type=str(fields.get("chunk_type", "")),
        article=str(fields.get("article", "")),
        article_title=str(fields.get("article_title", "")),
        paragraph=str(fields.get("paragraph", "")),
        letter=str(fields.get("letter", "")),
        start_idx=int(fields.get("start_index") or 0),
        end_idx=int(fields.get("end_index") or 0),
        last_modified=datetime.fromisoformat(last_modified) if last_modified else None,
        metadata={k: v for k, v in fields.items() if k not in _CHUNK_FIELDS},
    )
