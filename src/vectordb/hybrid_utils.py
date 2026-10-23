from typing import Any, Dict, Hashable, List, Sequence, Tuple, TypeVar
from src.chunks.legal_chunk import LegalChunk

T = TypeVar("T")


def _reciprocal_rank_fusion(rankings: Sequence[Sequence[T]], key=lambda item: item.id, k: int = 60) -> List[Tuple[T, float]]:
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


def _chunk_to_record_bm25(chunk: LegalChunk, vector: Sequence[float]) -> Dict[str, Any]:

    payload = chunk._json_payload()

    chunk_id = payload.pop("chunk_id")
    chunk_text = payload.pop("chunk_text", None)

    metadata = {key: value for key, value in payload.items() if value is not None}

    return {"id": chunk_id,
            "embedding": [float(x) for x in vector],
            "text": chunk_text,
            "metadata": metadata
        } 