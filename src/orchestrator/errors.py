from typing import Sequence
from uuid import UUID

class IndexingError(RuntimeError):
    """Pinecone non ha salvato i chunk: ripetere l'indicizzazione e' sicuro."""

    def __init__(self, document_ids: Sequence[UUID]):
        super().__init__(f"indicizzazione non riuscita per {len(document_ids)} documento/i")
        self.document_ids = list(document_ids)


class DeleteError(Exception):

    def __init__(self, document_id: UUID, namespace: str):
        super().__init__(f"Impossibile cancellare chunk del documento {document_id} nel namespace {namespace}")
        self.document_id = document_id
        self.namespace = namespace