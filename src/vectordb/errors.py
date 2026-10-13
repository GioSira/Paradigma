
class IndexNotFoundError(LookupError):
    """L'indice richiesto non esiste su Pinecone."""

class DimensionMismatchError(ValueError):
    """L'indice esiste ma ha una dimensione diversa da quella del modello."""