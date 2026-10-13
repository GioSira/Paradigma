from enum import Enum, auto

class IndexMode(Enum):

    BM25_HYB = auto()
    LLM = auto()
    PINECONE = auto()