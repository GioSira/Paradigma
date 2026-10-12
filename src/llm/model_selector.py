from enum import Enum


class ModelSelector(Enum):

    # Chat
    NEMOTRON_3_SUPER = "nvidia/nemotron-3-super-120b-a12b:free"
    NEMOTRON_3_5_LIGHTNING = "nvidia/nemotron-3.5-lightning:free"
    THINKING_MACHINE = "thinkingmachines/inkling:free"

    # Rerank
    VOYAGE_RERANK = "voyageai/rerank-2.5"

    # Embedding: un modello di chat non produce embedding.
    NEMOTRON_EMBED = "nvidia/nemotron-3-embed-1b:free"
    VOYAGE_EMBED_LITE = "voyageai/voyage-4-lite"   # 256, 512, 1024 o 2048 dimensioni