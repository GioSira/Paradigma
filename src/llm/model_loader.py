import os

from openrouter import OpenRouter
from openrouter.utils.retries import BackoffStrategy, RetryConfig

from dotenv import load_dotenv
from typing import List, Any, Tuple, Optional
from uuid import UUID

from src.collection.document import Document
from src.llm.model_selector import ModelSelector


class ModelLoader:

    def __init__(self,
                 llm_model: ModelSelector = ModelSelector.NEMOTRON_3_5_LIGHTNING,
                 rerank_model: ModelSelector = ModelSelector.VOYAGE_RERANK,
                 embed_model: ModelSelector = ModelSelector.NEMOTRON_EMBED,
                 embed_dimension: Optional[int] = None,
                 timeout_seconds: float = 30,
                 retry_seconds: float = 60,
                 client: Optional[OpenRouter] = None):

        if timeout_seconds <= 0 or retry_seconds < 0:
            raise ValueError("timeout_seconds deve essere > 0 e retry_seconds >= 0")
        
        self._model = llm_model
        self._reranker = rerank_model
        self._embed_model = embed_model
        self._embed_dimension = embed_dimension
 
        self._timeout_ms = int(timeout_seconds * 1000)
        self._retries = RetryConfig(
            "backoff",
            BackoffStrategy(initial_interval=500, max_interval=10_000, exponent=1.5,
                            max_elapsed_time=int(retry_seconds * 1000)),
            retry_connection_errors=True,
        )
 
        self._client = client or self._connect()


    @staticmethod
    def _connect():
        load_dotenv()
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY non impostata")
        
        return OpenRouter(api_key=key)


    # ============================== CHIUSURA ==============================
    # Il client tiene aperte le connessioni HTTP per riusarle, e ognuna
    # appartiene all'event loop che l'ha aperta. Va usato sempre dallo stesso
    # loop e chiuso prima che il loop finisca:
    #
    #     async with ModelLoader() as models:
    #         await models.embed_text("...")
 
 
    async def aclose(self) -> None:
        close = getattr(self._client, "__aexit__", None)
        if close is not None:
            await close(None, None, None)
 
 
    async def __aenter__(self) -> "ModelLoader":
        return self
 
 
    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.aclose()


    # ============================== UTILS ==============================
 
    @staticmethod
    def _connect() -> OpenRouter:
        load_dotenv()
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY non impostata")
        return OpenRouter(api_key=key)
 


    def _get_params(self, model: ModelSelector):

        return {
            "model": model.value,
            "retries": self._retries,
            "timeout_ms": self._timeout_ms 
        }


    # ============================== EMBEDDING ==============================

    async def embed_documents(self, documents: List[Document]) -> List[Tuple[UUID, Any]]:

        if not documents or len(documents) == 0:
            return []

        text_docs = []
        for doc in documents:
            text = doc.get_document_text()
            if not text or not text.strip():
                raise ValueError(f"il documento {doc.document_id} non ha testo da indicizzare")
            text_docs.append(text)


        params = self._get_params(self._embed_model)
        if self._embed_dimension is not None:
            params["dimension"] = self._embed_dimension

        response = await self._client.embeddings.generate_async(input=text_docs, **params)

        data = response.data

        if len(data) != len(documents):
            raise RuntimeError(f"attesi {len(documents)} embedding, ricevuti {len(response.data)}")

        return [(doc.document_id, item.embedding) for doc, item in zip(documents, data)]


    async def embed_text(self, text: str) -> Any:

        if not text or not text.strip():
            raise ValueError("il testo da trasformare in embedding e' vuoto")
 
        params = self._get_params(self._embed_model)
        if self._embed_dimension is not None:
            params["dimensions"] = self._embed_dimension

        response = await self._client.embeddings.generate_async(input=text, **params)

        return response.data[0].embedding


    # ============================== RERANK ==============================

    async def rerank(self, query: str, documents: List[Document], top_n: int = 3) -> List[Tuple[float, str]]:

        if top_n < 1:
            raise ValueError(f"top_n deve essere almeno 1, ricevuto {top_n}")
        
        if not documents:
            return []

        document_text = [d.get_document_text() or "" for d in documents]

        response = await self._client.rerank.rerank_async(documents=document_text, 
                                                          query=query, 
                                                          top_n=top_n, 
                                                          **self._get_params(self._reranker))

        ranked = sorted(response.results, key=lambda r: r.relevance_score, reverse=True)

        return [(documents[r.index], r.relevance_score) for r in ranked]


    # ============================== GENERATION ==============================


    async def generate(self, system_prompt: str, query: str, context: List[str], max_tokens: int = 1000, top_p=0.95):

        llm_context = "\n\n".join(context)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {query}\n\nContext:\n{llm_context}"}
        ]

        response = await self._client.chat.send_async(messages=messages, 
                                                      max_completion_tokens=max_tokens, 
                                                      top_p=top_p, 
                                                      **self._get_params(self._model))

        content = response.choices[0].message.content if response.choices else None

        if not content:
            raise RuntimeError("il modello non ha restituito testo")

        if isinstance(content, list):
            content = "".join(getattr(part, "text", None) or "" for part in content)
        
        return content
    