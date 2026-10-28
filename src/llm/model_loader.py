import os

from openrouter import OpenRouter
from openrouter.utils.retries import BackoffStrategy, RetryConfig
from openrouter.errors import (TooManyRequestsResponseError, BadGatewayResponseError, 
                               ProviderOverloadedResponseError, ServiceUnavailableResponseError,
                               RequestTimeoutResponseError)

from dotenv import load_dotenv
from typing import List, Any, Tuple, Optional, Dict
from uuid import UUID

from src.llm.model_selector import ModelSelector
from src.llm.errors import GenerationError
from src.llm.conversation import Chat
from src.llm.utils import retry_async

from src.chunks.document_chunk import DocumentChunk
from src.collection.document import Document


EXCEPTION_TUPLE = (TooManyRequestsResponseError, BadGatewayResponseError, 
                   ProviderOverloadedResponseError, ServiceUnavailableResponseError,
                   RequestTimeoutResponseError)


class ModelLoader:

    def __init__(self,
                 llm_model: ModelSelector = ModelSelector.NEMOTRON_3_ULTRA,
                 rerank_model: ModelSelector = ModelSelector.VOYAGE_RERANK,
                 embed_model: ModelSelector = ModelSelector.NEMOTRON_EMBED,
                 embed_dimension: Optional[int] = None,
                 reasoning_effort: Optional[str] = "none",
                 max_tokens: int = 1000,
                 timeout_seconds: float = 30,
                 retry_seconds: float = 60,
                 memory_size: int = 5,
                 client: Optional[OpenRouter] = None):

        if timeout_seconds <= 0 or retry_seconds < 0:
            raise ValueError("timeout_seconds deve essere > 0 e retry_seconds >= 0")

        if memory_size <= 0:
            raise ValueError("memory size must be greater than 0")
        
        self._model = llm_model
        self._reranker = rerank_model
        self._embed_model = embed_model
        self._embed_dimension = embed_dimension
        self._reasoning_effort = reasoning_effort
        self._max_tokens = max_tokens

        self._memory_size = memory_size
 
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
            "timeout_ms": self._timeout_ms,
        }


    # ============================== EMBEDDING ==============================

    @retry_async(EXCEPTION_TUPLE, max_retries=5, base_delay=5)
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
            params["dimensions"] = self._embed_dimension

        response = await self._client.embeddings.generate_async(input=text_docs, **params)

        # ensure order
        data = sorted(response.data, key=lambda item: item.index)

        if len(data) != len(documents):
            raise RuntimeError(f"attesi {len(documents)} embedding, ricevuti {len(response.data)}")

        return [(doc.document_id, item.embedding) for doc, item in zip(documents, data)]


    @retry_async(EXCEPTION_TUPLE, max_retries=5, base_delay=5)
    async def embed_text(self, text: str) -> Any:

        if not text or not text.strip():
            raise ValueError("il testo da trasformare in embedding e' vuoto")
 
        params = self._get_params(self._embed_model)
        if self._embed_dimension is not None:
            params["dimensions"] = self._embed_dimension

        response = await self._client.embeddings.generate_async(input=text, **params)

        return response.data[0].embedding


    @retry_async(EXCEPTION_TUPLE, max_retries=5, base_delay=5)
    async def embed_text_batch(self, text_batch: List[str]) -> List[Any]:

        if not text_batch or len(text_batch) == 0:
            return []

        params = self._get_params(self._embed_model)
        if self._embed_dimension is not None:
            params["dimensions"] = self._embed_dimension
        
        response = await self._client.embeddings.generate_async(input=text_batch, **params)
        
        data = sorted(response.data, key=lambda item: item.index)
        
        if len(data) != len(text_batch):
            raise RuntimeError(f"attesi {len(text_batch)} embedding, ricevuti {len(response.data)}")
        
        return [(doc, item.embedding) for doc, item in zip(text_batch, data)]
        


    # ============================== RERANK ==============================

    @retry_async(EXCEPTION_TUPLE, max_retries=5, base_delay=5)
    async def rerank(self, query: str, documents: List[DocumentChunk], top_n: int = 3) -> List[Tuple[float, str]]:

        if top_n < 1:
            raise ValueError(f"top_n deve essere almeno 1, ricevuto {top_n}")
        
        if not documents:
            return []

        document_text = [d.get_chunk_text() or "" for d in documents]

        response = await self._client.rerank.rerank_async(documents=document_text, 
                                                          query=query, 
                                                          top_n=top_n, 
                                                          **self._get_params(self._reranker))

        ranked = sorted(response.results, key=lambda r: r.relevance_score, reverse=True)

        return [(documents[r.index], r.relevance_score) for r in ranked]


    # ============================== GENERATION ==============================


    @retry_async(EXCEPTION_TUPLE, max_retries=5, base_delay=5)
    async def _send_message(self, messages: Dict[str, str], params: Dict[str, Any], top_p: float):

        response = await self._client.chat.send_async(messages=messages, 
                                                              max_completion_tokens=self._max_tokens, 
                                                              top_p=top_p, 
                                                              **params)
        
        choice = response.choices[0] if response.choices else None
        content = choice.message.content if choice else None
        finish_reason = getattr(choice, "finish_reason", None)

        if isinstance(content, list):
            content = "".join(getattr(part, "text", None) or "" for part in content)

        if not content or not content.strip():
            if finish_reason == "length":
                raise GenerationError(
                    f"il modello ha finito i {self._max_tokens} token prima di scrivere la risposta"
                )
            
            raise GenerationError(f"il modello non ha restituito testo (finish_reason={finish_reason})",
                                    finish_reason)

        return content


    @retry_async(EXCEPTION_TUPLE, max_retries=5, base_delay=5)
    async def generate(self, query: str, context: List[str], chat: Chat, system_prompt: str, top_p=0.95):

        llm_context = "\n\n".join(context)

        chat.add(content=f"Domanda: {query}\n\nContesto:\n{llm_context}", role="user")

        params = self._get_params(self._model)
        if self._reasoning_effort is not None:
            params["reasoning"] = {"effort": self._reasoning_effort}

        last_messages = chat.last_messages(self._memory_size + 1) # add last question
        content = await self._send_message(last_messages.to_model(system_prompt), params, top_p=top_p)
        
        return content


    @retry_async(EXCEPTION_TUPLE, max_retries=5, base_delay=5)
    async def expanse_question(self, context: str, 
                                system_prompt: str, 
                                chat: Chat,
                                top_p=0.95):

        conversation = []
        for message in chat.last_messages(self._memory_size).history:
            role = "Utente" if message.role == "user" else "Assistente"
            conversation.append(f"{role}: {message.content}")
        conversation = "\n".join(conversation)

        model_input = f"Conversazione:\n{conversation}\nUltima Domanda: {context}\n"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": model_input}
        ]

        params = self._get_params(self._model)
        if self._reasoning_effort is not None:
            params["reasoning"] = {"effort": self._reasoning_effort}

        content = await self._send_message(messages, params, top_p)

        return content
