import os
from dotenv import load_dotenv

from pydantic_settings import BaseSettings
from src.api.prompt import SYSTEM_PROMPT, EXPANSE_PROMPT

load_dotenv()

class Settings(BaseSettings):

    index_name: str = os.getenv("INDEX_NAME") or "paradigma"
    top_k: int = os.getenv("TOP_K") or 20
    top_n: int = os.getenv("TOP_N") or 5
    max_document_chars: int = os.getenv("MAX_DOCUMENT_CHARS") or 2_000_000
    max_question_chars: int = os.getenv("MAX_QUESTION_CHARS") or 2_000
    max_tokens: int = os.getenv("MAX_TOKENS") or 1000
    system_prompt: str = os.getenv("SYSTEM_PROMPT") or SYSTEM_PROMPT
    condens_prompt: str = os.getenv("CONDENSE_PROMPT") or EXPANSE_PROMPT

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"
    }
