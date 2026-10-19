import os

from pydantic import Field
from pydantic_settings import BaseSettings
from src.api.prompt import SYSTEM_PROMPT, CONDENSE_PROMPT


class Settings(BaseSettings):

    index_name: str = Field(default="paradigma", env="INDEX_NAME")
    top_k: int = Field(default=20, env="TOP_K")
    top_n: int = Field(default=5, env="TOP_N")
    max_document_chars: int = Field(default = 2_000_000, env="MAX_DOCUMENT_CHARS")
    max_question_chars: int = Field(default=2_000, env="MAX_QUESTION_CHARS")
    max_tokens: int = Field(default=1000, env="MAX_TOKENS")
    system_prompt: str = Field(default=SYSTEM_PROMPT, env="SYSTEM_PROMPT")
    condens_prompt: str = Field(default=CONDENSE_PROMPT, env="CONDENSE_PROMPT")

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"
    }
