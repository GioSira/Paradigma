import os

from dotenv import load_dotenv
from dataclasses import dataclass

from src.api.prompt import SYSTEM_PROMPT, CONDENSE_PROMPT


def _env_int(name: str, default: int) -> int:
    # os.getenv restituisce sempre stringhe: "10" non e' 10, e "10" <= "5" e' True.
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{name} deve essere un intero, trovato {value!r}") from None


@dataclass(frozen=True)
class Settings:

    index_name: str = "paradigma"
    top_k: int = 10
    top_n: int = 3
    max_document_chars: int = 2_000_000
    max_question_chars: int = 2_000
    max_tokens: int = 1000
    system_prompt: str = SYSTEM_PROMPT
    condens_prompt: str = CONDENSE_PROMPT

    @classmethod
    def from_env(cls) -> "Settings":
        """Legge l'ambiente quando viene chiamata, non all'import del modulo:
        cosi' i test possono costruire Settings(...) senza toccare il .env."""
        load_dotenv()
        return cls(
            index_name=os.getenv("INDEX_NAME") or cls.index_name,
            top_k=_env_int("TOP_K", cls.top_k),
            top_n=_env_int("TOP_N", cls.top_n),
            max_document_chars=_env_int("MAX_DOCUMENT_CHARS", cls.max_document_chars),
            max_question_chars=_env_int("MAX_QUESTION_CHARS", cls.max_question_chars),
            max_tokens=_env_int("MAX_TOKENS", cls.max_tokens)
        )
