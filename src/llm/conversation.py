from typing import List, Literal, Optional, Iterable
from pydantic import BaseModel, Field, ConfigDict, field_validator


Role = Literal["user", "system", "assistant"]


class Message(BaseModel):

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Role = "user"
    content: str = Field(..., min_length=1)

    @field_validator("content")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("il messaggio non puo' essere vuoto")
        return value


class Chat(BaseModel):

    model_config = ConfigDict(extra="forbid")

    history: List[Message] = Field(default_factory=list)


    @classmethod
    def of(cls, messages: Optional[Iterable[Message]]) -> "Chat":
        return cls(history=list(messages or []))


    def to_model(self, system_prompt: str):
        return [{"role": "system", "content": system_prompt}] + [{"role": message.role, "content": message.content} for message in self.history]


    def add(self, content: str, role: Role = "user"):
        self.history.append(Message(content=content, role=role))


    def last_messages(self, max_messages: int):

        if max_messages < 0:
            raise ValueError("max_messages cannot be negative")

        if max_messages == 0:
            return Chat()

        kept = self.history[-max_messages:]
        # find first user question in kept
        while len(kept) > 0 and kept[0].role != "user":
            kept = kept[1:]

        return Chat(history=list(kept))


    def last_user_message(self):

        for message in reversed(self.history):
            if message.role == "user":
                return message.content

        return None


    def __len__(self) -> int:
        return len(self.history)
