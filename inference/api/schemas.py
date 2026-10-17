from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Role = Literal["consciousness", "subconsciousness"]


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=262144)


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    messages: list[Message] = Field(min_length=1, max_length=256)
    max_new_tokens: int = Field(default=256, ge=1, le=4096)

    @model_validator(mode="after")
    def validate_messages(self):
        if self.messages[-1].role != "user":
            raise ValueError("messages должны заканчиваться репликой user")
        if any(m.role == "system" for m in self.messages[1:]):
            raise ValueError("system разрешён только первым сообщением")
        return self


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class GenerateResponse(BaseModel):
    request_id: str
    role: Role
    adapter_step: int
    content: str
    finish_reason: Literal["eot", "eos", "repetition", "max_new_tokens"]
    usage: Usage
    generation_seconds: float


class InputTooLong(Exception):
    pass


class GenerationFailed(Exception):
    pass
