from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    prompt: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None
    mode: Literal["chat", "code", "vision"] = "chat"
    model: Optional[str] = None

    @model_validator(mode="after")
    def validate_input(self) -> "ChatRequest":
        if not self.prompt and not self.messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")
        return self


class ChatResponse(BaseModel):
    model: str
    output: str
    source: str


class HealthCheckResult(BaseModel):
    url: str
    reachable: bool
    latency_ms: Optional[int] = None
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    service: str
    models: dict
    cuda: dict
    ollama: dict


class RagRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: Optional[int] = Field(default=None, ge=1, le=10)


class RagResponse(BaseModel):
    answer: str
    sources: List[str]
    retrieved_chunks: List[str]
    distances: List[float]
    used_rag: bool
    search_query: str
    timings: dict
