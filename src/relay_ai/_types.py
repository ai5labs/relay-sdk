"""Response types for the Relay AI SDK.

Pydantic models following the same conventions as the OpenAI and Anthropic SDKs.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0


class ToolCall(BaseModel):
    id: str = ""
    type: str = "function"
    function_name: str = ""
    function_arguments: str = ""


class ChatResponse(BaseModel):
    text: str = ""
    model: str = ""
    usage: Usage = Field(default_factory=Usage)
    finish_reason: Optional[str] = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    latency_ms: float = 0.0
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.usage.total_tokens


class ToolCallDelta(BaseModel):
    index: int = 0
    id: Optional[str] = None
    function_name: Optional[str] = None
    function_arguments_delta: str = ""


class StreamChunk(BaseModel):
    text: str = ""
    done: bool = False
    finish_reason: Optional[str] = None
    tool_call_deltas: list[ToolCallDelta] = Field(default_factory=list)
    usage: Optional[Usage] = None
    raw: dict[str, Any] = Field(default_factory=dict)


class ImageResponse(BaseModel):
    images: list[str] = Field(default_factory=list)
    model: str = ""
    raw: dict[str, Any] = Field(default_factory=dict)


class AudioResponse(BaseModel):
    text: str = ""
    model: str = ""
    latency_ms: float = 0.0
    raw: dict[str, Any] = Field(default_factory=dict)


class SpeechResponse(BaseModel):
    audio: bytes = b""
    content_type: str = "audio/mpeg"
    model: str = ""
    latency_ms: float = 0.0

    model_config = {"arbitrary_types_allowed": True}


class RouteAlternate(BaseModel):
    alias: str = ""
    confidence: float = 0.0


class RouteResponse(BaseModel):
    alias: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    alternates: list[RouteAlternate] = Field(default_factory=list)
    classified_intent: str = ""
    source: str = ""
    raw: dict[str, Any] = Field(default_factory=dict)


class CreditState(BaseModel):
    balance_cents: int = 0
    min_topup_cents: int = 0
    max_topup_cents: int = 0


class BatchResult(BaseModel):
    index: int = 0
    response: Optional[ChatResponse] = None
    error: Optional[str] = None
