"""Response types for the Relay AI SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatResponse:
    text: str = ""
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.usage.total_tokens


@dataclass
class ImageResponse:
    images: list[str] = field(default_factory=list)
    model: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamChunk:
    text: str = ""
    done: bool = False
    usage: Usage | None = None
    raw: dict[str, Any] = field(default_factory=dict)
