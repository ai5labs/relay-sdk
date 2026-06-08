"""Relay AI SDK — one key, every model.

Usage::

    from relay_ai import Relay

    client = Relay(api_key="sk-relay-...")
    response = client.chat("claude-sonnet-4.6", messages=[
        {"role": "user", "content": "Hello!"}
    ])
    print(response.text)
"""

from relay_ai._client import AsyncRelay, Relay
from relay_ai._errors import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ContentPolicyError,
    ContextWindowError,
    InsufficientCreditsError,
    InternalServerError,
    ModelNotFoundError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    RelayError,
)
from relay_ai._streaming import AsyncStream, Stream
from relay_ai._types import (
    AudioResponse,
    BatchResult,
    ChatResponse,
    CreditState,
    ImageResponse,
    RouteAlternate,
    RouteResponse,
    SpeechResponse,
    StreamChunk,
    ToolCall,
    ToolCallDelta,
    Usage,
)
from relay_ai._version import __version__

__all__ = [
    # Version
    "__version__",
    # Clients
    "Relay",
    "AsyncRelay",
    # Streaming
    "Stream",
    "AsyncStream",
    # Response types
    "ChatResponse",
    "StreamChunk",
    "ImageResponse",
    "AudioResponse",
    "SpeechResponse",
    "RouteResponse",
    "RouteAlternate",
    "CreditState",
    "BatchResult",
    "Usage",
    "ToolCall",
    "ToolCallDelta",
    # Errors
    "RelayError",
    "APIConnectionError",
    "APITimeoutError",
    "APIStatusError",
    "AuthenticationError",
    "InsufficientCreditsError",
    "PermissionDeniedError",
    "NotFoundError",
    "ModelNotFoundError",
    "RateLimitError",
    "BadRequestError",
    "ContentPolicyError",
    "ContextWindowError",
    "InternalServerError",
]
