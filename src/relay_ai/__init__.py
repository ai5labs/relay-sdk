"""Relay AI SDK — one key, every model.

Usage:
    from relay_ai import Relay

    client = Relay(api_key="sk-relay-...")
    response = client.chat("claude-sonnet-4.6", messages=[
        {"role": "user", "content": "Hello!"}
    ])
    print(response.text)
"""

from relay_ai._client import Relay, AsyncRelay
from relay_ai._types import ChatResponse, ImageResponse, StreamChunk

__version__ = "1.0.0"
__all__ = ["Relay", "AsyncRelay", "ChatResponse", "ImageResponse", "StreamChunk"]
