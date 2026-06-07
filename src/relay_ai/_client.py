"""Relay AI client — thin SDK that talks to the Relay gateway."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx

from relay_ai._types import ChatResponse, ImageResponse, StreamChunk, Usage

BASE_URL = "https://api.relay.ai5labs.com/v1"


class Relay:
    """Synchronous Relay client.

    Usage:
        client = Relay(api_key="sk-relay-...")
        resp = client.chat("claude-sonnet-4.6", messages=[
            {"role": "user", "content": "Hello!"}
        ])
        print(resp.text)
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ):
        self.api_key = api_key or os.environ.get("RELAY_API_KEY", "")
        self.base_url = (base_url or BASE_URL).rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=timeout,
        )

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatResponse | Iterator[StreamChunk]:
        body: dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if top_p is not None:
            body["top_p"] = top_p
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        body.update(kwargs)

        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
            return self._stream_chat(body)

        resp = self._client.post("/chat/completions", json=body)
        resp.raise_for_status()
        data = resp.json()
        return _parse_chat(data)

    def _stream_chat(self, body: dict[str, Any]) -> Iterator[StreamChunk]:
        with self._client.stream("POST", "/chat/completions", json=body) as resp:
            resp.raise_for_status()
            buffer = ""
            for chunk in resp.iter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    frame, buffer = buffer.split("\n\n", 1)
                    yield from _parse_sse_frame(frame)

    def images(
        self,
        model: str,
        prompt: str,
        *,
        n: int = 1,
        size: str = "1024x1024",
        **kwargs: Any,
    ) -> ImageResponse:
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
            **kwargs,
        }
        resp = self._client.post("/images/generations", json=body)
        resp.raise_for_status()
        data = resp.json()
        urls = [img.get("url", "") for img in data.get("data", [])]
        return ImageResponse(images=urls, model=model, raw=data)

    def models(self) -> list[str]:
        resp = self._client.get("/models")
        resp.raise_for_status()
        data = resp.json()
        return [m["id"] for m in data.get("data", [])]

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class AsyncRelay:
    """Async Relay client.

    Usage:
        async with AsyncRelay(api_key="sk-relay-...") as client:
            resp = await client.chat("claude-sonnet-4.6", messages=[...])
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ):
        self.api_key = api_key or os.environ.get("RELAY_API_KEY", "")
        self.base_url = (base_url or BASE_URL).rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=timeout,
        )

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatResponse | AsyncIterator[StreamChunk]:
        body: dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if top_p is not None:
            body["top_p"] = top_p
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        body.update(kwargs)

        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
            return self._stream_chat(body)

        resp = await self._client.post("/chat/completions", json=body)
        resp.raise_for_status()
        data = resp.json()
        return _parse_chat(data)

    async def _stream_chat(self, body: dict[str, Any]) -> AsyncIterator[StreamChunk]:
        async with self._client.stream("POST", "/chat/completions", json=body) as resp:
            resp.raise_for_status()
            buffer = ""
            async for chunk in resp.aiter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    frame, buffer = buffer.split("\n\n", 1)
                    for sc in _parse_sse_frame(frame):
                        yield sc

    async def images(
        self,
        model: str,
        prompt: str,
        *,
        n: int = 1,
        size: str = "1024x1024",
        **kwargs: Any,
    ) -> ImageResponse:
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
            **kwargs,
        }
        resp = await self._client.post("/images/generations", json=body)
        resp.raise_for_status()
        data = resp.json()
        urls = [img.get("url", "") for img in data.get("data", [])]
        return ImageResponse(images=urls, model=model, raw=data)

    async def models(self) -> list[str]:
        resp = await self._client.get("/models")
        resp.raise_for_status()
        data = resp.json()
        return [m["id"] for m in data.get("data", [])]

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()


def _parse_chat(data: dict[str, Any]) -> ChatResponse:
    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {})
    text = msg.get("content", "") or ""
    usage_raw = data.get("usage", {})
    usage = Usage(
        prompt_tokens=usage_raw.get("prompt_tokens", 0),
        completion_tokens=usage_raw.get("completion_tokens", 0),
        total_tokens=usage_raw.get("total_tokens", 0),
    )
    return ChatResponse(text=text, model=data.get("model", ""), usage=usage, raw=data)


def _parse_sse_frame(frame: str) -> list[StreamChunk]:
    chunks: list[StreamChunk] = []
    for line in frame.split("\n"):
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            chunks.append(StreamChunk(done=True))
            continue
        import json
        try:
            obj = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            continue
        choice = (obj.get("choices") or [{}])[0]
        delta = choice.get("delta", {})
        text = delta.get("content", "") or ""
        usage_raw = obj.get("usage")
        usage = None
        if usage_raw:
            usage = Usage(
                prompt_tokens=usage_raw.get("prompt_tokens", 0),
                completion_tokens=usage_raw.get("completion_tokens", 0),
                total_tokens=usage_raw.get("total_tokens", 0),
            )
        if text or usage:
            chunks.append(StreamChunk(text=text, usage=usage, raw=obj))
    return chunks
