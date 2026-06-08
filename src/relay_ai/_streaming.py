"""Streaming response wrappers.

``Stream`` (sync) and ``AsyncStream`` (async) are context managers that
parse SSE frames, accumulate the final response, and emit telemetry on close.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator

import httpx

from relay_ai._types import (
    ChatResponse,
    StreamChunk,
    ToolCall,
    ToolCallDelta,
    Usage,
)

if TYPE_CHECKING:
    from relay_ai._telemetry import TelemetrySink


# ---------------------------------------------------------------------------
# Shared SSE parsing
# ---------------------------------------------------------------------------


def _parse_sse_frame(frame: str) -> list[StreamChunk]:
    chunks: list[StreamChunk] = []
    for line in frame.split("\n"):
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            chunks.append(StreamChunk(done=True))
            continue
        try:
            obj = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            continue
        chunk = _parse_chunk_data(obj)
        if chunk is not None:
            chunks.append(chunk)
    return chunks


def _parse_chunk_data(obj: dict[str, Any]) -> StreamChunk | None:
    choice = (obj.get("choices") or [{}])[0]
    delta = choice.get("delta", {})
    text = delta.get("content") or ""
    finish_reason = choice.get("finish_reason")

    tool_call_deltas: list[ToolCallDelta] = []
    for tc in delta.get("tool_calls", []):
        fn = tc.get("function", {})
        tool_call_deltas.append(
            ToolCallDelta(
                index=tc.get("index", 0),
                id=tc.get("id"),
                function_name=fn.get("name"),
                function_arguments_delta=fn.get("arguments", ""),
            )
        )

    usage: Usage | None = None
    usage_raw = obj.get("usage")
    if usage_raw:
        ptd = usage_raw.get("prompt_tokens_details") or {}
        usage = Usage(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            completion_tokens=usage_raw.get("completion_tokens", 0),
            total_tokens=usage_raw.get("total_tokens", 0),
            cached_tokens=ptd.get("cached_tokens", 0),
        )

    if not text and not tool_call_deltas and usage is None and finish_reason is None:
        return None

    return StreamChunk(
        text=text,
        finish_reason=finish_reason,
        tool_call_deltas=tool_call_deltas,
        usage=usage,
        raw=obj,
    )


# ---------------------------------------------------------------------------
# Accumulation helpers (shared between sync & async)
# ---------------------------------------------------------------------------


class _Accumulator:
    """Accumulates streamed deltas into a final ``ChatResponse``."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.text = ""
        self.finish_reason: str | None = None
        self.usage: Usage | None = None
        self.tool_calls: dict[int, dict[str, str]] = {}
        self.error: str | None = None

    def feed(self, chunk: StreamChunk) -> None:
        if chunk.text:
            self.text += chunk.text
        if chunk.finish_reason:
            self.finish_reason = chunk.finish_reason
        if chunk.usage:
            self.usage = chunk.usage
        for tcd in chunk.tool_call_deltas:
            tc = self.tool_calls.setdefault(
                tcd.index, {"id": "", "name": "", "arguments": ""}
            )
            if tcd.id:
                tc["id"] = tcd.id
            if tcd.function_name:
                tc["name"] = tcd.function_name
            tc["arguments"] += tcd.function_arguments_delta

    def build(self) -> ChatResponse:
        tcs = [
            ToolCall(
                id=v.get("id", ""),
                function_name=v.get("name", ""),
                function_arguments=v.get("arguments", ""),
            )
            for _, v in sorted(self.tool_calls.items())
        ]
        return ChatResponse(
            text=self.text,
            model=self.model,
            usage=self.usage or Usage(),
            finish_reason=self.finish_reason,
            tool_calls=tcs,
        )


# ---------------------------------------------------------------------------
# Telemetry helper (no content — metadata only)
# ---------------------------------------------------------------------------


def _emit(
    sink: TelemetrySink | None,
    model: str,
    usage: Usage | None,
    finish_reason: str | None,
    latency_ms: float | None,
    *,
    error_code: str | None = None,
) -> None:
    if sink is None:
        return
    from relay_ai._version import __version__

    event: dict[str, Any] = {
        "model_alias": model,
        "source": "sdk",
        "sdk_version": __version__,
    }
    if usage:
        event["input_tokens"] = usage.prompt_tokens
        event["output_tokens"] = usage.completion_tokens
        if usage.cached_tokens:
            event["cached_tokens"] = usage.cached_tokens
    if finish_reason:
        event["finish_reason"] = finish_reason
    if latency_ms is not None:
        event["latency_ms"] = round(latency_ms, 1)
    if error_code:
        event["error_code"] = error_code
    sink.emit(event)


# ---------------------------------------------------------------------------
# Sync stream
# ---------------------------------------------------------------------------


class Stream:
    """Synchronous streaming response.  Use as a context manager.

    Example::

        with client.chat("claude-sonnet-4.6", messages=msgs, stream=True) as stream:
            for chunk in stream:
                print(chunk.text, end="", flush=True)
            final = stream.get_final_response()
    """

    def __init__(
        self,
        response: httpx.Response,
        model: str,
        *,
        telemetry: TelemetrySink | None = None,
    ) -> None:
        self._response = response
        self._acc = _Accumulator(model)
        self._telemetry = telemetry
        self._done = False
        self._t0 = time.monotonic()

    def __enter__(self) -> Stream:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __iter__(self) -> Iterator[StreamChunk]:
        try:
            buffer = ""
            for text in self._response.iter_text():
                buffer += text
                while "\n\n" in buffer:
                    frame, buffer = buffer.split("\n\n", 1)
                    for chunk in _parse_sse_frame(frame):
                        self._acc.feed(chunk)
                        yield chunk
            if buffer.strip():
                for chunk in _parse_sse_frame(buffer):
                    self._acc.feed(chunk)
                    yield chunk
        except Exception as exc:
            self._acc.error = type(exc).__name__
            raise

    def get_final_response(self) -> ChatResponse:
        """Return the accumulated ``ChatResponse`` after iteration."""
        return self._acc.build()

    def close(self) -> None:
        if not self._done:
            self._done = True
            latency_ms = (time.monotonic() - self._t0) * 1000
            _emit(
                self._telemetry,
                self._acc.model,
                self._acc.usage,
                self._acc.finish_reason,
                latency_ms,
                error_code=self._acc.error,
            )
            self._response.close()

    def __del__(self) -> None:
        if not self._done:
            try:
                self.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Async stream
# ---------------------------------------------------------------------------


class AsyncStream:
    """Asynchronous streaming response.  Use as an async context manager.

    Example::

        async with await client.chat("claude-sonnet-4.6", messages=msgs, stream=True) as stream:
            async for chunk in stream:
                print(chunk.text, end="", flush=True)
            final = stream.get_final_response()
    """

    def __init__(
        self,
        response: httpx.Response,
        model: str,
        *,
        telemetry: TelemetrySink | None = None,
    ) -> None:
        self._response = response
        self._acc = _Accumulator(model)
        self._telemetry = telemetry
        self._done = False
        self._t0 = time.monotonic()

    async def __aenter__(self) -> AsyncStream:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def __aiter__(self) -> AsyncIterator[StreamChunk]:
        try:
            buffer = ""
            async for text in self._response.aiter_text():
                buffer += text
                while "\n\n" in buffer:
                    frame, buffer = buffer.split("\n\n", 1)
                    for chunk in _parse_sse_frame(frame):
                        self._acc.feed(chunk)
                        yield chunk
            if buffer.strip():
                for chunk in _parse_sse_frame(buffer):
                    self._acc.feed(chunk)
                    yield chunk
        except Exception as exc:
            self._acc.error = type(exc).__name__
            raise

    def get_final_response(self) -> ChatResponse:
        """Return the accumulated ``ChatResponse`` after iteration."""
        return self._acc.build()

    async def close(self) -> None:
        if not self._done:
            self._done = True
            latency_ms = (time.monotonic() - self._t0) * 1000
            _emit(
                self._telemetry,
                self._acc.model,
                self._acc.usage,
                self._acc.finish_reason,
                latency_ms,
                error_code=self._acc.error,
            )
            await self._response.aclose()
