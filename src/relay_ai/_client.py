"""Relay AI client — production SDK for the Relay gateway.

Follows the same conventions as the OpenAI and Anthropic Python SDKs:
automatic retries, typed errors, streaming context managers, custom
``httpx`` client passthrough, and background telemetry.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, BinaryIO, Union

import httpx

from relay_ai._errors import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RelayError,
    _raise_for_status,
)
from relay_ai._streaming import AsyncStream, Stream
from relay_ai._telemetry import TelemetrySink
from relay_ai._types import (
    AudioResponse,
    BatchResult,
    ChatResponse,
    CreditState,
    ImageResponse,
    RouteAlternate,
    RouteResponse,
    SpeechResponse,
    ToolCall,
    Usage,
)
from relay_ai._version import __version__

_log = logging.getLogger("relay_ai")

BASE_URL = "https://api.relay.ai5labs.com/v1"

_RETRY_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
_DEFAULT_MAX_RETRIES = 2
_INITIAL_RETRY_DELAY = 0.5
_MAX_RETRY_DELAY = 8.0

FileInput = Union[str, Path, bytes, BinaryIO]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _build_chat_body(
    model: str,
    messages: list[dict[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    body: dict[str, Any] = {"model": model, "messages": messages}
    _optional = (
        "temperature",
        "max_tokens",
        "top_p",
        "tools",
        "tool_choice",
        "response_format",
        "parallel_tool_calls",
    )
    for key in _optional:
        if key in kwargs and kwargs[key] is not None:
            body[key] = kwargs.pop(key)
        else:
            kwargs.pop(key, None)
    body.update(kwargs)
    return body


def _parse_chat(data: dict[str, Any]) -> ChatResponse:
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message", {})

    usage_raw = data.get("usage", {})
    ptd = usage_raw.get("prompt_tokens_details") or {}
    usage = Usage(
        prompt_tokens=usage_raw.get("prompt_tokens", 0),
        completion_tokens=usage_raw.get("completion_tokens", 0),
        total_tokens=usage_raw.get("total_tokens", 0),
        cached_tokens=ptd.get("cached_tokens", 0),
    )

    tool_calls = [
        ToolCall(
            id=tc.get("id", ""),
            type=tc.get("type", "function"),
            function_name=tc.get("function", {}).get("name", ""),
            function_arguments=tc.get("function", {}).get("arguments", ""),
        )
        for tc in msg.get("tool_calls", [])
    ]

    return ChatResponse(
        text=msg.get("content") or "",
        model=data.get("model", ""),
        usage=usage,
        finish_reason=choice.get("finish_reason"),
        tool_calls=tool_calls,
        raw=data,
    )


def _retry_delay(
    attempt: int,
    response: httpx.Response | None = None,
) -> float:
    if response is not None:
        header = response.headers.get("retry-after")
        if header:
            try:
                return min(float(header), _MAX_RETRY_DELAY)
            except ValueError:
                pass
    delay = min(_INITIAL_RETRY_DELAY * (2**attempt), _MAX_RETRY_DELAY)
    return delay + delay * 0.25 * random.random()  # noqa: S311


def _prepare_file(file: FileInput) -> tuple[str, Any, bool]:
    """Return ``(filename, file_obj, should_close)``."""
    if isinstance(file, (str, Path)):
        path = Path(file)
        return path.name, open(path, "rb"), True  # noqa: SIM115
    if isinstance(file, bytes):
        return "audio.wav", file, False
    return getattr(file, "name", "audio"), file, False


def _make_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": f"relay-ai-python/{__version__}",
    }


# ------------------------------------------------------------------
# Telemetry emission (metadata only — no content)
# ------------------------------------------------------------------


def _emit_telemetry(
    sink: TelemetrySink | None,
    model: str,
    result: ChatResponse | None,
    latency_ms: float,
    *,
    error_code: str | None = None,
) -> None:
    if sink is None:
        return
    event: dict[str, Any] = {
        "model_alias": model,
        "latency_ms": round(latency_ms, 1),
        "sdk_version": __version__,
        "source": "sdk",
    }
    if result is not None:
        event["input_tokens"] = result.usage.prompt_tokens
        event["output_tokens"] = result.usage.completion_tokens
        if result.usage.cached_tokens:
            event["cached_tokens"] = result.usage.cached_tokens
        if result.finish_reason:
            event["finish_reason"] = result.finish_reason
    if error_code:
        event["error_code"] = error_code
    sink.emit(event)


def _error_code_from(exc: Exception) -> str:
    if isinstance(exc, APIStatusError):
        return exc.error_code or type(exc).__name__
    return type(exc).__name__


# ===================================================================
# Synchronous client
# ===================================================================


class Relay:
    """Synchronous Relay client.  One key, every model.

    Example::

        with Relay(api_key="sk-relay-...") as client:
            resp = client.chat("claude-sonnet-4.6", messages=[
                {"role": "user", "content": "Hello!"}
            ])
            print(resp.text)
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        send_telemetry: bool = True,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("RELAY_API_KEY", "")
        self.base_url = (base_url or BASE_URL).rstrip("/")
        self.max_retries = max_retries

        self._client = http_client or httpx.Client(
            base_url=self.base_url,
            headers=_make_headers(self.api_key),
            timeout=timeout,
        )
        self._owns_client = http_client is None
        self._telemetry: TelemetrySink | None = (
            TelemetrySink(self.api_key, self.base_url)
            if send_telemetry and self.api_key
            else None
        )

    # ---- chat -------------------------------------------------------

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatResponse | Stream:
        body = _build_chat_body(
            model,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            **kwargs,
        )

        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
            resp = self._request_stream("POST", "/chat/completions", json=body)
            return Stream(resp, model, telemetry=self._telemetry)

        t0 = time.monotonic()
        try:
            resp = self._request("POST", "/chat/completions", json=body)
        except RelayError as exc:
            _emit_telemetry(self._telemetry, model, None, (time.monotonic() - t0) * 1000, error_code=_error_code_from(exc))
            raise
        latency_ms = (time.monotonic() - t0) * 1000
        result = _parse_chat(resp.json())
        result.latency_ms = latency_ms
        _emit_telemetry(self._telemetry, model, result, latency_ms)
        return result

    # ---- images -----------------------------------------------------

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
        t0 = time.monotonic()
        try:
            resp = self._request("POST", "/images/generations", json=body)
        except RelayError as exc:
            _emit_telemetry(self._telemetry, model, None, (time.monotonic() - t0) * 1000, error_code=_error_code_from(exc))
            raise
        latency_ms = (time.monotonic() - t0) * 1000
        data = resp.json()
        urls = [img.get("url", "") for img in data.get("data", [])]
        _emit_telemetry(self._telemetry, model, None, latency_ms)
        return ImageResponse(images=urls, model=model, latency_ms=latency_ms, raw=data)

    # ---- audio ------------------------------------------------------

    def transcribe(
        self,
        model: str,
        file: FileInput,
        *,
        language: str | None = None,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> AudioResponse:
        filename, file_obj, should_close = _prepare_file(file)
        form: dict[str, Any] = {"model": model}
        if language:
            form["language"] = language
        if prompt:
            form["prompt"] = prompt
        form.update(kwargs)

        t0 = time.monotonic()
        try:
            resp = self._request(
                "POST",
                "/audio/transcriptions",
                files={"file": (filename, file_obj)},
                data=form,
            )
        except RelayError as exc:
            _emit_telemetry(self._telemetry, model, None, (time.monotonic() - t0) * 1000, error_code=_error_code_from(exc))
            raise
        finally:
            if should_close:
                file_obj.close()
        latency_ms = (time.monotonic() - t0) * 1000
        data = resp.json()
        _emit_telemetry(self._telemetry, model, None, latency_ms)
        return AudioResponse(
            text=data.get("text", ""),
            model=model,
            latency_ms=latency_ms,
            raw=data,
        )

    def speech(
        self,
        model: str,
        input: str,  # noqa: A002
        *,
        voice: str = "alloy",
        response_format: str = "mp3",
        **kwargs: Any,
    ) -> SpeechResponse:
        body: dict[str, Any] = {
            "model": model,
            "input": input,
            "voice": voice,
            "response_format": response_format,
            **kwargs,
        }
        t0 = time.monotonic()
        try:
            resp = self._request("POST", "/audio/speech", json=body)
        except RelayError as exc:
            _emit_telemetry(self._telemetry, model, None, (time.monotonic() - t0) * 1000, error_code=_error_code_from(exc))
            raise
        latency_ms = (time.monotonic() - t0) * 1000
        ct = resp.headers.get("content-type", f"audio/{response_format}")
        _emit_telemetry(self._telemetry, model, None, latency_ms)
        return SpeechResponse(
            audio=resp.content,
            content_type=ct,
            model=model,
            latency_ms=latency_ms,
        )

    # ---- routing ----------------------------------------------------

    def route(
        self,
        messages: list[dict[str, Any]],
        candidates: list[str],
        *,
        constraints: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> RouteResponse:
        body: dict[str, Any] = {
            "messages": messages,
            "candidates": candidates,
        }
        if constraints:
            body["constraints"] = constraints
        body.update(kwargs)

        t0 = time.monotonic()
        try:
            resp = self._request("POST", "/route", json=body)
        except RelayError as exc:
            _emit_telemetry(self._telemetry, "route", None, (time.monotonic() - t0) * 1000, error_code=_error_code_from(exc))
            raise
        latency_ms = (time.monotonic() - t0) * 1000
        data = resp.json()
        alternates = [
            RouteAlternate(alias=a[0], confidence=a[1])
            for a in data.get("alternates", [])
        ]
        alias = data.get("alias", "")
        _emit_telemetry(self._telemetry, alias or "route", None, latency_ms)
        return RouteResponse(
            alias=alias,
            confidence=data.get("confidence", 0.0),
            reasoning=data.get("reasoning", ""),
            alternates=alternates,
            classified_intent=data.get("classified_intent", ""),
            source=data.get("source", ""),
            latency_ms=latency_ms,
            raw=data,
        )

    # ---- billing ----------------------------------------------------

    def credits(self) -> CreditState:
        resp = self._request("GET", "/billing/credits/state")
        data = resp.json()
        return CreditState(
            balance_cents=data.get("balance_cents", 0),
            min_topup_cents=data.get("min_topup_cents", 0),
            max_topup_cents=data.get("max_topup_cents", 0),
        )

    # ---- models -----------------------------------------------------

    def models(self) -> list[str]:
        resp = self._request("GET", "/models")
        return [m["id"] for m in resp.json().get("data", [])]

    # ---- batch ------------------------------------------------------

    def batch(
        self,
        model: str,
        requests: list[dict[str, Any]],
        *,
        max_concurrent: int = 10,
    ) -> list[BatchResult]:
        from relay_ai._batch import batch_sync

        return batch_sync(self, model, requests, max_concurrent=max_concurrent)

    # ---- lifecycle ---------------------------------------------------

    def close(self) -> None:
        if self._telemetry:
            self._telemetry.close()
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Relay:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Relay(base_url={self.base_url!r})"

    # ---- transport ---------------------------------------------------

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Send a request with automatic retries on transient errors."""
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.request(method, path, **kwargs)
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    delay = _retry_delay(attempt)
                    _log.debug("retry %d/%d after %.1fs (timeout)", attempt + 1, self.max_retries, delay)
                    time.sleep(delay)
                    continue
                raise APITimeoutError(request=exc.request) from exc
            except httpx.ConnectError as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    delay = _retry_delay(attempt)
                    _log.debug("retry %d/%d after %.1fs (connect error)", attempt + 1, self.max_retries, delay)
                    time.sleep(delay)
                    continue
                raise APIConnectionError(message=str(exc), request=exc.request) from exc

            if resp.status_code in _RETRY_STATUS_CODES and attempt < self.max_retries:
                delay = _retry_delay(attempt, resp)
                _log.debug("retry %d/%d after %.1fs (HTTP %d)", attempt + 1, self.max_retries, delay, resp.status_code)
                time.sleep(delay)
                continue

            _raise_for_status(resp)
            return resp

        if isinstance(last_exc, httpx.TimeoutException):
            raise APITimeoutError(request=last_exc.request) from last_exc
        raise APIConnectionError(
            message=str(last_exc) if last_exc else "Connection error."
        )

    def _request_stream(
        self, method: str, path: str, **kwargs: Any
    ) -> httpx.Response:
        """Send a streaming request.  Retries before the first byte."""
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                request = self._client.build_request(method, path, **kwargs)
                resp = self._client.send(request, stream=True)
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    delay = _retry_delay(attempt)
                    _log.debug("retry %d/%d after %.1fs (timeout)", attempt + 1, self.max_retries, delay)
                    time.sleep(delay)
                    continue
                raise APITimeoutError(request=exc.request) from exc
            except httpx.ConnectError as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    delay = _retry_delay(attempt)
                    _log.debug("retry %d/%d after %.1fs (connect error)", attempt + 1, self.max_retries, delay)
                    time.sleep(delay)
                    continue
                raise APIConnectionError(message=str(exc), request=exc.request) from exc

            if resp.status_code in _RETRY_STATUS_CODES and attempt < self.max_retries:
                delay = _retry_delay(attempt, resp)
                _log.debug("retry %d/%d after %.1fs (HTTP %d)", attempt + 1, self.max_retries, delay, resp.status_code)
                resp.close()
                time.sleep(delay)
                continue

            if not resp.is_success:
                try:
                    resp.read()
                except Exception:
                    pass
                try:
                    _raise_for_status(resp)
                finally:
                    resp.close()

            return resp

        if isinstance(last_exc, httpx.TimeoutException):
            raise APITimeoutError(request=last_exc.request) from last_exc
        raise APIConnectionError(
            message=str(last_exc) if last_exc else "Connection error."
        )


# ===================================================================
# Asynchronous client
# ===================================================================


class AsyncRelay:
    """Asynchronous Relay client.  One key, every model.

    Example::

        async with AsyncRelay(api_key="sk-relay-...") as client:
            resp = await client.chat("claude-sonnet-4.6", messages=[
                {"role": "user", "content": "Hello!"}
            ])
            print(resp.text)
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        send_telemetry: bool = True,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("RELAY_API_KEY", "")
        self.base_url = (base_url or BASE_URL).rstrip("/")
        self.max_retries = max_retries

        self._client = http_client or httpx.AsyncClient(
            base_url=self.base_url,
            headers=_make_headers(self.api_key),
            timeout=timeout,
        )
        self._owns_client = http_client is None
        self._telemetry: TelemetrySink | None = (
            TelemetrySink(self.api_key, self.base_url)
            if send_telemetry and self.api_key
            else None
        )

    # ---- chat -------------------------------------------------------

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatResponse | AsyncStream:
        body = _build_chat_body(
            model,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            **kwargs,
        )

        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
            resp = await self._request_stream("POST", "/chat/completions", json=body)
            return AsyncStream(resp, model, telemetry=self._telemetry)

        t0 = time.monotonic()
        try:
            resp = await self._request("POST", "/chat/completions", json=body)
        except RelayError as exc:
            _emit_telemetry(self._telemetry, model, None, (time.monotonic() - t0) * 1000, error_code=_error_code_from(exc))
            raise
        latency_ms = (time.monotonic() - t0) * 1000
        result = _parse_chat(resp.json())
        result.latency_ms = latency_ms
        _emit_telemetry(self._telemetry, model, result, latency_ms)
        return result

    # ---- images -----------------------------------------------------

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
        t0 = time.monotonic()
        try:
            resp = await self._request("POST", "/images/generations", json=body)
        except RelayError as exc:
            _emit_telemetry(self._telemetry, model, None, (time.monotonic() - t0) * 1000, error_code=_error_code_from(exc))
            raise
        latency_ms = (time.monotonic() - t0) * 1000
        data = resp.json()
        urls = [img.get("url", "") for img in data.get("data", [])]
        _emit_telemetry(self._telemetry, model, None, latency_ms)
        return ImageResponse(images=urls, model=model, latency_ms=latency_ms, raw=data)

    # ---- audio ------------------------------------------------------

    async def transcribe(
        self,
        model: str,
        file: FileInput,
        *,
        language: str | None = None,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> AudioResponse:
        filename, file_obj, should_close = _prepare_file(file)
        form: dict[str, Any] = {"model": model}
        if language:
            form["language"] = language
        if prompt:
            form["prompt"] = prompt
        form.update(kwargs)

        t0 = time.monotonic()
        try:
            resp = await self._request(
                "POST",
                "/audio/transcriptions",
                files={"file": (filename, file_obj)},
                data=form,
            )
        except RelayError as exc:
            _emit_telemetry(self._telemetry, model, None, (time.monotonic() - t0) * 1000, error_code=_error_code_from(exc))
            raise
        finally:
            if should_close:
                file_obj.close()
        latency_ms = (time.monotonic() - t0) * 1000
        data = resp.json()
        _emit_telemetry(self._telemetry, model, None, latency_ms)
        return AudioResponse(
            text=data.get("text", ""),
            model=model,
            latency_ms=latency_ms,
            raw=data,
        )

    async def speech(
        self,
        model: str,
        input: str,  # noqa: A002
        *,
        voice: str = "alloy",
        response_format: str = "mp3",
        **kwargs: Any,
    ) -> SpeechResponse:
        body: dict[str, Any] = {
            "model": model,
            "input": input,
            "voice": voice,
            "response_format": response_format,
            **kwargs,
        }
        t0 = time.monotonic()
        try:
            resp = await self._request("POST", "/audio/speech", json=body)
        except RelayError as exc:
            _emit_telemetry(self._telemetry, model, None, (time.monotonic() - t0) * 1000, error_code=_error_code_from(exc))
            raise
        latency_ms = (time.monotonic() - t0) * 1000
        ct = resp.headers.get("content-type", f"audio/{response_format}")
        _emit_telemetry(self._telemetry, model, None, latency_ms)
        return SpeechResponse(
            audio=resp.content,
            content_type=ct,
            model=model,
            latency_ms=latency_ms,
        )

    # ---- routing ----------------------------------------------------

    async def route(
        self,
        messages: list[dict[str, Any]],
        candidates: list[str],
        *,
        constraints: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> RouteResponse:
        body: dict[str, Any] = {
            "messages": messages,
            "candidates": candidates,
        }
        if constraints:
            body["constraints"] = constraints
        body.update(kwargs)

        t0 = time.monotonic()
        try:
            resp = await self._request("POST", "/route", json=body)
        except RelayError as exc:
            _emit_telemetry(self._telemetry, "route", None, (time.monotonic() - t0) * 1000, error_code=_error_code_from(exc))
            raise
        latency_ms = (time.monotonic() - t0) * 1000
        data = resp.json()
        alternates = [
            RouteAlternate(alias=a[0], confidence=a[1])
            for a in data.get("alternates", [])
        ]
        alias = data.get("alias", "")
        _emit_telemetry(self._telemetry, alias or "route", None, latency_ms)
        return RouteResponse(
            alias=alias,
            confidence=data.get("confidence", 0.0),
            reasoning=data.get("reasoning", ""),
            alternates=alternates,
            classified_intent=data.get("classified_intent", ""),
            source=data.get("source", ""),
            latency_ms=latency_ms,
            raw=data,
        )

    # ---- billing ----------------------------------------------------

    async def credits(self) -> CreditState:
        resp = await self._request("GET", "/billing/credits/state")
        data = resp.json()
        return CreditState(
            balance_cents=data.get("balance_cents", 0),
            min_topup_cents=data.get("min_topup_cents", 0),
            max_topup_cents=data.get("max_topup_cents", 0),
        )

    # ---- models -----------------------------------------------------

    async def models(self) -> list[str]:
        resp = await self._request("GET", "/models")
        return [m["id"] for m in resp.json().get("data", [])]

    # ---- batch ------------------------------------------------------

    async def batch(
        self,
        model: str,
        requests: list[dict[str, Any]],
        *,
        max_concurrent: int = 10,
    ) -> list[BatchResult]:
        from relay_ai._batch import batch_async

        return await batch_async(self, model, requests, max_concurrent=max_concurrent)

    # ---- lifecycle ---------------------------------------------------

    async def close(self) -> None:
        if self._telemetry:
            self._telemetry.close()
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> AsyncRelay:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    def __repr__(self) -> str:
        return f"AsyncRelay(base_url={self.base_url!r})"

    # ---- transport ---------------------------------------------------

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._client.request(method, path, **kwargs)
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    delay = _retry_delay(attempt)
                    _log.debug("retry %d/%d after %.1fs (timeout)", attempt + 1, self.max_retries, delay)
                    await asyncio.sleep(delay)
                    continue
                raise APITimeoutError(request=exc.request) from exc
            except httpx.ConnectError as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    delay = _retry_delay(attempt)
                    _log.debug("retry %d/%d after %.1fs (connect error)", attempt + 1, self.max_retries, delay)
                    await asyncio.sleep(delay)
                    continue
                raise APIConnectionError(message=str(exc), request=exc.request) from exc

            if resp.status_code in _RETRY_STATUS_CODES and attempt < self.max_retries:
                delay = _retry_delay(attempt, resp)
                _log.debug("retry %d/%d after %.1fs (HTTP %d)", attempt + 1, self.max_retries, delay, resp.status_code)
                await asyncio.sleep(delay)
                continue

            _raise_for_status(resp)
            return resp

        if isinstance(last_exc, httpx.TimeoutException):
            raise APITimeoutError(request=last_exc.request) from last_exc
        raise APIConnectionError(
            message=str(last_exc) if last_exc else "Connection error."
        )

    async def _request_stream(
        self, method: str, path: str, **kwargs: Any
    ) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                request = self._client.build_request(method, path, **kwargs)
                resp = await self._client.send(request, stream=True)
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    delay = _retry_delay(attempt)
                    _log.debug("retry %d/%d after %.1fs (timeout)", attempt + 1, self.max_retries, delay)
                    await asyncio.sleep(delay)
                    continue
                raise APITimeoutError(request=exc.request) from exc
            except httpx.ConnectError as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    delay = _retry_delay(attempt)
                    _log.debug("retry %d/%d after %.1fs (connect error)", attempt + 1, self.max_retries, delay)
                    await asyncio.sleep(delay)
                    continue
                raise APIConnectionError(message=str(exc), request=exc.request) from exc

            if resp.status_code in _RETRY_STATUS_CODES and attempt < self.max_retries:
                delay = _retry_delay(attempt, resp)
                _log.debug("retry %d/%d after %.1fs (HTTP %d)", attempt + 1, self.max_retries, delay, resp.status_code)
                await resp.aclose()
                await asyncio.sleep(delay)
                continue

            if not resp.is_success:
                try:
                    await resp.aread()
                except Exception:
                    pass
                try:
                    _raise_for_status(resp)
                finally:
                    await resp.aclose()

            return resp

        if isinstance(last_exc, httpx.TimeoutException):
            raise APITimeoutError(request=last_exc.request) from last_exc
        raise APIConnectionError(
            message=str(last_exc) if last_exc else "Connection error."
        )
