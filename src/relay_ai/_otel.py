"""Optional OpenTelemetry instrumentation.

Install with::

    pip install ai5labs-relay[otel]

Provides ``instrument()`` to wrap a client with GenAI semantic-convention
spans, and ``RelaySpanExporter`` to forward metadata-only span data to
the Relay ``/v1/logs`` endpoint.

**Privacy**: no content attributes are ever set on spans.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

_HAS_OTEL = False
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

    _HAS_OTEL = True
except ImportError:
    pass

_F = TypeVar("_F", bound=Callable[..., Any])


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def instrument(
    client: Any,
    *,
    tracer_provider: Any | None = None,
) -> Any:
    """Add OpenTelemetry spans to every SDK call on *client*.

    Returns *client* (mutated in-place) so callers can chain::

        client = instrument(Relay())

    No-op if ``opentelemetry`` is not installed.
    """
    if not _HAS_OTEL:
        import warnings

        warnings.warn(
            "opentelemetry is not installed — pip install ai5labs-relay[otel]",
            stacklevel=2,
        )
        return client

    from relay_ai._version import __version__

    tp = tracer_provider or trace.get_tracer_provider()
    tracer = tp.get_tracer("relay_ai", __version__)

    import asyncio

    for method_name, operation in (
        ("chat", "chat"),
        ("images", "images.generate"),
        ("transcribe", "audio.transcribe"),
        ("speech", "audio.speech"),
        ("route", "route"),
    ):
        original = getattr(client, method_name, None)
        if original is None:
            continue
        if asyncio.iscoroutinefunction(original):
            setattr(client, method_name, _wrap_async(original, tracer, operation))
        else:
            setattr(client, method_name, _wrap_sync(original, tracer, operation))

    return client


# ------------------------------------------------------------------
# Span exporter
# ------------------------------------------------------------------

if _HAS_OTEL:

    class RelaySpanExporter(SpanExporter):
        """Export OTel spans as metadata-only events to ``/v1/logs``.

        **Privacy**: only ``gen_ai.*`` semantic-convention attributes and
        ``error.type`` are forwarded — no prompt or completion content.
        """

        _SAFE_ATTRS = frozenset(
            {
                "gen_ai.system",
                "gen_ai.request.model",
                "gen_ai.operation.name",
                "gen_ai.request.max_tokens",
                "gen_ai.request.temperature",
                "gen_ai.usage.input_tokens",
                "gen_ai.usage.output_tokens",
                "gen_ai.response.finish_reasons",
                "gen_ai.client.latency_ms",
                "error.type",
            }
        )

        def __init__(self, api_key: str, base_url: str) -> None:
            self._api_key = api_key
            self._url = base_url.rstrip("/") + "/logs"

        def export(self, spans: Any) -> SpanExportResult:
            import httpx as _httpx

            events: list[dict[str, Any]] = []
            for span in spans:
                attrs = {
                    k: v
                    for k, v in (span.attributes or {}).items()
                    if k in self._SAFE_ATTRS
                }
                duration_ms = (
                    (span.end_time - span.start_time) / 1_000_000
                    if span.end_time and span.start_time
                    else None
                )
                events.append(
                    {
                        "model_alias": attrs.get("gen_ai.request.model", ""),
                        "input_tokens": attrs.get("gen_ai.usage.input_tokens"),
                        "output_tokens": attrs.get("gen_ai.usage.output_tokens"),
                        "latency_ms": duration_ms,
                        "error_code": attrs.get("error.type"),
                        "source": "sdk-otel",
                    }
                )
            if not events:
                return SpanExportResult.SUCCESS
            try:
                _httpx.post(
                    self._url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={"events": events},
                    timeout=10.0,
                )
                return SpanExportResult.SUCCESS
            except Exception:
                return SpanExportResult.FAILURE

        def shutdown(self) -> None:
            pass

        def force_flush(self, timeout_millis: int = 30_000) -> bool:
            return True


# ------------------------------------------------------------------
# Internal wrapping helpers
# ------------------------------------------------------------------


def _set_request_attrs(span: Any, model: str, kwargs: dict[str, Any]) -> None:
    span.set_attribute("gen_ai.system", "relay")
    span.set_attribute("gen_ai.request.model", model)
    if (mt := kwargs.get("max_tokens")) is not None:
        span.set_attribute("gen_ai.request.max_tokens", mt)
    if (temp := kwargs.get("temperature")) is not None:
        span.set_attribute("gen_ai.request.temperature", temp)


def _set_response_attrs(span: Any, result: Any) -> None:
    if hasattr(result, "usage") and result.usage:
        span.set_attribute("gen_ai.usage.input_tokens", result.usage.prompt_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", result.usage.completion_tokens)
    if hasattr(result, "finish_reason") and result.finish_reason:
        span.set_attribute("gen_ai.response.finish_reasons", [result.finish_reason])
    if hasattr(result, "latency_ms"):
        span.set_attribute("gen_ai.client.latency_ms", result.latency_ms)


def _wrap_sync(original: _F, tracer: Any, operation: str) -> _F:
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        model = args[0] if args else kwargs.get("model", "unknown")
        with tracer.start_as_current_span(f"relay.{operation}") as span:
            span.set_attribute("gen_ai.operation.name", operation)
            _set_request_attrs(span, str(model), kwargs)
            try:
                result = original(*args, **kwargs)
                if not kwargs.get("stream"):
                    _set_response_attrs(span, result)
                return result
            except Exception as exc:
                span.record_exception(exc)
                span.set_attribute("error.type", type(exc).__name__)
                raise

    return wrapper  # type: ignore[return-value]


def _wrap_async(original: _F, tracer: Any, operation: str) -> _F:
    @functools.wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        model = args[0] if args else kwargs.get("model", "unknown")
        with tracer.start_as_current_span(f"relay.{operation}") as span:
            span.set_attribute("gen_ai.operation.name", operation)
            _set_request_attrs(span, str(model), kwargs)
            try:
                result = await original(*args, **kwargs)
                if not kwargs.get("stream"):
                    _set_response_attrs(span, result)
                return result
            except Exception as exc:
                span.record_exception(exc)
                span.set_attribute("error.type", type(exc).__name__)
                raise

    return wrapper  # type: ignore[return-value]
