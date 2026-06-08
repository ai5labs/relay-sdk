"""Background telemetry — metadata only, zero content retention.

Sends SDK usage events to the Relay ``/v1/logs`` endpoint.  Only whitelisted
metadata fields are ever transmitted — message content, tool arguments,
response text, and image data are **never** included.

Privacy is enforced client-side via ``_filter_event`` (defence-in-depth)
even though the router also strips content server-side.
"""

from __future__ import annotations

import atexit
import logging
import threading
from collections import deque
from typing import Any

import httpx

_log = logging.getLogger("relay_ai")

_METADATA_WHITELIST: frozenset[str] = frozenset(
    {
        "request_id",
        "ts",
        "model_alias",
        "provider",
        "input_tokens",
        "output_tokens",
        "cached_tokens",
        "cost_usd",
        "latency_ms",
        "error_code",
        "finish_reason",
        "source",
        "tags",
        "sdk_version",
    }
)


def _filter_event(raw: dict[str, Any]) -> dict[str, Any]:
    """Return **only** whitelisted metadata fields.  Everything else is dropped."""
    return {k: v for k, v in raw.items() if k in _METADATA_WHITELIST and v is not None}


class TelemetrySink:
    """Batches and sends metadata-only events to ``/v1/logs`` on a daemon thread.

    Thread-safe — ``emit()`` can be called from any thread (or from async code).
    Best-effort — network failures are silently dropped.
    """

    FLUSH_INTERVAL = 5.0
    FLUSH_BATCH_SIZE = 50
    BUFFER_MAX = 500

    def __init__(self, api_key: str, base_url: str) -> None:
        self._api_key = api_key
        self._url = base_url.rstrip("/") + "/logs"
        self._buffer: deque[dict[str, Any]] = deque(maxlen=self.BUFFER_MAX)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._closed = False
        self._thread: threading.Thread | None = None
        self._http = httpx.Client(
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=10.0,
        )

    def emit(self, event: dict[str, Any]) -> None:
        """Record a metadata-only event.  Non-blocking."""
        with self._lock:
            if self._closed:
                return
            safe = _filter_event(event)
            if not safe:
                return
            self._buffer.append(safe)
            if len(self._buffer) >= self.FLUSH_BATCH_SIZE:
                self._wake.set()
            self._ensure_started()

    def close(self) -> None:
        """Best-effort final flush, then stop the background thread."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        try:
            self._http.close()
        except Exception:
            pass
        atexit.unregister(self.close)

    # ------------------------------------------------------------------

    def _ensure_started(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="relay-telemetry",
        )
        self._thread.start()
        atexit.register(self.close)

    def _run(self) -> None:
        while not self._closed:
            self._wake.wait(timeout=self.FLUSH_INTERVAL)
            self._wake.clear()
            self._flush()
        self._flush()

    def _flush(self) -> None:
        with self._lock:
            if not self._buffer:
                return
            batch = list(self._buffer)
            self._buffer.clear()
        try:
            self._http.post(self._url, json={"events": batch})
        except Exception:
            _log.debug("telemetry flush failed", exc_info=True)
