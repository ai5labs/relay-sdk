"""Batch processing — concurrent fan-out through the gateway."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from relay_ai._errors import APIStatusError
from relay_ai._types import BatchResult

if TYPE_CHECKING:
    from relay_ai._client import AsyncRelay, Relay


def batch_sync(
    client: Relay,
    model: str,
    requests: list[dict[str, Any]],
    *,
    max_concurrent: int = 10,
) -> list[BatchResult]:
    """Fan out *requests* through ``client.chat()`` using a thread pool."""
    results: list[BatchResult | None] = [None] * len(requests)

    def _one(idx: int, req: dict[str, Any]) -> BatchResult:
        try:
            resp = client.chat(model, **req)
            return BatchResult(index=idx, response=resp)  # type: ignore[arg-type]
        except APIStatusError as exc:
            return BatchResult(
                index=idx, error=str(exc), status_code=exc.status_code, error_code=exc.error_code,
            )
        except Exception as exc:
            return BatchResult(index=idx, error=str(exc))

    workers = min(max_concurrent, len(requests)) or 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, i, r): i for i, r in enumerate(requests)}
        for future in as_completed(futures):
            result = future.result()
            results[result.index] = result

    return [r for r in results if r is not None]


async def batch_async(
    client: AsyncRelay,
    model: str,
    requests: list[dict[str, Any]],
    *,
    max_concurrent: int = 10,
) -> list[BatchResult]:
    """Fan out *requests* through ``client.chat()`` using an asyncio semaphore."""
    sem = asyncio.Semaphore(max_concurrent)

    async def _one(idx: int, req: dict[str, Any]) -> BatchResult:
        async with sem:
            try:
                resp = await client.chat(model, **req)
                return BatchResult(index=idx, response=resp)  # type: ignore[arg-type]
            except APIStatusError as exc:
                return BatchResult(
                    index=idx, error=str(exc), status_code=exc.status_code, error_code=exc.error_code,
                )
            except Exception as exc:
                return BatchResult(index=idx, error=str(exc))

    tasks = [_one(i, r) for i, r in enumerate(requests)]
    results = await asyncio.gather(*tasks)
    return sorted(results, key=lambda r: r.index)
