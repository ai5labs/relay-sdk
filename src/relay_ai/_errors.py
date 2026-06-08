"""Typed error hierarchy for the Relay AI SDK.

Mirrors the structure used by the OpenAI and Anthropic Python SDKs:
RelayError > APIConnectionError / APITimeoutError / APIStatusError > ...
"""

from __future__ import annotations

import json
from typing import Any

import httpx


class RelayError(Exception):
    """Base class for all Relay SDK errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.message!r})"


class APIConnectionError(RelayError):
    """Failed to connect to the Relay gateway."""

    def __init__(
        self,
        *,
        message: str = "Connection error.",
        request: httpx.Request | None = None,
    ) -> None:
        super().__init__(message)
        self.request = request


class APITimeoutError(APIConnectionError):
    """Request timed out."""

    def __init__(self, *, request: httpx.Request | None = None) -> None:
        super().__init__(message="Request timed out.", request=request)


class APIStatusError(RelayError):
    """HTTP response with an error status code."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        error_type: str | None = None,
        error_code: str | None = None,
        body: dict[str, Any] | None = None,
        response: httpx.Response,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.error_code = error_code
        self.body = body or {}
        self.response = response

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"message={self.message!r}, "
            f"status_code={self.status_code}, "
            f"error_code={self.error_code!r})"
        )


class AuthenticationError(APIStatusError):
    """Invalid or revoked API key (401)."""


class InsufficientCreditsError(APIStatusError):
    """Prepaid credit balance too low (402)."""


class PermissionDeniedError(APIStatusError):
    """Model not allowed for this API key (403)."""


class NotFoundError(APIStatusError):
    """Resource not found (404)."""


class ModelNotFoundError(NotFoundError):
    """Model alias does not exist in the catalogue."""


class RateLimitError(APIStatusError):
    """Rate limit or quota exceeded (429)."""

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class BadRequestError(APIStatusError):
    """Malformed or invalid request (400)."""


class ContentPolicyError(BadRequestError):
    """Content blocked by a safety filter."""


class ContextWindowError(BadRequestError):
    """Request exceeds the model's context window."""


class InternalServerError(APIStatusError):
    """Gateway or upstream provider error (5xx)."""


def _raise_for_status(response: httpx.Response) -> None:
    """Parse the router's OpenAI-compatible error envelope and raise a typed error.

    Error envelope: ``{"error": {"message": "...", "type": "...", "code": "..."}}``
    """
    if response.is_success:
        return

    body: dict[str, Any] = {}
    error_type: str | None = None
    error_code: str | None = None
    message = f"HTTP {response.status_code}"

    try:
        raw_text = response.text
    except httpx.ResponseNotRead:
        raw_text = ""

    if raw_text:
        try:
            body = json.loads(raw_text)
            err = body.get("error", {})
            if isinstance(err, dict):
                message = err.get("message") or message
                error_type = err.get("type")
                error_code = err.get("code")
        except (ValueError, TypeError):
            message = raw_text[:500]

    kwargs: dict[str, Any] = {
        "status_code": response.status_code,
        "error_type": error_type,
        "error_code": error_code,
        "body": body,
        "response": response,
    }

    status = response.status_code

    if status == 401:
        raise AuthenticationError(message, **kwargs)
    if status == 402:
        raise InsufficientCreditsError(message, **kwargs)
    if status == 403:
        raise PermissionDeniedError(message, **kwargs)
    if status == 404:
        if error_code == "model_not_found":
            raise ModelNotFoundError(message, **kwargs)
        raise NotFoundError(message, **kwargs)
    if status == 429:
        raw = response.headers.get("retry-after")
        retry_after: float | None = None
        if raw:
            try:
                retry_after = float(raw)
            except ValueError:
                pass
        raise RateLimitError(message, retry_after=retry_after, **kwargs)

    if error_code in ("context_length_exceeded", "request_too_large"):
        raise ContextWindowError(message, **kwargs)
    if error_type == "content_policy_violation" or error_code == "content_filter":
        raise ContentPolicyError(message, **kwargs)

    if status >= 500:
        raise InternalServerError(message, **kwargs)
    if status == 400:
        raise BadRequestError(message, **kwargs)

    raise APIStatusError(message, **kwargs)
