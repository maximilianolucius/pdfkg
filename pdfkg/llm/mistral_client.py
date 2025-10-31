"""Shared helpers for interacting with the Mistral chat API."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, Iterable

from dotenv import load_dotenv

load_dotenv()

try:
    from mistralai import Mistral
except ImportError as exc:  # pragma: no cover - handled at runtime when provider selected
    Mistral = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def get_model_name() -> str:
    """Return the configured Mistral model name."""
    return os.getenv("MISTRAL_MODEL", "mistral-medium-2508")


@lru_cache(maxsize=1)
def _get_client() -> "Mistral":
    if Mistral is None:
        raise ImportError(
            "mistralai is not installed. Install with 'pip install mistralai' to use the Mistral provider'."
        ) from _IMPORT_ERROR

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not configured in environment variables.")
    return Mistral(api_key=api_key)


def chat(
    messages: Iterable[Dict[str, Any]],
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    **kwargs: Any,
):
    """
    Execute a chat completion request against the configured Mistral model.

    Args:
        messages: Conversation messages to send.
        model: Optional model override. Defaults to env-configured model.
        max_tokens: Optional max tokens override.
        **kwargs: Additional parameters forwarded to the API.

    Returns:
        Mistral API response object.
    """
    client = _get_client()
    request_kwargs: Dict[str, Any] = dict(kwargs)
    if max_tokens is not None:
        request_kwargs["max_tokens"] = max_tokens

    return client.chat.complete(
        model=model or get_model_name(),
        messages=list(messages),
        **request_kwargs,
    )

