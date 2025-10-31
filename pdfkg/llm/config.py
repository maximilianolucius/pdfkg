"""LLM provider configuration helpers."""

from __future__ import annotations

import os
from typing import Optional

SUPPORTED_LLM_PROVIDERS = {"gemini", "mistral"}
_PROVIDER_API_ENV_MAP = {
    "gemini": "GEMINI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}


def resolve_llm_provider(provider: Optional[str]) -> str:
    """
    Resolve the desired LLM provider based on explicit input or environment configuration.

    Args:
        provider: Optional provider name override.

    Returns:
        Normalised provider string.

    Raises:
        ValueError: If the provider is not recognised.
    """
    candidate = (provider or os.getenv("DEFAULT_LLM_PROVIDER", "gemini")).strip().lower()
    if candidate not in SUPPORTED_LLM_PROVIDERS:
        raise ValueError(
            f"Unsupported LLM provider '{candidate}'. "
            f"Supported providers: {', '.join(sorted(SUPPORTED_LLM_PROVIDERS))}"
        )
    return candidate


def get_default_llm_provider() -> str:
    """Return the default LLM provider resolved from environment configuration."""
    return resolve_llm_provider(None)


def is_provider_configured(provider: str) -> bool:
    """Check if the required API key is present for the given provider."""
    env_var = _PROVIDER_API_ENV_MAP.get(provider.lower())
    if not env_var:
        return False
    return bool(os.getenv(env_var))


def require_provider_configuration(provider: str) -> None:
    """
    Ensure that the given provider has the required credentials configured.

    Raises:
        ValueError: If the required environment variable is missing.
    """
    env_var = _PROVIDER_API_ENV_MAP.get(provider.lower())
    if not env_var:
        raise ValueError(f"Unknown provider '{provider}' - cannot verify configuration.")
    if not os.getenv(env_var):
        raise ValueError(
            f"{env_var} not configured for provider '{provider}'. "
            "Set the environment variable in your .env file."
        )

