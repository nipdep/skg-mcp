from __future__ import annotations

try:
    from src.embedder.base import BaseEmbedder
except Exception:  # pragma: no cover - import-path compatibility
    from embedder.base import BaseEmbedder  # type: ignore


_PROVIDER_ALIASES = {
    "lms": "lms",
    "lmstudio": "lms",
    "lm_studio": "lms",
    "openai": "openai",
    "openai_endpoint": "openai",
    "lmstudio_openai": "openai",
}


def normalize_embedder_provider(provider: str | None) -> str:
    key = str(provider or "lms").strip().lower()
    return _PROVIDER_ALIASES.get(key, key)


def create_embedder(
    model_name: str,
    *,
    provider: str = "lms",
    api_key: str | None = None,
    base_url: str | None = None,
) -> BaseEmbedder:
    normalized_provider = normalize_embedder_provider(provider)

    if normalized_provider == "lms":
        try:
            from src.embedder.lms import LMSEmbedder
        except Exception:  # pragma: no cover - import-path compatibility
            from embedder.lms import LMSEmbedder  # type: ignore

        return LMSEmbedder(model_name=model_name, api_key=api_key)

    if normalized_provider == "openai":
        try:
            from src.embedder.openai import OpenAIEmbedder
        except Exception:  # pragma: no cover - import-path compatibility
            from embedder.openai import OpenAIEmbedder  # type: ignore

        return OpenAIEmbedder(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
        )

    raise ValueError(
        f"Unsupported embedder provider: {provider!r}. "
        "Supported providers: lms, openai."
    )
