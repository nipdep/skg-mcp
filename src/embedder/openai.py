from __future__ import annotations

from collections.abc import Sequence

try:
    from src.embedder.base import BaseEmbedder
except Exception:  # pragma: no cover - import-path compatibility
    from embedder.base import BaseEmbedder  # type: ignore


class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI-compatible embedding client.

    Works with OpenAI and OpenAI-compatible endpoints (LM Studio server mode,
    vLLM, NIM, etc.) that expose `/v1/embeddings`.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.base_url = base_url or "http://localhost:1234/v1"
        super().__init__(model_name=model_name, api_key=api_key)
        self._setup_client()

    def _setup_client(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "OpenAI embeddings require the `openai` package. "
                "Install it with `uv add openai`."
            ) from exc

        # For OpenAI-compatible local servers (e.g., LM Studio), any non-empty
        # key is commonly accepted. Real OpenAI usage should pass a real key.
        resolved_api_key = self.api_key or "lm-studio"
        self.client = OpenAI(base_url=self.base_url, api_key=resolved_api_key)

    def embed_texts(
        self,
        texts: str | Sequence[str],
    ) -> list[float] | list[list[float]]:
        is_single = isinstance(texts, str)
        batch = [texts] if is_single else list(texts)

        if not batch:
            return []

        response = self.client.embeddings.create(model=self.model_name, input=batch)
        ordered = sorted(response.data, key=lambda row: row.index)
        embeddings = [list(row.embedding) for row in ordered]
        return embeddings[0] if is_single else embeddings
