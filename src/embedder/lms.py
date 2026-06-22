from collections.abc import Sequence

try:
    import lmstudio as lms  # type: ignore
except Exception:  # pragma: no cover - depends on local runtime
    lms = None

try:
    from src.embedder.base import BaseEmbedder
except Exception:  # pragma: no cover - import-path compatibility
    from embedder.base import BaseEmbedder  # type: ignore

class LMSEmbedder(BaseEmbedder):

    def __init__(self, model_name: str, api_key: str | None = None):
        super().__init__(model_name, api_key)
        if lms is None:
            raise ImportError(
                "LM Studio embeddings require the `lmstudio` package."
            )
        self.client = lms.embedding_model(self.model_name)

    def embed_texts(
        self,
        texts: str | Sequence[str],
    ) -> list[float] | list[list[float]]:
        embeddings = self.client.embed(texts)
        return embeddings
