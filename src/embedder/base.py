import abc
from collections.abc import Sequence


class BaseEmbedder(abc.ABC):
    def __init__(self, model_name: str, api_key: str | None = None):
        self.model_name = model_name
        self.api_key = api_key

    @abc.abstractmethod
    def embed_texts(
        self,
        texts: str | Sequence[str],
    ) -> list[float] | list[list[float]]:
        """
        Embed one text or a batch of texts.

        Returns:
        - list[float] for single-string input
        - list[list[float]] for batch input
        """
        raise NotImplementedError("Subclasses must implement this method")
