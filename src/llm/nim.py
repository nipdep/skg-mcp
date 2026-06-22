import os

try:
    from src.llm.lms_gpt import OpenAIEndpointLLMGenerator
except Exception:  # pragma: no cover - import-path compatibility
    from llm.lms_gpt import OpenAIEndpointLLMGenerator  # type: ignore


class NIMLLMGenerator(OpenAIEndpointLLMGenerator):
    """NVIDIA NIM client via OpenAI-compatible `/v1/chat/completions`."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int = 120,
        system_prompt: str | None = None,
    ):
        super().__init__(
            model_name=model_name,
            base_url=base_url or os.getenv("NIM_BASE_URL", "http://localhost:8000"),
            api_key=api_key or os.getenv("NIM_API_KEY"),
            timeout=timeout,
            system_prompt=system_prompt,
        )
