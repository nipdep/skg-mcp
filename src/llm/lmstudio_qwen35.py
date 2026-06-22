import os
from copy import deepcopy
from typing import Any, Dict

try:
    from src.llm.lms_gpt import OpenAIEndpointLLMGenerator
except Exception:  # pragma: no cover - import-path compatibility
    from llm.lms_gpt import OpenAIEndpointLLMGenerator  # type: ignore


def _deep_merge_dict(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge_dict(base[key], value)
        else:
            base[key] = value
    return base


class LMStudioQwen35LLMGenerator(OpenAIEndpointLLMGenerator):
    """
    LM Studio chat-completions client tuned for Qwen3.5-family models.

    Defaults keep "thinking" disabled while allowing per-call config overrides.
    """

    DEFAULT_MODEL_NAME = "qwen3.5"
    DEFAULT_GENERATION_CONFIG: Dict[str, Any] = {
        "extra_body": {
            "top_k": 20,
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int = 120,
        system_prompt: str | None = None,
        default_generation_config: Dict[str, Any] | None = None,
    ):
        self.default_generation_config = deepcopy(self.DEFAULT_GENERATION_CONFIG)
        if default_generation_config:
            _deep_merge_dict(self.default_generation_config, deepcopy(default_generation_config))

        super().__init__(
            model_name=model_name or os.getenv("LMSTUDIO_QWEN_MODEL", self.DEFAULT_MODEL_NAME),
            base_url=base_url
            or os.getenv("LMSTUDIO_BASE_URL")
            or os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1"),
            api_key=api_key or os.getenv("LMSTUDIO_API_KEY") or os.getenv("OPENAI_API_KEY"),
            timeout=timeout,
            system_prompt=system_prompt,
        )

    def _merged_generation_config(self, config: Dict[str, Any] | None) -> Dict[str, Any]:
        merged = deepcopy(self.default_generation_config)
        if config:
            _deep_merge_dict(merged, deepcopy(config))
        return merged

    def set_system_prompt(self, system_prompt: str | None) -> None:
        self.system_prompt = system_prompt

    def _build_payload(self, prompt: str, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
        payload = super()._build_payload(prompt, config=self._merged_generation_config(config))
        # Mimic OpenAI client's `extra_body` behavior by flattening it into payload.
        extra_body = payload.pop("extra_body", None)
        if isinstance(extra_body, dict):
            payload.update(extra_body)
        return payload
