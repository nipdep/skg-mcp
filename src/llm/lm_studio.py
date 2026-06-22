from typing import Type
try:
    import lmstudio as lms  # type: ignore
except Exception:  # pragma: no cover - depends on local runtime
    lms = None
from pydantic import BaseModel
import re 
from typing import Dict, Any
try:
    from src.utils.wraps import handle_context_overflow  # type: ignore
except Exception:  # pragma: no cover - depends on local runtime
    try:
        from utils.wraps import handle_context_overflow  # type: ignore
    except Exception:
        def handle_context_overflow(func):  # type: ignore
            return func

try:
    from src.llm.base import BaseLLMGenerator
except Exception:  # pragma: no cover - import-path compatibility
    from llm.base import BaseLLMGenerator  # type: ignore

class LMStudioLLMGenerator(BaseLLMGenerator):

    def _setup_client(self):
        if lms is None:
            raise ImportError(
                "LM Studio support requires the `lmstudio` package."
            )
        self.llm_client = lms.llm(self.model_name)

    @handle_context_overflow
    def text_generate(self, prompt: str, config: dict = {}) -> str: 
        text_response = self.llm_client.respond(prompt, config=config).content
        return text_response

    @handle_context_overflow
    def structured_text_generate(self, prompt: str, response_format: Type[BaseModel], config: dict = {}) -> BaseModel:
        structured_response = self.llm_client.respond(prompt, config=config, response_format=response_format).parsed
        return structured_response
