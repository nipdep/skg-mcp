from typing import Type, Dict, Any
import os
import json
from urllib import error as urlerror
from urllib import request as urlrequest
from pydantic import BaseModel

try:
    from src.llm.base import BaseLLMGenerator
except Exception:  # pragma: no cover - import-path compatibility
    from llm.base import BaseLLMGenerator  # type: ignore

try:  # Optional dependency: needed only for LM Studio in-process mode.
    import lmstudio as lms  # type: ignore
except Exception:  # pragma: no cover - depends on local runtime
    lms = None

try:  # Optional utility decorator.
    from src.utils.wraps import handle_context_overflow  # type: ignore
except Exception:  # pragma: no cover - depends on local runtime
    try:
        from utils.wraps import handle_context_overflow  # type: ignore
    except Exception:  # pragma: no cover - import-path compatibility
        def handle_context_overflow(func):  # type: ignore
            return func

class LMSGPTGenerator(BaseLLMGenerator):

    def _setup_client(self):
        if lms is None:
            raise ImportError(
                "LM Studio support requires the `lmstudio` package. "
                "Install it or use OpenAIEndpointLLMGenerator."
            )
        self.llm_client = lms.llm(self.model_name)

    @handle_context_overflow
    def text_generate(self, prompt: str, config: dict = {}) -> str: 
        _response = self.llm_client.respond(prompt, config=config).content
        response = _response.split("<|message|>")[-1]
        return response

    @handle_context_overflow
    def structured_text_generate(self, prompt: str, response_format: Type[BaseModel], config: dict={}, add_type: bool=False) -> BaseModel:
        if add_type:
            structure_string = repr(response_format.model_dump_json(indent=2))
            prompt_with_structure = f"{prompt}\n\nThe response should match the following structure:\n{structure_string}"
            structured_response = self.llm_client.respond(prompt_with_structure, config=config)
        else:
            structured_response = self.llm_client.respond(prompt, config=config)
        # response parsing 
        # print(f"Raw structured response content: {structured_response.content}")
        response_content = structured_response.content.split("<|message|>")[-1]
        parsed_response = self._parse_structured_output(response_content, response_format)
        return parsed_response
    

class OpenAIEndpointLLMGenerator(BaseLLMGenerator):
    """
    Generic OpenAI-compatible chat-completions generator.

    Works with servers exposing `/v1/chat/completions` such as NIM, vLLM,
    LM Studio server mode, and compatible gateways.
    """

    CONFIG_KEY_ALIASES = {
        "maxTokens": "max_tokens",
        "maxGeneratedTokens": "max_tokens",
        "minTokens": "min_tokens",
        "minGeneratedTokens": "min_tokens",
        "stopStrings": "stop",
        "topPSampling": "top_p",
        "topKSampling": "top_k",
        "repeatPenalty": "frequency_penalty",
    }

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int = 120,
        system_prompt: str | None = None,
    ):
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "http://localhost:8000")
        self.timeout = timeout
        self.system_prompt = system_prompt
        super().__init__(model_name=model_name, api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def _setup_client(self):
        normalized = self.base_url.rstrip("/")
        if normalized.endswith("/v1/chat/completions"):
            self.endpoint = normalized
        elif normalized.endswith("/v1"):
            self.endpoint = f"{normalized}/chat/completions"
        else:
            self.endpoint = f"{normalized}/v1/chat/completions"

    def _normalize_config_keys(self, config: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        for key, value in config.items():
            normalized[self.CONFIG_KEY_ALIASES.get(key, key)] = value
        return normalized

    def _build_payload(self, prompt: str, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
        cfg = self._normalize_config_keys(dict(config or {}))
        messages = cfg.pop("messages", None)
        explicit_system = False
        if "system_message" in cfg:
            system_message = cfg.pop("system_message")
            explicit_system = True
        elif "system_prompt" in cfg:
            # Accept `system_prompt` alias for explicitness in runtime configs.
            system_message = cfg.pop("system_prompt")
            explicit_system = True
        else:
            system_message = self.system_prompt

        if messages is None:
            messages = []
            if explicit_system:
                if system_message is not None and str(system_message).strip():
                    messages.append({"role": "system", "content": system_message})
            elif self.system_prompt is not None and str(self.system_prompt).strip():
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
        }
        payload.update(cfg)
        return payload

    def _post_chat_completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urlrequest.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urlrequest.urlopen(req, timeout=self.timeout) as response:
                response_body = response.read().decode("utf-8")
        except urlerror.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI-compatible API HTTP {exc.code}: {body}") from exc
        except urlerror.URLError as exc:
            raise RuntimeError(f"Could not reach endpoint '{self.endpoint}': {exc.reason}") from exc

        try:
            return json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON returned by API: {response_body}") from exc

    def text_generate(self, prompt: str, config: dict | None = None) -> str:
        payload = self._build_payload(prompt=prompt, config=config)
        response = self._post_chat_completion(payload=payload)

        choices = response.get("choices", [])
        if not choices:
            raise RuntimeError(f"Response missing 'choices': {response}")

        message = choices[0].get("message", {})
        content = message.get("content")
        if content is None and message.get("reasoning_content") is not None:
            content = message["reasoning_content"]
        if content is None:
            raise RuntimeError(f"Response missing assistant content: {response}")
        return content

    def structured_text_generate(
        self,
        prompt: str,
        response_format: Type[BaseModel],
        config: dict | None = None,
    ) -> BaseModel:
        schema = json.dumps(response_format.model_json_schema(), indent=2)
        prompt_with_schema = (
            f"{prompt}\n\n"
            "Return only valid JSON that matches this JSON schema:\n"
            f"{schema}"
        )
        raw_response = self.text_generate(prompt=prompt_with_schema, config=config)
        return self._parse_structured_output(raw_response, response_format)
