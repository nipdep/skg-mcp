from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence


_MODEL_REGISTRY_LOCK = threading.Lock()
_MODEL_REGISTRY: Dict[str, "_LlamaHandle"] = {}


@dataclass
class _LlamaHandle:
    client: Any
    lock: threading.RLock


def _freeze_config(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _model_registry_key(model_path: str, llama_config: Dict[str, Any]) -> str:
    expanded_path = os.path.abspath(os.path.expanduser(model_path))
    return f"{expanded_path}::{_freeze_config(llama_config)}"


def _normalize_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for row in messages:
        role = str(row.get("role") or "user").strip() or "user"
        content = row.get("content")
        if isinstance(content, list):
            pieces = []
            for item in content:
                if isinstance(item, dict):
                    pieces.append(str(item.get("text") or item.get("content") or ""))
                else:
                    pieces.append(str(item))
            content_text = "".join(pieces)
        else:
            content_text = str(content or "")
        normalized.append({"role": role, "content": content_text})
    return normalized


def _allowed_chat_kwargs(generation_config: Dict[str, Any], model_name: str | None) -> Dict[str, Any]:
    config = dict(generation_config or {})
    extra_body = config.pop("extra_body", None)
    if isinstance(extra_body, dict):
        config.update(extra_body)

    if "max_completion_tokens" in config and "max_tokens" not in config:
        config["max_tokens"] = config.pop("max_completion_tokens")

    allowed = {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "typical_p",
        "stream",
        "stop",
        "seed",
        "response_format",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
        "repeat_penalty",
        "tfs_z",
        "mirostat_mode",
        "mirostat_tau",
        "mirostat_eta",
        "grammar",
        "logit_bias",
        "logprobs",
        "top_logprobs",
    }
    out = {key: value for key, value in config.items() if key in allowed}
    if model_name:
        out.setdefault("model", model_name)
    return out


def _resolve_model_path(llm_config: Dict[str, Any]) -> str:
    model_path = str(llm_config.get("model_path") or "").strip()
    if not model_path:
        model_name = str(llm_config.get("model_name") or "").strip()
        if model_name.lower().endswith(".gguf"):
            model_path = model_name
    if not model_path:
        raise ValueError("`model_path` is required for provider=llama_cpp_direct")
    return os.path.abspath(os.path.expanduser(model_path))


def _resolve_llama_config(llm_config: Dict[str, Any]) -> Dict[str, Any]:
    llama_config = dict(llm_config.get("llama_cpp", {}) or {})
    llama_config.setdefault("n_ctx", 8192)
    llama_config.setdefault("n_batch", 1024)
    llama_config.setdefault("n_ubatch", llama_config.get("n_batch", 1024))
    llama_config.setdefault("n_gpu_layers", -1)
    llama_config.setdefault("verbose", False)
    return llama_config


def _get_handle(llm_config: Dict[str, Any]) -> _LlamaHandle:
    try:
        import llama_cpp  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local runtime
        raise RuntimeError(
            "llama_cpp is required for provider=llama_cpp_direct. "
            "Install `llama-cpp-python` in the active environment."
        ) from exc

    model_path = _resolve_model_path(llm_config)
    llama_config = _resolve_llama_config(llm_config)
    key = _model_registry_key(model_path, llama_config)

    with _MODEL_REGISTRY_LOCK:
        existing = _MODEL_REGISTRY.get(key)
        if existing is not None:
            return existing

        client = llama_cpp.Llama(model_path=model_path, **llama_config)
        handle = _LlamaHandle(client=client, lock=threading.RLock())
        _MODEL_REGISTRY[key] = handle
        return handle


def chat_completion_from_messages(
    llm_config: Dict[str, Any],
    messages: Sequence[Dict[str, Any]],
) -> str:
    handle = _get_handle(llm_config)
    model_name = str(llm_config.get("model_name") or "").strip() or None
    generation_config = dict(llm_config.get("generation_config", {}) or {})
    normalized_messages = _normalize_messages(messages)
    kwargs = _allowed_chat_kwargs(generation_config, model_name=model_name)

    with handle.lock:
        response = handle.client.create_chat_completion(
            messages=normalized_messages,
            **kwargs,
        )

    choices = response.get("choices", []) if isinstance(response, dict) else []
    if not choices:
        raise RuntimeError(f"llama.cpp response missing choices: {response!r}")

    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = message.get("content")
    if isinstance(content, list):
        return "".join(str(item.get("text") or item.get("content") or "") for item in content if isinstance(item, dict))
    if content is None:
        raise RuntimeError(f"llama.cpp response missing assistant content: {response!r}")
    return str(content)


def call_llama_cpp_chat_completion(
    llm_config: Dict[str, Any],
    system_prompt: str,
    user_prompt: str,
) -> str:
    messages: List[Dict[str, str]] = []
    if str(system_prompt or "").strip():
        messages.append({"role": "system", "content": str(system_prompt)})
    messages.append({"role": "user", "content": str(user_prompt)})
    return chat_completion_from_messages(llm_config, messages)
