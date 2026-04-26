import os

from .mock_backend import LLMMockBackend
from .server import run_stdio, run_streamable_http
from .sudo_backend import SudoKGBackend


def main() -> None:
    backend_mode = os.getenv("SKG_BACKEND", "").strip().lower()
    if backend_mode in {"llm_mock", "mock_llm"}:
        backend = LLMMockBackend.from_env()
    elif backend_mode in {"sudo_kg", "sudo"}:
        backend = SudoKGBackend.from_env()
    else:
        backend = None

    transport = os.getenv("SKG_MCP_TRANSPORT", "stdio").strip().lower()
    if transport in {"streamable-http", "http"}:
        run_streamable_http(
            backend=backend,
            host=os.getenv("SKG_MCP_HOST", "127.0.0.1"),
            port=int(os.getenv("SKG_MCP_PORT", "8000")),
            path=os.getenv("SKG_MCP_PATH", "/mcp"),
        )
    else:
        run_stdio(backend=backend)


if __name__ == "__main__":
    main()
