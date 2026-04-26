from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, AsyncIterator, Mapping, TypeVar

from mcp import ClientSession, types
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from pydantic import AnyUrl, BaseModel

from .models import (
    ExpandContextArgs,
    ExpandContextResult,
    ExpandNeighborsArgs,
    ExpandNeighborsResult,
    FilterPapersArgs,
    FilterPapersResult,
    GetAttibutionArgs,
    GetAttibutionResult,
    GetProvenanceArgs,
    GetProvenanceResult,
    LexicalSearchArgs,
    ResolveConceptReferenceArgs,
    ResolveConceptReferenceResult,
    SearchResult,
    SemanticConstraintSearchArgs,
    SemanticSearchArgs,
)


TResult = TypeVar("TResult", bound=BaseModel)


class SKGMCPClientError(RuntimeError):
    """Raised when MCP server calls fail or return unexpected payloads."""


class SKGMCPClient:
    """Async MCP client wrapper for the SKG MCP server."""

    _TOOL_ALIASES: dict[str, tuple[str, ...]] = {
        "filter_papers": ("filter_papers",),
        "lexical_search": ("lexical_search",),
        "semantic_search": ("semantic_search",), 
        "semantic_constraint_search": ("semantic_constraint_search", ),
        "resolve_concept_reference": ("resolve_concept_reference",),
        "expand_context": ("expand_context",),
        "expand_neighbors": ("expand_neighbors",),
        "get_attibution": ("get_attibution",),
        "get_provenance": ("get_provenance",),
    }

    def __init__(self, session: ClientSession) -> None:
        self._session = session
        self._tool_name_index: dict[str, str] = {}

    @classmethod
    @asynccontextmanager
    async def connect_streamable_http(
        cls,
        url: str,
        *,
        read_timeout_seconds: float | None = 30.0,
    ) -> AsyncIterator["SKGMCPClient"]:
        read_timeout = (
            timedelta(seconds=read_timeout_seconds) if read_timeout_seconds is not None else None
        )
        async with streamable_http_client(url) as (read_stream, write_stream, _):
            async with ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=read_timeout,
            ) as session:
                client = cls(session=session)
                await client.initialize()
                yield client

    @classmethod
    @asynccontextmanager
    async def connect_stdio(
        cls,
        *,
        command: str = "uv",
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        read_timeout_seconds: float | None = 30.0,
    ) -> AsyncIterator["SKGMCPClient"]:
        resolved_args = list(args or ["run", "src/server.py"])
        resolved_cwd = cwd
        if resolved_cwd is None and args is None:
            resolved_cwd = cls._detect_project_root()

        params = StdioServerParameters(
            command=command,
            args=resolved_args,
            env=env,
            cwd=resolved_cwd,
        )
        read_timeout = (
            timedelta(seconds=read_timeout_seconds) if read_timeout_seconds is not None else None
        )
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=read_timeout,
            ) as session:
                client = cls(session=session)
                await client.initialize()
                yield client

    @staticmethod
    def _detect_project_root() -> str | None:
        """
        Best-effort project root discovery for source checkouts.

        Expected layout: <root>/src/skg_mcp/client.py and <root>/src/server.py
        """
        current_file = Path(__file__).resolve()
        candidate_root = current_file.parents[2]
        if (
            (candidate_root / "pyproject.toml").exists()
            and (candidate_root / "src" / "server.py").exists()
        ):
            return str(candidate_root)
        return None

    async def initialize(self) -> types.InitializeResult:
        result = await self._session.initialize()
        await self.refresh_tool_index()
        return result

    async def list_tools(self) -> list[types.Tool]:
        cursor: str | None = None
        tools: list[types.Tool] = []
        while True:
            result = await self._session.list_tools(cursor=cursor)
            tools.extend(result.tools)
            cursor = result.nextCursor
            if cursor is None:
                break
        return tools

    async def list_resources(self) -> list[types.Resource]:
        cursor: str | None = None
        resources: list[types.Resource] = []
        while True:
            result = await self._session.list_resources(cursor=cursor)
            resources.extend(result.resources)
            cursor = result.nextCursor
            if cursor is None:
                break
        return resources

    async def refresh_tool_index(self) -> dict[str, str]:
        tools = await self.list_tools()
        available = {tool.name for tool in tools}
        resolved: dict[str, str] = {}
        for canonical, aliases in self._TOOL_ALIASES.items():
            resolved[canonical] = next(
                (tool_name for tool_name in aliases if tool_name in available),
                aliases[0],
            )
        self._tool_name_index = resolved
        return dict(resolved)

    async def read_resource(self, uri: str) -> types.ReadResourceResult:
        return await self._session.read_resource(AnyUrl(uri))

    async def read_resource_json(self, uri: str) -> dict[str, Any]:
        result = await self.read_resource(uri)
        for content in result.contents:
            text = getattr(content, "text", None)
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        raise SKGMCPClientError(f"Resource '{uri}' did not return JSON object content.")

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> types.CallToolResult:
        result = await self._session.call_tool(name=name, arguments=arguments or {})
        if result.isError:
            raise SKGMCPClientError(self._tool_error_message(result, name))
        return result

    async def call_tool_structured(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = await self.call_tool(name=name, arguments=arguments)
        if isinstance(result.structuredContent, dict):
            return result.structuredContent

        parsed_content = self._parse_json_content(result.content)
        if isinstance(parsed_content, dict):
            return parsed_content

        raise SKGMCPClientError(
            f"Tool '{name}' returned no structured JSON object. "
            "Enable structured_output=True and return dict payloads."
        )

    async def server_metadata(self) -> dict[str, Any]:
        return await self.read_resource_json("skg://metadata/server")

    async def tool_catalog(self) -> dict[str, Any]:
        return await self.read_resource_json("skg://catalog/tools")

    async def filter_papers(
        self,
        args: FilterPapersArgs | Mapping[str, Any],
    ) -> FilterPapersResult:
        return await self._call_typed_tool("filter_papers", args, FilterPapersResult)

    async def lexical_search(
        self,
        args: LexicalSearchArgs | Mapping[str, Any],
    ) -> SearchResult:
        return await self._call_typed_tool("lexical_search", args, SearchResult)

    async def semantic_search(
        self,
        args: SemanticSearchArgs | Mapping[str, Any],
    ) -> SearchResult:
        return await self._call_typed_tool("semantic_search", args, SearchResult)

    async def semantic_constraint_search(
        self,
        args: SemanticConstraintSearchArgs | Mapping[str, Any],
    ) -> SearchResult:
        return await self._call_typed_tool("semantic_constraint_search", args, SearchResult)

    async def resolve_concept_reference(
        self,
        args: ResolveConceptReferenceArgs | Mapping[str, Any],
    ) -> ResolveConceptReferenceResult:
        return await self._call_typed_tool(
            "resolve_concept_reference",
            args,
            ResolveConceptReferenceResult,
        )

    async def expand_context(
        self,
        args: ExpandContextArgs | Mapping[str, Any],
    ) -> ExpandContextResult:
        return await self._call_typed_tool("expand_context", args, ExpandContextResult)

    async def expand_neighbors(
        self,
        args: ExpandNeighborsArgs | Mapping[str, Any],
    ) -> ExpandNeighborsResult:
        return await self._call_typed_tool("expand_neighbors", args, ExpandNeighborsResult)

    async def get_attibution(
        self,
        args: GetAttibutionArgs | Mapping[str, Any],
    ) -> GetAttibutionResult:
        return await self._call_typed_tool("get_attibution", args, GetAttibutionResult)

    async def get_provenance(
        self,
        args: GetProvenanceArgs | Mapping[str, Any],
    ) -> GetProvenanceResult:
        return await self._call_typed_tool("get_provenance", args, GetProvenanceResult)

    async def _call_typed_tool(
        self,
        canonical_name: str,
        args: BaseModel | Mapping[str, Any],
        result_model: type[TResult],
    ) -> TResult:
        tool_name = await self._resolve_tool_name(canonical_name)
        payload = {"args": self._to_dict(args)}
        structured = await self.call_tool_structured(tool_name, payload)
        return result_model.model_validate(structured)

    async def _resolve_tool_name(self, canonical_name: str) -> str:
        if canonical_name in self._tool_name_index:
            return self._tool_name_index[canonical_name]
        await self.refresh_tool_index()
        return self._tool_name_index.get(canonical_name, canonical_name)

    @staticmethod
    def _to_dict(data: BaseModel | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(data, BaseModel):
            return data.model_dump(exclude_none=True)
        return dict(data)

    @staticmethod
    def _parse_json_content(content: list[types.ContentBlock]) -> Any | None:
        for block in content:
            text = getattr(block, "text", None)
            if not text:
                continue
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                continue
        return None

    @staticmethod
    def _tool_error_message(result: types.CallToolResult, tool_name: str) -> str:
        parsed = SKGMCPClient._parse_json_content(result.content)
        if isinstance(parsed, dict) and "error" in parsed:
            return f"Tool '{tool_name}' failed: {parsed['error']}"

        text_parts = [getattr(block, "text", "") for block in result.content]
        text = "\n".join(part for part in text_parts if part).strip()
        if text:
            return f"Tool '{tool_name}' failed: {text}"
        return f"Tool '{tool_name}' failed with unknown error."
