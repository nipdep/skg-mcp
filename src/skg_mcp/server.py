from __future__ import annotations

from typing import Any, TypeVar

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from .backend import NotImplementedBackend, ScholarlyKnowledgeGraphBackend
from .models import (
    ExpandContextArgs,
    ExpandContextResult,
    ExpandNeighborsArgs,
    ExpandNeighborsResult,
    FilterPapersArgs,
    FilterPapersResult,
    GetAttributionArgs,
    GetAttributionResult,
    GetProvenanceArgs,
    GetProvenanceResult,
    LexicalSearchArgs,
    ResolveConceptReferenceArgs,
    ResolveConceptReferenceResult,
    SearchResult,
    SemanticConstraintSearchArgs,
    SemanticSearchArgs,
)


SERVER_NAME = "Scholarly Knowledge Graph MCP"
SERVER_VERSION = "0.3.0"

TArgs = TypeVar("TArgs", bound=BaseModel)
TResult = TypeVar("TResult", bound=BaseModel)


async def _invoke_backend(
    backend: ScholarlyKnowledgeGraphBackend,
    method_name: str,
    args: TArgs,
    result_model: type[TResult],
) -> TResult:
    method = getattr(backend, method_name)
    result = await method(args)
    return result_model.model_validate(result)


def create_mcp_server(backend: ScholarlyKnowledgeGraphBackend | None = None) -> FastMCP:
    backend = backend or NotImplementedBackend()
    mcp = FastMCP(SERVER_NAME)

    @mcp.tool(
        name="filter_papers",
        description="Return papers matching metadata constraints.",
        structured_output=True,
    )
    async def filter_papers(args: FilterPapersArgs) -> FilterPapersResult:
        return await _invoke_backend(backend, "filter_papers", args, FilterPapersResult)

    @mcp.tool(
        name="lexical_search",
        description="Unified lexical search across concept and/or statement nodes using node_types.",
        structured_output=True,
    )
    async def lexical_search(args: LexicalSearchArgs) -> SearchResult:
        return await _invoke_backend(backend, "lexical_search", args, SearchResult)

    @mcp.tool(
        name="semantic_search",
        description="Unified semantic search across concept and/or statement nodes using node_types.",
        structured_output=True,
    )
    async def semantic_search(args: SemanticSearchArgs) -> SearchResult:
        return await _invoke_backend(backend, "semantic_search", args, SearchResult)

    @mcp.tool(
        name="semantic_constraint_search",
        description="Unified constrained semantic search across concept and/or statement nodes using node_types.",
        structured_output=True,
    )
    async def semantic_constraint_search(args: SemanticConstraintSearchArgs) -> SearchResult:
        return await _invoke_backend(backend, "semantic_constraint_search", args, SearchResult)

    @mcp.tool(
        name="resolve_concept_reference",
        description="Disambiguate a concept mention using local context and optional paper scope.",
        structured_output=True,
    )
    async def resolve_concept_reference(
        args: ResolveConceptReferenceArgs,
    ) -> ResolveConceptReferenceResult:
        return await _invoke_backend(
            backend,
            "resolve_concept_reference",
            args,
            ResolveConceptReferenceResult,
        )

    @mcp.tool(
        name="expand_context",
        description="Expand linked context around any concept or statement node.",
        structured_output=True,
    )
    async def expand_context(args: ExpandContextArgs) -> ExpandContextResult:
        return await _invoke_backend(backend, "expand_context", args, ExpandContextResult)

    @mcp.tool(
        name="expand_neighbors",
        description="Expand neighbors from any node kind with configurable hop_count.",
        structured_output=True,
    )
    async def expand_neighbors(args: ExpandNeighborsArgs) -> ExpandNeighborsResult:
        return await _invoke_backend(
            backend,
            "expand_neighbors",
            args,
            ExpandNeighborsResult,
        )

    @mcp.tool(
        name="get_attribution",
        description="Return the source paper and precise document location (section, paragraph, sentence) for a given node.",
        structured_output=True,
    )
    async def get_attribution(args: GetAttributionArgs) -> GetAttributionResult:
        return await _invoke_backend(
            backend,
            "get_attribution",
            args,
            GetAttributionResult,
        )

    @mcp.tool(
        name="get_provenance",
        description="Return provenance data for a given node.",
        structured_output=True,
    )
    async def get_provenance(args: GetProvenanceArgs) -> GetProvenanceResult:
        return await _invoke_backend(
            backend,
            "get_provenance",
            args,
            GetProvenanceResult,
        )

    @mcp.resource(
        "skg://metadata/server",
        name="server_metadata",
        description="Server metadata and MCP surface summary.",
        mime_type="application/json",
    )
    async def server_metadata() -> dict[str, Any]:
        tools = await mcp.list_tools()
        resources = await mcp.list_resources()
        templates = await mcp.list_resource_templates()
        return {
            "name": SERVER_NAME,
            "version": SERVER_VERSION,
            "tool_count": len(tools),
            "tool_names": [tool.name for tool in tools],
            "resources": [resource.uri.unicode_string() for resource in resources],
            "resource_templates": [template.uriTemplate for template in templates],
        }

    @mcp.resource(
        "skg://catalog/tools",
        name="tool_catalog",
        description="Canonical MCP tool catalog exposed by this server.",
        mime_type="application/json",
    )
    async def tool_catalog() -> dict[str, Any]:
        tools = await mcp.list_tools()
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                    "output_schema": tool.outputSchema,
                }
                for tool in tools
            ]
        }

    @mcp.resource(
        "skg://catalog/tools/{tool_name}",
        name="tool_schema",
        description="Lookup for one MCP tool schema by tool name.",
        mime_type="application/json",
    )
    async def tool_schema(tool_name: str) -> dict[str, Any]:
        tools = await mcp.list_tools()
        for tool in tools:
            if tool.name == tool_name:
                return {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                    "output_schema": tool.outputSchema,
                }
        available = ", ".join(sorted(tool.name for tool in tools))
        raise ValueError(f"Unknown tool '{tool_name}'. Available tools: {available}")

    return mcp


def run_stdio(backend: ScholarlyKnowledgeGraphBackend | None = None) -> None:
    server = create_mcp_server(backend=backend)
    server.run(transport="stdio")


def run_streamable_http(
    backend: ScholarlyKnowledgeGraphBackend | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    path: str = "/mcp",
) -> None:
    server = create_mcp_server(backend=backend)
    server.settings.host = host
    server.settings.port = port
    server.settings.streamable_http_path = path
    server.run(transport="streamable-http")
