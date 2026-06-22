from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Awaitable, Callable

from skg_mcp.client import SKGMCPClient
from skg_mcp.models import (
    ConceptResult,
    ExpandContextArgs,
    ExpandNeighborsArgs,
    FilterPapersArgs,
    GetAttributionArgs,
    GetProvenanceArgs,
    LexicalSearchArgs,
    NodeKind,
    ResolveConceptReferenceArgs,
    SearchNodeType,
    SemanticConstraintSearchArgs,
    SemanticSearchArgs,
    SparqlTemplateFilter,
    StatementFilter,
    StatementResult,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ToolTestNode:
    name: str
    run: Callable[[SKGMCPClient, "SeedState"], Awaitable[Any]]


@dataclass
class ToolTestResult:
    name: str
    passed: bool
    elapsed_ms: int
    details: str


@dataclass
class SeedState:
    query: str
    paper_id: str | None = None
    concept: ConceptResult | None = None
    statement: StatementResult | None = None


def load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def join_host_port(host: str | None, port: str | None) -> str | None:
    if not host:
        return None
    base = host.strip().rstrip("/")
    if not port or base.rsplit(":", 1)[-1].isdigit():
        return base
    return f"{base}:{port.strip()}"


def build_stdio_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(load_dotenv(PROJECT_ROOT / ".env"))

    env["SKG_BACKEND"] = "sudo_kg"
    env.setdefault("SKG_FUSEKI_URL", join_host_port(env.get("FUSEKI_HOST"), env.get("FUSEKI_PORT")) or "")
    env.setdefault("SKG_FUSEKI_DATASET", env.get("FUSEKI_DATASET_NAME", "sudo_kg"))
    env.setdefault("SKG_MILVUS_URI", join_host_port(env.get("MILVUS_HOST"), env.get("MILVUS_PORT")) or "")
    env.setdefault("SKG_MILVUS_COLLECTION", env.get("MILVUS_COLLECTION_NAME", "sudo_kg"))
    env.setdefault("SKG_MILVUS_DB_NAME", env.get("MILVUS_DB_NAME", ""))
    env.setdefault("SKG_EMBEDDER_PROVIDER", env.get("MODEL_PROVIDER", "openai"))
    env.setdefault("SKG_EMBEDDER_BASE_URL", env.get("PROVIDER_URL", env.get("OPENAI_BASE_URL", "")))
    env.setdefault("SKG_EMBEDDER_API_KEY", env.get("OPENAI_API_KEY", "lmstudio"))
    env.setdefault("SKG_EMBEDDER_MODEL", env.get("EMBEDDING_MODEL", env.get("JINA_MODEL", "jina-v4-text-matching")))
    env.setdefault("SKG_BACKEND_TIMEOUT_SECONDS", "60")

    # Empty strings can surprise clients that distinguish unset from blank.
    return {key: value for key, value in env.items() if value != ""}


async def test_filter_papers(client: SKGMCPClient, state: SeedState) -> dict[str, Any]:
    result = await client.filter_papers(FilterPapersArgs(limit=3))
    if not result.papers:
        raise AssertionError("filter_papers returned no papers")
    state.paper_id = result.papers[0].id
    return {"paper_count": len(result.papers), "seed_paper_id": state.paper_id}


async def test_lexical_search(client: SKGMCPClient, state: SeedState) -> dict[str, Any]:
    result = await client.lexical_search(
        LexicalSearchArgs(
            query=state.query,
            node_types=[SearchNodeType.CONCEPT, SearchNodeType.STATEMENT],
            limit=6,
        )
    )
    if not result.concepts and not result.statements:
        raise AssertionError("lexical_search returned no concepts or statements")
    state.concept = result.concepts[0] if result.concepts else state.concept
    state.statement = result.statements[0] if result.statements else state.statement
    return {
        "concept_count": len(result.concepts),
        "statement_count": len(result.statements),
        "seed_concept_id": state.concept.id if state.concept else None,
        "seed_statement_id": state.statement.id if state.statement else None,
    }


async def test_semantic_search(client: SKGMCPClient, state: SeedState) -> dict[str, Any]:
    result = await client.semantic_search(
        SemanticSearchArgs(
            query=state.query,
            node_types=[SearchNodeType.CONCEPT, SearchNodeType.STATEMENT],
            limit=6,
        )
    )
    if not result.concepts and not result.statements:
        raise AssertionError("semantic_search returned no concepts or statements")
    state.concept = state.concept or (result.concepts[0] if result.concepts else None)
    state.statement = state.statement or (result.statements[0] if result.statements else None)
    return {"concept_count": len(result.concepts), "statement_count": len(result.statements)}


async def test_semantic_constraint_search(client: SKGMCPClient, state: SeedState) -> dict[str, Any]:
    filters = None
    if state.concept:
        concept_iri = f"https://purl.org/twc/sudo/kg/concept/{state.concept.id}"
        filters = StatementFilter(
            sparql_filters=[
                SparqlTemplateFilter(
                    where="?node sudo:mentions ?concept . FILTER(?concept IN ({{concepts}}))",
                    params={"concepts": [concept_iri]},
                    graph="sudo",
                )
            ]
        )

    result = await client.semantic_constraint_search(
        SemanticConstraintSearchArgs(
            query=state.query,
            node_types=[SearchNodeType.STATEMENT],
            statement_filters=filters,
            limit=5,
        )
    )
    return {"statement_count": len(result.statements), "used_sparql_filter": filters is not None}


async def test_resolve_concept_reference(client: SKGMCPClient, state: SeedState) -> dict[str, Any]:
    mention = state.concept.label if state.concept else state.query
    result = await client.resolve_concept_reference(
        ResolveConceptReferenceArgs(
            mention=mention,
            context_text=state.statement.text if state.statement else state.query,
            paper_id=state.paper_id,
            limit=3,
        )
    )
    if not result.resolutions:
        raise AssertionError("resolve_concept_reference returned no resolutions")
    return {"resolution_count": len(result.resolutions), "top_confidence": result.resolutions[0].confidence}


def seed_node(state: SeedState) -> tuple[str, NodeKind]:
    if state.concept:
        return state.concept.id, NodeKind.CONCEPT
    if state.statement:
        return state.statement.id, NodeKind.STATEMENT
    raise AssertionError("No seed concept or statement is available")


async def test_expand_context(client: SKGMCPClient, state: SeedState) -> dict[str, Any]:
    node_id, node_kind = seed_node(state)
    result = await client.expand_context(
        ExpandContextArgs(
            node_id=node_id,
            node_kind=node_kind,
            paper_id=state.paper_id,
            max_linked_nodes=5,
            max_neighbor_nodes=5,
        )
    )
    return {
        "node_id": result.node.id,
        "linked_count": len(result.linked_nodes or []),
        "neighbor_count": len(result.neighbor_nodes or []),
        "paper_usage_count": len(result.paper_usage or []),
    }


async def test_expand_neighbors(client: SKGMCPClient, state: SeedState) -> dict[str, Any]:
    node_id, node_kind = seed_node(state)
    result = await client.expand_neighbors(
        ExpandNeighborsArgs(
            node_id=node_id,
            node_kind=node_kind,
            paper_id=state.paper_id,
            hop_count=1,
            limit=10,
        )
    )
    return {"neighbor_count": len(result.neighbors)}


async def test_get_attibution(client: SKGMCPClient, state: SeedState) -> dict[str, Any]:
    node_id, node_kind = seed_node(state)
    result = await client.get_attibution(
        GetAttributionArgs(node_ids=[node_id], node_kind=node_kind)
    )
    return {"attribution_count": len(result.attributions)}


async def test_get_provenance(client: SKGMCPClient, state: SeedState) -> dict[str, Any]:
    node_id, node_kind = seed_node(state)
    result = await client.get_provenance(
        GetProvenanceArgs(node_ids=[node_id], node_kind=node_kind)
    )
    if not result.provenance:
        raise AssertionError("get_provenance returned no provenance")
    return {"provenance_count": len(result.provenance)}


def build_tool_test_nodes() -> list[ToolTestNode]:
    return [
        ToolTestNode("filter_papers", test_filter_papers),
        ToolTestNode("lexical_search", test_lexical_search),
        ToolTestNode("semantic_search", test_semantic_search),
        ToolTestNode("semantic_constraint_search", test_semantic_constraint_search),
        ToolTestNode("resolve_concept_reference", test_resolve_concept_reference),
        ToolTestNode("expand_context", test_expand_context),
        ToolTestNode("expand_neighbors", test_expand_neighbors),
        ToolTestNode("get_attibution", test_get_attibution),
        ToolTestNode("get_provenance", test_get_provenance),
    ]


async def run_tests(query: str, read_timeout_seconds: float) -> list[ToolTestResult]:
    state = SeedState(query=query)
    results: list[ToolTestResult] = []
    async with SKGMCPClient.connect_stdio(
        env=build_stdio_env(),
        args=["run", "src/server.py"],
        cwd=str(PROJECT_ROOT),
        read_timeout_seconds=read_timeout_seconds,
    ) as client:
        tool_names = {tool.name for tool in await client.list_tools()}
        expected = {node.name for node in build_tool_test_nodes()}
        missing = sorted(expected - tool_names)
        if missing:
            raise RuntimeError(f"MCP server is missing expected tools: {missing}")

        for node in build_tool_test_nodes():
            started = perf_counter()
            try:
                details = await node.run(client, state)
                results.append(
                    ToolTestResult(
                        name=node.name,
                        passed=True,
                        elapsed_ms=int((perf_counter() - started) * 1000),
                        details=json.dumps(details, sort_keys=True),
                    )
                )
            except Exception as exc:
                results.append(
                    ToolTestResult(
                        name=node.name,
                        passed=False,
                        elapsed_ms=int((perf_counter() - started) * 1000),
                        details=f"{type(exc).__name__}: {exc}",
                    )
                )
    return results


def print_report(results: list[ToolTestResult]) -> None:
    width = max(len(result.name) for result in results)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} {result.name.ljust(width)} {result.elapsed_ms:>6} ms  {result.details}")
    passed = sum(1 for result in results if result.passed)
    print(f"\n{passed}/{len(results)} tool tests passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live SUDO KG MCP tool tests.")
    parser.add_argument("--query", default="eventuality entailment graph")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a human report.")
    parser.add_argument("--no-fail", action="store_true", help="Exit 0 even if a tool test fails.")
    args = parser.parse_args()

    results = asyncio.run(run_tests(query=args.query, read_timeout_seconds=args.timeout))
    if args.json:
        print(json.dumps([result.__dict__ for result in results], indent=2))
    else:
        print_report(results)

    if not args.no_fail and any(not result.passed for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
