from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import httpx

try:
    from pymilvus import MilvusClient
except Exception:  # pragma: no cover - optional runtime dependency
    MilvusClient = None  # type: ignore[assignment]

try:
    from src.embedder.factory import create_embedder
except Exception:  # pragma: no cover - import-path compatibility
    from embedder.factory import create_embedder  # type: ignore

from .backend import ScholarlyKnowledgeGraphBackend
from .models import (
    ConceptFilter,
    ConceptResult,
    ContextNode,
    DocumentLocation,
    NodeAttribution,
    NodeProvenance,
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
    NeighborResult,
    NodeKind,
    NodeRef,
    PaperFilter,
    PaperRef,
    ProvenanceRef,
    Resolution,
    ResolveConceptReferenceArgs,
    ResolveConceptReferenceResult,
    SearchNodeType,
    SearchResult,
    SemanticConstraintSearchArgs,
    SemanticSearchArgs,
    SparqlTemplateFilter,
    StatementFilter,
    StatementResult,
)


PREFIXES = """
PREFIX dct: <http://purl.org/dc/terms/>
PREFIX fabio: <http://purl.org/spar/fabio/>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX pav: <http://purl.org/pav/>
PREFIX po: <http://www.essepuntato.it/2008/12/pattern#>
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX sudo: <https://w3id.org/twc/sudo/ontology#>
"""

KG_CONCEPT_BASE = "https://w3id.org/twc/sudo/kg/concept/"
KG_PROPOSITION_BASE = "https://w3id.org/twc/sudo/kg/proposition/"
PAPER_BASE = "https://w3id.org/twc/sudo/kg/paper/"
SENTENCE_BASE = "https://w3id.org/twc/sudo/kg/sentence/"

SUDO_META_TYPES = {"Artifact", "Argument", "Descriptor"}


def _load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def _join_host_port(host: str | None, port: str | None) -> str | None:
    if not host:
        return None
    base = host.strip().rstrip("/")
    if not port or re.search(r":\d+$", base):
        return base
    return f"{base}:{port.strip()}"


@dataclass(frozen=True)
class VectorHit:
    id: str
    text: str | None
    node_type: str | None
    paper_id: str | None
    score: float | None


class SudoKGBackend(ScholarlyKnowledgeGraphBackend):
    """Fuseki + Milvus backend for the SUDO scholarly KG."""

    def __init__(
        self,
        *,
        fuseki_query_url: str,
        milvus_uri: str | None = None,
        milvus_token: str | None = None,
        milvus_db_name: str | None = None,
        milvus_collection: str = "sudo_kg",
        graph_meta: str = "urn:meta",
        graph_struct: str = "urn:struct",
        graph_sudo: str = "urn:sudo",
        graph_prov: str = "urn:prov",
        graph_concept: str = "urn:concept",
        embedder_provider: str = "openai",
        embedder_model: str = "jina-v4-text-matching",
        embedder_api_key: str | None = None,
        embedder_base_url: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.fuseki_query_url = fuseki_query_url
        self.milvus_collection = milvus_collection
        self.graphs = {
            "meta": graph_meta,
            "struct": graph_struct,
            "sudo": graph_sudo,
            "prov": graph_prov,
            "concept": graph_concept,
        }
        self.timeout_seconds = timeout_seconds

        self._http = httpx.AsyncClient(timeout=timeout_seconds)
        self._milvus = None
        if milvus_uri:
            if MilvusClient is None:
                raise RuntimeError("pymilvus is required for SudoKGBackend Milvus search.")
            kwargs: dict[str, Any] = {"uri": milvus_uri}
            if milvus_token:
                kwargs["token"] = milvus_token
            if milvus_db_name:
                kwargs["db_name"] = milvus_db_name
            self._milvus = MilvusClient(**kwargs)

        try:
            self._embedder = create_embedder(
                model_name=embedder_model,
                provider=embedder_provider,
                api_key=embedder_api_key,
                base_url=embedder_base_url,
            )
        except Exception:
            self._embedder = None

    @classmethod
    def from_env(cls) -> "SudoKGBackend":
        _load_dotenv()
        fuseki_query_url = os.getenv("SKG_FUSEKI_QUERY_URL")
        if not fuseki_query_url:
            base = _join_host_port(
                os.getenv("SKG_FUSEKI_URL") or os.getenv("FUSEKI_HOST") or "http://localhost:3030",
                os.getenv("SKG_FUSEKI_PORT") or os.getenv("FUSEKI_PORT"),
            )
            dataset = (
                os.getenv("SKG_FUSEKI_DATASET")
                or os.getenv("FUSEKI_DATASET_NAME")
                or "sudo_kg"
            ).strip("/")
            fuseki_query_url = f"{base}/{dataset}/sparql"

        return cls(
            fuseki_query_url=fuseki_query_url,
            milvus_uri=os.getenv("SKG_MILVUS_URI")
            or _join_host_port(os.getenv("MILVUS_HOST"), os.getenv("MILVUS_PORT")),
            milvus_token=os.getenv("SKG_MILVUS_TOKEN"),
            milvus_db_name=os.getenv("SKG_MILVUS_DB_NAME") or os.getenv("MILVUS_DB_NAME"),
            milvus_collection=os.getenv("SKG_MILVUS_COLLECTION")
            or os.getenv("MILVUS_COLLECTION_NAME")
            or "sudo_kg",
            graph_meta=os.getenv("SKG_GRAPH_META", "urn:meta"),
            graph_struct=os.getenv("SKG_GRAPH_STRUCT", "urn:struct"),
            graph_sudo=os.getenv("SKG_GRAPH_SUDO", "urn:sudo"),
            graph_prov=os.getenv("SKG_GRAPH_PROV", "urn:prov"),
            graph_concept=os.getenv("SKG_GRAPH_CONCEPT", "urn:concept"),
            embedder_provider=os.getenv("SKG_EMBEDDER_PROVIDER")
            or os.getenv("MODEL_PROVIDER")
            or "openai",
            embedder_model=os.getenv("SKG_EMBEDDER_MODEL")
            or os.getenv("EMBEDDING_MODEL")
            or os.getenv("JINA_MODEL")
            or "jina-v4-text-matching",
            embedder_api_key=os.getenv("SKG_EMBEDDER_API_KEY") or os.getenv("OPENAI_API_KEY"),
            embedder_base_url=os.getenv("SKG_EMBEDDER_BASE_URL")
            or os.getenv("PROVIDER_URL")
            or os.getenv("OPENAI_BASE_URL"),
            timeout_seconds=float(os.getenv("SKG_BACKEND_TIMEOUT_SECONDS", "30")),
        )

    async def _sparql(self, query: str) -> list[dict[str, Any]]:
        response = await self._http.post(
            self.fuseki_query_url,
            data={"query": query},
            headers={"Accept": "application/sparql-results+json"},
        )
        response.raise_for_status()
        data = response.json()
        return data.get("results", {}).get("bindings", [])

    @staticmethod
    def _binding(row: dict[str, Any], key: str) -> str | None:
        value = row.get(key)
        if not value:
            return None
        return value.get("value")

    @staticmethod
    def _local_name(value: str | None) -> str | None:
        if not value:
            return None
        return re.split(r"[/#]", value.rstrip("/#"))[-1]

    @staticmethod
    def _literal(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int | float):
            return str(value)
        escaped = (
            str(value)
            .replace("\\", "\\\\")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
            .replace('"', '\\"')
        )
        return f'"{escaped}"'

    @classmethod
    def _term(cls, value: Any) -> str:
        if isinstance(value, str) and re.match(r"^https?://", value):
            return f"<{value}>"
        return cls._literal(value)

    @classmethod
    def _terms(cls, values: Sequence[Any]) -> str:
        return ", ".join(cls._term(value) for value in values)

    def _graph(self, name: str) -> str:
        graph = self.graphs[name]
        return graph if graph.startswith("<") and graph.endswith(">") else f"<{graph}>"

    def _node_iri(self, node_id: str, node_kind: NodeKind | SearchNodeType | str) -> str:
        if node_id.startswith("http://") or node_id.startswith("https://"):
            return node_id
        kind = node_kind.value if hasattr(node_kind, "value") else str(node_kind)
        if kind in {"statement", "proposition"}:
            return f"{KG_PROPOSITION_BASE}{node_id}"
        if kind == "paper":
            return f"{PAPER_BASE}{node_id}"
        return f"{KG_CONCEPT_BASE}{node_id}"

    def _paper_iri(self, paper_id: str) -> str:
        if paper_id.startswith("http://") or paper_id.startswith("https://"):
            return paper_id
        return f"{PAPER_BASE}{paper_id}"

    def _render_sparql_filters(
        self,
        filters: Iterable[SparqlTemplateFilter] | None,
        *,
        default_graph: str,
    ) -> str:
        chunks: list[str] = []
        for template in filters or []:
            chunk = template.where
            for key, value in template.params.items():
                rendered = self._terms(value) if isinstance(value, list) else self._term(value)
                chunk = chunk.replace("{{" + key + "}}", rendered)
            if re.search(r"{{\s*[\w-]+\s*}}", chunk):
                raise ValueError(f"Unresolved SPARQL filter placeholder in: {template.where}")
            graph = template.graph or default_graph
            chunks.append(f"GRAPH {self._graph(graph)} {{ {chunk} }}")
        return "\n".join(chunks)

    def _paper_filter_where(self, filters: PaperFilter | None) -> str:
        if not filters:
            return ""
        chunks: list[str] = []
        if filters.paper_ids:
            iris = self._terms([self._paper_iri(pid) for pid in filters.paper_ids])
            chunks.append(f"FILTER(?paper IN ({iris}))")
        if filters.authors:
            names = self._terms(filters.authors)
            chunks.append(
                "GRAPH "
                + self._graph("meta")
                + " { ?paper dct:creator/foaf:name ?authorName . "
                + f"FILTER(?authorName IN ({names})) }}"
            )
        if filters.domains:
            domains = self._terms(filters.domains)
            chunks.append(
                "GRAPH "
                + self._graph("meta")
                + " { ?paper dct:subject ?domain . FILTER(?domain IN ("
                + domains
                + ")) }"
            )
        chunks.append(self._render_sparql_filters(filters.sparql_filters, default_graph="meta"))
        return "\n".join(chunk for chunk in chunks if chunk)

    def _concept_filter_where(self, filters: ConceptFilter | None) -> str:
        if not filters:
            return ""
        chunks: list[str] = []
        if filters.paper_ids:
            iris = self._terms([self._paper_iri(pid) for pid in filters.paper_ids])
            chunks.append(f"GRAPH {self._graph('prov')} {{ ?node prov:hadPrimarySource ?paper . FILTER(?paper IN ({iris})) }}")
        if filters.concept_types:
            type_iris = self._terms([f"https://w3id.org/twc/sudo/ontology#{t}" for t in filters.concept_types])
            chunks.append(f"GRAPH {self._graph('sudo')} {{ ?node a ?type . FILTER(?type IN ({type_iris})) }}")
        if filters.canonical_only:
            chunks.append(f"GRAPH {self._graph('concept')} {{ ?node a ?type . }}")
        if filters.paper_local_only:
            chunks.append(f"GRAPH {self._graph('sudo')} {{ ?node a sudo:Artifact . }}")
        chunks.append(self._render_sparql_filters(filters.sparql_filters, default_graph="sudo"))
        return "\n".join(chunk for chunk in chunks if chunk)

    def _statement_filter_where(self, filters: StatementFilter | None) -> str:
        if not filters:
            return ""
        chunks: list[str] = []
        if filters.paper_ids:
            iris = self._terms([self._paper_iri(pid) for pid in filters.paper_ids])
            chunks.append(f"GRAPH {self._graph('prov')} {{ ?node prov:hadPrimarySource ?paper . FILTER(?paper IN ({iris})) }}")
        if filters.statement_types:
            type_iris = self._terms([f"https://w3id.org/twc/sudo/ontology#{t}" for t in filters.statement_types])
            chunks.append(f"GRAPH {self._graph('sudo')} {{ ?node a ?type . FILTER(?type IN ({type_iris})) }}")
        if filters.rhetorical_roles:
            type_iris = self._terms([f"https://w3id.org/twc/sudo/ontology#{t}" for t in filters.rhetorical_roles])
            chunks.append(f"GRAPH {self._graph('sudo')} {{ ?node a ?role . FILTER(?role IN ({type_iris})) }}")
        chunks.append(self._render_sparql_filters(filters.sparql_filters, default_graph="sudo"))
        return "\n".join(chunk for chunk in chunks if chunk)

    async def _hydrate_hits(
        self,
        hits: list[VectorHit],
        kind: SearchNodeType,
        *,
        concept_filters: ConceptFilter | None = None,
        statement_filters: StatementFilter | None = None,
        paper_filters: PaperFilter | None = None,
    ) -> list[dict[str, Any]]:
        if not hits:
            return []
        values = " ".join(
            f"(<{self._node_iri(hit.id, hit.node_type or kind.value)}> {self._literal(hit.id)} {self._literal(hit.score or 0.0)})"
            for hit in hits
        )
        custom_filters = []
        if kind == SearchNodeType.CONCEPT:
            custom_filters.append(self._concept_filter_where(concept_filters))
        else:
            custom_filters.append(self._statement_filter_where(statement_filters))
        custom_filters.append(self._paper_filter_where(paper_filters))
        filter_where = "\n".join(chunk for chunk in custom_filters if chunk)
        query = f"""
{PREFIXES}
SELECT ?id ?score ?node ?label ?type ?paper WHERE {{
  VALUES (?node ?id ?score) {{ {values} }}
  OPTIONAL {{ GRAPH {self._graph('sudo')} {{ ?node rdfs:label ?label . }} }}
  OPTIONAL {{ GRAPH {self._graph('concept')} {{ ?node rdfs:label ?label . }} }}
  OPTIONAL {{
    GRAPH {self._graph('sudo')} {{
      ?node a ?type .
      FILTER(STRSTARTS(STR(?type), "https://w3id.org/twc/sudo/ontology#"))
    }}
  }}
  OPTIONAL {{ GRAPH {self._graph('prov')} {{ ?node prov:hadPrimarySource ?paper . }} }}
  {filter_where}
}}
"""
        rows = await self._sparql(query)
        by_id: dict[str, Any] = {}
        for row in rows:
            rid = self._binding(row, "id")
            if rid is None:
                continue
            if rid not in by_id:
                by_id[rid] = row
            else:
                existing_type = self._local_name(self._binding(by_id[rid], "type")) or ""
                new_type = self._local_name(self._binding(row, "type")) or ""
                if existing_type in SUDO_META_TYPES and new_type not in SUDO_META_TYPES:
                    by_id[rid] = row
        hydrated: list[dict[str, Any]] = []
        for hit in hits:
            row = by_id.get(hit.id, {})
            if filter_where and not row:
                continue
            hydrated.append(
                {
                    "id": hit.id,
                    "label": self._binding(row, "label") or hit.text or hit.id,
                    "type": self._local_name(self._binding(row, "type")) or hit.node_type,
                    "paper_id": self._local_name(self._binding(row, "paper")) or hit.paper_id,
                    "score": hit.score,
                }
            )
        return hydrated

    def _milvus_expr(self, *, vector_types: list[str], paper_ids: list[str] | None = None) -> str:
        exprs = [f"type in [{', '.join(self._literal(t) for t in vector_types)}]"]
        if paper_ids:
            exprs.append(f"paper_id in [{', '.join(self._literal(pid) for pid in paper_ids)}]")
        return " and ".join(exprs)

    async def _milvus_search(
        self,
        *,
        query: str,
        vector_types: list[str],
        paper_ids: list[str] | None,
        limit: int,
        mode: str,
    ) -> list[VectorHit]:
        if self._milvus is None:
            return []

        expr = self._milvus_expr(vector_types=vector_types, paper_ids=paper_ids)

        def run_search() -> list[dict[str, Any]]:
            if mode == "lexical":
                raw = self._milvus.search(
                    collection_name=self.milvus_collection,
                    data=[query],
                    anns_field="text_sparse",
                    limit=limit,
                    filter=expr,
                    output_fields=["id", "text", "type", "paper_id"],
                    search_params={"metric_type": "BM25"},
                )
            else:
                if self._embedder is None:
                    raise RuntimeError(
                        "Semantic Milvus search requires a configured embedder. "
                        "Set SKG_EMBEDDER_* env vars for jina-v4-text-matching."
                    )
                vector = self._embedder.embed_texts([query])[0]
                raw = self._milvus.search(
                    collection_name=self.milvus_collection,
                    data=[vector],
                    anns_field="text_dense",
                    limit=limit,
                    filter=expr,
                    output_fields=["id", "text", "type", "paper_id"],
                    search_params={"metric_type": "COSINE", "params": {}},
                )
            return raw[0] if raw else []

        rows = await asyncio.to_thread(run_search)
        hits: list[VectorHit] = []
        for row in rows:
            entity = row.get("entity", row)
            hits.append(
                VectorHit(
                    id=str(entity.get("id") or row.get("id")),
                    text=entity.get("text"),
                    node_type=entity.get("type"),
                    paper_id=entity.get("paper_id"),
                    score=float(row.get("distance", row.get("score", 0.0))),
                )
            )
        return hits

    async def _lexical_fuseki_fallback(
        self,
        *,
        query_text: str,
        kind: SearchNodeType,
        filters: ConceptFilter | StatementFilter | None,
        paper_filters: PaperFilter | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        if kind == SearchNodeType.CONCEPT:
            type_filter = "FILTER EXISTS { GRAPH " + self._graph("sudo") + " { ?node a sudo:Artifact . } }"
            custom_filters = self._concept_filter_where(filters if isinstance(filters, ConceptFilter) else None)
        else:
            type_filter = (
                "GRAPH "
                + self._graph("sudo")
                + " { ?node a ?kind . FILTER(?kind IN (sudo:Argument, sudo:Descriptor)) }"
            )
            custom_filters = self._statement_filter_where(filters if isinstance(filters, StatementFilter) else None)
        paper_where = self._paper_filter_where(paper_filters)
        query_literal = self._literal(query_text)
        query = f"""
{PREFIXES}
SELECT ?node ?label ?type ?paper WHERE {{
  GRAPH {self._graph('sudo')} {{
    ?node rdfs:label ?label ;
          a ?type .
    FILTER(CONTAINS(LCASE(STR(?label)), LCASE({query_literal})))
  }}
  {type_filter}
  OPTIONAL {{ GRAPH {self._graph('prov')} {{ ?node prov:hadPrimarySource ?paper . }} }}
  {paper_where}
  {custom_filters}
}}
LIMIT {limit}
"""
        rows = await self._sparql(query)
        seen: dict[str, dict[str, Any]] = {}
        for row in rows:
            item = {
                "id": self._local_name(self._binding(row, "node")) or "",
                "label": self._binding(row, "label") or "",
                "type": self._local_name(self._binding(row, "type")),
                "paper_id": self._local_name(self._binding(row, "paper")),
                "score": None,
            }
            nid = item["id"]
            if nid not in seen:
                seen[nid] = item
            else:
                existing_type = seen[nid].get("type") or ""
                new_type = item.get("type") or ""
                if existing_type in SUDO_META_TYPES and new_type not in SUDO_META_TYPES:
                    seen[nid] = item
        return list(seen.values())

    @staticmethod
    def _concept_result(row: dict[str, Any]) -> ConceptResult:
        return ConceptResult(
            id=row["id"],
            label=row["label"],
            concept_type=row.get("type"),
            is_canonical=row.get("type") == "Concept",
            paper_id=row.get("paper_id"),
            score=row.get("score"),
        )

    @staticmethod
    def _statement_result(row: dict[str, Any]) -> StatementResult:
        node_type = row.get("type")
        return StatementResult(
            id=row["id"],
            text=row["label"],
            statement_type=node_type,
            rhetorical_role=node_type if node_type not in SUDO_META_TYPES else None,
            paper_id=row.get("paper_id"),
            score=row.get("score"),
        )

    async def filter_papers(self, args: FilterPapersArgs) -> FilterPapersResult:
        where = self._paper_filter_where(args.filters)
        query = f"""
{PREFIXES}
SELECT ?paper ?title ?year ?venue WHERE {{
  GRAPH {self._graph('meta')} {{
    ?paper a fabio:ResearchPaper .
    OPTIONAL {{ ?paper dct:title ?title . }}
    OPTIONAL {{ ?paper dct:issued ?year . }}
    OPTIONAL {{ ?paper dct:isPartOf/rdfs:label ?venue . }}
  }}
  {where}
}}
OFFSET {args.offset}
LIMIT {args.limit}
"""
        rows = await self._sparql(query)
        papers = [
            PaperRef(
                id=self._local_name(self._binding(row, "paper")) or "",
                title=self._binding(row, "title") or self._local_name(self._binding(row, "paper")) or "",
                venue=self._binding(row, "venue"),
            )
            for row in rows
        ]
        return FilterPapersResult(papers=papers, total_count=None)

    async def lexical_search(self, args: LexicalSearchArgs) -> SearchResult:
        return await self._search(query=args.query, node_types=args.node_types, limit=args.limit, mode="lexical", args=args)

    async def semantic_search(self, args: SemanticSearchArgs) -> SearchResult:
        return await self._search(query=args.query, node_types=args.node_types, limit=args.limit, mode="semantic", args=args)

    async def semantic_constraint_search(self, args: SemanticConstraintSearchArgs) -> SearchResult:
        return await self._search(
            query=args.query,
            node_types=args.node_types,
            limit=args.limit,
            mode="semantic" if args.match_mode.value != "lexical" else "lexical",
            args=args,
        )

    async def _search(self, *, query: str, node_types: list[SearchNodeType], limit: int, mode: str, args: Any) -> SearchResult:
        concepts: list[ConceptResult] = []
        statements: list[StatementResult] = []
        paper_ids = None
        if getattr(args, "paper_filters", None) and args.paper_filters.paper_ids:
            paper_ids = args.paper_filters.paper_ids

        per_kind_limit = max(1, limit // max(1, len(node_types)))
        if SearchNodeType.CONCEPT in node_types:
            try:
                hits = await self._milvus_search(
                    query=query,
                    vector_types=["concept"],
                    paper_ids=paper_ids or (args.concept_filters.paper_ids if getattr(args, "concept_filters", None) else None),
                    limit=per_kind_limit,
                    mode=mode,
                )
            except Exception:
                hits = []
            rows = await self._hydrate_hits(
                hits,
                SearchNodeType.CONCEPT,
                concept_filters=getattr(args, "concept_filters", None),
                paper_filters=getattr(args, "paper_filters", None),
            ) if hits else await self._lexical_fuseki_fallback(
                query_text=query,
                kind=SearchNodeType.CONCEPT,
                filters=getattr(args, "concept_filters", None),
                paper_filters=getattr(args, "paper_filters", None),
                limit=per_kind_limit,
            )
            concepts = [self._concept_result(row) for row in rows]

        if SearchNodeType.STATEMENT in node_types:
            try:
                hits = await self._milvus_search(
                    query=query,
                    vector_types=["proposition", "artifact"],
                    paper_ids=paper_ids or (args.statement_filters.paper_ids if getattr(args, "statement_filters", None) else None),
                    limit=per_kind_limit,
                    mode=mode,
                )
            except Exception:
                hits = []
            rows = await self._hydrate_hits(
                hits,
                SearchNodeType.STATEMENT,
                statement_filters=getattr(args, "statement_filters", None),
                paper_filters=getattr(args, "paper_filters", None),
            ) if hits else await self._lexical_fuseki_fallback(
                query_text=query,
                kind=SearchNodeType.STATEMENT,
                filters=getattr(args, "statement_filters", None),
                paper_filters=getattr(args, "paper_filters", None),
                limit=per_kind_limit,
            )
            statements = [self._statement_result(row) for row in rows]

        min_score = getattr(args, "min_score", None)
        if min_score is not None:
            concepts = [item for item in concepts if item.score is None or item.score >= min_score]
            statements = [item for item in statements if item.score is None or item.score >= min_score]
        return SearchResult(concepts=concepts, statements=statements)

    async def resolve_concept_reference(self, args: ResolveConceptReferenceArgs) -> ResolveConceptReferenceResult:
        search = await self.semantic_search(
            SemanticSearchArgs(
                query=args.mention,
                node_types=[SearchNodeType.CONCEPT],
                concept_filters=ConceptFilter(paper_ids=[args.paper_id] if args.paper_id else None),
                limit=args.limit,
            )
        )
        return ResolveConceptReferenceResult(
            resolutions=[
                Resolution(concept=concept, confidence=min(1.0, concept.score or 0.75))
                for concept in search.concepts
            ]
        )

    async def _node_ref(self, node_id: str, node_kind: NodeKind, paper_id: str | None = None) -> NodeRef:
        iri = self._node_iri(node_id, node_kind)
        query = f"""
{PREFIXES}
SELECT ?label ?paper WHERE {{
  OPTIONAL {{ GRAPH {self._graph('sudo')} {{ <{iri}> rdfs:label ?label . }} }}
  OPTIONAL {{ GRAPH {self._graph('concept')} {{ <{iri}> rdfs:label ?label . }} }}
  OPTIONAL {{ GRAPH {self._graph('prov')} {{ <{iri}> prov:hadPrimarySource ?paper . }} }}
}}
LIMIT 1
"""
        rows = await self._sparql(query)
        row = rows[0] if rows else {}
        return NodeRef(
            id=node_id,
            kind=node_kind,
            label=self._binding(row, "label"),
            paper_id=paper_id or self._local_name(self._binding(row, "paper")),
        )

    async def expand_context(self, args: ExpandContextArgs) -> ExpandContextResult:
        node = await self._node_ref(args.node_id, args.node_kind, args.paper_id)

        if not args.include_artifacts and not args.include_propositions:
            return ExpandContextResult(node=node)

        iri = self._node_iri(args.node_id, args.node_kind)

        type_filter_parts: list[str] = []
        if args.include_artifacts:
            type_filter_parts.append("?targetType = sudo:Artifact")
        if args.include_propositions:
            type_filter_parts.append(f'STRSTARTS(STR(?target), "{KG_PROPOSITION_BASE}")')
        type_filter = "FILTER(" + " || ".join(type_filter_parts) + ")"

        query = f"""
{PREFIXES}
SELECT ?relLocal ?target ?targetLabel ?targetType ?targetPaper WHERE {{
  GRAPH {self._graph('sudo')} {{
    {{
      <{iri}> ?rel ?target .
    }} UNION {{
      ?target ?rel <{iri}> .
    }}
    ?target a ?targetType .
    FILTER(?rel NOT IN (rdf:type, rdfs:label))
    {type_filter}
    OPTIONAL {{ ?target rdfs:label ?targetLabel . }}
    BIND(REPLACE(STR(?rel), "^.*[/#]", "") AS ?relLocal)
  }}
  OPTIONAL {{ GRAPH {self._graph('prov')} {{ ?target prov:hadPrimarySource ?targetPaper . }} }}
}}
LIMIT {args.limit}
"""
        rows = await self._sparql(query)

        seen: dict[str, dict[str, Any]] = {}
        for row in rows:
            target_iri = self._binding(row, "target")
            if not target_iri:
                continue
            if target_iri not in seen:
                seen[target_iri] = row
            else:
                existing_type = self._local_name(self._binding(seen[target_iri], "targetType")) or ""
                new_type = self._local_name(self._binding(row, "targetType")) or ""
                if existing_type in SUDO_META_TYPES and new_type not in SUDO_META_TYPES:
                    seen[target_iri] = row

        artifacts: list[ContextNode] = []
        propositions: list[ContextNode] = []
        for target_iri, row in seen.items():
            target_id = self._local_name(target_iri) or ""
            is_proposition = KG_PROPOSITION_BASE in target_iri
            kind = NodeKind.STATEMENT if is_proposition else NodeKind.CONCEPT
            ctx_node = ContextNode(
                id=target_id,
                kind=kind,
                label=self._binding(row, "targetLabel"),
                node_type=self._local_name(self._binding(row, "targetType")),
                paper_id=self._local_name(self._binding(row, "targetPaper")) or args.paper_id,
                relation=self._binding(row, "relLocal") if args.include_relations else None,
            )
            if is_proposition:
                propositions.append(ctx_node)
            else:
                artifacts.append(ctx_node)

        return ExpandContextResult(
            node=node,
            artifacts=artifacts if args.include_artifacts else None,
            propositions=propositions if args.include_propositions else None,
        )

    async def expand_neighbors(self, args: ExpandNeighborsArgs) -> ExpandNeighborsResult:
        source = await self._node_ref(args.node_id, args.node_kind, args.paper_id)
        iri = self._node_iri(args.node_id, args.node_kind)
        relation_filter = ""
        if args.relation_types:
            relation_filter = f"FILTER(?relLocal IN ({self._terms(args.relation_types)}))"
        query = f"""
{PREFIXES}
SELECT ?relLocal ?target ?targetLabel WHERE {{
  GRAPH {self._graph('sudo')} {{
    {{
      <{iri}> ?rel ?target .
    }} UNION {{
      ?target ?rel <{iri}> .
    }}
    OPTIONAL {{ ?target rdfs:label ?targetLabel . }}
    BIND(REPLACE(STR(?rel), "^.*[/#]", "") AS ?relLocal)
    FILTER(?rel NOT IN (rdf:type, rdfs:label))
    {relation_filter}
  }}
}}
LIMIT {args.limit}
"""
        rows = await self._sparql(query)
        neighbors: list[NeighborResult] = []
        for row in rows:
            target = self._binding(row, "target")
            target_id = self._local_name(target) or ""
            target_kind = NodeKind.STATEMENT if target and "/proposition/" in target else NodeKind.CONCEPT
            item = NeighborResult(
                relation_type=self._binding(row, "relLocal") or "related_to",
                target_id=target_id,
                target_kind=target_kind,
                target_label=self._binding(row, "targetLabel"),
                hop=1,
            )
            if args.include_node_kinds and item.target_kind not in args.include_node_kinds:
                continue
            neighbors.append(item)
        return ExpandNeighborsResult(source_node=source, hop_count=args.hop_count, neighbors=neighbors)

    async def get_attribution(self, args: GetAttributionArgs) -> GetAttributionResult:
        values = " ".join(
            f"(<{self._node_iri(nid, args.node_kind)}> {self._literal(nid)})"
            for nid in args.node_ids
        )
        paper_filter = ""
        if args.paper_id:
            paper_filter = f"FILTER(?paper = <{self._paper_iri(args.paper_id)}>)"
        query = f"""
{PREFIXES}
SELECT ?nodeId ?paper ?paperTitle ?sentence ?sentenceText ?parent ?grandparent ?sectionTitle WHERE {{
  VALUES (?node ?nodeId) {{ {values} }}
  GRAPH {self._graph('prov')} {{
    ?node prov:hadPrimarySource ?paper .
    OPTIONAL {{ ?node prov:wasDerivedFrom ?sentence . }}
    {paper_filter}
  }}
  OPTIONAL {{ GRAPH {self._graph('meta')} {{ ?paper dct:title ?paperTitle . }} }}
  OPTIONAL {{
    GRAPH {self._graph('struct')} {{
      ?parent po:contains ?sentence .
      OPTIONAL {{ ?sentence rdf:value ?sentenceText . }}
      OPTIONAL {{
        ?grandparent po:contains ?parent .
        OPTIONAL {{
          ?grandparent po:containsAsHeader ?sectionHeader .
          ?sectionHeader rdf:value ?sectionTitle .
        }}
      }}
    }}
  }}
}}
"""
        rows = await self._sparql(query)

        by_node: dict[str, dict[str, Any]] = {}
        for row in rows:
            nid = self._binding(row, "nodeId")
            if nid and nid not in by_node:
                by_node[nid] = row

        attributions: list[NodeAttribution] = []
        for nid in args.node_ids:
            row = by_node.get(nid, {})
            paper = None
            paper_iri = self._binding(row, "paper")
            if paper_iri:
                paper = PaperRef(
                    id=self._local_name(paper_iri) or paper_iri,
                    title=self._binding(row, "paperTitle") or self._local_name(paper_iri) or paper_iri,
                )
            location = None
            sentence_iri = self._binding(row, "sentence")
            if sentence_iri:
                parent_iri = self._binding(row, "parent")
                grandparent_iri = self._binding(row, "grandparent")
                if grandparent_iri:
                    section_id = self._local_name(grandparent_iri)
                    paragraph_id = self._local_name(parent_iri)
                else:
                    section_id = self._local_name(parent_iri)
                    paragraph_id = None
                location = DocumentLocation(
                    section_id=section_id,
                    section_title=self._binding(row, "sectionTitle"),
                    paragraph_id=paragraph_id,
                    sentence_id=self._local_name(sentence_iri),
                    sentence_text=self._binding(row, "sentenceText"),
                )
            attributions.append(NodeAttribution(node_id=nid, paper=paper, location=location))

        return GetAttributionResult(attributions=attributions)

    async def get_provenance(self, args: GetProvenanceArgs) -> GetProvenanceResult:
        values = " ".join(
            f"(<{self._node_iri(nid, args.node_kind)}> {self._literal(nid)})"
            for nid in args.node_ids
        )
        paper_filter = ""
        if args.paper_id:
            paper_filter = f"FILTER(?paper = <{self._paper_iri(args.paper_id)}>)"
        query = f"""
{PREFIXES}
SELECT ?nodeId ?paper ?sentence WHERE {{
  VALUES (?node ?nodeId) {{ {values} }}
  GRAPH {self._graph('prov')} {{
    ?node prov:hadPrimarySource ?paper .
    OPTIONAL {{ ?node prov:wasDerivedFrom ?sentence . }}
    {paper_filter}
  }}
}}
"""
        rows = await self._sparql(query)

        by_node: dict[str, list[ProvenanceRef]] = {nid: [] for nid in args.node_ids}
        for row in rows:
            nid = self._binding(row, "nodeId")
            paper_iri = self._binding(row, "paper")
            if nid and nid in by_node and paper_iri:
                by_node[nid].append(
                    ProvenanceRef(
                        paper_id=self._local_name(paper_iri) or paper_iri,
                        sentence_id=self._local_name(self._binding(row, "sentence")),
                    )
                )

        return GetProvenanceResult(
            provenance=[
                NodeProvenance(node_id=nid, provenance=refs)
                for nid, refs in by_node.items()
            ]
        )
