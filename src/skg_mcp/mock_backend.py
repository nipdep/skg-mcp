from __future__ import annotations

import asyncio
import json
import math
import os
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

try:
    from src.embedder.factory import create_embedder
except Exception:  # pragma: no cover - import-path compatibility
    from embedder.factory import create_embedder  # type: ignore

try:
    from src.llm.lms_gpt import OpenAIEndpointLLMGenerator
except Exception:  # pragma: no cover - import-path compatibility
    from llm.lms_gpt import OpenAIEndpointLLMGenerator  # type: ignore

try:
    from src.llm.nim import NIMLLMGenerator
except Exception:  # pragma: no cover - import-path compatibility
    from llm.nim import NIMLLMGenerator  # type: ignore

from .backend import ScholarlyKnowledgeGraphBackend
from .models import (
    ConceptResult,
    ContextNode,
    DocumentLocation,
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
    NodeAttribution,
    NodeProvenance,
    NodeRef,
    NodeKind,
    PaperRef,
    ProvenanceRef,
    Resolution,
    ResolveConceptReferenceArgs,
    ResolveConceptReferenceResult,
    SearchNodeType,
    SearchResult,
    SemanticConstraintSearchArgs,
    SemanticSearchArgs,
    StatementResult,
)


TModel = TypeVar("TModel", bound=BaseModel)


class LLMMockBackend(ScholarlyKnowledgeGraphBackend):
    """Mock backend backed by local LLM/embedder modules for contract testing."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None,
        temperature: float = 0.2,
        timeout_seconds: float = 45.0,
        strict_llm: bool = False,
        llm_provider: str = "openai_endpoint",
        embedder_provider: str = "openai",
        embedder_model: str = "text-embedding-3-small",
        embedder_api_key: str | None = None,
        embedder_base_url: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.strict_llm = strict_llm

        self.llm_provider = llm_provider
        self.embedder_provider = embedder_provider
        self.embedder_model = embedder_model
        self.embedder_api_key = embedder_api_key
        self.embedder_base_url = embedder_base_url

        self.llm_generator = self._build_llm_generator()
        self.embedder = self._build_embedder()

    @classmethod
    def from_env(cls) -> "LLMMockBackend":
        return cls(
            model=os.getenv("SKG_LLM_MODEL", "openai/gpt-oss-20b"),
            base_url=os.getenv("SKG_LLM_BASE_URL", "https://spark-6d47:1234/v1"),
            api_key=os.getenv("SKG_LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            temperature=float(os.getenv("SKG_LLM_TEMPERATURE", "0.2")),
            timeout_seconds=float(os.getenv("SKG_LLM_TIMEOUT_SECONDS", "45")),
            strict_llm=os.getenv("SKG_LLM_STRICT", "false").lower() in {"1", "true", "yes"},
            llm_provider=os.getenv("SKG_LLM_PROVIDER", "lmstudio"),
            embedder_provider=os.getenv("SKG_EMBEDDER_PROVIDER", "lmstudio"),
            embedder_model=os.getenv("SKG_EMBEDDER_MODEL", "text-embedding-bge-base-en-v1.5"),
            embedder_api_key=os.getenv("SKG_EMBEDDER_API_KEY")
            or os.getenv("SKG_LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY"),
            embedder_base_url=os.getenv("SKG_EMBEDDER_BASE_URL") or os.getenv("SKG_LLM_BASE_URL"),
        )

    def _build_llm_generator(self) -> Any | None:
        provider = (self.llm_provider or "openai_endpoint").strip().lower()

        try:
            if provider in {"nim", "nvidia_nim"}:
                return NIMLLMGenerator(
                    model_name=self.model,
                    base_url=self.base_url,
                    api_key=self.api_key,
                    timeout=int(self.timeout_seconds),
                    system_prompt=(
                        "You are a scholarly knowledge graph mock backend. "
                        "Always return concise, valid JSON that matches the requested schema."
                    ),
                )

            # default to the generic OpenAI-compatible client from user's llm module.
            return OpenAIEndpointLLMGenerator(
                model_name=self.model,
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=int(self.timeout_seconds),
                system_prompt=(
                    "You are a scholarly knowledge graph mock backend. "
                    "Always return concise, valid JSON that matches the requested schema."
                ),
            )
        except Exception:
            if self.strict_llm:
                raise
            return None

    def _build_embedder(self) -> Any | None:
        try:
            return create_embedder(
                model_name=self.embedder_model,
                provider=self.embedder_provider,
                api_key=self.embedder_api_key,
                base_url=self.embedder_base_url,
            )
        except Exception:
            return None

    async def _invoke_with_llm(
        self,
        *,
        operation: str,
        args: BaseModel,
        result_model: type[TModel],
        fallback: Callable[[], TModel],
    ) -> TModel:
        if self.llm_generator is None:
            if self.strict_llm:
                raise RuntimeError(
                    "LLM generator could not be initialized from configured llm module/provider."
                )
            return fallback()

        try:
            prompt = self._build_generation_prompt(operation=operation, args=args, result_model=result_model)
            config: dict[str, Any] = {
                "temperature": self.temperature,
                "max_tokens": 2000,
            }
            parsed = await asyncio.to_thread(
                self.llm_generator.structured_text_generate,
                prompt,
                result_model,
                config,
            )
            return result_model.model_validate(parsed)
        except Exception:
            if self.strict_llm:
                raise
            return fallback()

    @staticmethod
    def _build_generation_prompt(*, operation: str, args: BaseModel, result_model: type[TModel]) -> str:
        schema = json.dumps(result_model.model_json_schema(), indent=2)
        return (
            "Generate a synthetic test response for a scholarly knowledge graph backend.\n"
            f"Operation: {operation}\n"
            f"Input args JSON:\n{args.model_dump_json(indent=2)}\n\n"
            "Return ONLY JSON that matches this schema exactly (no markdown, no explanation):\n"
            f"{schema}"
        )

    @staticmethod
    def _make_paper_ref(paper_id: str, title: str) -> PaperRef:
        return PaperRef(id=paper_id, title=title, year=2024, venue="MockConf")

    @staticmethod
    def _make_concept(concept_id: str, label: str, paper_id: str | None = None, score: float = 0.75) -> ConceptResult:
        return ConceptResult(
            id=concept_id,
            label=label,
            concept_type="mock_concept",
            is_canonical=True,
            paper_id=paper_id,
            score=score,
            summary=f"Synthetic concept generated for '{label}'.",
        )

    @staticmethod
    def _make_statement(statement_id: str, text: str, paper_id: str | None = None, score: float = 0.72) -> StatementResult:
        return StatementResult(
            id=statement_id,
            text=text,
            statement_type="mock_statement",
            rhetorical_role="method",
            paper_id=paper_id,
            score=score,
        )

    @staticmethod
    def _make_node_ref(
        node_id: str,
        node_kind: NodeKind,
        label: str | None = None,
        paper_id: str | None = None,
        score: float | None = None,
    ) -> NodeRef:
        return NodeRef(
            id=node_id,
            kind=node_kind,
            label=label,
            paper_id=paper_id,
            score=score,
        )

    @staticmethod
    def _make_provenance(paper_id: str) -> ProvenanceRef:
        return ProvenanceRef(
            paper_id=paper_id,
            sentence_id=f"{paper_id}-s1",
        )

    @staticmethod
    def _split_limit(total: int, include_concepts: bool, include_statements: bool) -> tuple[int, int]:
        if include_concepts and include_statements:
            c = max(1, total // 2)
            s = max(1, total - c)
            return c, s
        if include_concepts:
            return max(1, total), 0
        if include_statements:
            return 0, max(1, total)
        return 0, 0

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)

    def _rank_with_embedder(self, query: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []

        if self.embedder is None:
            # deterministic fallback ranking
            return [max(0.5, 0.9 - idx * 0.08) for idx in range(len(candidates))]

        try:
            vectors = self.embedder.embed_texts([query] + candidates)
            if not isinstance(vectors, list) or not vectors:
                return [max(0.5, 0.9 - idx * 0.08) for idx in range(len(candidates))]

            query_vec = vectors[0]
            cand_vecs = vectors[1:]
            scores: list[float] = []
            for vec in cand_vecs:
                if not isinstance(vec, list):
                    scores.append(0.5)
                    continue
                cos = self._cosine_similarity(query_vec, vec)
                # map [-1, 1] -> [0, 1]
                scores.append((cos + 1.0) / 2.0)

            # keep numeric stability and minimum signal
            return [round(max(0.3, min(0.99, s)), 3) for s in scores]
        except Exception:
            return [max(0.5, 0.9 - idx * 0.08) for idx in range(len(candidates))]

    async def filter_papers(self, args: FilterPapersArgs) -> FilterPapersResult:
        def fallback() -> FilterPapersResult:
            pid = (args.filters.paper_ids or ["p-mock-1"])[0]
            paper = self._make_paper_ref(pid, f"Mock Paper for '{pid}'")
            return FilterPapersResult(papers=[paper], total_count=1)

        return await self._invoke_with_llm(
            operation="filter_papers",
            args=args,
            result_model=FilterPapersResult,
            fallback=fallback,
        )

    async def lexical_search(self, args: LexicalSearchArgs) -> SearchResult:
        def fallback() -> SearchResult:
            wants_concepts = SearchNodeType.CONCEPT in args.node_types
            wants_statements = SearchNodeType.STATEMENT in args.node_types
            concept_limit, statement_limit = self._split_limit(args.limit, wants_concepts, wants_statements)

            concept_texts = [f"{args.query}", f"{args.query} method", f"{args.query} concept"][:concept_limit]
            statement_texts = [
                f"Lexical match for '{args.query}'.",
                f"'{args.query}' appears in method section.",
                f"Keyword evidence for '{args.query}'.",
            ][:statement_limit]

            concept_scores = self._rank_with_embedder(args.query, concept_texts)
            statement_scores = self._rank_with_embedder(args.query, statement_texts)

            concepts = [
                self._make_concept(f"c-lex-{idx+1}", text, score=concept_scores[idx])
                for idx, text in enumerate(concept_texts)
            ]
            statements = [
                self._make_statement(f"s-lex-{idx+1}", text, score=statement_scores[idx])
                for idx, text in enumerate(statement_texts)
            ]

            return SearchResult(concepts=concepts if wants_concepts else [], statements=statements if wants_statements else [])

        return await self._invoke_with_llm(
            operation="lexical_search",
            args=args,
            result_model=SearchResult,
            fallback=fallback,
        )

    async def semantic_search(self, args: SemanticSearchArgs) -> SearchResult:
        def fallback() -> SearchResult:
            wants_concepts = SearchNodeType.CONCEPT in args.node_types
            wants_statements = SearchNodeType.STATEMENT in args.node_types
            concept_limit, statement_limit = self._split_limit(args.limit, wants_concepts, wants_statements)

            concept_texts = [
                f"{args.query}",
                f"Semantic abstraction of {args.query}",
                f"Related representation for {args.query}",
            ][:concept_limit]
            statement_texts = [
                f"Semantic match for '{args.query}'.",
                f"This statement is semantically close to '{args.query}'.",
                f"Equivalent claim related to '{args.query}'.",
            ][:statement_limit]

            concept_scores = self._rank_with_embedder(args.query, concept_texts)
            statement_scores = self._rank_with_embedder(args.query, statement_texts)

            concepts = [
                self._make_concept(f"c-sem-{idx+1}", text, score=concept_scores[idx])
                for idx, text in enumerate(concept_texts)
            ]
            statements = [
                self._make_statement(f"s-sem-{idx+1}", text, score=statement_scores[idx])
                for idx, text in enumerate(statement_texts)
            ]

            return SearchResult(concepts=concepts if wants_concepts else [], statements=statements if wants_statements else [])

        return await self._invoke_with_llm(
            operation="semantic_search",
            args=args,
            result_model=SearchResult,
            fallback=fallback,
        )

    async def semantic_constraint_search(self, args: SemanticConstraintSearchArgs) -> SearchResult:
        def fallback() -> SearchResult:
            wants_concepts = SearchNodeType.CONCEPT in args.node_types
            wants_statements = SearchNodeType.STATEMENT in args.node_types
            concept_limit, statement_limit = self._split_limit(args.limit, wants_concepts, wants_statements)

            mode = str(args.match_mode)
            concept_texts = [
                f"{args.query} ({mode})",
                f"Constrained concept for {args.query}",
                f"Filtered semantic candidate for {args.query}",
            ][:concept_limit]
            statement_texts = [
                f"Constrained semantic match for '{args.query}'.",
                f"Filtered statement candidate for '{args.query}'.",
                f"Constraint-compatible claim for '{args.query}'.",
            ][:statement_limit]

            concept_scores = self._rank_with_embedder(args.query, concept_texts)
            statement_scores = self._rank_with_embedder(args.query, statement_texts)

            concepts = [
                self._make_concept(f"c-constr-{idx+1}", text, score=concept_scores[idx])
                for idx, text in enumerate(concept_texts)
            ]
            statements = [
                self._make_statement(f"s-constr-{idx+1}", text, score=statement_scores[idx])
                for idx, text in enumerate(statement_texts)
            ]

            return SearchResult(concepts=concepts if wants_concepts else [], statements=statements if wants_statements else [])

        return await self._invoke_with_llm(
            operation="semantic_constraint_search",
            args=args,
            result_model=SearchResult,
            fallback=fallback,
        )

    async def resolve_concept_reference(
        self, args: ResolveConceptReferenceArgs
    ) -> ResolveConceptReferenceResult:
        def fallback() -> ResolveConceptReferenceResult:
            concept = self._make_concept("c-resolve-1", args.mention, paper_id=args.paper_id, score=0.9)
            return ResolveConceptReferenceResult(
                resolutions=[
                    Resolution(
                        concept=concept,
                        confidence=0.87,
                        rationale="Mock disambiguation based on mention and optional paper scope.",
                    )
                ]
            )

        return await self._invoke_with_llm(
            operation="resolve_concept_reference",
            args=args,
            result_model=ResolveConceptReferenceResult,
            fallback=fallback,
        )

    async def expand_context(self, args: ExpandContextArgs) -> ExpandContextResult:
        def fallback() -> ExpandContextResult:
            node = self._make_node_ref(
                node_id=args.node_id,
                node_kind=args.node_kind,
                label=f"{args.node_kind.value}-{args.node_id}",
                paper_id=args.paper_id,
                score=0.82,
            )

            artifacts = None
            if args.include_artifacts:
                artifacts = [
                    ContextNode(
                        id="ctx-artifact-1",
                        kind=NodeKind.CONCEPT,
                        label="mock-artifact-1",
                        node_type="Artifact",
                        paper_id=args.paper_id,
                        relation="mentions" if args.include_relations else None,
                    )
                ][: args.limit]

            propositions = None
            if args.include_propositions:
                propositions = [
                    ContextNode(
                        id="ctx-proposition-1",
                        kind=NodeKind.STATEMENT,
                        label="Mock proposition text.",
                        node_type="Argument",
                        paper_id=args.paper_id,
                        relation="supports" if args.include_relations else None,
                    )
                ][: args.limit]

            return ExpandContextResult(node=node, artifacts=artifacts, propositions=propositions)

        return await self._invoke_with_llm(
            operation="expand_context",
            args=args,
            result_model=ExpandContextResult,
            fallback=fallback,
        )

    async def expand_neighbors(self, args: ExpandNeighborsArgs) -> ExpandNeighborsResult:
        def fallback() -> ExpandNeighborsResult:
            source_node = self._make_node_ref(
                node_id=args.node_id,
                node_kind=args.node_kind,
                label=f"source-{args.node_id}",
                paper_id=args.paper_id,
                score=0.84,
            )

            candidates = [
                NeighborResult(
                    relation_type="related_to",
                    target_id="neighbor-c-1",
                    target_kind=NodeKind.CONCEPT,
                    target_label="neighbor concept",
                    score=0.78,
                    hop=min(args.hop_count, 1),
                ),
                NeighborResult(
                    relation_type="described_by",
                    target_id="neighbor-s-1",
                    target_kind=NodeKind.STATEMENT,
                    target_label="neighbor statement",
                    score=0.74,
                    hop=min(args.hop_count, 2),
                ),
                NeighborResult(
                    relation_type="published_in",
                    target_id="neighbor-p-1",
                    target_kind=NodeKind.PAPER,
                    target_label="neighbor paper",
                    score=0.7,
                    hop=min(args.hop_count, 2),
                ),
            ]

            if args.include_node_kinds:
                allowed = set(args.include_node_kinds)
                candidates = [n for n in candidates if n.target_kind in allowed]
            if args.relation_types:
                relation_set = set(args.relation_types)
                candidates = [n for n in candidates if n.relation_type in relation_set]

            return ExpandNeighborsResult(
                source_node=source_node,
                hop_count=args.hop_count,
                neighbors=candidates[: args.limit],
            )

        return await self._invoke_with_llm(
            operation="expand_neighbors",
            args=args,
            result_model=ExpandNeighborsResult,
            fallback=fallback,
        )

    async def get_attribution(self, args: GetAttributionArgs) -> GetAttributionResult:
        def fallback() -> GetAttributionResult:
            paper = self._make_paper_ref(args.paper_id or "p-mock-1", "Mock Attribution Paper")
            attributions = [
                NodeAttribution(
                    node_id=nid,
                    paper=paper,
                    location=DocumentLocation(
                        section_id="sec-mock-1",
                        section_title="Mock Section",
                        paragraph_id="para-mock-1",
                        sentence_id="sent-mock-1",
                        sentence_text=f"Node {nid} appears in this synthetic sentence.",
                    ),
                )
                for nid in args.node_ids
            ]
            return GetAttributionResult(attributions=attributions)

        return await self._invoke_with_llm(
            operation="get_attribution",
            args=args,
            result_model=GetAttributionResult,
            fallback=fallback,
        )

    async def get_provenance(self, args: GetProvenanceArgs) -> GetProvenanceResult:
        def fallback() -> GetProvenanceResult:
            return GetProvenanceResult(
                provenance=[
                    NodeProvenance(
                        node_id=nid,
                        provenance=[self._make_provenance(args.paper_id or "p-mock-1")],
                    )
                    for nid in args.node_ids
                ]
            )

        return await self._invoke_with_llm(
            operation="get_provenance",
            args=args,
            result_model=GetProvenanceResult,
            fallback=fallback,
        )
