from __future__ import annotations

from abc import ABC, abstractmethod

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


class ScholarlyKnowledgeGraphBackend(ABC):
    """Abstract SKG backend contract for MCP tools.

    Implement this interface with your concrete storage/query layer
    (SPARQL, property graph, SQL + vector index, etc.).
    """

    @abstractmethod
    async def filter_papers(self, args: FilterPapersArgs) -> FilterPapersResult:
        """Return papers matching metadata constraints."""

    @abstractmethod
    async def lexical_search(self, args: LexicalSearchArgs) -> SearchResult:
        """Unified lexical search over concept and/or statement nodes."""

    @abstractmethod
    async def resolve_concept_reference(
        self, args: ResolveConceptReferenceArgs
    ) -> ResolveConceptReferenceResult:
        """Disambiguate a concept mention using context and optional paper scope."""

    @abstractmethod
    async def semantic_search(self, args: SemanticSearchArgs) -> SearchResult:
        """Unified semantic search over concept and/or statement nodes."""

    @abstractmethod
    async def semantic_constraint_search(
        self, args: SemanticConstraintSearchArgs
    ) -> SearchResult:
        """Unified constrained semantic search over concept and/or statement nodes."""

    @abstractmethod
    async def expand_context(self, args: ExpandContextArgs) -> ExpandContextResult:
        """Return linked and neighboring context for a concept or statement node."""

    @abstractmethod
    async def expand_neighbors(self, args: ExpandNeighborsArgs) -> ExpandNeighborsResult:
        """Traverse neighbors from any node type with configurable hop count."""

    @abstractmethod
    async def get_attibution(self, args: GetAttibutionArgs) -> GetAttibutionResult:
        """Return attribution metadata for a node."""

    @abstractmethod
    async def get_provenance(self, args: GetProvenanceArgs) -> GetProvenanceResult:
        """Return provenance data for a node."""


class NotImplementedBackend(ScholarlyKnowledgeGraphBackend):
    """Placeholder backend that keeps the MCP surface discoverable before wiring storage."""

    async def _not_implemented(self, method: str) -> None:
        raise NotImplementedError(
            f"Backend method '{method}' is abstract and must be implemented in your SKG adapter."
        )

    async def filter_papers(self, args: FilterPapersArgs) -> FilterPapersResult:
        await self._not_implemented("filter_papers")

    async def lexical_search(self, args: LexicalSearchArgs) -> SearchResult:
        await self._not_implemented("lexical_search")

    async def resolve_concept_reference(
        self, args: ResolveConceptReferenceArgs
    ) -> ResolveConceptReferenceResult:
        await self._not_implemented("resolve_concept_reference")

    async def semantic_search(self, args: SemanticSearchArgs) -> SearchResult:
        await self._not_implemented("semantic_search")

    async def semantic_constraint_search(
        self, args: SemanticConstraintSearchArgs
    ) -> SearchResult:
        await self._not_implemented("semantic_constraint_search")

    async def expand_context(self, args: ExpandContextArgs) -> ExpandContextResult:
        await self._not_implemented("expand_context")

    async def expand_neighbors(self, args: ExpandNeighborsArgs) -> ExpandNeighborsResult:
        await self._not_implemented("expand_neighbors")

    async def get_attibution(self, args: GetAttibutionArgs) -> GetAttibutionResult:
        await self._not_implemented("get_attibution")

    async def get_provenance(self, args: GetProvenanceArgs) -> GetProvenanceResult:
        await self._not_implemented("get_provenance")
