from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


ID = str
Score = float
Text = str


class NodeKind(str, Enum):
    CONCEPT = "concept"
    STATEMENT = "statement"
    PAPER = "paper"


class MatchMode(str, Enum):
    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class SearchNodeType(str, Enum):
    CONCEPT = "concept"
    STATEMENT = "statement"


class MentionGranularity(str, Enum):
    TOKEN = "token"
    SPAN = "span"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SECTION = "section"


JSONScalar = Union[str, int, float, bool]
JSONValue = Union[JSONScalar, List[JSONScalar]]


class SparqlTemplateFilter(BaseModel):
    """Small SPARQL WHERE-fragment filter with safe named parameters.

    The fragment is inserted inside the relevant tool query and may refer to
    variables exposed by that query, such as ?paper, ?node, ?type, and ?label.
    Parameter placeholders use {{name}} and are replaced with escaped literals,
    IRIs, or lists.
    """

    where: str = Field(
        description=(
            "SPARQL WHERE fragment. Example: "
            "'?node sudo:mentions ?concept . FILTER(?concept IN ({{concepts}}))'"
        )
    )
    params: Dict[str, JSONValue] = Field(default_factory=dict)
    graph: Optional[Literal["meta", "struct", "sudo", "prov", "concept"]] = None


class YearRange(BaseModel):
    start: Optional[int] = None
    end: Optional[int] = None


class NumericRange(BaseModel):
    min: Optional[int] = None
    max: Optional[int] = None


class PaperFilter(BaseModel):
    paper_ids: Optional[List[ID]] = None
    years: Optional[List[int]] = None
    year_range: Optional[YearRange] = None
    venues: Optional[List[str]] = None
    authors: Optional[List[str]] = None
    domains: Optional[List[str]] = None
    citation_count_range: Optional[NumericRange] = None
    metadata: Optional[Dict[str, JSONValue]] = None
    sparql_filters: Optional[List[SparqlTemplateFilter]] = None


class ConceptFilter(BaseModel):
    paper_ids: Optional[List[ID]] = None
    concept_types: Optional[List[str]] = None
    canonical_only: bool = False
    paper_local_only: bool = False
    sparql_filters: Optional[List[SparqlTemplateFilter]] = None


class StatementFilter(BaseModel):
    paper_ids: Optional[List[ID]] = None
    statement_types: Optional[List[str]] = None
    section_types: Optional[List[str]] = None
    rhetorical_roles: Optional[List[str]] = None
    sparql_filters: Optional[List[SparqlTemplateFilter]] = None


class PaperRef(BaseModel):
    id: ID
    title: str
    year: Optional[int] = None
    venue: Optional[str] = None


class ProvenanceRef(BaseModel):
    paper_id: ID
    section_id: Optional[ID] = None
    section_title: Optional[str] = None
    paragraph_id: Optional[ID] = None
    sentence_id: Optional[ID] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    exact_text: Optional[str] = None


class ConceptResult(BaseModel):
    id: ID
    label: str
    aliases: Optional[List[str]] = None
    concept_type: Optional[str] = None
    is_canonical: Optional[bool] = None
    paper_id: Optional[ID] = None
    score: Optional[Score] = None
    summary: Optional[str] = None
    provenance: Optional[List[ProvenanceRef]] = None


class StatementResult(BaseModel):
    id: ID
    text: str
    statement_type: Optional[str] = None
    rhetorical_role: Optional[str] = None
    paper_id: Optional[ID] = None
    score: Optional[Score] = None
    provenance: Optional[List[ProvenanceRef]] = None


class NeighborResult(BaseModel):
    relation_type: str
    target_id: ID
    target_kind: NodeKind
    target_label: Optional[str] = None
    score: Optional[Score] = None
    hop: Optional[int] = None


class NodeRef(BaseModel):
    id: ID
    kind: NodeKind
    label: Optional[str] = None
    paper_id: Optional[ID] = None
    score: Optional[Score] = None


class Resolution(BaseModel):
    concept: ConceptResult
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: Optional[str] = None


class ConceptCondition(BaseModel):
    related_to_concepts: Optional[List[str]] = None
    exclude_concepts: Optional[List[str]] = None
    required_statement_patterns: Optional[List[str]] = None
    paper_filters: Optional[PaperFilter] = None


class StatementCondition(BaseModel):
    must_reference_concept_ids: Optional[List[ID]] = None
    exclude_concept_ids: Optional[List[ID]] = None
    required_relations: Optional[List[str]] = None
    rhetorical_roles: Optional[List[str]] = None


# Request models
class FilterPapersArgs(BaseModel):
    filters: PaperFilter = Field(default_factory=PaperFilter)
    limit: int = Field(default=20, ge=1, le=200)
    offset: int = Field(default=0, ge=0)


class LexicalSearchArgs(BaseModel):
    query: str
    node_types: List[SearchNodeType] = Field(
        default_factory=lambda: [SearchNodeType.CONCEPT, SearchNodeType.STATEMENT],
        min_length=1,
    )
    concept_filters: Optional[ConceptFilter] = None
    statement_filters: Optional[StatementFilter] = None
    paper_filters: Optional[PaperFilter] = None
    limit: int = Field(default=10, ge=1, le=100)
    include_aliases: bool = True
    fuzzy: bool = True


class SemanticSearchArgs(BaseModel):
    query: str
    node_types: List[SearchNodeType] = Field(
        default_factory=lambda: [SearchNodeType.CONCEPT, SearchNodeType.STATEMENT],
        min_length=1,
    )
    concept_filters: Optional[ConceptFilter] = None
    statement_filters: Optional[StatementFilter] = None
    paper_filters: Optional[PaperFilter] = None
    limit: int = Field(default=10, ge=1, le=100)
    embedding_model: Optional[str] = None
    min_score: Optional[float] = None


class SemanticConstraintSearchArgs(BaseModel):
    query: str
    node_types: List[SearchNodeType] = Field(
        default_factory=lambda: [SearchNodeType.CONCEPT, SearchNodeType.STATEMENT],
        min_length=1,
    )
    match_mode: MatchMode = MatchMode.HYBRID
    concept_filters: Optional[ConceptFilter] = None
    statement_filters: Optional[StatementFilter] = None
    paper_filters: Optional[PaperFilter] = None
    concept_semantic_conditions: Optional[ConceptCondition] = None
    statement_semantic_conditions: Optional[StatementCondition] = None
    limit: int = Field(default=10, ge=1, le=100)
    min_score: Optional[float] = None


class ResolveConceptReferenceArgs(BaseModel):
    mention: str
    context_text: Optional[str] = None
    paper_id: Optional[ID] = None
    candidate_concept_ids: Optional[List[ID]] = None
    limit: int = Field(default=5, ge=1, le=20)


class ExpandContextArgs(BaseModel):
    node_id: ID
    node_kind: NodeKind
    paper_id: Optional[ID] = None
    include_linked_nodes: bool = True
    include_neighbor_nodes: bool = True
    include_document_context: bool = True
    include_paper_usage: bool = True
    max_linked_nodes: int = Field(default=8, ge=1, le=100)
    max_neighbor_nodes: int = Field(default=8, ge=1, le=100)


class ExpandNeighborsArgs(BaseModel):
    node_id: ID
    node_kind: NodeKind
    paper_id: Optional[ID] = None
    relation_types: Optional[List[str]] = None
    hop_count: int = Field(default=1, ge=1, le=5)
    limit: int = Field(default=20, ge=1, le=200)
    include_node_kinds: Optional[List[NodeKind]] = None


class GetAttibutionArgs(BaseModel):
    node_id: ID
    node_kind: NodeKind
    paper_id: Optional[ID] = None
    max_items: int = Field(default=10, ge=1, le=100)


class GetProvenanceArgs(BaseModel):
    node_id: ID
    node_kind: NodeKind
    paper_id: Optional[ID] = None
    max_items: int = Field(default=10, ge=1, le=200)


# Response models
class FilterPapersResult(BaseModel):
    papers: List[PaperRef]
    total_count: Optional[int] = None


class ConceptsResult(BaseModel):
    concepts: List[ConceptResult]


class SearchResult(BaseModel):
    concepts: List[ConceptResult] = Field(default_factory=list)
    statements: List[StatementResult] = Field(default_factory=list)


class ResolveConceptReferenceResult(BaseModel):
    resolutions: List[Resolution]


class StatementsResult(BaseModel):
    statements: List[StatementResult]


class PaperUsageItem(BaseModel):
    paper: PaperRef
    usage_summary: Optional[str] = None


class ExpandContextResult(BaseModel):
    node: NodeRef
    linked_nodes: Optional[List[NodeRef]] = None
    neighbor_nodes: Optional[List[NodeRef]] = None
    paper_usage: Optional[List[PaperUsageItem]] = None
    document_context: Optional[DocumentContext] = None


class DocumentContext(BaseModel):
    preceding_text: Optional[str] = None
    following_text: Optional[str] = None
    section_title: Optional[str] = None


class ExpandNeighborsResult(BaseModel):
    source_node: NodeRef
    hop_count: int
    neighbors: List[NeighborResult]


class AttributionItem(BaseModel):
    source_paper: Optional[PaperRef] = None
    source_author: Optional[str] = None
    attribution_type: Optional[str] = None
    statement: Optional[str] = None
    confidence: Optional[float] = None


class GetAttibutionResult(BaseModel):
    node: NodeRef
    attributions: List[AttributionItem]


class GetProvenanceResult(BaseModel):
    node: NodeRef
    provenance: List[ProvenanceRef]
