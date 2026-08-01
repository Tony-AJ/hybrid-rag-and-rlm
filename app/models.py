from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# ==================== Node Models ====================

class NodeCreate(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    regen_embedding: bool = True

class NodeResponse(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    edges: Optional[List[Dict[str, Any]]] = None

class NodeCreateResponse(BaseModel):
    status: str
    id: str
    embedding_dim: Optional[int] = None

class NodeUpdate(BaseModel):
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    regen_embedding: bool = False

class NodeUpdateResponse(BaseModel):
    status: str
    id: str
    embedding_regenerated: bool

class NodeDeleteResponse(BaseModel):
    status: str
    id: str
    removed_edges_count: int

# ==================== Edge Models ====================

class EdgeCreate(BaseModel):
    source: str
    target: str
    type: str
    weight: float = 1.0

class EdgeResponse(BaseModel):
    status: Optional[str] = None
    edge_id: str
    source: str
    target: str
    type: Optional[str] = None # Make optional to handle create response which might not have it or have it differently
    weight: Optional[float] = None
    # Adjusting to match exact Create Edge response: {status, edge_id, source, target}
    # And Get Edge response: {edge_id, source, target, type, weight}

class EdgeCreateResponse(BaseModel):
    status: str
    edge_id: str
    source: str
    target: str

class EdgeGetResponse(BaseModel):
    edge_id: str
    source: str
    target: str
    type: str
    weight: float

class EdgeUpdate(BaseModel):
    weight: float

class EdgeUpdateResponse(BaseModel):
    status: str
    edge_id: str
    new_weight: float

class EdgeDeleteResponse(BaseModel):
    status: str
    edge_id: str

# ==================== Search Models ====================

class VectorSearchRequest(BaseModel):
    query_text: str
    top_k: int = 5
    metadata_filter: Optional[Dict[str, Any]] = None

class VectorSearchResultItem(BaseModel):
    id: str
    vector_score: float

class VectorSearchResponse(BaseModel):
    query_text: str
    results: List[VectorSearchResultItem]

class GraphTraversalNode(BaseModel):
    id: str
    hop: int
    edge: Optional[str] = None
    weight: Optional[float] = None
    edge_path: Optional[List[str]] = None
    weights: Optional[List[float]] = None

class GraphTraversalResponse(BaseModel):
    start_id: str
    depth: int
    nodes: List[GraphTraversalNode]

class HybridSearchRequest(BaseModel):
    query_text: str
    vector_weight: float = 0.6
    graph_weight: float = 0.4
    top_k: int = 5
    max_depth: int = 2
    path_distance_penalty: float = 0.1


class RetrievalReason(BaseModel):
    """Structured, machine-parseable retrieval explanation."""
    rank: Optional[int] = None          # Position in the final fused ranking (1-indexed)
    vector_rank: Optional[int] = None   # Position in vector search results (1-indexed), None if not vector-retrieved
    graph_rank: Optional[int] = None    # Position in graph expansion order (1-indexed), None if not graph-retrieved
    hop: int = 0                        # Graph hop distance from nearest vector-hit node
    edge: Optional[str] = None          # Edge type used to reach this node (e.g. "next", "goal_achieved")
    retrieval_method: List[str] = Field(default_factory=list)  # e.g. ["vector", "graph"]


class RetrievalMetadata(BaseModel):
    """Per-query provenance for benchmarking and reproducibility."""
    pipeline_version: str = "2.0"
    timestamp: str                              # ISO-8601 UTC
    max_depth: int = 2
    path_distance_penalty: float = 0.1
    normalization: str = "min-max"
    distance_conversion: str = "cosine = 1 - (L2^2 / 2)"


class HybridSearchResultItem(BaseModel):
    """Research-grade hybrid search result with full explainability."""
    node_id: str
    raw_vector_score: Optional[float] = None         # True cosine similarity = 1 - (L2²/2). None if graph-only.
    normalized_vector_score: Optional[float] = None   # Min-max normalized within this query's results. None if graph-only.
    graph_score: float = 0.0                          # 1/(hop+1), already in (0,1]
    vector_contribution: float = 0.0                  # α × normalized_vector_score
    graph_contribution: float = 0.0                   # β × graph_score
    penalty_contribution: float = 0.0                 # γ × (hop / max_depth)
    # Exact identity: vector_contribution + graph_contribution - penalty_contribution == unclamped_final_score
    unclamped_final_score: float = 0.0
    final_score: float = 0.0                          # max(0, unclamped_final_score) — the value used for ranking
    retrieval_sources: List[str] = Field(default_factory=list)  # ["vector"], ["graph"], or ["vector", "graph"]
    reason: RetrievalReason = Field(default_factory=RetrievalReason)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    text: str = ""
    info: Dict[str, Any] = Field(default_factory=dict)  # Optional extensibility field for debugging/experiments


class HybridSearchResponse(BaseModel):
    query_text: str
    vector_weight: float
    graph_weight: float
    pipeline_version: str = "2.0"
    retrieval_metadata: Optional[RetrievalMetadata] = None
    results: List[HybridSearchResultItem]


# Legacy model kept for backward compatibility with main.py CLI
class HybridSearchResult(BaseModel):
    node_id: str
    final_score: float
    cosine_similarity: float
    graph_score: float
    text: str

