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

class HybridSearchResultItem(BaseModel):
    id: str
    vector_score: float
    graph_score: float
    final_score: float
    info: Dict[str, Any]

class HybridSearchResponse(BaseModel):
    query_text: str
    vector_weight: float
    graph_weight: float
    results: List[HybridSearchResultItem]

class HybridSearchResult(BaseModel):
    node_id: str
    final_score: float
    cosine_similarity: float
    graph_score: float
    text: str
