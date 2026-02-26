from typing import List, Optional
from fastapi import APIRouter, Depends
from app.models import (
    VectorSearchRequest, VectorSearchResponse,
    GraphTraversalResponse,
    HybridSearchRequest, HybridSearchResponse
)
from app.service import HybridRetrievalService
from app.dependencies import get_service

router = APIRouter(prefix="/search", tags=["search"])

@router.post("/vector", response_model=VectorSearchResponse)
def vector_search(request: VectorSearchRequest, service: HybridRetrievalService = Depends(get_service)):
    return service.vector_search(request.query_text, request.top_k, request.metadata_filter)

@router.get("/graph", response_model=GraphTraversalResponse)
def graph_traversal(start_id: str, depth: int = 2, type_filter: Optional[str] = None, service: HybridRetrievalService = Depends(get_service)):
    return service.graph_traversal(start_id, depth, type_filter)

@router.post("/hybrid", response_model=HybridSearchResponse)
def hybrid_search(request: HybridSearchRequest, service: HybridRetrievalService = Depends(get_service)):
    return service.hybrid_search(
        request.query_text,
        request.vector_weight,
        request.graph_weight,
        request.top_k
    )
