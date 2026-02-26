from fastapi import APIRouter, Depends, HTTPException
from app.models import EdgeCreate, EdgeCreateResponse, EdgeGetResponse, EdgeUpdate, EdgeUpdateResponse, EdgeDeleteResponse
from app.service import HybridRetrievalService
from app.dependencies import get_service

router = APIRouter(prefix="/edges", tags=["edges"])

@router.post("", response_model=EdgeCreateResponse)
def create_edge(edge: EdgeCreate, service: HybridRetrievalService = Depends(get_service)):
    response = service.create_edge(edge)
    if not response:
        raise HTTPException(status_code=400, detail="Source or Target node not found")
    return response

@router.get("/{edge_id}", response_model=EdgeGetResponse)
def get_edge(edge_id: str, service: HybridRetrievalService = Depends(get_service)):
    edge = service.get_edge(edge_id)
    if not edge:
        raise HTTPException(status_code=404, detail="Edge not found")
    return edge

@router.put("/{edge_id}", response_model=EdgeUpdateResponse)
def update_edge(edge_id: str, update: EdgeUpdate, service: HybridRetrievalService = Depends(get_service)):
    response = service.update_edge(edge_id, update)
    if not response:
        raise HTTPException(status_code=404, detail="Edge not found")
    return response

@router.delete("/{edge_id}", response_model=EdgeDeleteResponse)
def delete_edge(edge_id: str, service: HybridRetrievalService = Depends(get_service)):
    response = service.delete_edge(edge_id)
    if not response:
        raise HTTPException(status_code=404, detail="Edge not found")
    return response
