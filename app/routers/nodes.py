from fastapi import APIRouter, Depends, HTTPException
from app.models import NodeCreate, NodeResponse, NodeCreateResponse, NodeUpdate, NodeUpdateResponse, NodeDeleteResponse
from app.service import HybridRetrievalService
from app.dependencies import get_service

router = APIRouter(prefix="/nodes", tags=["nodes"])

@router.post("", response_model=NodeCreateResponse)
def create_node(node: NodeCreate, service: HybridRetrievalService = Depends(get_service)):
    return service.create_node(node)

@router.get("/{node_id}", response_model=NodeResponse)
def get_node(node_id: str, service: HybridRetrievalService = Depends(get_service)):
    node = service.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return node

@router.put("/{node_id}", response_model=NodeUpdateResponse)
def update_node(node_id: str, update: NodeUpdate, service: HybridRetrievalService = Depends(get_service)):
    response = service.update_node(node_id, update)
    if not response:
        raise HTTPException(status_code=404, detail="Node not found")
    return response

@router.delete("/{node_id}", response_model=NodeDeleteResponse)
def delete_node(node_id: str, service: HybridRetrievalService = Depends(get_service)):
    response = service.delete_node(node_id)
    if not response:
        raise HTTPException(status_code=404, detail="Node not found")
    return response
