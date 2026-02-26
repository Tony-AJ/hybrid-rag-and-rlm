from fastapi import APIRouter, Depends
from typing import List, Dict, Any
from app.dependencies import get_service
from app.service import HybridRetrievalService

router = APIRouter(prefix="/graph", tags=["graph"])

@router.get("")
def get_graph_data(service: HybridRetrievalService = Depends(get_service)):
    """
    Get all nodes and edges for graph visualization.
    Returns the complete graph structure.
    """
    nodes = []
    edges = []
    
    # Get all nodes from the graph
    for node_id in service.graph_db.graph.nodes():
        node = service.graph_db.get_node(node_id)
        if node:
            nodes.append({
                "id": node.id,
                "label": node.text[:50] + "..." if len(node.text) > 50 else node.text,
                "text": node.text,
                "metadata": node.metadata
            })
    
    # Get all edges from the graph
    for source, target, key, edge_data in service.graph_db.graph.edges(keys=True, data=True):
        edges.append({
            "id": edge_data.get("id"),
            "from": source,
            "to": target,
            "label": edge_data.get("type", ""),
            "weight": edge_data.get("weight", 1.0),
            "type": edge_data.get("type", "")
        })
    
    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges)
        }
    }


