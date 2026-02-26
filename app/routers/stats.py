from fastapi import APIRouter, Depends
from app.dependencies import get_service
from app.service import HybridRetrievalService

router = APIRouter(prefix="/stats", tags=["stats"])

@router.get("")
def get_stats(service: HybridRetrievalService = Depends(get_service)):
    """
    Get system statistics including node count, edge count, and other metrics.
    """
    # Get graph stats
    graph_stats = service.graph_db.get_stats()
    
    # Get vector DB stats
    vector_count = 0
    try:
        # Try to get count from ChromaDB collection
        # ChromaDB stores data in a collection, we can try to get all IDs
        collection = service.vector_db.db._collection
        if collection:
            # Get all IDs to count documents
            result = collection.get()
            if result and 'ids' in result:
                vector_count = len(result['ids'])
    except Exception as e:
        # If we can't get the count, use graph node count as approximation
        # (assuming nodes with embeddings are in vector DB)
        vector_count = graph_stats["nodes"]
    
    return {
        "nodes": graph_stats["nodes"],
        "edges": graph_stats["edges"],
        "vector_documents": vector_count,
        "graph_stats": graph_stats
    }

