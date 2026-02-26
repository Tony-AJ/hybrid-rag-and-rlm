"""
Graph Database Module

A lightweight graph database implementation using NetworkX for hybrid Vector + Graph retrieval.

This module provides:
- Node and edge data models
- CRUD operations for nodes and relationships
- Graph traversal (BFS)
- Graph-based relevance scoring
- JSON persistence

Usage:
    from graph_db import GraphDatabase, GraphNode, GraphRelationship
    
    db = GraphDatabase()
    node = db.create_node("Document text", {"source": "file.pdf"})
    db.save("graph.json")
"""

from graph_db.models import GraphNode, GraphRelationship
from graph_db.graph_db import GraphDatabase

__all__ = ["GraphNode", "GraphRelationship", "GraphDatabase"]
__version__ = "1.0.0"
