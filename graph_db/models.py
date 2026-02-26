"""
Data models for Graph Database entities.

Classes:
    GraphNode: Represents a node in the graph
    GraphRelationship: Represents an edge/relationship between nodes
"""

import uuid
from typing import Optional, Dict, List, Any


class GraphNode:
    """
    Represents a node in the graph database.
    
    Attributes:
        id (str): Unique identifier (UUID)
        text (str): Text content of the node
        metadata (dict): Additional metadata
        embedding (list[float] | None): Optional vector embedding
    """
    
    def __init__(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
        node_id: Optional[str] = None
    ):
        """
        Initialize a GraphNode.
        
        Args:
            text: Text content of the node
            metadata: Additional metadata (default: {})
            embedding: Optional vector embedding
            node_id: Optional custom ID (generates UUID if not provided)
        """
        self.id = node_id or str(uuid.uuid4())
        self.text = text
        self.metadata = metadata or {}
        self.embedding = embedding
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert node to dictionary for serialization.
        
        Returns:
            Dictionary representation of the node
        """
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "embedding": self.embedding
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphNode":
        """
        Create GraphNode from dictionary.
        
        Args:
            data: Dictionary containing node data
            
        Returns:
            GraphNode instance
        """
        return cls(
            text=data["text"],
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
            node_id=data["id"]
        )
    
    def __repr__(self) -> str:
        return f"GraphNode(id={self.id[:8]}..., text={self.text[:30]}...)"


class GraphRelationship:
    """
    Represents an edge/relationship in the graph database.
    
    Attributes:
        id (str): Unique identifier (UUID)
        source (str): Source node ID
        target (str): Target node ID
        type (str): Relationship type (e.g., "references", "contains")
        weight (float): Edge weight for scoring
    """
    
    def __init__(
        self,
        source: str,
        target: str,
        rel_type: str,
        weight: float = 1.0,
        edge_id: Optional[str] = None
    ):
        """
        Initialize a GraphRelationship.
        
        Args:
            source: Source node ID
            target: Target node ID
            rel_type: Type of relationship
            weight: Edge weight (default: 1.0)
            edge_id: Optional custom ID (generates UUID if not provided)
        """
        self.id = edge_id or str(uuid.uuid4())
        self.source = source
        self.target = target
        self.type = rel_type
        self.weight = weight
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert relationship to dictionary for serialization.
        
        Returns:
            Dictionary representation of the relationship
        """
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "weight": self.weight
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphRelationship":
        """
        Create GraphRelationship from dictionary.
        
        Args:
            data: Dictionary containing relationship data
            
        Returns:
            GraphRelationship instance
        """
        return cls(
            source=data["source"],
            target=data["target"],
            rel_type=data["type"],
            weight=data.get("weight", 1.0),
            edge_id=data["id"]
        )
    
    def __repr__(self) -> str:
        return f"GraphRelationship(id={self.id[:8]}..., {self.source[:8]}... --[{self.type}]--> {self.target[:8]}...)"
