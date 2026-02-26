"""
Core Graph Database implementation using NetworkX.

Provides CRUD operations, traversal, scoring, and persistence for graph data.
"""

import json
import networkx as nx
from collections import deque
from typing import Optional, Dict, List, Any
from pathlib import Path

from graph_db.models import GraphNode, GraphRelationship


class GraphDatabase:
    """
    Graph database using NetworkX MultiDiGraph.
    
    Provides:
        - Node CRUD operations
        - Edge CRUD operations
        - Graph traversal (BFS)
        - Graph-based relevance scoring
        - JSON persistence
    """
    
    def __init__(self, db_path: str = "db/graph_data.json", auto_persist: bool = True):
        """
        Initialize graph database.
        
        Args:
            db_path: Path to persist graph data (default: db/graph_data.json)
            auto_persist: If True, automatically load existing data on init
        """
        self.graph = nx.MultiDiGraph()
        self._edge_id_map: Dict[str, tuple] = {}  # edge_id -> (source, target, key)
        self.db_path = db_path
        self.auto_persist = auto_persist
        
        # Auto-load if file exists
        if auto_persist and Path(db_path).exists():
            try:
                self.load(db_path)
            except Exception:
                pass  # Start fresh if load fails
    
    # ==================== Persistence ====================
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save graph to JSON file.
        
        Args:
            path: File path to save to (defaults to self.db_path)
            
        Raises:
            IOError: If file cannot be written
        """
        if path is None:
            path = self.db_path
        data = {
            "nodes": [],
            "edges": []
        }
        
        # Serialize nodes
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            node = GraphNode(
                text=node_data["text"],
                metadata=node_data["metadata"],
                embedding=node_data.get("embedding"),
                node_id=node_id
            )
            data["nodes"].append(node.to_dict())
        
        # Serialize edges
        for source, target, key, edge_data in self.graph.edges(keys=True, data=True):
            edge = GraphRelationship(
                source=source,
                target=target,
                rel_type=edge_data["type"],
                weight=edge_data["weight"],
                edge_id=edge_data["id"]
            )
            data["edges"].append(edge.to_dict())
        
        # Write to file
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load(self, path: Optional[str] = None) -> None:
        """
        Load graph from JSON file.
        
        Args:
            path: File path to load from (defaults to self.db_path)
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
        """
        if path is None:
            path = self.db_path
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Clear existing graph
        self.graph.clear()
        self._edge_id_map.clear()
        
        # Load nodes
        for node_data in data.get("nodes", []):
            node = GraphNode.from_dict(node_data)
            self.graph.add_node(
                node.id,
                text=node.text,
                metadata=node.metadata,
                embedding=node.embedding
            )
        
        # Load edges
        for edge_data in data.get("edges", []):
            edge = GraphRelationship.from_dict(edge_data)
            key = self.graph.add_edge(
                edge.source,
                edge.target,
                id=edge.id,
                type=edge.type,
                weight=edge.weight
            )
            self._edge_id_map[edge.id] = (edge.source, edge.target, key)
    
    def persist(self) -> None:
        """
        Persist graph to default database path.
        
        Convenience method equivalent to save() with no arguments.
        """
        self.save()
    
    # ==================== Node CRUD ====================
    
    def create_node(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
        node_id: Optional[str] = None
    ) -> GraphNode:
        """
        Create a new node in the graph.
        
        Args:
            text: Text content of the node
            metadata: Additional metadata
            embedding: Optional vector embedding
            node_id: Optional specific ID for the node
            
        Returns:
            Created GraphNode
        """
        node = GraphNode(text=text, metadata=metadata, embedding=embedding, node_id=node_id)
        self.graph.add_node(
            node.id,
            text=node.text,
            metadata=node.metadata,
            embedding=node.embedding
        )
        if self.auto_persist:
            self.persist()
        return node
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """
        Get node by ID.
        
        Args:
            node_id: Node identifier
            
        Returns:
            GraphNode if found, None otherwise
        """
        if node_id not in self.graph.nodes:
            return None
        
        node_data = self.graph.nodes[node_id]
        return GraphNode(
            text=node_data["text"],
            metadata=node_data["metadata"],
            embedding=node_data.get("embedding"),
            node_id=node_id
        )
    
    def update_node(
        self,
        node_id: str,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None
    ) -> bool:
        """
        Update node attributes.
        
        Args:
            node_id: Node identifier
            text: New text (optional)
            metadata: New metadata (optional)
            embedding: New embedding (optional)
            
        Returns:
            True if updated, False if node doesn't exist
        """
        if node_id not in self.graph.nodes:
            return False
        
        if text is not None:
            self.graph.nodes[node_id]["text"] = text
        
        if metadata is not None:
            self.graph.nodes[node_id]["metadata"] = metadata
        
        if embedding is not None:
            self.graph.nodes[node_id]["embedding"] = embedding
        
        if self.auto_persist:
            self.persist()
        return True
    
    def delete_node(self, node_id: str) -> bool:
        """
        Delete node and all connected edges.
        
        Args:
            node_id: Node identifier
            
        Returns:
            True if deleted, False if node doesn't exist
        """
        if node_id not in self.graph.nodes:
            return False
        
        # Remove edge mappings
        edges_to_remove = []
        for edge_id, (source, target, key) in self._edge_id_map.items():
            if source == node_id or target == node_id:
                edges_to_remove.append(edge_id)
        
        for edge_id in edges_to_remove:
            del self._edge_id_map[edge_id]
        
        # Remove node (automatically removes edges)
        self.graph.remove_node(node_id)
        if self.auto_persist:
            self.persist()
        return True
    
    # ==================== Edge CRUD ====================
    
    def create_edge(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        weight: float = 1.0
    ) -> Optional[GraphRelationship]:
        """
        Create a new edge between nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            rel_type: Relationship type
            weight: Edge weight
            
        Returns:
            Created GraphRelationship if successful, None if nodes don't exist
        """
        if source_id not in self.graph.nodes or target_id not in self.graph.nodes:
            return None
        
        edge = GraphRelationship(
            source=source_id,
            target=target_id,
            rel_type=rel_type,
            weight=weight
        )
        
        key = self.graph.add_edge(
            source_id,
            target_id,
            id=edge.id,
            type=edge.type,
            weight=edge.weight
        )
        
        self._edge_id_map[edge.id] = (source_id, target_id, key)
        if self.auto_persist:
            self.persist()
        return edge
    
    def get_edge(self, edge_id: str) -> Optional[GraphRelationship]:
        """
        Get edge by ID.
        
        Args:
            edge_id: Edge identifier
            
        Returns:
            GraphRelationship if found, None otherwise
        """
        if edge_id not in self._edge_id_map:
            return None
        
        source, target, key = self._edge_id_map[edge_id]
        edge_data = self.graph[source][target][key]
        
        return GraphRelationship(
            source=source,
            target=target,
            rel_type=edge_data["type"],
            weight=edge_data["weight"],
            edge_id=edge_data["id"]
        )
    
    def delete_edge(self, edge_id: str) -> bool:
        """
        Delete an edge.
        
        Args:
            edge_id: Edge identifier
            
        Returns:
            True if deleted, False if edge doesn't exist
        """
        if edge_id not in self._edge_id_map:
            return False
        
        source, target, key = self._edge_id_map[edge_id]
        self.graph.remove_edge(source, target, key)
        del self._edge_id_map[edge_id]
        if self.auto_persist:
            self.persist()
        return True
    
    # ==================== Graph Operations ====================
    
    def traverse(self, start_id: str, depth: int) -> List[str]:
        """
        Traverse graph using BFS up to specified depth.
        
        Args:
            start_id: Starting node ID
            depth: Maximum traversal depth
            
        Returns:
            List of node IDs reachable within depth
        """
        if start_id not in self.graph.nodes:
            return []
        
        visited = set()
        result = []
        queue = deque([(start_id, 0)])  # (node_id, current_depth)
        
        while queue:
            node_id, current_depth = queue.popleft()
            
            if node_id in visited:
                continue
            
            visited.add(node_id)
            result.append(node_id)
            
            # Only explore neighbors if we haven't reached max depth
            if current_depth < depth:
                # Get both outgoing and incoming neighbors (for directed graph)
                neighbors = set(self.graph.successors(node_id)) | set(self.graph.predecessors(node_id))
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append((neighbor, current_depth + 1))
        
        return result
    
    def compute_graph_scores(
        self,
        start_id: str,
        depth: int
    ) -> Dict[str, float]:
        """
        Compute graph-based relevance scores for nodes.
        
        Score formula: (total_edge_weight) / (hop_distance + 1)
        
        Args:
            start_id: Starting node ID
            depth: Maximum traversal depth
            
        Returns:
            Dictionary mapping node_id to relevance score
        """
        if start_id not in self.graph.nodes:
            return {}
        
        scores = {}
        queue = deque([(start_id, 0, 0)])  # (node_id, hop_distance, accumulated_weight)
        visited = {}  # node_id -> (min_distance, max_weight)
        
        while queue:
            node_id, hop_distance, accumulated_weight = queue.popleft()
            
            # Skip if we've exceeded max depth
            if hop_distance > depth:
                continue
            
            # Update visited tracking
            if node_id not in visited:
                visited[node_id] = (hop_distance, accumulated_weight)
            else:
                prev_distance, prev_weight = visited[node_id]
                # Keep the path with shorter distance or higher weight
                if hop_distance < prev_distance or (hop_distance == prev_distance and accumulated_weight > prev_weight):
                    visited[node_id] = (hop_distance, accumulated_weight)
                else:
                    continue  # Skip this path
            
            # Calculate score for this node
            if hop_distance == 0:
                scores[node_id] = float('inf')  # Starting node has infinite relevance
            else:
                scores[node_id] = accumulated_weight / hop_distance
            
            # Explore neighbors if within depth
            if hop_distance < depth:
                # Outgoing edges
                for neighbor in self.graph.successors(node_id):
                    for key, edge_data in self.graph[node_id][neighbor].items():
                        new_weight = accumulated_weight + edge_data.get("weight", 1.0)
                        queue.append((neighbor, hop_distance + 1, new_weight))
                
                # Incoming edges
                for neighbor in self.graph.predecessors(node_id):
                    for key, edge_data in self.graph[neighbor][node_id].items():
                        new_weight = accumulated_weight + edge_data.get("weight", 1.0)
                        queue.append((neighbor, hop_distance + 1, new_weight))
        
        return scores
    
    # ==================== Utility Methods ====================
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get graph statistics.
        
        Returns:
            Dictionary with node count and edge count
        """
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges()
        }
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"GraphDatabase(nodes={stats['nodes']}, edges={stats['edges']})"
