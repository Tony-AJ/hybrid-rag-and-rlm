#!/usr/bin/env python3
"""
Graph Database - Quick Reference Guide
========================================

This file contains quick copy-paste snippets for common operations.
"""

from graph_db import GraphDatabase, GraphNode, GraphRelationship

# =============================================================================
# INITIALIZATION
# =============================================================================

# Create new database
db = GraphDatabase()

# Load existing database
db = GraphDatabase()
db.load("my_graph.json")


# =============================================================================
# NODE OPERATIONS
# =============================================================================

# Create a simple node
node = db.create_node(
    text="Sample text content",
    metadata={"source": "example", "category": "demo"}
)

# Create node with embedding (for vector integration)
node_with_embedding = db.create_node(
    text="Document with embedding",
    metadata={"page": 1},
    embedding=[0.1, 0.2, 0.3, 0.4]  # Your vector embedding
)

# Get node by ID
retrieved_node = db.get_node(node_id="some-uuid")
if retrieved_node:
    print(f"Text: {retrieved_node.text}")
    print(f"Metadata: {retrieved_node.metadata}")

# Update node
db.update_node(
    node_id="some-uuid",
    text="New text",  # optional
    metadata={"updated": True},  # optional
    embedding=[0.5, 0.6, 0.7]  # optional
)

# Delete node (also deletes connected edges)
success = db.delete_node("some-uuid")


# =============================================================================
# EDGE OPERATIONS
# =============================================================================

# Create relationship between nodes
edge = db.create_edge(
    source_id=node1_id,
    target_id=node2_id,
    rel_type="references",  # Custom type
    weight=2.5  # Higher weight = stronger relationship
)

# Common relationship types:
# - "references", "mentions", "cites"
# - "precedes", "follows", "next"
# - "contains", "part_of", "belongs_to"
# - "related_to", "similar_to", "linked_to"
# - "uses", "requires", "depends_on"

# Get edge by ID
retrieved_edge = db.get_edge(edge_id="some-uuid")
if retrieved_edge:
    print(f"Type: {retrieved_edge.type}")
    print(f"Weight: {retrieved_edge.weight}")

# Delete edge
success = db.delete_edge("edge-uuid")


# =============================================================================
# GRAPH TRAVERSAL
# =============================================================================

# Find all nodes reachable within N hops (BFS)
reachable = db.traverse(
    start_id="starting-node-uuid",
    depth=2  # Traverse up to 2 hops away
)
# Returns: list of node IDs

# Example: Find related documents
for node_id in reachable:
    node = db.get_node(node_id)
    print(f"Related: {node.text}")


# =============================================================================
# GRAPH SCORING
# =============================================================================

# Compute relevance scores based on graph structure
scores = db.compute_graph_scores(
    start_id="starting-node-uuid",
    depth=2
)
# Returns: dict mapping node_id -> score

# Sort by relevance
sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
for node_id, score in sorted_nodes[:10]:  # Top 10
    node = db.get_node(node_id)
    print(f"Score {score:.2f}: {node.text}")


# =============================================================================
# PERSISTENCE
# =============================================================================

# Save graph to JSON file
db.save("knowledge_graph.json")

# Load graph from JSON file
db = GraphDatabase()
db.load("knowledge_graph.json")

# Get statistics
stats = db.get_stats()
print(f"Nodes: {stats['nodes']}, Edges: {stats['edges']}")


# =============================================================================
# HYBRID SEARCH PATTERN (Vector + Graph)
# =============================================================================

def hybrid_search(query_embedding, vector_db, graph_db, k=10, depth=2):
    """
    Combine vector similarity with graph relationships.
    
    Args:
        query_embedding: Query vector
        vector_db: Vector database (e.g., ChromaDB)
        graph_db: Graph database
        k: Number of initial vector results
        depth: Graph traversal depth
        
    Returns:
        List of (node_id, combined_score) tuples
    """
    # Step 1: Vector search
    vector_results = vector_db.search(query_embedding, k=k)
    # Returns: [(node_id, vector_score), ...]
    
    # Step 2: Expand with graph
    all_candidates = set()
    for node_id, _ in vector_results:
        related = graph_db.traverse(node_id, depth=depth)
        all_candidates.update(related)
    
    # Step 3: Compute graph scores
    graph_scores = {}
    for node_id, _ in vector_results:
        scores = graph_db.compute_graph_scores(node_id, depth=depth)
        for gid, gscore in scores.items():
            if gid not in graph_scores or scores[gid] > graph_scores[gid]:
                graph_scores[gid] = gscore
    
    # Step 4: Combine scores
    vector_score_map = dict(vector_results)
    combined = []
    for node_id in all_candidates:
        vs = vector_score_map.get(node_id, 0)
        gs = graph_scores.get(node_id, 0)
        if gs == float('inf'):
            gs = 10.0  # Cap infinite scores
        combined_score = 0.7 * vs + 0.3 * gs  # Weighted combination
        combined.append((node_id, combined_score))
    
    # Sort by combined score
    combined.sort(key=lambda x: x[1], reverse=True)
    return combined


# =============================================================================
# KNOWLEDGE GRAPH BUILDING
# =============================================================================

def build_document_graph(documents):
    """
    Build a knowledge graph from a list of documents.
    
    Args:
        documents: List of dicts with 'id', 'text', 'metadata'
        
    Returns:
        GraphDatabase instance
    """
    db = GraphDatabase()
    
    # Add documents as nodes
    node_map = {}
    for doc in documents:
        node = db.create_node(
            text=doc['text'],
            metadata=doc.get('metadata', {}),
            embedding=doc.get('embedding')
        )
        node_map[doc['id']] = node.id
    
    # Add relationships
    # Example 1: Sequential documents
    for i in range(len(documents) - 1):
        db.create_edge(
            node_map[documents[i]['id']],
            node_map[documents[i + 1]['id']],
            rel_type="precedes",
            weight=1.0
        )
    
    # Example 2: Concept linking (if documents share tags)
    from collections import defaultdict
    tag_to_docs = defaultdict(list)
    for doc in documents:
        for tag in doc.get('metadata', {}).get('tags', []):
            tag_to_docs[tag].append(doc['id'])
    
    for tag, doc_ids in tag_to_docs.items():
        # Link documents with same tag
        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                db.create_edge(
                    node_map[doc_ids[i]],
                    node_map[doc_ids[j]],
                    rel_type="related_by_" + tag,
                    weight=1.5
                )
    
    return db


# =============================================================================
# EXAMPLE: COMPLETE WORKFLOW
# =============================================================================

if __name__ == "__main__":
    # Initialize
    db = GraphDatabase()
    
    # Create knowledge graph
    doc1 = db.create_node("Introduction to Python", {"section": "intro", "page": 1})
    doc2 = db.create_node("Python Data Structures", {"section": "basics", "page": 5})
    doc3 = db.create_node("NumPy Arrays", {"section": "libraries", "page": 15})
    
    concept = db.create_node("Python Programming", {"type": "concept"})
    
    # Add relationships
    db.create_edge(doc1.id, concept.id, "introduces", weight=3.0)
    db.create_edge(doc2.id, concept.id, "explains", weight=2.5)
    db.create_edge(doc3.id, concept.id, "uses", weight=2.0)
    db.create_edge(doc1.id, doc2.id, "precedes", weight=1.5)
    db.create_edge(doc2.id, doc3.id, "precedes", weight=1.5)
    
    # Query: Find documents related to doc1
    print("Documents related to 'Introduction to Python':")
    related = db.traverse(doc1.id, depth=2)
    scores = db.compute_graph_scores(doc1.id, depth=2)
    
    # Show results sorted by relevance
    doc_scores = {nid: sc for nid, sc in scores.items() 
                  if db.get_node(nid).metadata.get('type') != 'concept'}
    
    for node_id in sorted(doc_scores, key=doc_scores.get, reverse=True):
        node = db.get_node(node_id)
        score = doc_scores[node_id]
        score_str = "∞" if score == float('inf') else f"{score:.2f}"
        print(f"  [{score_str}] {node.text} (page {node.metadata['page']})")
    
    # Save for later use
    db.save("python_docs_graph.json")
    print(f"\nSaved graph: {db.get_stats()}")
