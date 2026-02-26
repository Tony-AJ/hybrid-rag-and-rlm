"""
Graph Database Examples

Comprehensive examples demonstrating all features of the graph_db module:
- Node CRUD operations
- Edge CRUD operations
- Graph traversal
- Graph-based scoring
- Persistence (save/load)
"""

from graph_db import GraphDatabase, GraphNode, GraphRelationship


def example_basic_operations():
    """Example 1: Basic CRUD operations"""
    print("\n" + "="*60)
    print("Example 1: Basic Node and Edge CRUD Operations")
    print("="*60)
    
    # Initialize database
    db = GraphDatabase()
    
    # Create nodes
    print("\n Creating nodes...")
    node1 = db.create_node(
        text="Python is a programming language",
        metadata={"source": "wikipedia", "category": "programming"}
    )
    node2 = db.create_node(
        text="Django is a Python web framework",
        metadata={"source": "documentation", "category": "framework"}
    )
    node3 = db.create_node(
        text="Flask is a lightweight Python framework",
        metadata={"source": "documentation", "category": "framework"}
    )
    
    print(f"  ✓ Created {node1}")
    print(f"  ✓ Created {node2}")
    print(f"  ✓ Created {node3}")
    
    # Create relationships
    print("\n Creating relationships...")
    edge1 = db.create_edge(node2.id, node1.id, "uses", weight=2.0)
    edge2 = db.create_edge(node3.id, node1.id, "uses", weight=1.5)
    edge3 = db.create_edge(node2.id, node3.id, "related_to", weight=1.0)
    
    print(f"  ✓ Created {edge1}")
    print(f"  ✓ Created {edge2}")
    print(f"  ✓ Created {edge3}")
    
    # Read operations
    print("\n Reading data...")
    retrieved_node = db.get_node(node1.id)
    print(f"  Retrieved: {retrieved_node}")
    print(f"  Text: {retrieved_node.text}")
    print(f"  Metadata: {retrieved_node.metadata}")
    
    # Update operations
    print("\n Updating node...")
    db.update_node(
        node1.id,
        metadata={"source": "wikipedia", "category": "programming", "verified": True}
    )
    updated_node = db.get_node(node1.id)
    print(f"  Updated metadata: {updated_node.metadata}")
    
    # Stats
    print("\nDatabase statistics:")
    stats = db.get_stats()
    print(f"  Nodes: {stats['nodes']}")
    print(f"  Edges: {stats['edges']}")
    
    return db


def example_traversal(db: GraphDatabase):
    """Example 2: Graph traversal"""
    print("\n" + "="*60)
    print("Example 2: Graph Traversal (BFS)")
    print("="*60)
    
    # Get all nodes for demonstration
    stats = db.get_stats()
    all_node_ids = list(db.graph.nodes())
    
    if len(all_node_ids) < 3:
        print("    Need at least 3 nodes for traversal demo")
        return
    
    start_node_id = all_node_ids[1]  # Start from Django node
    start_node = db.get_node(start_node_id)
    
    print(f"\n Starting traversal from: {start_node.text}")
    
    # Traverse with different depths
    for depth in [1, 2, 3]:
        reachable = db.traverse(start_node_id, depth)
        print(f"\n  Depth {depth}: Found {len(reachable)} reachable nodes")
        for node_id in reachable:
            node = db.get_node(node_id)
            print(f"    → {node.text[:50]}")


def example_scoring(db: GraphDatabase):
    """Example 3: Graph-based relevance scoring"""
    print("\n" + "="*60)
    print("Example 3: Graph-Based Relevance Scoring")
    print("="*60)
    
    all_node_ids = list(db.graph.nodes())
    if not all_node_ids:
        print("   No nodes in database")
        return
    
    start_node_id = all_node_ids[0]
    start_node = db.get_node(start_node_id)
    
    print(f"\nComputing scores from: {start_node.text}")
    
    scores = db.compute_graph_scores(start_node_id, depth=2)
    
    # Sort by score
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n Relevance scores (higher = more relevant):")
    for node_id, score in sorted_scores:
        node = db.get_node(node_id)
        if score == float('inf'):
            print(f"    {score:.2f}: {node.text[:50]} (starting node)")
        else:
            print(f"    {score:.2f}: {node.text[:50]}")


def example_persistence():
    """Example 4: Save and load from JSON"""
    print("\n" + "="*60)
    print("Example 4: Persistence (Save/Load)")
    print("="*60)
    
    # Create a new database
    db = GraphDatabase()
    
    print("\nCreating sample graph...")
    n1 = db.create_node("Machine Learning", {"topic": "AI"})
    n2 = db.create_node("Deep Learning", {"topic": "AI"})
    n3 = db.create_node("Neural Networks", {"topic": "AI"})
    n4 = db.create_node("Computer Vision", {"topic": "AI"})
    
    db.create_edge(n2.id, n1.id, "is_subset_of", weight=3.0)
    db.create_edge(n3.id, n2.id, "foundation_of", weight=2.5)
    db.create_edge(n4.id, n3.id, "uses", weight=2.0)
    
    print(f"  Created graph with {db.get_stats()['nodes']} nodes and {db.get_stats()['edges']} edges")
    
    # Save to file
    filepath = "test_graph.json"
    print(f"\nSaving graph to {filepath}...")
    db.save(filepath)
    print("  ✓ Saved successfully")
    
    # Load from file
    print(f"\nLoading graph from {filepath}...")
    new_db = GraphDatabase()
    new_db.load(filepath)
    print("  ✓ Loaded successfully")
    
    # Verify
    print(f"\nVerification:")
    print(f"  Original: {db.get_stats()}")
    print(f"  Loaded:   {new_db.get_stats()}")
    
    # Show loaded content
    print("\n  Loaded nodes:")
    for node_id in new_db.graph.nodes():
        node = new_db.get_node(node_id)
        print(f"    → {node.text}")
    
    return new_db


def example_knowledge_graph():
    """Example 5: Real-world knowledge graph for document retrieval"""
    print("\n" + "="*60)
    print("Example 5: Document Knowledge Graph (Real-World Use Case)")
    print("="*60)
    
    db = GraphDatabase()
    
    print("\n Building knowledge graph for document retrieval...")
    
    # Create document nodes
    doc1 = db.create_node(
        "Introduction to Python programming and its applications in data science",
        metadata={"type": "document", "section": "intro", "page": 1},
        embedding=[0.1, 0.2, 0.3]  # Mock embedding
    )
    
    doc2 = db.create_node(
        "Python data structures: lists, dictionaries, and sets",
        metadata={"type": "document", "section": "basics", "page": 5},
        embedding=[0.15, 0.25, 0.28]
    )
    
    doc3 = db.create_node(
        "NumPy and Pandas for data manipulation and analysis",
        metadata={"type": "document", "section": "libraries", "page": 12},
        embedding=[0.2, 0.3, 0.25]
    )
    
    doc4 = db.create_node(
        "Building machine learning models with scikit-learn",
        metadata={"type": "document", "section": "ml", "page": 25},
        embedding=[0.3, 0.4, 0.2]
    )
    
    doc5 = db.create_node(
        "Deep learning with TensorFlow and PyTorch",
        metadata={"type": "document", "section": "deep_learning", "page": 45},
        embedding=[0.35, 0.45, 0.18]
    )
    
    # Create concept nodes
    concept_python = db.create_node(
        "Python Programming Language",
        metadata={"type": "concept"}
    )
    
    concept_data_science = db.create_node(
        "Data Science",
        metadata={"type": "concept"}
    )
    
    concept_ml = db.create_node(
        "Machine Learning",
        metadata={"type": "concept"}
    )
    
    # Create relationships
    print("\nCreating semantic relationships...")
    
    # Documents mention concepts
    db.create_edge(doc1.id, concept_python.id, "mentions", weight=3.0)
    db.create_edge(doc1.id, concept_data_science.id, "mentions", weight=2.5)
    db.create_edge(doc2.id, concept_python.id, "mentions", weight=3.0)
    db.create_edge(doc3.id, concept_data_science.id, "mentions", weight=3.0)
    db.create_edge(doc4.id, concept_ml.id, "mentions", weight=3.0)
    db.create_edge(doc5.id, concept_ml.id, "mentions", weight=3.0)
    
    # Sequential document flow
    db.create_edge(doc1.id, doc2.id, "precedes", weight=1.5)
    db.create_edge(doc2.id, doc3.id, "precedes", weight=1.5)
    db.create_edge(doc3.id, doc4.id, "precedes", weight=1.5)
    db.create_edge(doc4.id, doc5.id, "precedes", weight=1.5)
    
    # Related concepts
    db.create_edge(concept_data_science.id, concept_ml.id, "related_to", weight=2.0)
    db.create_edge(concept_python.id, concept_data_science.id, "used_in", weight=2.5)
    
    stats = db.get_stats()
    print(f"  Created {stats['nodes']} nodes and {stats['edges']} edges")
    
    # Simulate hybrid search scenario
    print("\nHybrid Search Simulation:")
    print("  Query: 'machine learning with Python'")
    print("\n  Step 1: Vector search finds doc4 as most relevant")
    print(f"    → {doc4.text}")
    
    print("\n  Step 2: Graph expansion from doc4 (depth=2)")
    related_nodes = db.traverse(doc4.id, depth=2)
    print(f"    → Found {len(related_nodes)} related documents/concepts")
    
    print("\n  Step 3: Compute graph-based relevance scores")
    scores = db.compute_graph_scores(doc4.id, depth=2)
    
    # Filter and sort document nodes only
    doc_scores = {
        node_id: score for node_id, score in scores.items()
        if db.get_node(node_id).metadata.get("type") == "document"
    }
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("\n  Ranked results (combining vector + graph scores):")
    for i, (node_id, score) in enumerate(sorted_docs[:5], 1):
        node = db.get_node(node_id)
        score_str = "∞" if score == float('inf') else f"{score:.2f}"
        print(f"    {i}. [Score: {score_str}] {node.text}")
        print(f"       Page {node.metadata['page']}, Section: {node.metadata['section']}")
    
    # Save the knowledge graph
    print("\n Saving knowledge graph...")
    db.save("knowledge_graph.json")
    print("  ✓ Saved to knowledge_graph.json")
    
    return db


def example_delete_operations():
    """Example 6: Delete operations"""
    print("\n" + "="*60)
    print("Example 6: Delete Operations")
    print("="*60)
    
    db = GraphDatabase()
    
    # Create test data
    n1 = db.create_node("Node 1", {"test": True})
    n2 = db.create_node("Node 2", {"test": True})
    n3 = db.create_node("Node 3", {"test": True})
    
    e1 = db.create_edge(n1.id, n2.id, "links_to", weight=1.0)
    e2 = db.create_edge(n2.id, n3.id, "links_to", weight=1.0)
    
    print(f"\n Initial state: {db.get_stats()}")
    
    # Delete an edge
    print(f"\n Deleting edge {e1.id[:8]}...")
    success = db.delete_edge(e1.id)
    print(f"  {'yes' if success else '✗'} Edge deleted: {success}")
    print(f"  New stats: {db.get_stats()}")
    
    # Verify edge is gone
    retrieved_edge = db.get_edge(e1.id)
    print(f"  Edge retrieval: {retrieved_edge}")
    
    # Delete a node (also deletes connected edges)
    print(f"\n Deleting node {n2.id[:8]}...")
    success = db.delete_node(n2.id)
    print(f"  {'yes' if success else '✗'} Node deleted: {success}")
    print(f"  New stats: {db.get_stats()}")
    
    # Verify node is gone
    retrieved_node = db.get_node(n2.id)
    print(f"  Node retrieval: {retrieved_node}")


def main():
    """Run all examples"""
    print("\n" + ""*30)
    print("GRAPH DATABASE MODULE - COMPREHENSIVE EXAMPLES")
    print(""*30)
    
    # Example 1: Basic CRUD
    db = example_basic_operations()
    
    # Example 2: Traversal
    example_traversal(db)
    
    # Example 3: Scoring
    example_scoring(db)
    
    # Example 4: Persistence
    loaded_db = example_persistence()
    
    # Example 5: Real-world knowledge graph
    kg_db = example_knowledge_graph()
    
    # Example 6: Delete operations
    example_delete_operations()
    
    print("\n" + "="*60)
    print(" ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("="*60)
    print("\nGenerated files:")
    print("  - test_graph.json")
    print("  - knowledge_graph.json")
    print("\n Use these examples as templates for your hybrid search system!")
    print()


if __name__ == "__main__":
    main()
