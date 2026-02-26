#!/usr/bin/env python3
"""
Quick validation test for Graph Database module
Run: python test_graph_db.py
"""

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from graph_db import GraphDatabase, GraphNode, GraphRelationship
        print(" All imports successful")
        return True
    except ImportError as e:
        print(f"Import failed: {e}")
        return False


def test_node_operations():
    """Test node CRUD operations"""
    print("\nTesting node operations...")
    from graph_db import GraphDatabase
    
    db = GraphDatabase(auto_persist=False)  # Disable auto-persist for tests
    
    # Create
    node = db.create_node("Test node", {"key": "value"})
    assert node is not None, "Node creation failed"
    print(f" Created node: {node.id[:8]}...")
    
    # Read
    retrieved = db.get_node(node.id)
    assert retrieved is not None, "Node retrieval failed"
    assert retrieved.text == "Test node", "Text mismatch"
    print(f" Retrieved node correctly")
    
    # Update
    success = db.update_node(node.id, text="Updated text")
    assert success, "Node update failed"
    updated = db.get_node(node.id)
    assert updated is not None, "Update not saved"
    assert updated.text == "Updated text", "Update not saved"
    print(f" Updated node successfully")
    
    # Delete
    success = db.delete_node(node.id)
    assert success, "Node deletion failed"
    deleted = db.get_node(node.id)
    assert deleted is None, "Node still exists after deletion"
    print(f" Deleted node successfully")
    
    return True


def test_edge_operations():
    """Test edge CRUD operations"""
    print("\nTesting edge operations...")
    from graph_db import GraphDatabase
    
    db = GraphDatabase(auto_persist=False)
    
    # Create nodes first
    n1 = db.create_node("Node 1", {})
    n2 = db.create_node("Node 2", {})
    
    # Create edge
    edge = db.create_edge(n1.id, n2.id, "links_to", weight=2.5)
    assert edge is not None, "Edge creation failed"
    print(f" Created edge: {edge.id[:8]}...")
    
    # Read edge
    retrieved = db.get_edge(edge.id)
    assert retrieved is not None, "Edge retrieval failed"
    assert retrieved.source == n1.id, "Source mismatch"
    assert retrieved.target == n2.id, "Target mismatch"
    assert retrieved.weight == 2.5, "Weight mismatch"
    print(f" Retrieved edge correctly")
    
    # Delete edge
    success = db.delete_edge(edge.id)
    assert success, "Edge deletion failed"
    deleted = db.get_edge(edge.id)
    assert deleted is None, "Edge still exists after deletion"
    print(f" Deleted edge successfully")
    
    return True


def test_traversal():
    """Test graph traversal"""
    print("\nTesting graph traversal...")
    from graph_db import GraphDatabase
    
    db = GraphDatabase(auto_persist=False)
    
    # Create a small graph: n1 -> n2 -> n3
    n1 = db.create_node("Start", {})
    n2 = db.create_node("Middle", {})
    n3 = db.create_node("End", {})
    
    db.create_edge(n1.id, n2.id, "next", weight=1.0)
    db.create_edge(n2.id, n3.id, "next", weight=1.0)
    
    # Traverse from n1
    nodes_depth1 = db.traverse(n1.id, depth=1)
    assert len(nodes_depth1) >= 2, f"Expected >=2 nodes at depth 1, got {len(nodes_depth1)}"
    print(f" Depth 1 traversal: {len(nodes_depth1)} nodes")
    
    nodes_depth2 = db.traverse(n1.id, depth=2)
    assert len(nodes_depth2) == 3, f"Expected 3 nodes at depth 2, got {len(nodes_depth2)}"
    print(f" Depth 2 traversal: {len(nodes_depth2)} nodes")
    
    return True


def test_scoring():
    """Test graph-based scoring"""
    print("\nTesting graph scoring...")
    from graph_db import GraphDatabase
    
    db = GraphDatabase(auto_persist=False)
    
    # Create nodes
    n1 = db.create_node("Center", {})
    n2 = db.create_node("Neighbor 1", {})
    n3 = db.create_node("Neighbor 2", {})
    
    # Create weighted edges
    db.create_edge(n1.id, n2.id, "related", weight=3.0)
    db.create_edge(n1.id, n3.id, "related", weight=1.5)
    
    # Compute scores
    scores = db.compute_graph_scores(n1.id, depth=1)
    
    assert n1.id in scores, "Starting node not in scores"
    assert n2.id in scores, "Neighbor 1 not in scores"
    assert n3.id in scores, "Neighbor 2 not in scores"
    
    # Starting node should have highest score
    assert scores[n1.id] == float('inf'), "Starting node should have infinite score"
    
    # Higher weight edge should have higher score
    assert scores[n2.id] > scores[n3.id], "Score should correlate with edge weight"
    
    print(f" Scoring works correctly")
    print(f"  Scores: {n1.id[:8]}=∞, {n2.id[:8]}={scores[n2.id]:.2f}, {n3.id[:8]}={scores[n3.id]:.2f}")
    
    return True


def test_persistence():
    """Test save and load"""
    print("\nTesting persistence...")
    from graph_db import GraphDatabase
    import os
    
    db = GraphDatabase(auto_persist=False)
    
    # Create test data
    n1 = db.create_node("Persistent node 1", {"persistent": True})
    n2 = db.create_node("Persistent node 2", {"persistent": True})
    db.create_edge(n1.id, n2.id, "persists", weight=2.0)
    
    original_stats = db.get_stats()
    
    # Save
    filepath = "test_persistence.json"
    db.save(filepath)
    assert os.path.exists(filepath), "Save file not created"
    print(f" Saved to {filepath}")
    
    # Load
    new_db = GraphDatabase(db_path=filepath, auto_persist=False)
    new_db.load(filepath)
    loaded_stats = new_db.get_stats()
    
    assert original_stats == loaded_stats, "Stats mismatch after load"
    print(f" Loaded successfully")
    print(f"  Stats: {loaded_stats}")
    
    # Cleanup
    os.remove(filepath)
    
    return True
def test_large_chunk_ingestion():
    """Test graph creation from 4 long text chunks (~1000 chars each)"""
    print("\nTesting ingestion of 4 large chunks...")

    from graph_db import GraphDatabase
    import uuid
    
    db = GraphDatabase(auto_persist=False)

    # Generate 4 fake 1000-character chunks
    chunks = [
        ("Chunk 1: " + ("A" * 980)),
        ("Chunk 2: " + ("B" * 980)),
        ("Chunk 3: " + ("C" * 980)),
        ("Chunk 4: " + ("D" * 980)),
    ]

    node_ids = []

    # Create nodes for each chunk
    for i, text in enumerate(chunks):
        metadata = {"chunk_index": i + 1, "length": len(text)}
        node = db.create_node(text, metadata)
        node_ids.append(node.id)
        print(f" Created node {i+1}: {node.id[:8]}, length={metadata['length']}")

    # Create edges : chunk1 -> chunk2 -> chunk3 -> chunk4
    edge_ids = []
    for i in range(len(node_ids) - 1):
        edge = db.create_edge(
            node_ids[i],
            node_ids[i + 1],
            rel_type="next_chunk",
            weight=1.0
        )
        assert edge is not None, "Edge creation failed"
        edge_ids.append(edge.id)
        print(f" Linked {node_ids[i][:8]} -> {node_ids[i+1][:8]}")

    # Validate graph structure
    assert len(node_ids) == 4, "Expected 4 nodes"
    assert len(edge_ids) == 3, "Expected 3 edges"

    # Traverse from first chunk
    reachable = db.traverse(node_ids[0], depth=3)
    print(f"\nReachable nodes from Chunk 1 (depth=3):")
    for nid in reachable:
        print(" -", nid[:8])

    assert len(reachable) == 4, "Traversal mismatch, expected all 4 nodes"

    print("\nFinal Graph Output:")
    stats = db.get_stats()
    print(stats)

    print(" Large chunk ingestion test passed!")
    return True



def main():
    test_large_chunk_ingestion()
    """Run all tests"""
    
    # print("="*60)
    # print("Graph Database Module - Validation Tests")
    # print("="*60)
    
    # tests = [
    #     ("Imports", test_imports),
    #     ("Node Operations", test_node_operations),
    #     ("Edge Operations", test_edge_operations),
    #     ("Graph Traversal", test_traversal),
    #     ("Graph Scoring", test_scoring),
    #     ("Persistence", test_persistence),
    # ]
    
    # passed = 0
    # failed = 0
    
    # for name, test_func in tests:
    #     try:
    #         print(f"\n[Test: {name}]")
    #         result = test_func()
    #         if result:
    #             passed += 1
    #             print(f"{name} PASSED")
    #         else:
    #             failed += 1
    #             print(f" {name} FAILED")
    #     except Exception as e:
    #         failed += 1
    #         print(f" {name} FAILED with exception: {e}")
    #         import traceback
    #         traceback.print_exc()
    
    # print("\n" + "="*60)
    # print(f"Test Results: {passed} passed, {failed} failed")
    # print("="*60)
    
    # if failed == 0:
    #     print("\nAll tests passed! Graph database is working correctly.")
    #     return 0
    # else:
    #     print(f"\n{failed} test(s) failed. Please review the errors above.")
    #     return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
