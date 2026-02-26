from fastapi.testclient import TestClient
from app.main import app
import os
import shutil


# Clean up DBs for testing to ensure fresh state
if os.path.exists("db/test_graph_data.json"):
    os.remove("db/test_graph_data.json")
if os.path.exists("db/test_chroma_db"):
    shutil.rmtree("db/test_chroma_db")

client = TestClient(app)

def run_tests():
    print("\n" + "="*50)
    print("STARTING FINAL SYSTEM TEST")
    print("="*50 + "\n")

    # ==========================================
    # 1. Node CRUD Tests
    # ==========================================
    print("--- Testing Node CRUD ---")
    
    # A. Create Node 1
    print("1. Creating Node 1...")
    response = client.post("/nodes", json={
        "text": "Machine learning is a subset of AI.",
        "metadata": { "source": "doc1" },
        "auto_embed": True
    })
    assert response.status_code == 200
    node1 = response.json()
    node1_id = node1["id"]
    print(f"   Success: Created Node {node1_id}")

    # Create Node 2
    print("2. Creating Node 2...")
    response = client.post("/nodes", json={
        "text": "Supervised learning uses labeled data.",
        "metadata": { "source": "doc2" },
        "auto_embed": True
    })
    assert response.status_code == 200
    node2 = response.json()
    node2_id = node2["id"]
    print(f"   Success: Created Node {node2_id}")

    # B. Get Node
    print(f"3. Getting Node {node1_id}...")
    response = client.get(f"/nodes/{node1_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "Machine learning is a subset of AI."
    print("   Success: Retrieved Node")

    # C. Update Node
    print(f"4. Updating Node {node1_id}...")
    response = client.put(f"/nodes/{node1_id}", json={
        "text": "Machine learning is a key subset of Artificial Intelligence.",
        "metadata": { "source": "doc1_updated" },
        "regenerate_embedding": True
    })
    assert response.status_code == 200
    assert response.json()["embedding_regenerated"] is True
    print("   Success: Updated Node")

    # Verify Update
    response = client.get(f"/nodes/{node1_id}")
    assert response.json()["text"] == "Machine learning is a key subset of Artificial Intelligence."

    # ==========================================
    # 2. Edge CRUD Tests
    # ==========================================
    print("\n--- Testing Edge CRUD ---")

    # E. Create Relationship
    print("5. Creating Edge between Node 1 and Node 2...")
    response = client.post("/edges", json={
        "source": node1_id,
        "target": node2_id,
        "type": "related_to",
        "weight": 0.9
    })
    assert response.status_code == 200
    edge = response.json()
    edge_id = edge["edge_id"]
    print(f"   Success: Created Edge {edge_id}")

    # F. Get Relationship
    print(f"6. Getting Edge {edge_id}...")
    response = client.get(f"/edges/{edge_id}")
    assert response.status_code == 200
    assert response.json()["weight"] == 0.9
    print("   Success: Retrieved Edge")

    # ==========================================
    # 3. Search Tests
    # ==========================================
    print("\n--- Testing Search APIs ---")

    # H. Vector Search
    print("7. Testing Vector Search...")
    response = client.post("/search/vector", json={
        "query_text": "Artificial Intelligence",
        "top_k": 5
    })
    assert response.status_code == 200
    results = response.json()
    assert len(results) > 0
    print(f"   Success: Found {len(results)} results")

    # I. Graph Traversal
    print("8. Testing Graph Traversal...")
    response = client.get(f"/search/graph?start_id={node1_id}&depth=2")
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) >= 1 # Should find node2
    print(f"   Success: Traversed graph from {node1_id}")

    # J. Hybrid Search
    print("9. Testing Hybrid Search...")
    response = client.post("/search/hybrid", json={
        "query_text": "labeled data",
        "vector_weight": 0.5,
        "graph_weight": 0.5,
        "top_k": 5
    })
    assert response.status_code == 200
    results = response.json()
    assert len(results) > 0
    print(f"   Success: Hybrid search returned {len(results)} results")
    print(f"   Top Result Score: {results[0]['final_score']}")

    # ==========================================
    # 4. Deletion Tests
    # ==========================================
    print("\n--- Testing Deletion ---")

    # G. Delete Relationship
    print(f"10. Deleting Edge {edge_id}...")
    response = client.delete(f"/edges/{edge_id}")
    assert response.status_code == 200
    print("   Success: Deleted Edge")

    # Verify Edge Gone
    response = client.get(f"/edges/{edge_id}")
    assert response.status_code == 404
    print("   Verified: Edge Not Found")

    # D. Delete Node
    print(f"11. Deleting Node {node1_id}...")
    response = client.delete(f"/nodes/{node1_id}")
    assert response.status_code == 200
    print("   Success: Deleted Node")

    # Verify Node Gone
    response = client.get(f"/nodes/{node1_id}")
    assert response.status_code == 404
    print("   Verified: Node Not Found")

    # ==========================================
    # 5. Error Handling / Negative Tests
    # ==========================================
    print("\n--- Testing Error Handling ---")

    # Get non-existent node
    print("12. Getting non-existent node...")
    response = client.get("/nodes/non_existent_id")
    assert response.status_code == 404
    print("   Success: Correctly returned 404")

    # Create edge with non-existent nodes
    print("13. Creating edge with invalid nodes...")
    response = client.post("/edges", json={
        "source": "fake_1",
        "target": "fake_2",
        "type": "test",
        "weight": 1.0
    })
    assert response.status_code == 400
    print("   Success: Correctly returned 400")

    print("\n" + "="*50)
    print("ALL TESTS PASSED SUCCESSFULLY")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_tests()
