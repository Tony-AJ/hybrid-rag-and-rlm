import pytest
from fastapi.testclient import TestClient
from app.main import app
import os
import shutil

# Setup/Teardown
@pytest.fixture(scope="module")
def client():
    # Clean up DBs before test
    if os.path.exists("db/graph_data_service.json"):
        os.remove("db/graph_data_service.json")
    if os.path.exists("db/chroma_db_service"):
        shutil.rmtree("db/chroma_db_service")
        
    with TestClient(app) as c:
        yield c
        
    # Clean up after test
    if os.path.exists("db/graph_data_service.json"):
        os.remove("db/graph_data_service.json")
    if os.path.exists("db/chroma_db_service"):
        shutil.rmtree("db/chroma_db_service")

def test_1_create_node(client):
    payload = {
        "id": "doc1",
        "text": "Redis caching strategies",
        "metadata": { "type": "article", "tags": ["cache", "redis"] },
        "embedding": None,
        "regen_embedding": True
    }
    response = client.post("/nodes", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "created"
    assert data["id"] == "doc1"
    assert data["embedding_dim"] == 384

def test_create_other_nodes(client):
    # Helper to create nodes for relationships
    nodes = [
        {"id": "doc2", "text": "Redis performance tuning", "metadata": {"type": "guide"}},
        {"id": "doc4", "text": "Advanced Redis topics", "metadata": {"type": "article"}},
        {"id": "doc6", "text": "Database scaling", "metadata": {"type": "book"}},
        {"id": "doc7", "text": "To be deleted", "metadata": {"type": "temp"}}
    ]
    for n in nodes:
        client.post("/nodes", json={**n, "regen_embedding": True})

def test_5_create_edge(client):
    payload = {
        "source": "doc1",
        "target": "doc4",
        "type": "related_to",
        "weight": 0.8
    }
    response = client.post("/edges", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "created"
    assert data["edge_id"] is not None
    assert data["source"] == "doc1"
    assert data["target"] == "doc4"
    # Store edge_id for later tests
    pytest.edge_id = data["edge_id"]

def test_2_get_node(client):
    response = client.get("/nodes/doc1")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "doc1"
    assert data["text"] == "Redis caching strategies"
    assert data["metadata"] == { "type": "article", "tags": ["cache", "redis"] }
    assert len(data["embedding"]) == 384 # Check dim instead of exact values
    assert isinstance(data["edges"], list)
    # Check edge content
    edge = next((e for e in data["edges"] if e["target"] == "doc4"), None)
    assert edge is not None
    assert edge["edge_id"] == pytest.edge_id
    assert edge["type"] == "related_to"
    assert edge["weight"] == 0.8

def test_3_update_node(client):
    payload = {
        "text": "Updated redis caching guide",
        "metadata": { "type": "guide" },
        "regen_embedding": True
    }
    response = client.put("/nodes/doc1", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "updated"
    assert data["id"] == "doc1"
    assert data["embedding_regenerated"] == True
    
    # Verify update
    r = client.get("/nodes/doc1")
    assert r.json()["text"] == "Updated redis caching guide"
    assert r.json()["metadata"] == { "type": "guide" }

def test_6_get_edge(client):
    response = client.get(f"/edges/{pytest.edge_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["edge_id"] == pytest.edge_id
    assert data["source"] == "doc1"
    assert data["target"] == "doc4"
    assert data["type"] == "related_to"
    assert data["weight"] == 0.8

def test_7_update_edge(client):
    payload = { "weight": 0.95 }
    response = client.put(f"/edges/{pytest.edge_id}", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "updated"
    assert data["edge_id"] == pytest.edge_id
    assert data["new_weight"] == 0.95
    
    # Verify
    r = client.get(f"/edges/{pytest.edge_id}")
    assert r.json()["weight"] == 0.95

def test_8_delete_edge(client):
    response = client.delete(f"/edges/{pytest.edge_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "deleted"
    assert data["edge_id"] == pytest.edge_id
    
    # Verify gone
    r = client.get(f"/edges/{pytest.edge_id}")
    assert r.status_code == 404

def test_4_delete_node(client):
    # Create edges for doc7 to test cascade
    # doc7 -> doc2
    # doc6 -> doc7
    client.post("/edges", json={"source": "doc7", "target": "doc2", "type": "rel", "weight": 1.0})
    client.post("/edges", json={"source": "doc6", "target": "doc7", "type": "rel", "weight": 1.0})
    
    response = client.delete("/nodes/doc7")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "deleted"
    assert data["id"] == "doc7"
    assert data["removed_edges_count"] == 2

def test_9_vector_search(client):
    # doc1: Updated redis caching guide
    # doc4: Advanced Redis topics
    # doc2: Redis performance tuning
    
    payload = {
        "query_text": "redis caching",
        "top_k": 5,
        "metadata_filter": { "type": "guide" } # doc1 matches
    }
    response = client.post("/search/vector", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query_text"] == "redis caching"
    assert isinstance(data["results"], list)
    assert len(data["results"]) > 0
    # Check structure
    res = data["results"][0]
    assert "id" in res
    assert "vector_score" in res
    # doc1 should be top result due to filter and text match
    assert data["results"][0]["id"] == "doc1"

def test_10_graph_search(client):
    # Setup graph structure
    # doc6 -> doc2 (mentions, 0.9)
    # doc6 -> doc1 (references, 0.6)
    # doc1 -> doc4 (related_to, 0.8) - we deleted edge earlier, recreate
    
    client.post("/edges", json={"source": "doc6", "target": "doc2", "type": "mentions", "weight": 0.9})
    client.post("/edges", json={"source": "doc6", "target": "doc1", "type": "references", "weight": 0.6})
    client.post("/edges", json={"source": "doc1", "target": "doc4", "type": "related_to", "weight": 0.8})
    
    # Search
    response = client.get("/search/graph?start_id=doc6&depth=2")
    assert response.status_code == 200
    data = response.json()
    assert data["start_id"] == "doc6"
    assert data["depth"] == 2
    assert isinstance(data["nodes"], list)
    
    # Check specific nodes
    ids = [n["id"] for n in data["nodes"]]
    assert "doc2" in ids
    assert "doc1" in ids
    assert "doc4" in ids
    
    # Check doc4 (hop 2)
    doc4 = next(n for n in data["nodes"] if n["id"] == "doc4")
    assert doc4["hop"] == 2
    assert doc4["edge_path"] == ["references", "related_to"]
    assert doc4["weights"] == [0.6, 0.8]

def test_11_hybrid_search(client):
    payload = {
        "query_text": "redis caching",
        "vector_weight": 0.6,
        "graph_weight": 0.4,
        "top_k": 5
    }
    response = client.post("/search/hybrid", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query_text"] == "redis caching"
    assert data["vector_weight"] == 0.6
    assert data["graph_weight"] == 0.4
    assert isinstance(data["results"], list)
    
    # Check results
    # doc1 should be high (vector match + graph connectivity?)
    # doc1 is connected to doc4.
    # doc6 is start of graph.
    
    res = data["results"][0]
    assert "id" in res
    assert "vector_score" in res
    assert "graph_score" in res
    assert "final_score" in res
    assert "info" in res
    assert "hop" in res["info"]

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))
