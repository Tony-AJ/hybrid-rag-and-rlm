"""
Hybrid Retrieval System: ChromaDB (Vector Search) + NetworkX (Graph Relations)

End-to-end implementation combining cosine similarity from vector search
and graph-based relevance scoring for enhanced retrieval.
"""

import os
import json
from typing import List, Dict, Any, Tuple
from collections import defaultdict

import networkx as nx
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

from graph_db.graph_db import GraphDatabase


# ==================== Configuration ====================

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_PATH = os.path.join(CURRENT_DIR, "vector_db", "db", "chroma_db_with_metadata")
GRAPH_DB_PATH = os.path.join(CURRENT_DIR, "db", "graph_data.json")

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Hybrid scoring hyperparameters
COSINE_WEIGHT = 0.7  # α
GRAPH_WEIGHT = 0.3   # β
PATH_PENALTY = 0.1   # γ

# Graph traversal depth
GRAPH_DEPTH = 2


# ==================== Initialization ====================

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Initialize ChromaDB vector store
chroma_db = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=embeddings
)

# Initialize Graph Database
graph_db = GraphDatabase(db_path=GRAPH_DB_PATH, auto_persist=True)


# ==================== 1. Vector Search ====================

def vector_search(query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
    """
    Perform vector search using ChromaDB with cosine similarity.

    Returns:
        List of tuples: (chunk_id, chunk_text, cosine_score)
    """
    results = chroma_db.similarity_search_with_score(query, k=top_k)

    vector_results = []
    for doc, score in results:
        chunk_id = doc.metadata.get('id', doc.metadata.get('source', str(hash(doc.page_content))))
        chunk_text = doc.page_content

        # ChromaDB returns distance, convert to similarity
        cosine_score = 1.0 / (1.0 + score)

        vector_results.append((chunk_id, chunk_text, cosine_score))

    return vector_results


# ==================== 2. Graph Expansion & Scoring ====================

def graph_score(graph: nx.MultiDiGraph, chunk_id: str, depth: int = GRAPH_DEPTH) -> Dict[str, Dict[str, Any]]:
    """
    Compute graph-based scores for nodes related to chunk_id.

    Score: 1 / (hop_distance + 1), already normalized to (0, 1].

    Returns:
        Dictionary mapping related_chunk_id -> {"score": float, "hop": int, "via_edge": str, "from_node": str}
    """
    if chunk_id not in graph.nodes:
        return {}

    result = {}
    visited = set()
    queue = [(chunk_id, 0, None, None)]  # (node_id, current_depth, via_edge, from_node)

    while queue:
        node_id, current_depth, via_edge, from_node = queue.pop(0)

        if node_id in visited:
            continue

        visited.add(node_id)

        score = 1.0 / (current_depth + 1)
        result[node_id] = {
            "score": score,
            "hop": current_depth,
            "via_edge": via_edge,
            "from_node": from_node
        }

        if current_depth < depth:
            neighbors = set(graph.successors(node_id)) | set(graph.predecessors(node_id))
            for neighbor in neighbors:
                if neighbor not in visited:
                    # Get edge type for explainability
                    edge_type = "link"
                    if graph.has_edge(node_id, neighbor):
                        for key, edge_data in graph[node_id][neighbor].items():
                            edge_type = edge_data.get("type", "link")
                            break
                    elif graph.has_edge(neighbor, node_id):
                        for key, edge_data in graph[neighbor][node_id].items():
                            edge_type = edge_data.get("type", "link")
                            break
                    queue.append((neighbor, current_depth + 1, edge_type, node_id))

    return result


# ==================== 3. Hybrid Scoring ====================

def hybrid_rank(
    cosine_results: List[Tuple[str, str, float]],
    graph_data: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Combine cosine similarity and graph scores using normalized hybrid formula.

    Formula:
        final_score = (α × normalized_vector_score)
                    + (β × normalized_graph_score)
                    - (γ × path_distance_penalty)
    """
    chunk_data = {}

    # Add vector search results
    for chunk_id, chunk_text, cos_score in cosine_results:
        if chunk_id not in chunk_data:
            chunk_data[chunk_id] = {
                'chunk_id': chunk_id,
                'text': chunk_text,
                'raw_cosine': cos_score,
                'graph_score': 0.0,
                'hop': 0,
                'via_edge': None,
                'from_node': None,
            }
        else:
            chunk_data[chunk_id]['raw_cosine'] = max(
                chunk_data[chunk_id]['raw_cosine'],
                cos_score
            )

    # Add graph expansion results
    for chunk_id, g_info in graph_data.items():
        if chunk_id not in chunk_data:
            node = graph_db.get_node(chunk_id)
            chunk_text = node.text if node else ""

            chunk_data[chunk_id] = {
                'chunk_id': chunk_id,
                'text': chunk_text,
                'raw_cosine': 0.0,
                'graph_score': g_info["score"],
                'hop': g_info["hop"],
                'via_edge': g_info["via_edge"],
                'from_node': g_info["from_node"],
            }
        else:
            chunk_data[chunk_id]['graph_score'] = g_info["score"]
            chunk_data[chunk_id]['hop'] = g_info["hop"]
            chunk_data[chunk_id]['via_edge'] = g_info["via_edge"]
            chunk_data[chunk_id]['from_node'] = g_info["from_node"]

    # ── Min-Max Normalize Vector Scores ──
    raw_scores = [d['raw_cosine'] for d in chunk_data.values() if d['raw_cosine'] > 0]
    v_min = min(raw_scores) if raw_scores else 0.0
    v_max = max(raw_scores) if raw_scores else 1.0
    v_range = v_max - v_min if v_max != v_min else 1.0

    for data in chunk_data.values():
        if data['raw_cosine'] > 0:
            data['norm_cosine'] = (data['raw_cosine'] - v_min) / v_range
        else:
            data['norm_cosine'] = 0.0

    # ── Compute Final Scores ──
    α = COSINE_WEIGHT
    β = GRAPH_WEIGHT
    γ = PATH_PENALTY

    final_results = []
    for chunk_id, data in chunk_data.items():
        norm_v = data['norm_cosine']
        norm_g = data['graph_score']
        hop = data['hop']

        penalty = (hop / GRAPH_DEPTH) if GRAPH_DEPTH > 0 else 0.0

        final_score = (α * norm_v) + (β * norm_g) - (γ * penalty)
        final_score = max(0.0, final_score)

        # ── Build Explanation Reason ──
        reason_parts = []
        if norm_v > 0:
            reason_parts.append(f"Vector match (score {norm_v:.2f})")
        if hop == 0 and norm_v > 0:
            reason_parts.append("Direct vector hit")
        elif hop > 0:
            via = data.get('via_edge') or 'link'
            from_node = data.get('from_node') or '?'
            reason_parts.append(f"Connected to {from_node} via '{via}' at hop {hop}")
        if not reason_parts:
            reason_parts.append("Graph-only node")

        result = {
            'chunk_id': chunk_id,
            'cosine_similarity': round(norm_v, 4),
            'graph_score': round(norm_g, 4),
            'final_score': round(final_score, 4),
            'reason': "; ".join(reason_parts),
            'text': data['text']
        }
        final_results.append(result)

    final_results.sort(key=lambda x: x['final_score'], reverse=True)

    return final_results


# ==================== 4. Final Retrieval Pipeline ====================

def hybrid_retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    End-to-end hybrid retrieval pipeline.

    Process:
        1. Vector search → top_k chunks
        2. Graph expansion → related nodes with edge metadata
        3. Normalized hybrid scoring with explainability
        4. Return sorted results
    """
    # Step 1: Vector search
    vector_results = vector_search(query, top_k)

    # Step 2: Graph expansion
    graph_data_combined = {}

    for chunk_id, _, cos_score in vector_results:
        related = graph_score(graph_db.graph, chunk_id, depth=GRAPH_DEPTH)

        for node_id, g_info in related.items():
            if node_id not in graph_data_combined:
                graph_data_combined[node_id] = g_info
            else:
                if g_info["score"] > graph_data_combined[node_id]["score"]:
                    graph_data_combined[node_id] = g_info

    # Step 3: Hybrid ranking
    final_results = hybrid_rank(vector_results, graph_data_combined)

    return final_results


# ==================== 5. Example Usage ====================

if __name__ == "__main__":
    # Example query
    test_query = "How did Juliet die? Who killed Juliet?"
    
    print(f"\n{'='*60}")
    print(f"HYBRID RETRIEVAL SYSTEM")
    print(f"{'='*60}")
    print(f"\nQuery: {test_query}")
    print(f"\nConfiguration:")
    print(f"  - Cosine Weight: {COSINE_WEIGHT}")
    print(f"  - Graph Weight: {GRAPH_WEIGHT}")
    print(f"  - Graph Depth: {GRAPH_DEPTH}")
    print(f"\n{'='*60}\n")
    
    # Perform hybrid retrieval
    results = hybrid_retrieve(test_query, top_k=5)
    
    # Output as JSON
    print("Results (JSON):")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    
    # Also print human-readable format
    print(f"\n{'='*60}")
    print(f"Top {len(results)} Results:")
    print(f"{'='*60}\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. Chunk ID: {result['chunk_id']}")
        print(f"   Cosine Similarity: {result['cosine_similarity']}")
        print(f"   Graph Score: {result['graph_score']}")
        print(f"   Final Score: {result['final_score']}")
        print(f"   Text: {result['text'][:100]}...")
        print()
