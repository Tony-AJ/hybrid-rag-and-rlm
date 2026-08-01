"""
Hybrid Retrieval System: ChromaDB (Vector Search) + NetworkX (Graph Relations)

Research-grade implementation combining cosine similarity from vector search
and graph-based relevance scoring for enhanced retrieval.

Pipeline Version: 2.0

Scoring Pipeline:
    Query → Embedding → ChromaDB Search → L2² Distance → Cosine Similarity
    → Min-Max Normalization → Split Hybrid Formula → Ranked Results

Key Design Decisions:
    - Distance conversion: cosine = 1 - (L2² / 2)  (exact for unit vectors)
    - Dual scores: raw_cosine (truth) + normalized_cosine (ranking)
    - Split formula: vector-retrieved nodes vs graph-only nodes
    - Normalization: (score - min) / (range + ε), no floor clamp
    - Distinction: retrieved_by_vector boolean, not fake scores
"""

import os
import json
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from datetime import datetime, timezone

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
COSINE_WEIGHT = 0.7  # alpha
GRAPH_WEIGHT = 0.3   # beta
PATH_PENALTY = 0.1   # gamma

# Graph traversal depth
GRAPH_DEPTH = 2

# Pipeline version
PIPELINE_VERSION = "2.0"

# Normalization epsilon (prevents division by zero)
NORM_EPSILON = 1e-6


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
    Perform vector search using ChromaDB.

    Distance conversion: cosine_similarity = 1 - (L2_squared / 2)
    This is mathematically exact for unit-normalized embeddings.

    Returns:
        List of tuples: (chunk_id, chunk_text, cosine_similarity)
        Ordered by similarity descending.
    """
    results = chroma_db.similarity_search_with_score(query, k=top_k)

    vector_results = []
    for doc, distance in results:
        chunk_id = doc.metadata.get('id', doc.metadata.get('source', str(hash(doc.page_content))))
        chunk_text = doc.page_content

        # ChromaDB returns L2² distance. Convert to cosine similarity.
        # For unit-normalized vectors: cosine = 1 - (L2² / 2)
        cosine_score = max(0.0, 1.0 - (distance / 2.0))

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
    Combine cosine similarity and graph scores using research-grade hybrid formula.

    Design:
        - Vector-retrieved nodes: final = alpha * norm_v + beta * g - gamma * penalty
        - Graph-only nodes:       final = beta * g - gamma * penalty
        - No fake vector scores for graph-only nodes.

    Normalization:
        normalized = (score - v_min) / (v_range + epsilon)
        No floor clamp. retrieved_by_vector boolean distinguishes 0.0 from null.
    """
    chunk_data = {}

    # ── Collect vector search results (with rank) ──
    for vector_rank, (chunk_id, chunk_text, cos_score) in enumerate(cosine_results, 1):
        if chunk_id not in chunk_data:
            chunk_data[chunk_id] = {
                'chunk_id': chunk_id,
                'text': chunk_text,
                'raw_cosine': cos_score,
                'retrieved_by_vector': True,
                'vector_rank': vector_rank,
                'graph_score': 0.0,
                'graph_rank': None,
                'hop': 0,
                'via_edge': None,
                'from_node': None,
            }
        else:
            # Keep best cosine score and earliest rank
            if cos_score > chunk_data[chunk_id]['raw_cosine']:
                chunk_data[chunk_id]['raw_cosine'] = cos_score
                chunk_data[chunk_id]['vector_rank'] = vector_rank

    # ── Collect graph expansion results (with rank) ──
    graph_rank_counter = 0
    for chunk_id, g_info in graph_data.items():
        graph_rank_counter += 1

        if chunk_id not in chunk_data:
            # Graph-only node — never retrieved by vector search
            node = graph_db.get_node(chunk_id)
            chunk_text = node.text if node else ""

            chunk_data[chunk_id] = {
                'chunk_id': chunk_id,
                'text': chunk_text,
                'raw_cosine': None,            # null, not 0.0
                'retrieved_by_vector': False,
                'vector_rank': None,
                'graph_score': g_info["score"],
                'graph_rank': graph_rank_counter,
                'hop': g_info["hop"],
                'via_edge': g_info["via_edge"],
                'from_node': g_info["from_node"],
            }
        else:
            # Node found by both vector AND graph
            chunk_data[chunk_id]['graph_score'] = g_info["score"]
            chunk_data[chunk_id]['graph_rank'] = graph_rank_counter
            chunk_data[chunk_id]['hop'] = g_info["hop"]
            chunk_data[chunk_id]['via_edge'] = g_info["via_edge"]
            chunk_data[chunk_id]['from_node'] = g_info["from_node"]

    # ── Min-Max Normalize Vector Scores ──
    # Only normalize across vector-retrieved nodes
    raw_scores = [d['raw_cosine'] for d in chunk_data.values() if d['retrieved_by_vector']]
    v_min = min(raw_scores) if raw_scores else 0.0
    v_max = max(raw_scores) if raw_scores else 0.0
    v_range = v_max - v_min

    # Min-max is undefined with a single hit or with all scores tied: (score - min) is
    # 0 for every candidate, which would silently zero out the entire vector component.
    # There is no relative ordering to express, so every hit is treated as best-available.
    degenerate = len(raw_scores) < 2 or v_range < NORM_EPSILON

    for data in chunk_data.values():
        if not data['retrieved_by_vector']:
            data['norm_cosine'] = None  # null, not 0.0
        elif degenerate:
            data['norm_cosine'] = 1.0
        else:
            data['norm_cosine'] = (data['raw_cosine'] - v_min) / (v_range + NORM_EPSILON)

    # ── Compute Final Scores with Split Formula ──
    alpha = COSINE_WEIGHT
    beta = GRAPH_WEIGHT
    gamma = PATH_PENALTY

    final_results = []
    for chunk_id, data in chunk_data.items():
        norm_g = data['graph_score']
        hop = data['hop']
        penalty = (hop / GRAPH_DEPTH) if GRAPH_DEPTH > 0 else 0.0

        # Split formula: vector-retrieved vs graph-only
        if data['retrieved_by_vector']:
            norm_v = data['norm_cosine']
            vector_contrib = alpha * norm_v
            graph_contrib = beta * norm_g
            penalty_contrib = gamma * penalty
            final_score = vector_contrib + graph_contrib - penalty_contrib
        else:
            vector_contrib = 0.0
            graph_contrib = beta * norm_g
            penalty_contrib = gamma * penalty
            final_score = graph_contrib - penalty_contrib

        # Keep the pre-clamp value so the contribution breakdown always reconstructs
        # exactly; a heavy penalty can drive a deep graph-only node below zero.
        unclamped_final = final_score
        final_score = max(0.0, final_score)

        # ── Build retrieval_sources ──
        sources = []
        if data['retrieved_by_vector']:
            sources.append("vector")
        if data['graph_score'] > 0:
            sources.append("graph")

        # ── Build structured reason (mirrors app.models.RetrievalReason) ──
        reason = {
            "rank": None,               # filled in after the final sort
            "vector_rank": data['vector_rank'],
            "graph_rank": data['graph_rank'],
            "hop": hop,
            "edge": data.get('via_edge'),
            "retrieval_method": sources,
        }

        result = {
            'node_id': chunk_id,
            'raw_vector_score': round(data['raw_cosine'], 4) if data['raw_cosine'] is not None else None,
            'normalized_vector_score': round(data['norm_cosine'], 4) if data['norm_cosine'] is not None else None,
            'graph_score': round(norm_g, 4),
            'vector_contribution': round(vector_contrib, 4),
            'graph_contribution': round(graph_contrib, 4),
            'penalty_contribution': round(penalty_contrib, 4),
            'unclamped_final_score': round(unclamped_final, 4),
            'final_score': round(final_score, 4),
            'retrieval_sources': sources,
            'reason': reason,
            'metadata': {},
            'text': data['text'],
            'info': {},
        }
        final_results.append(result)

    # Sort by final_score descending
    final_results.sort(key=lambda x: x['final_score'], reverse=True)

    # Assign overall rank
    for rank, result in enumerate(final_results, 1):
        result['reason']['rank'] = rank

    return final_results


# ==================== 4. Final Retrieval Pipeline ====================

def build_retrieval_metadata() -> Dict[str, Any]:
    """Per-query provenance for benchmarking and reproducibility."""
    return {
        "pipeline_version": PIPELINE_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "max_depth": GRAPH_DEPTH,
        "path_distance_penalty": PATH_PENALTY,
        "normalization": "min-max",
        "distance_conversion": "cosine = 1 - (L2^2 / 2)",
    }


def hybrid_retrieve(query: str, top_k: int = 5, with_metadata: bool = False) -> Any:
    """
    End-to-end hybrid retrieval pipeline (v2.0).

    Process:
        1. Vector search -> top_k chunks with true cosine similarity
        2. Graph expansion -> related nodes with edge metadata
        3. Normalized hybrid scoring with split formula
        4. Return sorted results with full explainability

    Returns:
        with_metadata=False (default): List[Dict] of ranked results.
        with_metadata=True: {"query_text", "retrieval_metadata", "results"} envelope.
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

    if with_metadata:
        return {
            "query_text": query,
            "retrieval_metadata": build_retrieval_metadata(),
            "results": final_results,
        }

    return final_results


# ==================== 5. Example Usage ====================

if __name__ == "__main__":
    # Example query
    test_query = "How did Juliet die? Who killed Juliet?"

    print(f"\n{'='*60}")
    print(f"HYBRID RETRIEVAL SYSTEM v{PIPELINE_VERSION}")
    print(f"{'='*60}")
    print(f"\nQuery: {test_query}")
    print(f"\nConfiguration:")
    print(f"  - Cosine Weight (alpha): {COSINE_WEIGHT}")
    print(f"  - Graph Weight (beta):   {GRAPH_WEIGHT}")
    print(f"  - Path Penalty (gamma):  {PATH_PENALTY}")
    print(f"  - Graph Depth:           {GRAPH_DEPTH}")
    print(f"\n{'='*60}\n")

    # Perform hybrid retrieval (envelope form includes retrieval_metadata)
    envelope = hybrid_retrieve(test_query, top_k=5, with_metadata=True)
    results = envelope["results"]

    # Output as JSON
    print("Results (JSON):")
    print(json.dumps(envelope, indent=2, ensure_ascii=False))

    # Also print human-readable format
    print(f"\n{'='*60}")
    print(f"Top {len(results)} Results:")
    print(f"{'='*60}\n")

    for result in results:
        rank = result['reason'].get('rank', '?')
        sources = ", ".join(result['retrieval_sources'])
        raw_v = result['raw_vector_score']
        norm_v = result['normalized_vector_score']
        raw_v_str = f"{raw_v:.4f}" if raw_v is not None else "N/A"
        norm_v_str = f"{norm_v:.4f}" if norm_v is not None else "N/A"

        print(f"{rank}. [{result['node_id']}]  final={result['final_score']:.4f}  via: {sources}")
        print(f"   Raw cosine: {raw_v_str}  |  Normalized: {norm_v_str}  |  Graph: {result['graph_score']:.4f}")
        print(f"   Contributions: vector={result['vector_contribution']:.4f}  graph={result['graph_contribution']:.4f}  penalty={result['penalty_contribution']:.4f}")
        print(f"   Reason: hop={result['reason']['hop']}, edge={result['reason'].get('edge')}")
        print(f"   Text: {result['text'][:100]}...")
        print()
