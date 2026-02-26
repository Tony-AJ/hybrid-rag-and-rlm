"""
Evaluation Benchmark for Hybrid Retrieval System

Compares three retrieval modes:
  1. Pure Vector Retrieval (ChromaDB cosine similarity only)
  2. Pure Graph Retrieval (NetworkX graph traversal only)
  3. Hybrid Retrieval (Normalized α·vector + β·graph - γ·penalty)

Metrics:
  - Precision@K: fraction of retrieved items that are relevant
  - Recall@K: fraction of relevant items that are retrieved
  - MRR (Mean Reciprocal Rank): 1 / rank of first relevant result

Usage:
  python benchmark.py
"""

import json
import os
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict

import networkx as nx


# =====================================================================
# Configuration
# =====================================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_PATH = os.path.join(CURRENT_DIR, "vector_db", "db", "chroma_db_with_metadata")
GRAPH_DB_PATH = os.path.join(CURRENT_DIR, "db", "graph_data.json")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5
GRAPH_DEPTH = 2

# Hybrid hyperparameters (α, β, γ)
ALPHA = 0.7
BETA = 0.3
GAMMA = 0.1


# =====================================================================
# Ground Truth: queries → expected relevant chunk IDs
# =====================================================================

GROUND_TRUTH: List[Dict[str, Any]] = [
    {
        "query": "How did the feud end?",
        "relevant": {"chunk_8", "chunk_1"},
        "description": "The feud ended when both families grieved together (chunk_8), feud context (chunk_1)"
    },
    {
        "query": "Who killed Tybalt?",
        "relevant": {"chunk_5", "chunk_4"},
        "description": "Romeo killed Tybalt (chunk_5), Tybalt fought Mercutio (chunk_4)"
    },
    {
        "query": "Why did Juliet take the potion?",
        "relevant": {"chunk_6", "chunk_3"},
        "description": "To avoid marrying Paris (chunk_6), secret marriage context (chunk_3)"
    },
    {
        "query": "How did Romeo die?",
        "relevant": {"chunk_7", "chunk_6"},
        "description": "Romeo drank poison (chunk_7), Juliet's fake death plan (chunk_6)"
    },
    {
        "query": "Where is the play set?",
        "relevant": {"chunk_1"},
        "description": "Verona, Italy (chunk_1)"
    },
    {
        "query": "Who married Romeo and Juliet?",
        "relevant": {"chunk_3", "chunk_2"},
        "description": "Friar Lawrence (chunk_3), they met at the ball (chunk_2)"
    },
]


# =====================================================================
# Metric Functions
# =====================================================================

def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Fraction of top-K retrieved items that are relevant."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(top_k)


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Fraction of relevant items that appear in top-K retrieved."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """1 / rank of the first relevant result (0 if none found)."""
    for i, item in enumerate(retrieved, 1):
        if item in relevant:
            return 1.0 / i
    return 0.0


# =====================================================================
# Retrieval Functions
# =====================================================================

def run_vector_only(chroma_db, query: str, top_k: int) -> List[str]:
    """Pure vector retrieval using ChromaDB."""
    results = chroma_db.similarity_search_with_score(query, k=top_k)
    ids = []
    for doc, score in results:
        chunk_id = doc.metadata.get('id', doc.metadata.get('source', ''))
        if chunk_id:
            ids.append(chunk_id)
    return ids


def run_graph_only(graph_db, start_ids: List[str], depth: int, top_k: int) -> List[str]:
    """
    Pure graph retrieval: BFS from start_ids, score = 1/(hop+1).
    start_ids are seeded from the vector results (we need at least a starting point).
    """
    scores = {}
    visited = set()
    queue = []

    for sid in start_ids:
        if sid in graph_db.graph.nodes:
            queue.append((sid, 0))
            visited.add(sid)

    while queue:
        node_id, hop = queue.pop(0)
        g_score = 1.0 / (hop + 1)

        if node_id not in scores or g_score > scores[node_id]:
            scores[node_id] = g_score

        if hop < depth:
            for neighbor in graph_db.graph.successors(node_id):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, hop + 1))

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [nid for nid, _ in ranked[:top_k]]


def run_hybrid(chroma_db, graph_db, query: str, top_k: int, depth: int,
               alpha: float, beta: float, gamma: float) -> List[str]:
    """Normalized hybrid retrieval with α, β, γ."""
    # Vector search
    vector_results = chroma_db.similarity_search_with_score(query, k=top_k)

    candidates = {}
    start_nodes = []

    for doc, score in vector_results:
        nid = doc.metadata.get('id', '')
        if not nid:
            continue
        sim = 1.0 / (1.0 + score)
        candidates[nid] = {
            "raw_vector": sim,
            "graph_score": 0.0,
            "hop": 0,
        }
        start_nodes.append(nid)

    # Graph expansion
    visited = {}
    queue = []
    for nid in start_nodes:
        if nid not in visited and nid in graph_db.graph.nodes:
            visited[nid] = 0
            queue.append((nid, 0))

    while queue:
        curr_id, hops = queue.pop(0)
        g_score = 1.0 / (1.0 + hops)

        if curr_id not in candidates:
            candidates[curr_id] = {
                "raw_vector": 0.0,
                "graph_score": g_score,
                "hop": hops,
            }
        else:
            if g_score > candidates[curr_id]["graph_score"]:
                candidates[curr_id]["graph_score"] = g_score
                candidates[curr_id]["hop"] = hops

        if hops < depth:
            if curr_id in graph_db.graph:
                for neighbor in graph_db.graph.successors(curr_id):
                    if neighbor not in visited:
                        visited[neighbor] = hops + 1
                        queue.append((neighbor, hops + 1))

    # Normalize vector scores (min-max)
    raw_scores = [d["raw_vector"] for d in candidates.values() if d["raw_vector"] > 0]
    v_min = min(raw_scores) if raw_scores else 0.0
    v_max = max(raw_scores) if raw_scores else 1.0
    v_range = v_max - v_min if v_max != v_min else 1.0

    # Final scoring
    scored = []
    for nid, data in candidates.items():
        norm_v = (data["raw_vector"] - v_min) / v_range if data["raw_vector"] > 0 else 0.0
        norm_g = data["graph_score"]
        hop = data["hop"]
        penalty = (hop / depth) if depth > 0 else 0.0

        final = (alpha * norm_v) + (beta * norm_g) - (gamma * penalty)
        final = max(0.0, final)
        scored.append((nid, final))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [nid for nid, _ in scored[:top_k]]


# =====================================================================
# Main Benchmark
# =====================================================================

def run_benchmark():
    print("\n" + "=" * 70)
    print("  HYBRID RETRIEVAL BENCHMARK")
    print("  Comparing: Pure Vector | Pure Graph | Hybrid (Normalized)")
    print("=" * 70)

    # ── Initialize databases ──
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from graph_db.graph_db import GraphDatabase

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    chroma_db = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )
    graph = GraphDatabase(db_path=GRAPH_DB_PATH, auto_persist=True)

    # ── Accumulators ──
    modes = ["Vector Only", "Graph Only", "Hybrid"]
    metrics = {mode: {"precision": [], "recall": [], "rr": []} for mode in modes}

    print(f"\n  Hyperparameters: α={ALPHA}, β={BETA}, γ={GAMMA}")
    print(f"  Top-K={TOP_K}, Graph Depth={GRAPH_DEPTH}")
    print(f"  Queries: {len(GROUND_TRUTH)}")
    print()

    for i, gt in enumerate(GROUND_TRUTH, 1):
        query = gt["query"]
        relevant = gt["relevant"]

        print(f"  ── Query {i}: \"{query}\"")
        print(f"     Relevant: {relevant}")

        # Vector Only
        vec_ids = run_vector_only(chroma_db, query, TOP_K)

        # Graph Only (seeded from vector top-1 to have a starting point)
        seed = vec_ids[:1] if vec_ids else []
        graph_ids = run_graph_only(graph, seed, GRAPH_DEPTH, TOP_K)

        # Hybrid
        hybrid_ids = run_hybrid(chroma_db, graph, query, TOP_K, GRAPH_DEPTH, ALPHA, BETA, GAMMA)

        for mode, retrieved in zip(modes, [vec_ids, graph_ids, hybrid_ids]):
            p = precision_at_k(retrieved, relevant, TOP_K)
            r = recall_at_k(retrieved, relevant, TOP_K)
            rr = reciprocal_rank(retrieved, relevant)
            metrics[mode]["precision"].append(p)
            metrics[mode]["recall"].append(r)
            metrics[mode]["rr"].append(rr)

        print(f"     Vector: {vec_ids[:TOP_K]}")
        print(f"     Graph:  {graph_ids[:TOP_K]}")
        print(f"     Hybrid: {hybrid_ids[:TOP_K]}")
        print()

    # ── Summary Table ──
    print("=" * 70)
    print(f"  {'Metric':<25} {'Vector Only':>12} {'Graph Only':>12} {'Hybrid':>12}")
    print("  " + "-" * 64)

    summary = {}
    for mode in modes:
        avg_p = sum(metrics[mode]["precision"]) / len(metrics[mode]["precision"])
        avg_r = sum(metrics[mode]["recall"]) / len(metrics[mode]["recall"])
        avg_mrr = sum(metrics[mode]["rr"]) / len(metrics[mode]["rr"])
        summary[mode] = {"Precision@K": round(avg_p, 4), "Recall@K": round(avg_r, 4), "MRR": round(avg_mrr, 4)}

    for metric_name in ["Precision@K", "Recall@K", "MRR"]:
        row = f"  {metric_name:<25}"
        for mode in modes:
            row += f" {summary[mode][metric_name]:>12.4f}"
        print(row)

    print("=" * 70)

    # ── Determine Winner ──
    hybrid_mrr = summary["Hybrid"]["MRR"]
    vec_mrr = summary["Vector Only"]["MRR"]
    graph_mrr = summary["Graph Only"]["MRR"]

    if hybrid_mrr >= vec_mrr and hybrid_mrr >= graph_mrr:
        print("\n  ✅ HYBRID outperforms or matches both Pure Vector and Pure Graph!")
    elif vec_mrr > hybrid_mrr:
        print("\n  ⚠️  Pure Vector outperforms Hybrid – consider tuning α, β, γ.")
    else:
        print("\n  ⚠️  Pure Graph outperforms Hybrid – consider tuning α, β, γ.")

    # ── Save Results ──
    output = {
        "hyperparameters": {"alpha": ALPHA, "beta": BETA, "gamma": GAMMA, "top_k": TOP_K, "graph_depth": GRAPH_DEPTH},
        "queries": len(GROUND_TRUTH),
        "results": summary
    }

    output_path = os.path.join(CURRENT_DIR, "benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  📄 Results saved to: {output_path}")
    print("=" * 70 + "\n")

    return summary


if __name__ == "__main__":
    run_benchmark()
