"""
Evaluation Benchmark for Hybrid Retrieval System

Compares three retrieval modes:
  1. Pure Vector Retrieval (ChromaDB cosine similarity only)
  2. Pure Graph Retrieval (NetworkX graph traversal only)
  3. Hybrid Retrieval (Normalized alpha*vector + beta*graph - gamma*penalty)

Evaluates on a 20-query ground-truth benchmark dataset across:
  - Precision@5
  - Recall@5
  - MRR (Mean Reciprocal Rank)
  - NDCG@5 (Normalized Discounted Cumulative Gain at 5)

Usage:
  python benchmark.py
"""

import os
import sys
import json
import math
from typing import List, Dict, Any, Set, Tuple

# Suppress posthog telemetry warnings
import posthog
posthog.disabled = True
posthog.capture = lambda *args, **kwargs: None


# =====================================================================
# Configuration & Paths
# =====================================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

VECTOR_DB_PATH = os.path.join(CURRENT_DIR, "vector_db", "db", "chroma_db_with_metadata")
GRAPH_DB_PATH = os.path.join(CURRENT_DIR, "db", "graph_data.json")
DATASET_PATH = os.path.join(CURRENT_DIR, "benchmark_dataset.json")
RESULTS_PATH = os.path.join(CURRENT_DIR, "benchmark_results.json")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5
GRAPH_DEPTH = 2

# Hybrid hyperparameters (alpha: vector weight, beta: graph weight, gamma: distance penalty)
ALPHA = 0.6
BETA = 0.4
GAMMA = 0.1
NORM_EPSILON = 1e-6


# =====================================================================
# 20 Ground Truth Benchmark Queries
# =====================================================================

FALLBACK_GROUND_TRUTH: List[Dict[str, Any]] = [
    {
        "id": 1,
        "query": "Where does the story take place and which families are involved?",
        "relevant": ["chunk_1"],
        "category": "single-hop factual",
        "description": "Verona, Italy; Montagues and Capulets violent feud (chunk_1)"
    },
    {
        "id": 2,
        "query": "Where did Romeo first meet Juliet?",
        "relevant": ["chunk_2"],
        "category": "single-hop factual",
        "description": "Capulet ball in disguise (chunk_2)"
    },
    {
        "id": 3,
        "query": "Who performed the secret wedding ceremony for Romeo and Juliet?",
        "relevant": ["chunk_3"],
        "category": "single-hop factual",
        "description": "Friar Lawrence secretly marries Romeo and Juliet (chunk_3)"
    },
    {
        "id": 4,
        "query": "Who challenged Romeo to a duel and who was killed?",
        "relevant": ["chunk_4"],
        "category": "single-hop factual",
        "description": "Tybalt challenges Romeo; Mercutio fights and is killed (chunk_4)"
    },
    {
        "id": 5,
        "query": "Why was Romeo banished from Verona by the Prince?",
        "relevant": ["chunk_5"],
        "category": "single-hop factual",
        "description": "Romeo kills Tybalt and is banished from Verona (chunk_5)"
    },
    {
        "id": 6,
        "query": "What did Juliet do to avoid marrying Count Paris?",
        "relevant": ["chunk_6"],
        "category": "single-hop factual",
        "description": "Takes potion to appear dead (chunk_6)"
    },
    {
        "id": 7,
        "query": "How did Romeo die in Juliet's tomb?",
        "relevant": ["chunk_7"],
        "category": "single-hop factual",
        "description": "Drinks poison and dies by Juliet's side (chunk_7)"
    },
    {
        "id": 8,
        "query": "How did Juliet die after waking up?",
        "relevant": ["chunk_8"],
        "category": "single-hop factual",
        "description": "Stabs herself with Romeo's dagger (chunk_8)"
    },
    {
        "id": 9,
        "query": "How did Tybalt's challenge lead to Romeo's banishment?",
        "relevant": ["chunk_4", "chunk_5"],
        "category": "2-hop causal",
        "description": "Tybalt duel / Mercutio death (chunk_4) -> Romeo kills Tybalt & banishment (chunk_5)"
    },
    {
        "id": 10,
        "query": "What sequence of events followed Mercutio's death?",
        "relevant": ["chunk_4", "chunk_5", "chunk_6"],
        "category": "multi-hop sequential",
        "description": "Mercutio death (chunk_4) -> Romeo kills Tybalt (chunk_5) -> Juliet forced to marry Paris (chunk_6)"
    },
    {
        "id": 11,
        "query": "Why did Friar Lawrence marry Romeo and Juliet, and did his hope come true?",
        "relevant": ["chunk_3", "chunk_8"],
        "category": "semantic graph relation (goal_achieved)",
        "description": "Friar marries them to reconcile families (chunk_3) -> feud ends in grief (chunk_8)"
    },
    {
        "id": 12,
        "query": "How did the family feud escalate into violence and death in the streets?",
        "relevant": ["chunk_1", "chunk_4"],
        "category": "semantic graph relation (foreshadows)",
        "description": "Feud context (chunk_1) foreshadows duel and Mercutio killed (chunk_4)"
    },
    {
        "id": 13,
        "query": "What tragic miscommunication caused Romeo's death at the tomb?",
        "relevant": ["chunk_6", "chunk_7"],
        "category": "2-hop causal",
        "description": "Juliet takes potion plan (chunk_6) -> Romeo hears of death but not the plan (chunk_7)"
    },
    {
        "id": 14,
        "query": "How did Romeo's presence at the Capulet ball lead to Juliet taking the sleeping potion?",
        "relevant": ["chunk_2", "chunk_6"],
        "category": "semantic graph relation (character_arc)",
        "description": "Ball meeting (chunk_2) leads to forced Paris marriage and potion (chunk_6)"
    },
    {
        "id": 15,
        "query": "What were the consequences of Romeo killing Tybalt on Juliet's marriage situation?",
        "relevant": ["chunk_5", "chunk_6"],
        "category": "2-hop causal",
        "description": "Romeo banished (chunk_5) -> father forces Paris marriage and potion (chunk_6)"
    },
    {
        "id": 16,
        "query": "What happened when Juliet woke up from her potion-induced sleep?",
        "relevant": ["chunk_6", "chunk_7", "chunk_8"],
        "category": "multi-hop sequential",
        "description": "Potion sleep (chunk_6) -> finds Romeo dead from poison (chunk_7) -> stabs herself (chunk_8)"
    },
    {
        "id": 17,
        "query": "How did the ancient feud between Montagues and Capulets finally come to an end?",
        "relevant": ["chunk_1", "chunk_8"],
        "category": "narrative arc (beginning to end)",
        "description": "Feud begins story (chunk_1) -> families end feud in shared grief (chunk_8)"
    },
    {
        "id": 18,
        "query": "Trace the chain of events from the secret wedding to Romeo's exile.",
        "relevant": ["chunk_3", "chunk_4", "chunk_5"],
        "category": "multi-hop sequential",
        "description": "Secret wedding (chunk_3) -> Tybalt duel (chunk_4) -> Romeo kills Tybalt & exiled (chunk_5)"
    },
    {
        "id": 19,
        "query": "How did Mercutio's fight lead ultimately to the tragic deaths of the two lovers?",
        "relevant": ["chunk_4", "chunk_5", "chunk_7", "chunk_8"],
        "category": "extended multi-hop causal",
        "description": "Mercutio duel (chunk_4) -> Romeo exiled (chunk_5) -> Romeo poison (chunk_7) -> Juliet dagger (chunk_8)"
    },
    {
        "id": 20,
        "query": "What was the overarching narrative arc from Romeo attending the ball to the lovers' secret union?",
        "relevant": ["chunk_2", "chunk_3"],
        "category": "sequential narrative",
        "description": "Ball meeting (chunk_2) leads to secret marriage by Friar Lawrence (chunk_3)"
    }
]


def load_dataset() -> List[Dict[str, Any]]:
    """Load benchmark dataset from JSON file or fallback to embedded queries."""
    if os.path.exists(DATASET_PATH):
        try:
            with open(DATASET_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data
        except Exception:
            pass
    return FALLBACK_GROUND_TRUTH


# =====================================================================
# Metric Functions
# =====================================================================

def precision_at_k(retrieved: List[str], relevant: Set[str], k: int = 5) -> float:
    """Fraction of top-K retrieved items that are relevant."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int = 5) -> float:
    """Fraction of relevant items that appear in top-K retrieved."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """1 / rank of the first relevant result (0.0 if none found)."""
    for i, item in enumerate(retrieved, 1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int = 5) -> float:
    """
    Normalized Discounted Cumulative Gain at rank K.
    DCG = sum(rel_i / log2(i + 1))
    IDCG = sum(1 / log2(i + 1)) for ideal rank of size min(len(relevant), K)
    """
    top_k = retrieved[:k]
    if not top_k or not relevant:
        return 0.0

    dcg = 0.0
    for i, item in enumerate(top_k, 1):
        if item in relevant:
            dcg += 1.0 / math.log2(i + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


# =====================================================================
# Retrieval Implementations
# =====================================================================

def run_vector_only(chroma_db, query: str, top_k: int = 5) -> List[str]:
    """Pure vector retrieval using ChromaDB cosine similarity."""
    results = chroma_db.similarity_search_with_score(query, k=top_k)
    ids = []
    for doc, _ in results:
        chunk_id = doc.metadata.get('id', doc.metadata.get('source', ''))
        if chunk_id:
            ids.append(chunk_id)
    return ids


def run_graph_only(graph_db, start_ids: List[str], depth: int = 2, top_k: int = 5) -> List[str]:
    """
    Pure graph retrieval: BFS traversal from seed start_ids.
    Traverses both forward (successors) and backward (predecessors) relationships.
    Score = 1 / (hop + 1).
    """
    scores = {}
    visited = set()
    queue = []

    for sid in start_ids:
        if sid in graph_db.graph.nodes:
            queue.append((sid, 0))
            visited.add(sid)
            scores[sid] = 1.0

    while queue:
        node_id, hop = queue.pop(0)
        g_score = 1.0 / (hop + 1)

        if node_id not in scores or g_score > scores[node_id]:
            scores[node_id] = g_score

        if hop < depth:
            neighbors = set(graph_db.graph.successors(node_id)) | set(graph_db.graph.predecessors(node_id))
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, hop + 1))

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [nid for nid, _ in ranked[:top_k]]


def run_hybrid(chroma_db, graph_db, query: str, top_k: int = 5, depth: int = 2,
               alpha: float = ALPHA, beta: float = BETA, gamma: float = GAMMA) -> List[str]:
    """
    Research-grade Hybrid Retrieval:
      1. ChromaDB vector search with exact cosine distance conversion.
      2. Graph expansion seeded from top vector hits.
      3. Min-Max normalization of vector similarity.
      4. Hybrid scoring: final = (alpha * norm_v) + (beta * graph_score) - (gamma * penalty)
    """
    vector_results = chroma_db.similarity_search_with_score(query, k=top_k)

    candidates = {}
    start_nodes = []

    for doc, dist in vector_results:
        nid = doc.metadata.get('id', '')
        if not nid:
            continue
        # Convert L2 distance to cosine similarity: cosine = 1 - (L2^2 / 2)
        cosine_sim = max(0.0, 1.0 - (dist / 2.0))
        candidates[nid] = {
            "raw_vector": cosine_sim,
            "graph_score": 0.0,
            "hop": 0,
        }
        start_nodes.append(nid)

    # Graph expansion from top vector hits
    visited = {}
    queue = []
    for nid in start_nodes[:2]:
        if nid in graph_db.graph.nodes and nid not in visited:
            visited[nid] = 0
            queue.append((nid, 0))
            if nid in candidates:
                candidates[nid]["graph_score"] = 1.0

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
            neighbors = set(graph_db.graph.successors(curr_id)) | set(graph_db.graph.predecessors(curr_id))
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited[neighbor] = hops + 1
                    queue.append((neighbor, hops + 1))

    # Min-max normalization for vector scores
    raw_scores = [d["raw_vector"] for d in candidates.values() if d["raw_vector"] > 0]
    v_min = min(raw_scores) if raw_scores else 0.0
    v_max = max(raw_scores) if raw_scores else 1.0
    v_range = v_max - v_min if (v_max - v_min) > NORM_EPSILON else 1.0

    scored = []
    for nid, data in candidates.items():
        norm_v = (data["raw_vector"] - v_min) / v_range if data["raw_vector"] > 0 else 0.0
        norm_g = data["graph_score"]
        hop = data["hop"]
        penalty = (hop / depth) if depth > 0 else 0.0

        final = (alpha * norm_v) + (beta * norm_g) - (gamma * penalty)
        scored.append((nid, max(0.0, final)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [nid for nid, _ in scored[:top_k]]


# =====================================================================
# Main Benchmark Runner
# =====================================================================

def run_benchmark():
    dataset = load_dataset()

    print("\n" + "=" * 70)
    print("  HYBRID RETRIEVAL BENCHMARK")
    print("  Comparing: Pure Vector | Pure Graph | Hybrid (Normalized)")
    print(f"  Dataset: {len(dataset)} Ground-Truth Queries")
    print(f"  Hyperparameters: alpha={ALPHA}, beta={BETA}, gamma={GAMMA}")
    print(f"  Top-K={TOP_K}, Graph Depth={GRAPH_DEPTH}")
    print("=" * 70 + "\n")

    # Initialize databases
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from graph_db.graph_db import GraphDatabase

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    chroma_db = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )
    graph = GraphDatabase(db_path=GRAPH_DB_PATH, auto_persist=True)

    modes = ["vector", "graph", "hybrid"]
    metrics = {m: {"precision@5": [], "recall@5": [], "mrr": [], "ndcg@5": []} for m in modes}

    for i, item in enumerate(dataset, 1):
        query = item["query"]
        relevant = set(item["relevant"])

        # Vector Only
        vec_ids = run_vector_only(chroma_db, query, TOP_K)

        # Graph Only (seeded from vector top-1 hit)
        seed = vec_ids[:1] if vec_ids else []
        graph_ids = run_graph_only(graph, seed, GRAPH_DEPTH, TOP_K)

        # Hybrid
        hybrid_ids = run_hybrid(chroma_db, graph, query, TOP_K, GRAPH_DEPTH, ALPHA, BETA, GAMMA)

        for mode, retrieved in zip(modes, [vec_ids, graph_ids, hybrid_ids]):
            p = precision_at_k(retrieved, relevant, TOP_K)
            r = recall_at_k(retrieved, relevant, TOP_K)
            rr = reciprocal_rank(retrieved, relevant)
            ndcg = ndcg_at_k(retrieved, relevant, TOP_K)

            metrics[mode]["precision@5"].append(p)
            metrics[mode]["recall@5"].append(r)
            metrics[mode]["mrr"].append(rr)
            metrics[mode]["ndcg@5"].append(ndcg)

        print(f"  [{i:2d}/{len(dataset)}] Query: \"{query}\"")
        print(f"       Relevant: {sorted(list(relevant))}")
        print(f"       Vector:   {vec_ids} (P@5={p:.2f}, NDCG@5={ndcg:.2f})")
        print(f"       Graph:    {graph_ids}")
        print(f"       Hybrid:   {hybrid_ids}")
        print()

    # Aggregate results into target JSON structure
    benchmark_output: Dict[str, Dict[str, float]] = {}
    for mode in modes:
        n_queries = len(dataset)
        avg_p = sum(metrics[mode]["precision@5"]) / n_queries
        avg_r = sum(metrics[mode]["recall@5"]) / n_queries
        avg_mrr = sum(metrics[mode]["mrr"]) / n_queries
        avg_ndcg = sum(metrics[mode]["ndcg@5"]) / n_queries

        benchmark_output[mode] = {
            "precision@5": round(avg_p, 4),
            "recall@5": round(avg_r, 4),
            "mrr": round(avg_mrr, 4),
            "ndcg@5": round(avg_ndcg, 4)
        }

    # Summary Table
    print("=" * 70)
    print(f"  {'Metric':<18} {'Vector':>14} {'Graph':>14} {'Hybrid':>14}")
    print("  " + "-" * 64)
    for metric_name in ["precision@5", "recall@5", "mrr", "ndcg@5"]:
        row = f"  {metric_name:<18}"
        for mode in modes:
            row += f" {benchmark_output[mode][metric_name]:>14.4f}"
        print(row)
    print("=" * 70)

    # Verification of Hybrid approach improvement
    vec_ndcg = benchmark_output["vector"]["ndcg@5"]
    hybrid_ndcg = benchmark_output["hybrid"]["ndcg@5"]
    graph_ndcg = benchmark_output["graph"]["ndcg@5"]

    print("\n--- BENCHMARK RESULTS JSON ---")
    print(json.dumps(benchmark_output, indent=2))

    if hybrid_ndcg >= vec_ndcg and hybrid_ndcg >= graph_ndcg:
        print("\n  [EVIDENCE] Hybrid retrieval successfully improves/matches retrieval quality!")
    else:
        print("\n  [NOTE] Consider tuning hyperparameters (alpha, beta, gamma).")

    # Save to benchmark_results.json
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark_output, f, indent=2)

    print(f"\n  [SAVED] Output saved to: {RESULTS_PATH}")
    print("=" * 70 + "\n")

    return benchmark_output


if __name__ == "__main__":
    run_benchmark()
