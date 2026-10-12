"""Benchmark the production FastAPI hybrid retrieval service.

Run from the project root (the directory containing app/, db/, and
benchmark_dataset.json):
    python benchmark.py

This script imports app.service.HybridRetrievalService and calls its actual
hybrid_search implementation. It does not reimplement hybrid scoring.

Modes:
  - vector: production service's vector database search
  - graph_expansion: outgoing graph neighbors of the vector top-k seeds,
    excluding the vector hits themselves. This is a graph-expansion baseline,
    not an independent query-to-graph retrieval method.
  - hybrid: production service's HybridRetrievalService.hybrid_search

Optional environment variables:
  BENCHMARK_TOP_K=5
  BENCHMARK_GRAPH_DEPTH=2
  BENCHMARK_ALPHA=0.6
  BENCHMARK_BETA=0.4
  BENCHMARK_GAMMA=0.1
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET_PATH = ROOT / "benchmark_dataset.json"
RESULTS_PATH = ROOT / "benchmark_results.json"
PER_QUERY_PATH = ROOT / "benchmark_per_query_results.json"

TOP_K = int(os.getenv("BENCHMARK_TOP_K", "5"))
GRAPH_DEPTH = int(os.getenv("BENCHMARK_GRAPH_DEPTH", "2"))
ALPHA = float(os.getenv("BENCHMARK_ALPHA", "0.6"))
BETA = float(os.getenv("BENCHMARK_BETA", "0.4"))
GAMMA = float(os.getenv("BENCHMARK_GAMMA", "0.1"))


def load_dataset() -> List[Dict[str, Any]]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Benchmark dataset not found: {DATASET_PATH}. "
            "Place benchmark_dataset.json beside benchmark.py."
        )
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)
    if not isinstance(dataset, list) or not dataset:
        raise ValueError("benchmark_dataset.json must contain a non-empty JSON list.")
    for index, item in enumerate(dataset, 1):
        if not isinstance(item, dict) or "query" not in item or "relevant" not in item:
            raise ValueError(f"Dataset entry {index} must contain 'query' and 'relevant'.")
        if not isinstance(item["relevant"], list) or not item["relevant"]:
            raise ValueError(f"Dataset entry {index} must have a non-empty 'relevant' list.")
    return dataset


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    return sum(item in relevant for item in retrieved[:k]) / k if k else 0.0


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    return sum(item in relevant for item in retrieved[:k]) / len(relevant) if relevant else 0.0


def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    for rank, item in enumerate(retrieved, 1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(retrieved[:k], 1)
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def extract_vector_ids(service: Any, query: str, k: int) -> List[str]:
    """Use the production service's configured vector database."""
    rows = service.vector_db.search(query, k)
    ids: List[str] = []
    for doc_id, _text, _score, metadata in rows:
        node_id = doc_id or (metadata or {}).get("id")
        if node_id and node_id not in ids:
            ids.append(str(node_id))
    return ids


def graph_expansion_ids(
    service: Any,
    seed_ids: List[str],
    depth: int,
    top_k: int,
) -> List[str]:
    """Rank graph-only outgoing neighbors, matching production edge direction.

    Seeds come from the production vector search. Seed nodes are excluded from
    the returned list so this baseline measures the added graph candidates,
    rather than counting vector hits as graph retrieval successes.
    """
    graph = service.graph_db.graph
    seed_set = set(seed_ids)
    best_hop: Dict[str, int] = {}
    queue = deque()

    for seed in seed_ids:
        if seed in graph and seed not in best_hop:
            best_hop[seed] = 0
            queue.append((seed, 0))

    while queue:
        node_id, hop = queue.popleft()
        if hop >= depth:
            continue
        # Sorting neighbors makes equal-score runs reproducible.
        for neighbor in sorted(graph.successors(node_id), key=str):
            next_hop = hop + 1
            if neighbor not in best_hop or next_hop < best_hop[neighbor]:
                best_hop[neighbor] = next_hop
                queue.append((neighbor, next_hop))

    candidates = [
        (node_id, 1.0 / (hop + 1), hop)
        for node_id, hop in best_hop.items()
        if node_id not in seed_set and hop > 0
    ]
    candidates.sort(key=lambda row: (-row[1], row[2], str(row[0])))
    return [str(node_id) for node_id, _score, _hop in candidates[:top_k]]


def result_ids(results: Any) -> List[str]:
    ids: List[str] = []
    for result in results:
        node_id = getattr(result, "node_id", None)
        if node_id is None and isinstance(result, dict):
            node_id = result.get("node_id") or result.get("id")
        if node_id is not None:
            ids.append(str(node_id))
    return ids


def summarize(retrieved: List[str], relevant: Set[str], k: int) -> Dict[str, float]:
    return {
        f"precision@{k}": precision_at_k(retrieved, relevant, k),
        f"recall@{k}": recall_at_k(retrieved, relevant, k),
        "mrr": reciprocal_rank(retrieved, relevant),
        f"ndcg@{k}": ndcg_at_k(retrieved, relevant, k),
    }


def run_benchmark() -> Dict[str, Any]:
    dataset = load_dataset()

    # Import the same service class used by the FastAPI application.
    try:
        from app.service import HybridRetrievalService
    except Exception as exc:
        raise RuntimeError(
            "Could not import app.service.HybridRetrievalService. Run this script "
            "from the project root and verify that FastAPI imports this service."
        ) from exc

    service = HybridRetrievalService()
    modes = ["vector", "graph_expansion", "hybrid"]
    metric_names = [f"precision@{TOP_K}", f"recall@{TOP_K}", "mrr", f"ndcg@{TOP_K}"]
    metric_values: Dict[str, Dict[str, List[float]]] = {
        mode: {name: [] for name in metric_names} for mode in modes
    }
    per_query: List[Dict[str, Any]] = []

    graph_nodes = set(service.graph_db.graph.nodes)
    all_relevant = {str(node_id) for row in dataset for node_id in row["relevant"]}
    graph_overlap = all_relevant & {str(node_id) for node_id in graph_nodes}

    print("\n" + "=" * 88)
    print("PRODUCTION SERVICE RETRIEVAL BENCHMARK")
    print("Implementation: app.service.HybridRetrievalService")
    print(f"Queries={len(dataset)} | K={TOP_K} | depth={GRAPH_DEPTH}")
    print(f"Weights: alpha={ALPHA}, beta={BETA}, gamma={GAMMA}")
    print(f"Graph nodes={len(graph_nodes)} | relevant IDs found in graph={len(graph_overlap)}/{len(all_relevant)}")
    if not graph_overlap:
        print("WARNING: No ground-truth relevant IDs exist in the service graph. Results may be invalid for this corpus.")
    elif len(graph_overlap) < len(all_relevant):
        missing = sorted(all_relevant - graph_overlap)
        print(f"WARNING: Some ground-truth IDs are absent from the service graph: {missing[:10]}")
    print("=" * 88)

    for index, item in enumerate(dataset, 1):
        query = str(item["query"])
        relevant = {str(value) for value in item["relevant"]}

        started = time.perf_counter()
        vec_ids = extract_vector_ids(service, query, TOP_K)
        vector_latency_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        graph_ids = graph_expansion_ids(service, vec_ids, GRAPH_DEPTH, TOP_K)
        graph_latency_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        hybrid_response = service.hybrid_search(
            query=query,
            vector_weight=ALPHA,
            graph_weight=BETA,
            top_k=TOP_K,
            max_depth=GRAPH_DEPTH,
            path_distance_penalty=GAMMA,
        )
        hybrid_latency_ms = (time.perf_counter() - started) * 1000
        hybrid_ids = result_ids(hybrid_response.results)

        rankings = {
            "vector": vec_ids,
            "graph_expansion": graph_ids,
            "hybrid": hybrid_ids,
        }
        mode_latency = {
            "vector": round(vector_latency_ms, 3),
            "graph_expansion": round(graph_latency_ms, 3),
            "hybrid": round(hybrid_latency_ms, 3),
        }
        query_metrics: Dict[str, Dict[str, float]] = {}
        for mode, ids in rankings.items():
            scores = summarize(ids, relevant, TOP_K)
            query_metrics[mode] = {key: round(value, 6) for key, value in scores.items()}
            for key, value in scores.items():
                metric_values[mode][key].append(value)

        per_query.append({
            "id": item.get("id", index),
            "query": query,
            "category": item.get("category"),
            "relevant": sorted(relevant),
            "rankings": rankings,
            "metrics": query_metrics,
            "latency_ms": mode_latency,
        })

        print(f"\n[{index:02d}/{len(dataset)}] {query}")
        print(f"  Relevant: {sorted(relevant)}")
        for mode in modes:
            values = query_metrics[mode]
            print(
                f"  {mode:16s} {rankings[mode]} | "
                f"P@{TOP_K}={values[f'precision@{TOP_K}']:.3f} "
                f"R@{TOP_K}={values[f'recall@{TOP_K}']:.3f} "
                f"MRR={values['mrr']:.3f} "
                f"NDCG@{TOP_K}={values[f'ndcg@{TOP_K}']:.3f} "
                f"latency={mode_latency[mode]:.2f}ms"
            )

    aggregate: Dict[str, Dict[str, float]] = {}
    for mode in modes:
        aggregate[mode] = {
            metric: round(sum(values) / len(values), 4) if values else 0.0
            for metric, values in metric_values[mode].items()
        }
        aggregate[mode]["mean_latency_ms"] = round(
            sum(row["latency_ms"][mode] for row in per_query) / len(per_query), 3
        ) if per_query else 0.0

    output = {
        "benchmark_version": "2.0-service-aligned",
        "implementation": "app.service.HybridRetrievalService.hybrid_search",
        "configuration": {
            "top_k": TOP_K,
            "graph_depth": GRAPH_DEPTH,
            "alpha": ALPHA,
            "beta": BETA,
            "gamma": GAMMA,
            "vector_db_path_expected": "db/chroma_db_service",
            "graph_db_path_expected": "db/graph_data_service.json",
        },
        "ground_truth_validation": {
            "unique_relevant_ids": len(all_relevant),
            "relevant_ids_found_in_graph": len(graph_overlap),
            "relevant_ids_missing_from_graph": sorted(all_relevant - graph_overlap),
        },
        "aggregate": aggregate,
        "per_query": per_query,
    }

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump({"aggregate": aggregate, "configuration": output["configuration"],
                   "implementation": output["implementation"],
                   "ground_truth_validation": output["ground_truth_validation"]}, file, indent=2)
    with PER_QUERY_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 88)
    print(f"{'Metric':<18}{'Vector':>16}{'Graph expansion':>20}{'Hybrid':>16}")
    print("-" * 70)
    for metric in metric_names + ["mean_latency_ms"]:
        print(
            f"{metric:<18}"
            f"{aggregate['vector'].get(metric, 0):>16.4f}"
            f"{aggregate['graph_expansion'].get(metric, 0):>20.4f}"
            f"{aggregate['hybrid'].get(metric, 0):>16.4f}"
        )
    print("=" * 88)
    print(f"Aggregate results saved to: {RESULTS_PATH}")
    print(f"Per-query rankings and metrics saved to: {PER_QUERY_PATH}")
    print("Note: graph_expansion is seeded by vector results; it is not an independent graph-only retriever.")
    return output


if __name__ == "__main__":
    run_benchmark()
