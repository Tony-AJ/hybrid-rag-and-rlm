"""Benchmark the production FastAPI hybrid retrieval service.

Run from the project root (the directory containing app/, db/, and
benchmark_dataset.json):
    python benchmark.py

This script imports app.service.HybridRetrievalService and calls its actual
hybrid_search implementation. It does not reimplement hybrid scoring.

Validity rules (a benchmark that cannot be trusted must not print metrics):
  1. The selected corpus is declared explicitly. There is no silent fallback to
     whatever databases the service happens to default to.
  2. The relevance dataset is bound to that corpus by corpus id, dataset file
     digest and a content fingerprint of the indexed documents. Labels from one
     corpus are never scored against another, even when the node IDs collide.
  3. Every ground-truth ID must exist in the graph and in the vector index
     before a single query is run. IDs are never rewritten or remapped.
  4. Retrieval exceptions propagate. An empty ranking means retrieval returned
     nothing, never that an error was swallowed.
  5. If validation fails, the run reports {"status": "invalid", "metrics": null}
     and exits non-zero. It never reports precision/recall/NDCG of 0.0.

Modes:
  - vector: production service's vector database search
  - graph_expansion: outgoing graph neighbors of the vector top-k seeds,
    excluding the vector hits themselves. This is a graph-expansion baseline,
    not an independent query-to-graph retrieval method.
  - hybrid: production service's HybridRetrievalService.hybrid_search

Optional environment variables:
  BENCHMARK_CORPUS_ID=romeo-and-juliet-summary-v1
  BENCHMARK_TOP_K=5
  BENCHMARK_GRAPH_DEPTH=2
  BENCHMARK_ALPHA=0.6
  BENCHMARK_BETA=0.4
  BENCHMARK_GAMMA=0.1
  BENCHMARK_PRINT_FINGERPRINT=1   # print the live corpus fingerprint and exit
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# If executed with system Python where dependencies are missing, transparently delegate to .venv
_venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
if _venv_python.exists() and Path(sys.executable).resolve() != _venv_python.resolve():
    try:
        import fastapi  # type: ignore
    except ImportError:
        import subprocess
        result = subprocess.run([str(_venv_python)] + sys.argv)
        sys.exit(result.returncode)


RESULTS_PATH = ROOT / "benchmark_results.json"
PER_QUERY_PATH = ROOT / "benchmark_per_query_results.json"

TOP_K = int(os.getenv("BENCHMARK_TOP_K", "5"))
GRAPH_DEPTH = int(os.getenv("BENCHMARK_GRAPH_DEPTH", "2"))
ALPHA = float(os.getenv("BENCHMARK_ALPHA", "0.6"))
BETA = float(os.getenv("BENCHMARK_BETA", "0.4"))
GAMMA = float(os.getenv("BENCHMARK_GAMMA", "0.1"))

# ---------------------------------------------------------------------------
# Corpus registry: which relevance dataset belongs to which indexed corpus.
#
# A corpus entry pins:
#   vector_db_path / graph_db_path  - the stores to evaluate against
#   dataset / dataset_sha256        - the relevance judgments made FOR this corpus
#   vector_doc_count / fingerprint  - content identity of the indexed documents
#
# `corpus_fingerprint` is a SHA-256 over the sorted (chroma_id, metadata_id,
# document_text) triples of the whole vector collection, so the same node IDs
# carrying different text in a different corpus cannot pass validation.
# Regenerate a fingerprint with: BENCHMARK_PRINT_FINGERPRINT=1 python benchmark.py
# ---------------------------------------------------------------------------
DEFAULT_CORPUS_ID = "romeo-and-juliet-summary-v1"

CORPUS_REGISTRY: Dict[str, Dict[str, Any]] = {
    "romeo-and-juliet-summary-v1": {
        "description": (
            "8-chunk Romeo and Juliet plot summary, node IDs chunk_1..chunk_8, "
            "with a 'next'/semantic relation graph over those chunks."
        ),
        "vector_db_path": "vector_db/db/chroma_db_with_metadata",
        "graph_db_path": "db/graph_data.json",
        "dataset": "benchmark_dataset.json",
        "dataset_version": "1.0",
        "dataset_sha256": "47d0d867f4b6fcb4666b0b081f31091a7c20cacf3109cf9356644928e6d33f83",
        "vector_doc_count": 8,
        "corpus_fingerprint": "8b9c97acc24e1920cc2d4f57b7b9b551285e40ecc64a1d4d8e0eb959cffb379b",
    },
    # Production upload corpus (resume + user manuals). No relevance judgments
    # exist for it yet. Writing them is Option B; until then this corpus cannot
    # be benchmarked, and the Romeo labels must NOT be reused for it.
    "production-uploads-v1": {
        "description": (
            "Documents uploaded through the FastAPI PDF flow (Tony AJ resume, "
            "user manuals). IDs look like '<filename>.pdf_chunk_<n>'."
        ),
        "vector_db_path": "db/chroma_db_service",
        "graph_db_path": "db/graph_data_service.json",
        "dataset": None,
        "dataset_version": None,
        "dataset_sha256": None,
        "vector_doc_count": None,
        "corpus_fingerprint": None,
    },
}

CORPUS_ID = os.getenv("BENCHMARK_CORPUS_ID", DEFAULT_CORPUS_ID)


class BenchmarkInvalid(ValueError):
    """The benchmark cannot produce trustworthy metrics.

    Subclasses ValueError so `raise`-based validation stays compatible with
    callers that catch ValueError. Carries a machine-readable reason so the
    runner can emit {"status": "invalid", "metrics": null}.
    """

    def __init__(self, reason: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}


# ============================== Dataset loading ==============================


def resolve_corpus(corpus_id: str) -> Dict[str, Any]:
    if corpus_id not in CORPUS_REGISTRY:
        raise BenchmarkInvalid(
            f"Unknown corpus id {corpus_id!r}. Known ids: {sorted(CORPUS_REGISTRY)}.",
            {"corpus_id": corpus_id, "known_corpora": sorted(CORPUS_REGISTRY)},
        )
    corpus = dict(CORPUS_REGISTRY[corpus_id])
    corpus["corpus_id"] = corpus_id
    if not corpus.get("dataset"):
        raise BenchmarkInvalid(
            f"Corpus {corpus_id!r} has no relevance dataset. Create questions and "
            "relevance labels for the documents actually indexed in this corpus. "
            "Do not reuse another corpus's labels.",
            {"corpus_id": corpus_id},
        )
    return corpus


def load_dataset(corpus: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load the relevance dataset and verify it was written for this corpus."""
    dataset_path = ROOT / str(corpus["dataset"])
    if not dataset_path.exists():
        raise BenchmarkInvalid(
            f"Benchmark dataset not found: {dataset_path}.",
            {"dataset_path": str(dataset_path)},
        )

    raw_bytes = dataset_path.read_bytes()
    actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    expected_sha256 = corpus.get("dataset_sha256")
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise BenchmarkInvalid(
            "Relevance dataset changed since it was bound to this corpus "
            f"({dataset_path.name}: expected sha256 {expected_sha256}, found "
            f"{actual_sha256}). Re-verify the judgments against the indexed "
            "documents, then update dataset_sha256 in CORPUS_REGISTRY.",
            {
                "dataset_path": str(dataset_path),
                "expected_sha256": expected_sha256,
                "actual_sha256": actual_sha256,
            },
        )

    payload = json.loads(raw_bytes.decode("utf-8"))

    # Two accepted shapes: a bare list (legacy), or a wrapper that declares its
    # own corpus identity. A wrapper that names a different corpus is fatal.
    declared_corpus_id = None
    declared_version = None
    if isinstance(payload, dict):
        declared_corpus_id = payload.get("corpus_id")
        declared_version = payload.get("dataset_version")
        dataset = payload.get("queries")
    else:
        dataset = payload

    if not isinstance(dataset, list) or not dataset:
        raise BenchmarkInvalid(
            f"{dataset_path.name} must contain a non-empty JSON list of queries "
            "(or an object with a non-empty 'queries' list).",
            {"dataset_path": str(dataset_path)},
        )

    if declared_corpus_id and declared_corpus_id != corpus["corpus_id"]:
        raise BenchmarkInvalid(
            f"Dataset {dataset_path.name} declares corpus_id "
            f"{declared_corpus_id!r} but the run selected {corpus['corpus_id']!r}. "
            "Relevance labels from one corpus must never be scored against another.",
            {
                "dataset_corpus_id": declared_corpus_id,
                "selected_corpus_id": corpus["corpus_id"],
            },
        )

    for index, item in enumerate(dataset, 1):
        if not isinstance(item, dict) or "query" not in item or "relevant" not in item:
            raise BenchmarkInvalid(
                f"Dataset entry {index} must contain 'query' and 'relevant'.",
                {"entry": index},
            )
        if not isinstance(item["relevant"], list) or not item["relevant"]:
            raise BenchmarkInvalid(
                f"Dataset entry {index} must have a non-empty 'relevant' list.",
                {"entry": index},
            )
        for node_id in item["relevant"]:
            if not isinstance(node_id, str) or not node_id.strip():
                raise BenchmarkInvalid(
                    f"Dataset entry {index} has an empty or non-string relevant ID.",
                    {"entry": index, "relevant": item["relevant"]},
                )

    provenance = {
        "dataset_path": dataset_path.name,
        "dataset_sha256": actual_sha256,
        "dataset_version": declared_version or corpus.get("dataset_version"),
        "declared_corpus_id": declared_corpus_id,
    }
    return dataset, provenance


# ============================ Corpus validation =============================


def corpus_fingerprint(service: Any) -> Tuple[int, str, List[str]]:
    """Content identity of the indexed vector collection.

    Returns (doc_count, sha256, sorted vector document IDs). The hash covers
    text, so two corpora that share node IDs still produce different digests.
    """
    stored = service.vector_db.db.get(include=["documents", "metadatas"])
    raw_ids = stored.get("ids") or []
    documents = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []

    rows: List[Tuple[str, str, str]] = []
    vector_ids: List[str] = []
    for index, raw_id in enumerate(raw_ids):
        metadata = metadatas[index] if index < len(metadatas) else {}
        metadata_id = (metadata or {}).get("id")
        text = documents[index] if index < len(documents) else ""
        rows.append((str(raw_id), str(metadata_id), text or ""))
        # The service resolves a hit to metadata['id'], falling back to the
        # Chroma ID, so validation must consider both as addressable.
        vector_ids.append(str(metadata_id) if metadata_id else str(raw_id))

    rows.sort()
    blob = "\n".join("\t".join(row) for row in rows)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return len(rows), digest, sorted(set(vector_ids))


def validate_corpus_and_ground_truth(
    service: Any,
    dataset: List[Dict[str, Any]],
    corpus: Dict[str, Any],
) -> Dict[str, Any]:
    """Fail before any query runs if the labels do not match the indexed corpus."""
    graph_ids = set(service.graph_db.graph.nodes)
    relevant_ids = {
        node_id
        for item in dataset
        for node_id in item["relevant"]
    }

    # --- 1. Ground-truth IDs must exist in the graph. No remapping, ever. ---
    missing_ids = relevant_ids - graph_ids
    if missing_ids:
        raise BenchmarkInvalid(
            "INVALID BENCHMARK: Ground-truth IDs are missing "
            f"from the production graph: {sorted(missing_ids)}. "
            "Use a matching corpus and relevance dataset.",
            {
                "missing_from_graph": sorted(missing_ids),
                "graph_node_count": len(graph_ids),
                "graph_db_path": service.graph_db_path,
            },
        )

    # --- 2. Corpus identity: same IDs can live in unrelated datasets. ---
    doc_count, digest, vector_ids = corpus_fingerprint(service)

    missing_from_vector = sorted(relevant_ids - set(vector_ids))
    if missing_from_vector:
        raise BenchmarkInvalid(
            "INVALID BENCHMARK: Ground-truth IDs are missing from the vector "
            f"index: {missing_from_vector}. Use a matching corpus and relevance "
            "dataset.",
            {
                "missing_from_vector": missing_from_vector,
                "vector_doc_count": doc_count,
                "vector_db_path": service.vector_db_path,
            },
        )

    expected_count = corpus.get("vector_doc_count")
    if expected_count is not None and doc_count != expected_count:
        raise BenchmarkInvalid(
            f"Corpus {corpus['corpus_id']!r} should contain {expected_count} "
            f"indexed documents but the vector store holds {doc_count}. The "
            "relevance judgments were not made for this index.",
            {"expected_vector_doc_count": expected_count, "actual_vector_doc_count": doc_count},
        )

    expected_fingerprint = corpus.get("corpus_fingerprint")
    if not expected_fingerprint:
        raise BenchmarkInvalid(
            f"Corpus {corpus['corpus_id']!r} has no pinned corpus_fingerprint, so "
            "the ground-truth judgments cannot be tied to the indexed documents. "
            "Record one with BENCHMARK_PRINT_FINGERPRINT=1 after confirming the "
            "labels describe these exact documents.",
            {"actual_corpus_fingerprint": digest},
        )
    if digest != expected_fingerprint:
        raise BenchmarkInvalid(
            f"Corpus content mismatch for {corpus['corpus_id']!r}: expected "
            f"fingerprint {expected_fingerprint}, found {digest}. The indexed "
            "documents are not the ones these relevance labels were written for.",
            {
                "expected_corpus_fingerprint": expected_fingerprint,
                "actual_corpus_fingerprint": digest,
                "vector_db_path": service.vector_db_path,
                "graph_db_path": service.graph_db_path,
            },
        )

    # --- Non-fatal integrity note: IDs that exist only as edge endpoints. ---
    # NetworkX creates a node implicitly when an edge references it, so a
    # ground-truth ID can be "in the graph" while carrying no stored text.
    # Graph expansion still traverses it, so this is reported, not fatal.
    textless_ids = sorted(
        node_id
        for node_id in relevant_ids
        if not (service.graph_db.graph.nodes[node_id].get("text") or "").strip()
    )

    return {
        "corpus_id": corpus["corpus_id"],
        "vector_db_path": service.vector_db_path,
        "graph_db_path": service.graph_db_path,
        "unique_relevant_ids": len(relevant_ids),
        "relevant_ids_found_in_graph": len(relevant_ids),
        "relevant_ids_missing_from_graph": [],
        "relevant_ids_missing_from_vector_index": [],
        "graph_node_count": len(graph_ids),
        "vector_doc_count": doc_count,
        "corpus_fingerprint": digest,
        "relevant_ids_without_graph_text": textless_ids,
    }


# ================================== Metrics ==================================


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


def summarize(retrieved: List[str], relevant: Set[str], k: int) -> Dict[str, float]:
    return {
        f"precision@{k}": precision_at_k(retrieved, relevant, k),
        f"recall@{k}": recall_at_k(retrieved, relevant, k),
        "mrr": reciprocal_rank(retrieved, relevant),
        f"ndcg@{k}": ndcg_at_k(retrieved, relevant, k),
    }


# ================================= Retrieval =================================


def validate_ranking(ids: List[Any], query: str, mode: str) -> List[Any]:
    """Reject rankings that cannot be scored honestly.

    The production result objects expose `node_id`; IDs are extracted before
    this call, so the checks run on the exact list that will be scored.
    """
    if len(ids) != len(set(ids)):
        duplicates = sorted({node_id for node_id in ids if ids.count(node_id) > 1})
        raise ValueError(
            f"Duplicate result IDs in {mode} ranking for query: {query!r} "
            f"(duplicates: {duplicates})"
        )
    if any(node_id is None or not str(node_id).strip() for node_id in ids):
        raise ValueError(
            f"Invalid result ID in {mode} ranking for query: {query!r} (ids: {ids!r})"
        )
    return ids


def extract_vector_ids(service: Any, query: str, k: int) -> List[str]:
    """Use the production service's configured vector database."""
    rows = service.vector_db.search(query, k)
    ids: List[str] = []
    for doc_id, _text, _score, metadata in rows:
        node_id = doc_id or (metadata or {}).get("id")
        if node_id is None or not str(node_id).strip():
            raise ValueError(
                f"Vector search returned a row with no usable node ID for query: "
                f"{query!r} (metadata: {metadata!r})"
            )
        node_id = str(node_id)
        if node_id not in ids:
            ids.append(node_id)
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


def result_ids(results: Any, query: str) -> List[str]:
    """Extract node IDs from HybridSearchResultItem objects (or dict rows)."""
    ids: List[str] = []
    for result in results:
        node_id = getattr(result, "node_id", None)
        if node_id is None and isinstance(result, dict):
            node_id = result.get("node_id") or result.get("id")
        if node_id is None or not str(node_id).strip():
            raise ValueError(
                f"Hybrid search returned a result with no node_id for query: "
                f"{query!r} (result: {result!r})"
            )
        ids.append(str(node_id))
    return ids


# ================================== Runner ===================================


def build_service(corpus: Dict[str, Any]) -> Any:
    """Construct the production service against the explicitly selected corpus.

    No fallback to the service defaults: an absent store is a validation
    failure, not a reason to benchmark a different corpus.
    """
    try:
        from app.service import HybridRetrievalService
    except Exception as exc:
        raise RuntimeError(
            "Could not import app.service.HybridRetrievalService. Run this script "
            "from the project root and verify that FastAPI imports this service."
        ) from exc

    vector_path = ROOT / str(corpus["vector_db_path"])
    graph_path = ROOT / str(corpus["graph_db_path"])
    missing_paths = [str(path) for path in (vector_path, graph_path) if not path.exists()]
    if missing_paths:
        raise BenchmarkInvalid(
            f"Corpus {corpus['corpus_id']!r} stores are missing: {missing_paths}. "
            "Restore the matching vector index and graph, or select a corpus that "
            "exists via BENCHMARK_CORPUS_ID.",
            {"corpus_id": corpus["corpus_id"], "missing_paths": missing_paths},
        )

    return HybridRetrievalService(
        vector_db_path=str(vector_path),
        graph_db_path=str(graph_path),
    )


def print_fingerprint() -> int:
    """Report the live corpus identity so a fingerprint can be pinned."""
    corpus = dict(CORPUS_REGISTRY.get(CORPUS_ID, {}))
    corpus["corpus_id"] = CORPUS_ID
    if "vector_db_path" not in corpus:
        print(f"Unknown corpus id {CORPUS_ID!r}. Known: {sorted(CORPUS_REGISTRY)}")
        return 2
    service = build_service(corpus)
    doc_count, digest, vector_ids = corpus_fingerprint(service)
    print(f"corpus_id            : {CORPUS_ID}")
    print(f"vector_db_path       : {service.vector_db_path}")
    print(f"graph_db_path        : {service.graph_db_path}")
    print(f"vector_doc_count     : {doc_count}")
    print(f"corpus_fingerprint   : {digest}")
    print(f"graph_node_count     : {service.graph_db.graph.number_of_nodes()}")
    print(f"vector_ids[:10]      : {vector_ids[:10]}")
    return 0


def run_benchmark() -> Dict[str, Any]:
    corpus = resolve_corpus(CORPUS_ID)
    dataset, dataset_provenance = load_dataset(corpus)
    service = build_service(corpus)

    # Nothing is measured until the ground truth is proven to describe this corpus.
    validation = validate_corpus_and_ground_truth(service, dataset, corpus)
    validation.update(dataset_provenance)

    modes = ["vector", "graph_expansion", "hybrid"]
    metric_names = [f"precision@{TOP_K}", f"recall@{TOP_K}", "mrr", f"ndcg@{TOP_K}"]
    metric_values: Dict[str, Dict[str, List[float]]] = {
        mode: {name: [] for name in metric_names} for mode in modes
    }
    per_query: List[Dict[str, Any]] = []

    print("\n" + "=" * 88)
    print("PRODUCTION SERVICE RETRIEVAL BENCHMARK")
    print("Implementation: app.service.HybridRetrievalService")
    print(f"Corpus={corpus['corpus_id']} | dataset={dataset_provenance['dataset_path']} "
          f"(version {dataset_provenance['dataset_version']})")
    print(f"Vector store={service.vector_db_path} ({validation['vector_doc_count']} docs)")
    print(f"Graph store={service.graph_db_path} ({validation['graph_node_count']} nodes)")
    print(f"Corpus fingerprint={validation['corpus_fingerprint']}")
    print(f"Queries={len(dataset)} | K={TOP_K} | depth={GRAPH_DEPTH}")
    print(f"Weights: alpha={ALPHA}, beta={BETA}, gamma={GAMMA}")
    print(f"Ground truth: {validation['unique_relevant_ids']} unique relevant IDs, all present")
    if validation["relevant_ids_without_graph_text"]:
        print(
            "NOTE: these ground-truth IDs exist in the graph only as edge endpoints "
            f"and carry no stored text: {validation['relevant_ids_without_graph_text']}. "
            "Graph expansion traverses them; graph-only text retrieval cannot."
        )
    print("=" * 88)

    for index, item in enumerate(dataset, 1):
        query = str(item["query"])
        relevant = {str(value) for value in item["relevant"]}

        # Retrieval failures must surface. An empty ranking may only mean
        # "retrieval found nothing", never "an exception was hidden".
        started = time.perf_counter()
        try:
            vec_ids = extract_vector_ids(service, query, TOP_K)
        except Exception as exc:
            raise RuntimeError(
                f"Vector retrieval failed for query: {query!r}"
            ) from exc
        vector_latency_ms = (time.perf_counter() - started) * 1000
        validate_ranking(vec_ids, query, "vector")

        started = time.perf_counter()
        try:
            graph_ids = graph_expansion_ids(service, vec_ids, GRAPH_DEPTH, TOP_K)
        except Exception as exc:
            raise RuntimeError(
                f"Graph expansion failed for query: {query!r}"
            ) from exc
        graph_latency_ms = (time.perf_counter() - started) * 1000
        validate_ranking(graph_ids, query, "graph_expansion")

        started = time.perf_counter()
        try:
            hybrid_response = service.hybrid_search(
                query=query,
                vector_weight=ALPHA,
                graph_weight=BETA,
                top_k=TOP_K,
                max_depth=GRAPH_DEPTH,
                path_distance_penalty=GAMMA,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Retrieval failed for query: {query!r}"
            ) from exc
        hybrid_latency_ms = (time.perf_counter() - started) * 1000
        hybrid_ids = result_ids(hybrid_response.results, query)
        validate_ranking(hybrid_ids, query, "hybrid")

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

    if not per_query:
        raise BenchmarkInvalid(
            "No query produced a ranking, so no metrics can be reported.",
            {"corpus_id": corpus["corpus_id"]},
        )

    aggregate: Dict[str, Dict[str, float]] = {}
    for mode in modes:
        aggregate[mode] = {
            metric: round(sum(values) / len(values), 4)
            for metric, values in metric_values[mode].items()
        }
        aggregate[mode]["mean_latency_ms"] = round(
            sum(row["latency_ms"][mode] for row in per_query) / len(per_query), 3
        )

    configuration = {
        "corpus_id": corpus["corpus_id"],
        "corpus_description": corpus.get("description"),
        "dataset": dataset_provenance["dataset_path"],
        "dataset_version": dataset_provenance["dataset_version"],
        "dataset_sha256": dataset_provenance["dataset_sha256"],
        "top_k": TOP_K,
        "graph_depth": GRAPH_DEPTH,
        "alpha": ALPHA,
        "beta": BETA,
        "gamma": GAMMA,
        "vector_db_path": service.vector_db_path,
        "graph_db_path": service.graph_db_path,
    }

    output = {
        "status": "valid",
        "benchmark_version": "2.1-corpus-validated",
        "implementation": "app.service.HybridRetrievalService.hybrid_search",
        "configuration": configuration,
        "ground_truth_validation": validation,
        "aggregate": aggregate,
        "per_query": per_query,
    }

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump({
            "status": "valid",
            "aggregate": aggregate,
            "configuration": configuration,
            "implementation": output["implementation"],
            "ground_truth_validation": validation,
        }, file, indent=2)
    with PER_QUERY_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 88)
    print(f"{'Metric':<18}{'Vector':>16}{'Graph expansion':>20}{'Hybrid':>16}")
    print("-" * 70)
    for metric in metric_names + ["mean_latency_ms"]:
        print(
            f"{metric:<18}"
            f"{aggregate['vector'][metric]:>16.4f}"
            f"{aggregate['graph_expansion'][metric]:>20.4f}"
            f"{aggregate['hybrid'][metric]:>16.4f}"
        )
    print("=" * 88)
    print(f"Aggregate results saved to: {RESULTS_PATH}")
    print(f"Per-query rankings and metrics saved to: {PER_QUERY_PATH}")
    print("Note: graph_expansion is seeded by vector results; it is not an independent graph-only retriever.")
    return output


def write_invalid_report(reason: str, details: Dict[str, Any]) -> Dict[str, Any]:
    """Record an explicitly invalid run. Metrics are null, never zero."""
    report = {
        "status": "invalid",
        "reason": reason,
        "metrics": None,
        "benchmark_version": "2.1-corpus-validated",
        "corpus_id": CORPUS_ID,
        "details": details,
    }
    for path in (RESULTS_PATH, PER_QUERY_PATH):
        with path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)
    print("\n" + "=" * 88)
    print("BENCHMARK INVALID — no metrics reported")
    print(reason)
    print("=" * 88)
    print(json.dumps({"status": "invalid", "reason": reason, "metrics": None}, indent=2))
    print(f"Invalid-run report written to: {RESULTS_PATH} and {PER_QUERY_PATH}")
    return report


if __name__ == "__main__":
    if os.getenv("BENCHMARK_PRINT_FINGERPRINT"):
        sys.exit(print_fingerprint())
    try:
        run_benchmark()
    except BenchmarkInvalid as exc:
        write_invalid_report(exc.reason, exc.details)
        sys.exit(2)
