"""Step 2: fusion ablation on a frozen corpus, retrieval metrics only.

What this does and does not do
------------------------------
It does NOT introduce a second retrieval implementation. Candidate generation is
the production path: `service.vector_db.search` for seeds and the same directed
`graph.successors` BFS with the same visited-once rule, so every variant scores
the identical candidate set from the identical stores.

Only the *fusion arithmetic* varies. The control variant re-implements
`HybridRetrievalService.hybrid_search` steps 3-5 exactly, including the
`round(x, 4)` applied before sorting and the stable insertion-order tie-break,
and the harness refuses to report anything until the control reproduces
production `hybrid_search` output id-for-id on every query. That assertion is
what makes the other variants trustworthy: they differ from production in one
named mechanism and nothing else.

app/service.py is never modified. The production formula stays the control.

Variants (one mechanism each, all at the production candidate budget):
  control            alpha*norm_v + beta*g - gamma*(hop/max_depth), min-max norm
  no_hop0_graph      same, but hop-0 vector seeds get no graph credit
  max_norm           same, but norm_v = raw/v_max instead of min-max
  rrf                reciprocal rank fusion over vector_rank and graph_rank

Separate dimension (candidate budget, control formula):
  seed budget 5 (production, tied to top_k) vs 25

Run:
    python experiments/fusion_ablation.py
    BENCHMARK_CORPUS_ID=romeo-and-juliet-summary-v1 python experiments/fusion_ablation.py
"""

from __future__ import annotations

import collections
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import benchmark as bench  # noqa: E402  (frozen validation + metrics live there)

ABLATION_VERSION = "1.0"
NORM_EPSILON = 1e-6  # mirrors app.service.NORM_EPSILON
RRF_K = 60  # standard reciprocal-rank-fusion constant

CORPUS_ID = os.getenv("BENCHMARK_CORPUS_ID", "meridian-transit-v1")
TOP_K = int(os.getenv("ABLATION_TOP_K", "5"))
MAX_DEPTH = int(os.getenv("ABLATION_MAX_DEPTH", "2"))
ALPHA = float(os.getenv("ABLATION_ALPHA", "0.6"))
BETA = float(os.getenv("ABLATION_BETA", "0.4"))
GAMMA = float(os.getenv("ABLATION_GAMMA", "0.1"))
WIDE_SEED_K = int(os.getenv("ABLATION_WIDE_SEED_K", "25"))

RESULTS_PATH = ROOT / f"experiment_fusion_ablation_{CORPUS_ID}.json"


# ========================= Candidate generation ============================


def collect_candidates(service: Any, query: str, seed_k: int, max_depth: int) -> Dict[str, Dict[str, Any]]:
    """Production candidate set: vector seeds then outgoing BFS.

    Mirrors app/service.py hybrid_search steps 1-2 exactly: seeds keep their
    vector rank, BFS enqueues a node only the first time it is seen, every
    popped node takes g = 1/(1+hop), and seeds therefore end at
    graph_score 1.0 / hop 0. Insertion order is seeds in rank order followed by
    graph-only nodes in BFS pop order, which is the tie-break production relies on.
    """
    rows = service.vector_db.search(query, seed_k)
    candidates: Dict[str, Dict[str, Any]] = {}
    start_nodes: List[str] = []

    for vector_rank, (doc_id, text, score, metadata) in enumerate(rows, 1):
        node_id = doc_id or (metadata or {}).get("id")
        if not node_id:
            continue
        node_id = str(node_id)
        if node_id in candidates:
            continue
        candidates[node_id] = {
            "raw_vector_score": score,
            "retrieved_by_vector": True,
            "vector_rank": vector_rank,
            "graph_score": 0.0,
            "graph_rank": None,
            "hop": 0,
            "reached_from": None,
            "text": text,
        }
        start_nodes.append(node_id)

    graph = service.graph_db.graph
    queue = collections.deque()
    visited: Dict[str, int] = {}
    graph_rank_counter = 0

    for node_id in start_nodes:
        if node_id not in visited:
            visited[node_id] = 0
            queue.append((node_id, 0, None))

    while queue:
        current, hops, reached_from = queue.popleft()
        g_score = 1.0 / (1.0 + hops)
        graph_rank_counter += 1

        if current not in candidates:
            node = service.graph_db.get_node(current)
            candidates[current] = {
                "raw_vector_score": None,
                "retrieved_by_vector": False,
                "vector_rank": None,
                "graph_score": g_score,
                "graph_rank": graph_rank_counter,
                "hop": hops,
                "reached_from": reached_from,
                "text": node.text if node else "",
            }
        else:
            if g_score > candidates[current]["graph_score"]:
                candidates[current]["graph_score"] = g_score
                candidates[current]["hop"] = hops
                if candidates[current]["graph_rank"] is None:
                    candidates[current]["graph_rank"] = graph_rank_counter

        if hops < max_depth and current in graph:
            for neighbor in graph.successors(current):
                if neighbor not in visited:
                    visited[neighbor] = hops + 1
                    queue.append((neighbor, hops + 1, current))

    # Convergence: how many independent seeds reach each node within max_depth.
    # The production BFS is multi-source and visits a node once, so this count is
    # not recoverable from it. Running one BFS per seed produces exactly the same
    # node set (asserted below) and adds only this statistic.
    seed_reach: Dict[str, int] = {node_id: 0 for node_id in candidates}
    for seed in start_nodes:
        seen_from_seed = {seed}
        frontier = {seed}
        for _ in range(max_depth):
            nxt = set()
            for node_id in frontier:
                if node_id in graph:
                    for target in graph.successors(node_id):
                        if target not in seen_from_seed:
                            seen_from_seed.add(target)
                            nxt.add(target)
            frontier = nxt
        for node_id in seen_from_seed:
            if node_id in seed_reach:
                seed_reach[node_id] += 1
    assert set(seed_reach) == set(candidates), "per-seed traversal changed the candidate set"
    for node_id, row in candidates.items():
        row["seed_reach_count"] = seed_reach[node_id]
        row["seed_count"] = len(start_nodes)

    # Resolve each graph-only node to the seed its path started from, by walking
    # `reached_from` back to a vector-retrieved node. Same candidate set, only
    # provenance is recorded.
    for node_id, row in candidates.items():
        if row["retrieved_by_vector"]:
            row["origin_seed"] = node_id
            continue
        cursor = row["reached_from"]
        guard = 0
        while cursor is not None and guard < 16:
            parent = candidates.get(cursor)
            if parent is None:
                break
            if parent["retrieved_by_vector"]:
                break
            cursor = parent["reached_from"]
            guard += 1
        row["origin_seed"] = cursor

    return candidates


def normalize_min_max(candidates: Dict[str, Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """app/service.py step 3, including the degenerate branch and the epsilon."""
    raw = [row["raw_vector_score"] for row in candidates.values() if row["retrieved_by_vector"]]
    v_min = min(raw) if raw else 0.0
    v_max = max(raw) if raw else 0.0
    v_range = v_max - v_min
    degenerate = len(raw) < 2 or v_range < NORM_EPSILON

    norms: Dict[str, Optional[float]] = {}
    for node_id, row in candidates.items():
        if not row["retrieved_by_vector"]:
            norms[node_id] = None
        elif degenerate:
            norms[node_id] = 1.0
        else:
            norms[node_id] = (row["raw_vector_score"] - v_min) / (v_range + NORM_EPSILON)
    return norms


def normalize_max_only(candidates: Dict[str, Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Variant normalization: divide by the max, so the weakest hit is not pinned to 0."""
    raw = [row["raw_vector_score"] for row in candidates.values() if row["retrieved_by_vector"]]
    v_max = max(raw) if raw else 0.0
    norms: Dict[str, Optional[float]] = {}
    for node_id, row in candidates.items():
        if not row["retrieved_by_vector"]:
            norms[node_id] = None
        elif v_max < NORM_EPSILON:
            norms[node_id] = 1.0
        else:
            norms[node_id] = row["raw_vector_score"] / v_max
    return norms


# ================================ Variants =================================


def _rank(scored: List[Tuple[str, float]], top_k: int) -> List[str]:
    """Stable descending sort on the rounded score, then truncate.

    Production sorts the already-rounded `final_score` with `reverse=True` on a
    list built in candidate insertion order, so equal scores keep insertion
    order. `sorted` on the negated score is the same stable ordering.
    """
    ordered = sorted(scored, key=lambda row: -row[1])
    return [node_id for node_id, _score in ordered[:top_k]]


def variant_control(candidates: Dict[str, Dict[str, Any]], top_k: int, max_depth: int) -> List[str]:
    norms = normalize_min_max(candidates)
    scored: List[Tuple[str, float]] = []
    for node_id, row in candidates.items():
        penalty = (row["hop"] / max_depth) if max_depth > 0 else 0.0
        if row["retrieved_by_vector"]:
            final = ALPHA * (norms[node_id] or 0.0) + BETA * row["graph_score"] - GAMMA * penalty
        else:
            final = BETA * row["graph_score"] - GAMMA * penalty
        scored.append((node_id, round(max(0.0, final), 4)))
    return _rank(scored, top_k)


def variant_no_hop0_graph(candidates: Dict[str, Dict[str, Any]], top_k: int, max_depth: int) -> List[str]:
    """Mechanism under test: hop-0 seeds no longer collect the free beta credit.

    In the control every vector seed ends with graph_score 1.0, so beta is added
    uniformly to all of them and graph-only nodes cap at beta/2 - gamma/2. This
    variant gives graph credit only to nodes the traversal actually had to walk to.
    """
    norms = normalize_min_max(candidates)
    scored: List[Tuple[str, float]] = []
    for node_id, row in candidates.items():
        penalty = (row["hop"] / max_depth) if max_depth > 0 else 0.0
        if row["retrieved_by_vector"]:
            graph_credit = 0.0 if row["hop"] == 0 else BETA * row["graph_score"]
            final = ALPHA * (norms[node_id] or 0.0) + graph_credit - GAMMA * penalty
        else:
            final = BETA * row["graph_score"] - GAMMA * penalty
        scored.append((node_id, round(max(0.0, final), 4)))
    return _rank(scored, top_k)


def variant_max_norm(candidates: Dict[str, Dict[str, Any]], top_k: int, max_depth: int) -> List[str]:
    """Mechanism under test: normalization floor. Everything else is the control."""
    norms = normalize_max_only(candidates)
    scored: List[Tuple[str, float]] = []
    for node_id, row in candidates.items():
        penalty = (row["hop"] / max_depth) if max_depth > 0 else 0.0
        if row["retrieved_by_vector"]:
            final = ALPHA * (norms[node_id] or 0.0) + BETA * row["graph_score"] - GAMMA * penalty
        else:
            final = BETA * row["graph_score"] - GAMMA * penalty
        scored.append((node_id, round(max(0.0, final), 4)))
    return _rank(scored, top_k)


def variant_rrf(candidates: Dict[str, Dict[str, Any]], top_k: int, max_depth: int) -> List[str]:
    """Mechanism under test: rank fusion instead of weighted score fusion.

    Scores never enter the formula, so neither normalization nor the hop-0
    constant can dominate. A node absent from a list contributes nothing.
    """
    scored: List[Tuple[str, float]] = []
    for node_id, row in candidates.items():
        total = 0.0
        if row["vector_rank"] is not None:
            total += 1.0 / (RRF_K + row["vector_rank"])
        if row["graph_rank"] is not None:
            total += 1.0 / (RRF_K + row["graph_rank"])
        scored.append((node_id, round(total, 6)))
    return _rank(scored, top_k)


def variant_no_hop0_no_penalty(candidates: Dict[str, Dict[str, Any]], top_k: int, max_depth: int) -> List[str]:
    """Round 2, mechanism: the distance penalty, on top of no_hop0_graph.

    Round 1 fixed the 1-hop families but pushed the 2-hop family to zero: with
    gamma=0.1 a hop-2 node scores beta/3 - gamma, barely above zero, so hop-1
    neighbours fill the top-K. This sets gamma=0 and changes nothing else.
    """
    norms = normalize_min_max(candidates)
    scored: List[Tuple[str, float]] = []
    for node_id, row in candidates.items():
        if row["retrieved_by_vector"]:
            graph_credit = 0.0 if row["hop"] == 0 else BETA * row["graph_score"]
            final = ALPHA * (norms[node_id] or 0.0) + graph_credit
        else:
            final = BETA * row["graph_score"]
        scored.append((node_id, round(max(0.0, final), 4)))
    return _rank(scored, top_k)


def variant_no_hop0_flat_hop(candidates: Dict[str, Dict[str, Any]], top_k: int, max_depth: int) -> List[str]:
    """Round 2, mechanism: hop decay, on top of no_hop0_graph.

    Traversed nodes all receive the same beta credit regardless of depth, so a
    hop-2 answer is no longer structurally behind every hop-1 neighbour. The
    distance penalty is retained, which is what still separates hop 1 from hop 2.
    """
    norms = normalize_min_max(candidates)
    scored: List[Tuple[str, float]] = []
    for node_id, row in candidates.items():
        penalty = (row["hop"] / max_depth) if max_depth > 0 else 0.0
        traversed = row["hop"] > 0
        graph_credit = BETA if traversed else 0.0
        if row["retrieved_by_vector"]:
            final = ALPHA * (norms[node_id] or 0.0) + graph_credit - GAMMA * penalty
        else:
            final = graph_credit - GAMMA * penalty
        scored.append((node_id, round(max(0.0, final), 4)))
    return _rank(scored, top_k)


def variant_no_hop0_seed_weighted(candidates: Dict[str, Dict[str, Any]], top_k: int, max_depth: int) -> List[str]:
    """Round 2, mechanism: graph credit inherits the strength of its origin seed.

    In no_hop0_graph every hop-1 node scores identically, so ties are broken by
    BFS pop order and a neighbour of the best seed ranks no higher than a
    neighbour of the worst. Here the credit is scaled by the origin seed's
    normalized vector score, which is evidence the query already produced.
    """
    norms = normalize_min_max(candidates)
    scored: List[Tuple[str, float]] = []
    for node_id, row in candidates.items():
        penalty = (row["hop"] / max_depth) if max_depth > 0 else 0.0
        origin = row.get("origin_seed")
        origin_strength = norms.get(origin) if origin else None
        if origin_strength is None:
            origin_strength = 0.0
        if row["retrieved_by_vector"]:
            graph_credit = 0.0 if row["hop"] == 0 else BETA * row["graph_score"] * origin_strength
            final = ALPHA * (norms[node_id] or 0.0) + graph_credit - GAMMA * penalty
        else:
            final = BETA * row["graph_score"] * origin_strength - GAMMA * penalty
        scored.append((node_id, round(max(0.0, final), 4)))
    return _rank(scored, top_k)


def variant_no_hop0_convergence(candidates: Dict[str, Dict[str, Any]], top_k: int, max_depth: int) -> List[str]:
    """Round 3, mechanism: graph credit scales with how many seeds reach the node.

    In no_hop0_graph every hop-1 node scores identically, so an incidental
    neighbour of one seed ties with a node several seeds agree on. Here the
    credit is multiplied by the share of seeds that reach it, which is evidence
    the retrieval already produced and does not encode any answer path.
    """
    norms = normalize_min_max(candidates)
    scored: List[Tuple[str, float]] = []
    for node_id, row in candidates.items():
        penalty = (row["hop"] / max_depth) if max_depth > 0 else 0.0
        seed_count = max(1, row.get("seed_count", 1))
        convergence = row.get("seed_reach_count", 0) / seed_count
        if row["retrieved_by_vector"]:
            graph_credit = 0.0 if row["hop"] == 0 else BETA * row["graph_score"] * convergence
            final = ALPHA * (norms[node_id] or 0.0) + graph_credit - GAMMA * penalty
        else:
            final = BETA * row["graph_score"] * convergence - GAMMA * penalty
        scored.append((node_id, round(max(0.0, final), 4)))
    return _rank(scored, top_k)


def variant_no_hop0_query_rescore(candidates: Dict[str, Dict[str, Any]], top_k: int, max_depth: int) -> List[str]:
    """Round 3, mechanism: graph-only nodes get a real vector score.

    The production pipeline deliberately gives graph-only nodes no vector score
    ("no fake vector scores"), so they are ranked purely structurally and every
    hop-1 neighbour is interchangeable. This scores them with the same embedding
    model against the same query, using embeddings already stored in the index,
    so graph evidence and lexical evidence are finally on one scale.

    Requires `attach_query_scores` to have run; the scores are attached to the
    candidate rows rather than computed here, so the extra cost is measured.
    """
    raw = {
        node_id: row.get("query_score")
        for node_id, row in candidates.items()
        if row.get("query_score") is not None
    }
    v_min = min(raw.values()) if raw else 0.0
    v_max = max(raw.values()) if raw else 0.0
    v_range = v_max - v_min
    degenerate = len(raw) < 2 or v_range < NORM_EPSILON

    scored: List[Tuple[str, float]] = []
    for node_id, row in candidates.items():
        penalty = (row["hop"] / max_depth) if max_depth > 0 else 0.0
        score = row.get("query_score")
        if score is None:
            norm_v = 0.0
        elif degenerate:
            norm_v = 1.0
        else:
            norm_v = (score - v_min) / (v_range + NORM_EPSILON)
        graph_credit = 0.0 if row["hop"] == 0 else BETA * row["graph_score"]
        final = ALPHA * norm_v + graph_credit - GAMMA * penalty
        scored.append((node_id, round(max(0.0, final), 4)))
    return _rank(scored, top_k)


def attach_query_scores(service: Any, query: str, candidates: Dict[str, Dict[str, Any]]) -> float:
    """Cosine of every candidate against the query, from stored embeddings.

    Returns the milliseconds spent, so the variant's extra cost is reported
    rather than hidden. Vector-retrieved nodes keep the score the production
    search already gave them where available.
    """
    started = time.perf_counter()
    node_ids = list(candidates)
    stored = service.vector_db.db.get(ids=node_ids, include=["embeddings"])
    embeddings = stored.get("embeddings")
    returned_ids = stored.get("ids") or []
    query_vector = service.vector_db.embedding_function.embed_query(query)
    query_norm = sum(value * value for value in query_vector) ** 0.5 or 1.0

    by_id = {}
    for index, node_id in enumerate(returned_ids):
        if embeddings is not None and index < len(embeddings):
            by_id[node_id] = embeddings[index]

    for node_id, row in candidates.items():
        vector = by_id.get(node_id)
        if vector is None or len(vector) == 0:
            row["query_score"] = row["raw_vector_score"]
            continue
        dot = sum(a * b for a, b in zip(query_vector, vector))
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        row["query_score"] = dot / (query_norm * norm)
    return (time.perf_counter() - started) * 1000


# Variants that need query-side rescoring before they can be ranked.
NEEDS_QUERY_SCORES = {"no_hop0_query_rescore"}


VARIANTS = {
    "control": variant_control,
    "no_hop0_graph": variant_no_hop0_graph,
    "max_norm": variant_max_norm,
    "rrf": variant_rrf,
    "no_hop0_no_penalty": variant_no_hop0_no_penalty,
    "no_hop0_flat_hop": variant_no_hop0_flat_hop,
    "no_hop0_seed_weighted": variant_no_hop0_seed_weighted,
    "no_hop0_convergence": variant_no_hop0_convergence,
    "no_hop0_query_rescore": variant_no_hop0_query_rescore,
}

# Which single mechanism each variant changes, and relative to what.
VARIANT_BASE = {
    "control": None,
    "no_hop0_graph": "control",
    "max_norm": "control",
    "rrf": "control",
    "no_hop0_no_penalty": "no_hop0_graph",
    "no_hop0_flat_hop": "no_hop0_graph",
    "no_hop0_seed_weighted": "no_hop0_graph",
    "no_hop0_convergence": "no_hop0_graph",
    "no_hop0_query_rescore": "no_hop0_graph",
}


# ================================ Harness ==================================


def verify_control_is_faithful(
    service: Any,
    dataset: List[Dict[str, Any]],
    top_k: int,
    max_depth: int,
) -> Dict[str, Any]:
    """Refuse to run the ablation unless the control equals production output."""
    mismatches: List[Dict[str, Any]] = []
    for item in dataset:
        query = str(item["query"])
        production = bench.result_ids(
            service.hybrid_search(
                query=query,
                vector_weight=ALPHA,
                graph_weight=BETA,
                top_k=top_k,
                max_depth=max_depth,
                path_distance_penalty=GAMMA,
            ).results,
            query,
        )
        candidates = collect_candidates(service, query, top_k, max_depth)
        reproduced = variant_control(candidates, top_k, max_depth)
        if production != reproduced:
            mismatches.append({
                "query_id": item.get("id"),
                "query": query,
                "production": production,
                "control": reproduced,
            })
    if mismatches:
        raise RuntimeError(
            "Control variant does not reproduce production hybrid_search. The "
            f"ablation would measure the harness, not the formula. Mismatches: "
            f"{json.dumps(mismatches[:3], indent=2)}"
        )
    return {"queries_checked": len(dataset), "mismatches": 0}


def evaluate(
    service: Any,
    dataset: List[Dict[str, Any]],
    variant_name: str,
    seed_k: int,
    top_k: int,
    max_depth: int,
) -> Dict[str, Any]:
    scorer = VARIANTS[variant_name]
    metric_names = [f"precision@{top_k}", f"recall@{top_k}", "mrr", f"ndcg@{top_k}"]
    totals: Dict[str, List[float]] = {name: [] for name in metric_names}
    per_query: List[Dict[str, Any]] = []
    latencies: List[float] = []

    for item in dataset:
        query = str(item["query"])
        relevant = {str(value) for value in item["relevant"]}

        started = time.perf_counter()
        try:
            candidates = collect_candidates(service, query, seed_k, max_depth)
            if variant_name in NEEDS_QUERY_SCORES:
                attach_query_scores(service, query, candidates)
            ranking = scorer(candidates, top_k, max_depth)
        except Exception as exc:
            raise RuntimeError(
                f"Variant {variant_name!r} failed for query: {query!r}"
            ) from exc
        latencies.append((time.perf_counter() - started) * 1000)

        bench.validate_ranking(ranking, query, variant_name)
        scores = bench.summarize(ranking, relevant, top_k)
        for name, value in scores.items():
            totals[name].append(value)

        vector_only = {
            node_id for node_id, row in candidates.items() if row["retrieved_by_vector"]
        }
        vector_baseline = [
            node_id
            for node_id, row in sorted(
                ((n, r) for n, r in candidates.items() if r["retrieved_by_vector"]),
                key=lambda pair: pair[1]["vector_rank"],
            )
        ][:top_k]
        per_query.append({
            "id": item.get("id"),
            "category": item.get("category"),
            "hops": item.get("hops"),
            "relevant": sorted(relevant),
            "ranking": ranking,
            "metrics": {name: round(value, 6) for name, value in scores.items()},
            "graph_only_in_top_k": len([n for n in ranking if n not in vector_only]),
            "recovered_in_top_k": round(
                len((relevant - set(vector_baseline)) & set(ranking)) / len(relevant), 6
            ),
        })

    aggregate = {name: round(statistics.fmean(values), 4) for name, values in totals.items()}
    aggregate["mean_latency_ms"] = round(statistics.fmean(latencies), 3)
    aggregate["graph_only_in_top_k"] = round(
        statistics.fmean([row["graph_only_in_top_k"] for row in per_query]), 4
    )
    aggregate["recovered_in_top_k"] = round(
        statistics.fmean([row["recovered_in_top_k"] for row in per_query]), 4
    )

    by_category: Dict[str, Any] = {}
    for row in per_query:
        bucket = by_category.setdefault(
            row["category"] or "uncategorised", {"queries": 0, "metrics": collections.defaultdict(float)}
        )
        bucket["queries"] += 1
        for name, value in row["metrics"].items():
            bucket["metrics"][name] += value
        bucket["metrics"]["recovered_in_top_k"] += row["recovered_in_top_k"]
    for bucket in by_category.values():
        count = bucket["queries"]
        bucket["metrics"] = {name: round(total / count, 4) for name, total in bucket["metrics"].items()}

    return {
        "variant": variant_name,
        "seed_k": seed_k,
        "aggregate": aggregate,
        "by_category": by_category,
        "per_query": per_query,
    }


def compare(control: Dict[str, Any], candidate: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Per-query win/loss/tie against the control, plus the mean delta.

    A sign count is reported rather than a p-value: 60 queries over a
    deterministic retriever is a census of this dataset, not a sample, so the
    useful question is how many queries moved and in which direction.
    """
    wins = losses = ties = 0
    deltas: List[float] = []
    regressions: List[Dict[str, Any]] = []
    for control_row, candidate_row in zip(control["per_query"], candidate["per_query"]):
        assert control_row["id"] == candidate_row["id"], "per-query alignment broken"
        delta = candidate_row["metrics"][key] - control_row["metrics"][key]
        deltas.append(delta)
        if delta > 1e-9:
            wins += 1
        elif delta < -1e-9:
            losses += 1
            regressions.append({
                "id": control_row["id"],
                "category": control_row["category"],
                "control": control_row["metrics"][key],
                "variant": candidate_row["metrics"][key],
            })
        else:
            ties += 1
    return {
        "metric": key,
        "mean_delta": round(statistics.fmean(deltas), 4),
        "queries_improved": wins,
        "queries_regressed": losses,
        "queries_unchanged": ties,
        "regressions": regressions[:10],
    }


def main() -> int:
    corpus = bench.resolve_corpus(CORPUS_ID)
    dataset, provenance = bench.load_dataset(corpus)
    service = bench.build_service(corpus)
    validation = bench.validate_corpus_and_ground_truth(service, dataset, corpus)

    faithfulness = verify_control_is_faithful(service, dataset, TOP_K, MAX_DEPTH)
    print(f"control reproduces production hybrid_search on {faithfulness['queries_checked']}/"
          f"{len(dataset)} queries, 0 mismatches\n")

    runs: Dict[str, Any] = {}
    for name in VARIANTS:
        runs[name] = evaluate(service, dataset, name, TOP_K, TOP_K, MAX_DEPTH)

    # Separate dimension: candidate budget, control formula only.
    wide_name = f"control_seed_k_{WIDE_SEED_K}"
    runs[wide_name] = evaluate(service, dataset, "control", WIDE_SEED_K, TOP_K, MAX_DEPTH)
    VARIANT_BASE[wide_name] = "control"

    metric_names = [f"precision@{TOP_K}", f"recall@{TOP_K}", "mrr", f"ndcg@{TOP_K}"]
    print(f"{'variant':<24}{'seed_k':>7}" + "".join(f"{name:>13}" for name in metric_names)
          + f"{'graph_in_K':>12}{'recovered':>11}{'ms':>9}")
    for name, run in runs.items():
        agg = run["aggregate"]
        print(
            f"{name:<24}{run['seed_k']:>7}"
            + "".join(f"{agg[metric]:>13.4f}" for metric in metric_names)
            + f"{agg['graph_only_in_top_k']:>12.4f}{agg['recovered_in_top_k']:>11.4f}"
            + f"{agg['mean_latency_ms']:>9.2f}"
        )

    # Each variant is compared against the base it modifies, so a delta isolates
    # one mechanism instead of stacking two.
    comparisons: Dict[str, Any] = {}
    for name, run in runs.items():
        base_name = VARIANT_BASE.get(name)
        if base_name is None:
            continue
        comparisons[name] = {
            "base": base_name,
            "metrics": {metric: compare(runs[base_name], run, metric) for metric in metric_names},
        }

    print("\nversus the base each variant modifies:")
    for name, entry in comparisons.items():
        recall = entry["metrics"][f"recall@{TOP_K}"]
        ndcg = entry["metrics"][f"ndcg@{TOP_K}"]
        print(f"  {name:<24}vs {entry['base']:<16}"
              f"d_recall={recall['mean_delta']:+.4f} d_ndcg={ndcg['mean_delta']:+.4f}  "
              f"improved={recall['queries_improved']:<3} regressed={recall['queries_regressed']:<3} "
              f"unchanged={recall['queries_unchanged']}")

    print(f"\nper-category recall@{TOP_K}:")
    categories = list(runs["control"]["by_category"])
    print(f"{'variant':<24}" + "".join(f"{cat.split(' (')[0][:17]:>19}" for cat in categories))
    for name, run in runs.items():
        print(f"{name:<24}" + "".join(
            f"{run['by_category'][cat]['metrics'][f'recall@{TOP_K}']:>19.3f}" for cat in categories
        ))

    output = {
        "status": "valid",
        "experiment": "fusion-ablation",
        "ablation_version": ABLATION_VERSION,
        "production_implementation_modified": False,
        "control_faithfulness": faithfulness,
        "configuration": {
            "corpus_id": corpus["corpus_id"],
            "dataset": provenance["dataset_path"],
            "dataset_sha256": provenance["dataset_sha256"],
            "corpus_fingerprint": validation["corpus_fingerprint"],
            "top_k": TOP_K,
            "max_depth": MAX_DEPTH,
            "alpha": ALPHA,
            "beta": BETA,
            "gamma": GAMMA,
            "rrf_k": RRF_K,
            "wide_seed_k": WIDE_SEED_K,
            "vector_db_path": service.vector_db_path,
            "graph_db_path": service.graph_db_path,
        },
        "variant_descriptions": {
            "control": "production formula, reproduced bit-for-bit",
            "no_hop0_graph": "hop-0 vector seeds receive no graph credit",
            "no_hop0_convergence": "graph credit scaled by the share of seeds reaching the node",
            "no_hop0_query_rescore": "graph-only nodes scored against the query with the production embedding model",
            "max_norm": "vector scores normalized by max instead of min-max",
            "rrf": "reciprocal rank fusion over vector_rank and graph_rank",
            f"control_seed_k_{WIDE_SEED_K}": "control formula, wider vector seed budget",
        },
        "runs": runs,
        "comparisons": comparisons,
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
