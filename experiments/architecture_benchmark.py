"""Architecture benchmark: Vector RAG vs Graph RAG vs Hybrid RAG vs Hybrid RAG + RLM.

Five architectures run on the same corpus and the same query set:

  A  Vector RAG        production vector search (benchmark.extract_vector_ids)
  B  Graph RAG         benchmark.graph_expansion_ids over the vector top-k seeds
  B2 Graph-only RAG    query entity linking into the graph, then traversal; no
                       embeddings anywhere (EntityLinkingGraphRetriever)
  C  Hybrid RAG        the frozen step-3 retriever (rlm.loop.FrozenRetriever)
  D  Hybrid RAG + RLM  the bounded RLM loop on the frozen retriever

Two phases keep retrieval quality measurable apart from answer generation:

  retrieval  No LLM. Every query through A, B, B2 and C; P@k, R@k, MRR and NDCG@k via
             benchmark.summarize, plus retrieval latency. D retrieves through C,
             so its first retrieval is C's ranking and it has no separate row.
             Reproduction checks: A and B must equal the saved baseline
             (benchmark_baseline_<corpus>.json -> aggregate.vector and
             aggregate.graph_expansion) and C must equal the frozen config's
             selected_metrics, to 4 dp. A failed check is reported, never patched.
             B2 has no saved baseline. Its depth-3 ranking is reported as a
             sensitivity row only, never used for answers.
  answers    A, B, B2 and C each run rlm.loop.run_loop with max_steps=1 and
             grounding off: one retrieval, one ANSWER_TEMPLATE call. D runs the
             evaluator defaults (3 steps, 3 retrievals, merged decide/answer,
             compact links) with lean grounding. Scoring and evidence coverage
             are exactly rlm.evaluate's.

Nothing is reimplemented. Each architecture is an adapter with the
FrozenRetriever interface (retrieve / fetch / relations), so the same run_loop
and the same prompt templates serve all four, and the only difference between
A, B, B2 and C answers is the evidence they were given.

What each adapter shows the model:
  - Vector chunks carry `relations = {}`. Vector RAG has no graph, so no
    `Links:` line can leak graph structure into its prompts. This is checked on
    every query in the retrieval phase.
  - Graph and hybrid chunks carry their outgoing typed edges, as in the loop.

Graph RAG caveat: B is vector-seeded expansion with the seeds excluded,
identical to benchmark.py's graph_expansion baseline (seed count = top_k,
depth = benchmark.GRAPH_DEPTH). It measures what graph traversal adds beyond
the vector hits, not a vector-free graph retriever. A vector-free variant
(query entity linking into the graph, then traversal) is B2. Its
latency here includes the seeding vector search, since that is part of the
cost of producing the ranking; benchmark.py times the expansion alone.

Answer runs interleave the architectures per query (query 1: A, B, C, D;
query 2: ...), so a quota stop leaves all four on nearly the same query
subset. Each architecture has its own resume cache keyed by a signature over
everything that can change its answers, and aggregates are computed over the
queries every architecture completed, so a partial run still compares like
with like.

Run:
    python -m experiments.architecture_benchmark --phase retrieval
    python -m experiments.architecture_benchmark --phase answers --provider mock --limit 3
    python -m experiments.architecture_benchmark --provider groq --model openai/gpt-oss-120b --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import benchmark as bench  # noqa: E402
from rlm import evaluate as rlm_eval  # noqa: E402
from rlm import loop as rlm_loop  # noqa: E402
from rlm.providers import ProviderUnavailable, QuotaExceeded, build_provider  # noqa: E402

BENCHMARK_VERSION = "arch-bench-1.1"
REPRODUCTION_DECIMALS = 4

ARCHITECTURES: Dict[str, Dict[str, str]] = {
    "A": {"name": "vector_rag", "label": "Vector RAG", "retriever": "vector"},
    "B": {"name": "graph_rag", "label": "Graph RAG", "retriever": "graph"},
    "B2": {"name": "graph_only_rag", "label": "Graph-only RAG", "retriever": "graph_entity"},
    "C": {"name": "hybrid_rag", "label": "Hybrid RAG", "retriever": "hybrid"},
    "D": {"name": "hybrid_rag_rlm", "label": "Hybrid RAG + RLM", "retriever": "hybrid"},
}
RETRIEVAL_ARCHITECTURES = ("A", "B", "B2", "C")
# B2 at one more hop: entity linking lands on the named station, one hop before
# the incident a vector seed would hit. Reported for transparency only.
SENSITIVITY_DEPTH = 3


# ============================ Retriever adapters ============================


def graph_chunk(
    service: Any, node_id: str, relations: Dict[str, List[str]], hop: Optional[int], by_vector: bool
) -> Dict[str, Any]:
    """A chunk in FrozenRetriever's shape, text read from the graph node."""
    graph = service.graph_db.graph
    text = (graph.nodes[node_id].get("text") or "") if node_id in graph else ""
    return {
        "node_id": node_id,
        "text": text,
        "hop": hop,
        "retrieved_by_vector": by_vector,
        "relations": relations,
    }


class VectorRetriever:
    """A: production vector search. Chunks carry no graph relations."""

    name = "vector"

    def __init__(self, service: Any):
        self.service = service

    def ranking(self, query: str, top_k: int) -> List[str]:
        ids = bench.extract_vector_ids(self.service, query, top_k)
        return bench.validate_ranking(ids, query, self.name)

    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        return [graph_chunk(self.service, node_id, {}, 0, True) for node_id in self.ranking(query, top_k)]

    def relations(self, node_id: str) -> Dict[str, List[str]]:
        return {}

    def fetch(self, node_id: str) -> Optional[Dict[str, Any]]:
        if node_id not in self.service.graph_db.graph:
            return None
        return graph_chunk(self.service, node_id, {}, None, False)

    def as_record(self) -> Dict[str, Any]:
        return {"retriever": self.name, "implementation": "benchmark.extract_vector_ids",
                "relations_in_prompt": False}


class GraphExpansionRetriever:
    """B: benchmark.py's graph_expansion ranking, seeds excluded, graph-aware chunks."""

    name = "graph"

    def __init__(self, service: Any, depth: int = bench.GRAPH_DEPTH):
        self.service = service
        self.depth = depth

    def ranking(self, query: str, top_k: int) -> List[str]:
        # run_benchmark seeds with the vector top-k at the same k it scores.
        seeds = bench.extract_vector_ids(self.service, query, top_k)
        ids = bench.graph_expansion_ids(self.service, seeds, self.depth, top_k)
        return bench.validate_ranking(ids, query, self.name)

    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        return [
            graph_chunk(self.service, node_id, self.relations(node_id), None, False)
            for node_id in self.ranking(query, top_k)
        ]

    def relations(self, node_id: str) -> Dict[str, List[str]]:
        # FrozenRetriever's edge grouping only reads self.service; reuse it.
        return rlm_loop.FrozenRetriever.relations(self, node_id)  # type: ignore[arg-type]

    def fetch(self, node_id: str) -> Optional[Dict[str, Any]]:
        if node_id not in self.service.graph_db.graph:
            return None
        return graph_chunk(self.service, node_id, self.relations(node_id), None, False)

    def as_record(self) -> Dict[str, Any]:
        return {"retriever": self.name, "implementation": "benchmark.graph_expansion_ids",
                "seeds": "benchmark.extract_vector_ids at top_k, excluded from the ranking",
                "depth": self.depth, "relations_in_prompt": True}


STOPWORDS = frozenset(
    "a an and are at do does for from have how in is it its of on or own the to was were what when "
    "which who why with along either side sit immediately carry out needed".split()
)


def content_tokens(text: str) -> set:
    return {
        token for token in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", rlm_eval.fold_typography(text).lower())
        if len(token) > 2 and token not in STOPWORDS
    }


class EntityLinkingGraphRetriever:
    """B2: graph-only retrieval. No embeddings and no vector search at any point.

    1. Entity linking. Every graph node gets aliases read from its own id and
       lead sentence: station names ("Alder Vale is a through station"), line
       names ("The Azure Line runs"), codes from node ids (TX-440, MP-101,
       PL-A), procedure titles ("Maintenance procedure MP-101: <title>."),
       policy names and incident references (2026/01). A node is a seed when
       one of its aliases appears in the query as a whole phrase. The alias
       patterns are specific to this synthetic corpus's text conventions.
    2. Traversal. Outgoing edges from the seeds up to `depth` hops, the same
       edge direction as production and graph_expansion. Seeds stay in the
       ranking at hop 0: unlike B there is no vector search whose hits they
       would double count.
    3. Ranking. Hop distance first, then the number of seeds that reach the
       node (convergence), then keyword overlap between the query and the node
       text, then node id. The overlap step only orders nodes the graph already
       reached at the same distance; it never adds a node.

    A query that names no linkable entity gets an empty ranking, which is
    scored as a miss, not patched with a fallback.
    """

    name = "graph_entity"

    ALIAS_PATTERNS = (
        re.compile(r"^(.+?) is a (?:terminus|through station)"),
        re.compile(r"^The (.+? Line) runs"),
        re.compile(r"^Maintenance procedure \S+: (.+?)\."),
        re.compile(r"^Authorisation policy \S+, (.+?), governs"),
        re.compile(r"^Service incident (\d{4}/\d{2})"),
    )
    CODE_PREFIXES = ("equipment_", "procedure_", "policy_")

    def __init__(self, service: Any, depth: int = bench.GRAPH_DEPTH):
        self.service = service
        self.depth = depth
        self.aliases = self._build_aliases()

    def _build_aliases(self) -> List[tuple]:
        aliases: List[tuple] = []
        for node_id, data in self.service.graph_db.graph.nodes(data=True):
            node_id = str(node_id)
            text = data.get("text") or ""
            names = {match.group(1) for pattern in self.ALIAS_PATTERNS
                     for match in [pattern.match(text)] if match}
            if node_id.startswith(self.CODE_PREFIXES):
                names.add(node_id.split("_", 1)[1])
            for alias in names:
                folded = rlm_eval.fold_typography(alias).lower().strip()
                if folded:
                    aliases.append((re.compile(rf"(?<![\w-]){re.escape(folded)}(?![\w-])"), node_id))
        return aliases

    def link(self, query: str) -> List[str]:
        folded = rlm_eval.fold_typography(query).lower()
        return sorted({node_id for pattern, node_id in self.aliases if pattern.search(folded)})

    def ranking(self, query: str, top_k: int) -> List[str]:
        graph = self.service.graph_db.graph
        seeds = self.link(query)
        best_hop: Dict[str, int] = {}
        reached_from: Dict[str, set] = {}
        for seed in seeds:
            frontier = {seed}
            seen = {seed: 0}
            for hop in range(1, self.depth + 1):
                frontier = {
                    str(neighbor) for node in frontier for neighbor in graph.successors(node)
                    if str(neighbor) not in seen
                }
                for node in frontier:
                    seen[node] = hop
            for node, hop in seen.items():
                best_hop[node] = min(hop, best_hop.get(node, hop))
                reached_from.setdefault(node, set()).add(seed)
        query_tokens = content_tokens(query)

        def key(node: str) -> tuple:
            overlap = len(query_tokens & content_tokens(graph.nodes[node].get("text") or ""))
            return (best_hop[node], -len(reached_from[node]), -overlap, node)

        ids = sorted(best_hop, key=key)[:top_k]
        return bench.validate_ranking(ids, query, self.name)

    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        return [
            graph_chunk(self.service, node_id, self.relations(node_id), None, False)
            for node_id in self.ranking(query, top_k)
        ]

    def relations(self, node_id: str) -> Dict[str, List[str]]:
        return rlm_loop.FrozenRetriever.relations(self, node_id)  # type: ignore[arg-type]

    def fetch(self, node_id: str) -> Optional[Dict[str, Any]]:
        if node_id not in self.service.graph_db.graph:
            return None
        return graph_chunk(self.service, node_id, self.relations(node_id), None, False)

    def as_record(self) -> Dict[str, Any]:
        return {"retriever": self.name, "implementation": "EntityLinkingGraphRetriever",
                "seeds": "query entity linking against node aliases; no embeddings",
                "ranking": "hop, seed convergence, query keyword overlap, node id",
                "depth": self.depth, "aliases": len(self.aliases), "relations_in_prompt": True}


class HybridRetriever(rlm_loop.FrozenRetriever):
    """C and D: the frozen retriever, unchanged, plus a ranking helper."""

    name = "hybrid"

    def ranking(self, query: str, top_k: int) -> List[str]:
        ids = [chunk["node_id"] for chunk in self.retrieve(query, top_k)]
        return bench.validate_ranking(ids, query, self.name)

    def as_record(self) -> Dict[str, Any]:
        return {"retriever": self.name, "implementation": "rlm.loop.FrozenRetriever",
                "variant": self.variant_name, "weights": self.weights, "seed_k": self.seed_k,
                "max_depth": self.max_depth, "relations_in_prompt": True}


def build_retrievers(service: Any, frozen: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "vector": VectorRetriever(service),
        "graph": GraphExpansionRetriever(service),
        "graph_entity": EntityLinkingGraphRetriever(service),
        "hybrid": HybridRetriever(service, frozen),
    }


# ============================= Shared helpers ===============================


def mean(values: List[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


def group_by_category(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row.get("category") or "uncategorised", []).append(row)
    return groups


# ============================= Retrieval phase ==============================


def metric_names(top_k: int) -> List[str]:
    return [f"precision@{top_k}", f"recall@{top_k}", "mrr", f"ndcg@{top_k}"]


def aggregate_retrieval(rows: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"queries": len(rows)}
    for name in metric_names(top_k):
        summary[name] = mean([row["metrics"][name] for row in rows])
    summary["mean_latency_ms"] = round(statistics.fmean([row["latency_ms"] for row in rows]), 3)
    return summary


def run_retrieval_phase(
    dataset: List[Dict[str, Any]], retrievers: Dict[str, Any], top_k: int
) -> Dict[str, Any]:
    """Rank every query with A, B and C and score the rankings. No LLM."""
    per_arch: Dict[str, List[Dict[str, Any]]] = {arch: [] for arch in RETRIEVAL_ARCHITECTURES}
    vector_prompts_with_links: List[Any] = []
    empty_text_chunks: Dict[str, int] = {arch: 0 for arch in RETRIEVAL_ARCHITECTURES}

    for index, item in enumerate(dataset, 1):
        query = str(item["query"])
        relevant = {str(value) for value in item["relevant"]}
        for arch in RETRIEVAL_ARCHITECTURES:
            retriever = retrievers[ARCHITECTURES[arch]["retriever"]]
            started = time.perf_counter()
            chunks = retriever.retrieve(query, top_k)
            latency_ms = (time.perf_counter() - started) * 1000
            ids = bench.validate_ranking([chunk["node_id"] for chunk in chunks], query, arch)
            empty_text_chunks[arch] += sum(not chunk["text"].strip() for chunk in chunks)
            if arch == "A" and "Links:" in rlm_loop.format_evidence(chunks):
                vector_prompts_with_links.append(item.get("id", index))
            per_arch[arch].append({
                "id": item.get("id", index),
                "category": item.get("category"),
                "hops": item.get("hops"),
                "query": query,
                "relevant": sorted(relevant),
                "ranking": ids,
                "metrics": {key: round(value, 6) for key, value in bench.summarize(ids, relevant, top_k).items()},
                "latency_ms": round(latency_ms, 3),
            })
        print(f"[{index:02d}/{len(dataset)}] "
              + " ".join(f"{arch} R@{top_k}={per_arch[arch][-1]['metrics'][f'recall@{top_k}']:.2f}"
                         for arch in RETRIEVAL_ARCHITECTURES)
              + f" | {query[:56]}")

    deep = EntityLinkingGraphRetriever(retrievers["graph_entity"].service, depth=SENSITIVITY_DEPTH)
    deep_rows = []
    for row in per_arch["B2"]:
        ids = deep.ranking(row["query"], top_k)
        deep_rows.append({**row, "ranking": ids, "metrics": {
            key: round(value, 6) for key, value in bench.summarize(ids, set(row["relevant"]), top_k).items()
        }})
    linked = [len(retrievers["graph_entity"].link(row["query"])) for row in per_arch["B2"]]

    return {
        "top_k": top_k,
        "entity_linking": {
            "queries_with_no_linked_entity": sum(count == 0 for count in linked),
            "mean_linked_entities": mean([float(count) for count in linked]),
        },
        "sensitivity": {
            f"B2_depth{SENSITIVITY_DEPTH}": {
                "description": "B2 traversal one hop deeper; reported only, not used for answers",
                "aggregate": aggregate_retrieval(deep_rows, top_k),
                "by_category": {
                    category: aggregate_retrieval(group, top_k)
                    for category, group in group_by_category(deep_rows).items()
                },
            },
        },
        "architectures": {
            arch: {
                **ARCHITECTURES[arch],
                "retriever_config": retrievers[ARCHITECTURES[arch]["retriever"]].as_record(),
                "aggregate": aggregate_retrieval(rows, top_k),
                "by_category": {
                    category: aggregate_retrieval(group, top_k)
                    for category, group in group_by_category(rows).items()
                },
                "chunks_without_text": empty_text_chunks[arch],
                "per_query": rows,
            }
            for arch, rows in per_arch.items()
        },
        "note": "D retrieves through C's retriever; its first retrieval is C's ranking.",
        "vector_prompt_link_check": {
            "passed": not vector_prompts_with_links,
            "queries_with_links": vector_prompts_with_links,
            "detail": "format_evidence over every Vector RAG ranking must contain no 'Links:' line.",
        },
    }


def compare_metrics(
    label: str, measured: Dict[str, Any], expected: Optional[Dict[str, Any]], names: List[str], source: str
) -> Dict[str, Any]:
    if expected is None:
        return {"check": label, "status": "unavailable", "source": source}
    mismatches = {
        name: {"measured": measured[name], "expected": expected.get(name)}
        for name in names
        if expected.get(name) is None
        or round(measured[name], REPRODUCTION_DECIMALS) != round(expected[name], REPRODUCTION_DECIMALS)
    }
    return {
        "check": label,
        "status": "passed" if not mismatches else "FAILED",
        "source": source,
        "measured": {name: measured[name] for name in names},
        "expected": {name: expected.get(name) for name in names},
        "mismatches": mismatches,
    }


def reproduction_checks(
    retrieval: Dict[str, Any], corpus_id: str, frozen: Dict[str, Any], top_k: int
) -> List[Dict[str, Any]]:
    """A and B against the saved baseline, C against the frozen selection."""
    names = metric_names(top_k)
    measured = {arch: retrieval["architectures"][arch]["aggregate"] for arch in RETRIEVAL_ARCHITECTURES}
    baseline_path = ROOT / f"benchmark_baseline_{corpus_id}.json"
    baseline: Dict[str, Any] = {}
    if baseline_path.exists():
        payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        if int(payload.get("configuration", {}).get("top_k", -1)) == top_k:
            baseline = payload.get("aggregate") or {}
    frozen_metrics = None
    if frozen.get("frozen_at_corpus") == corpus_id and int(frozen["selected"]["top_k"]) == top_k:
        frozen_metrics = frozen.get("selected_metrics")
    return [
        compare_metrics("A vector == baseline aggregate.vector", measured["A"],
                        baseline.get("vector"), names, baseline_path.name),
        compare_metrics("B graph == baseline aggregate.graph_expansion", measured["B"],
                        baseline.get("graph_expansion"), names, baseline_path.name),
        compare_metrics("C hybrid == frozen selected_metrics", measured["C"],
                        frozen_metrics, names, rlm_loop.FROZEN_CONFIG_PATH.name),
    ]


# ============================== Answer phase ================================


def answer_bounds(top_k: int, max_steps: int, max_retrievals: int) -> Dict[str, rlm_loop.LoopBounds]:
    single_shot = dict(max_steps=1, top_k=top_k, grounding="off")
    return {
        "A": rlm_loop.LoopBounds(**single_shot),
        "B": rlm_loop.LoopBounds(**single_shot),
        "B2": rlm_loop.LoopBounds(**single_shot),
        "C": rlm_loop.LoopBounds(**single_shot),
        # The evaluator defaults, with the lean grounding of the reference runs.
        "D": rlm_loop.LoopBounds(max_steps=max_steps, max_retrievals=max_retrievals, top_k=top_k,
                                 merge_decide_answer=True, full_links=False, grounding="lean"),
    }


def architecture_signature(
    arch: str,
    corpus_id: str,
    dataset_sha: str,
    retriever_record: Dict[str, Any],
    provider_name: str,
    model: str,
    bounds_record: Dict[str, Any],
    generation_options: Dict[str, Any],
) -> str:
    """Identity of one architecture's answer configuration, for its resume cache."""
    payload = json.dumps(
        {
            "benchmark_version": BENCHMARK_VERSION,
            "architecture": arch,
            "retriever": retriever_record,
            "corpus_id": corpus_id,
            "dataset_sha256": dataset_sha,
            "provider": provider_name,
            "model": model,
            "generation_options": generation_options,
            "bounds": bounds_record,
            "prompts": rlm_loop.TEMPLATE_HASHES,
            "loop_version": rlm_loop.LOOP_VERSION,
            "evaluation_code_version": rlm_eval.EVALUATION_CODE_VERSION,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def answer_row(arch: str, item: Dict[str, Any], index: int, run: rlm_loop.LoopRun,
               universe: List[str]) -> Dict[str, Any]:
    """One scored episode, in rlm.evaluate's row shape."""
    question = str(item["query"])
    relevant = {str(value) for value in item["relevant"]}
    evidence = set(run.evidence_ids)
    return {
        "architecture": arch,
        "id": item.get("id", index),
        "category": item.get("category"),
        "hops": item.get("hops"),
        "question": question,
        "answer": run.answer,
        "answer_evaluation": rlm_eval.score_answer(run.answer, item["gold_answer"], universe, question),
        "evidence_coverage": {
            "relevant_chunks": sorted(relevant),
            "evidence_chunks": run.evidence_ids,
            "coverage": round(len(relevant & evidence) / len(relevant), 4),
            "all_relevant_present": relevant <= evidence,
        },
        "cost": {
            "retrieval_calls": run.retrieval_calls,
            "generation_calls": run.generation_calls,
            "input_tokens": run.input_tokens,
            "output_tokens": run.output_tokens,
            "retrieval_latency_ms": round(run.retrieval_latency_ms, 3),
            "generation_latency_ms": round(run.generation_latency_ms, 3),
            "by_stage": run.stage_costs,
        },
        "stop_reason": run.stop_reason,
        "steps": run.steps,
    }


def aggregate_answers(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def avg(path: Callable[[Dict[str, Any]], float]) -> float:
        return mean([float(path(row)) for row in rows])

    return {
        "queries": len(rows),
        "accuracy": avg(lambda row: row["answer_evaluation"]["correct"]),
        "abstention_rate": avg(lambda row: row["answer_evaluation"]["abstained"]),
        "evidence_coverage": avg(lambda row: row["evidence_coverage"]["coverage"]),
        "all_relevant_present_rate": avg(lambda row: row["evidence_coverage"]["all_relevant_present"]),
        "mean_input_tokens": avg(lambda row: row["cost"]["input_tokens"]),
        "mean_output_tokens": avg(lambda row: row["cost"]["output_tokens"]),
        "total_input_tokens": sum(row["cost"]["input_tokens"] for row in rows),
        "total_output_tokens": sum(row["cost"]["output_tokens"] for row in rows),
        "mean_retrieval_calls": avg(lambda row: row["cost"]["retrieval_calls"]),
        "mean_generation_calls": avg(lambda row: row["cost"]["generation_calls"]),
        "mean_retrieval_latency_ms": avg(lambda row: row["cost"]["retrieval_latency_ms"]),
        "mean_generation_latency_ms": avg(lambda row: row["cost"]["generation_latency_ms"]),
    }


def run_answer_phase(
    args: argparse.Namespace,
    corpus: Dict[str, Any],
    provenance: Dict[str, Any],
    dataset: List[Dict[str, Any]],
    retrievers: Dict[str, Any],
    universe: List[str],
) -> Dict[str, Any]:
    missing_gold = [item.get("id") for item in dataset if not item.get("gold_answer")]
    if missing_gold:
        raise SystemExit(f"Dataset has no gold_answer for queries {missing_gold[:10]}.")
    try:
        provider = build_provider(args.provider, model=args.model, temperature=args.temperature,
                                  max_tokens=args.max_tokens, reasoning_effort=args.reasoning_effort)
    except (ProviderUnavailable, ValueError) as exc:
        raise SystemExit(str(exc))
    verification = provider.verify_model()
    evaluative = bool(getattr(provider, "EVALUATIVE", True))

    bounds = answer_bounds(args.top_k, args.max_steps, args.max_retrievals)
    queries = dataset[: args.limit] if args.limit else dataset

    state: Dict[str, Dict[str, Any]] = {}
    for arch, spec in ARCHITECTURES.items():
        signature = architecture_signature(
            arch, corpus["corpus_id"], provenance["dataset_sha256"],
            retrievers[spec["retriever"]].as_record(), args.provider, provider.config.model,
            bounds[arch].as_record(), {"reasoning_effort": args.reasoning_effort},
        )
        cache_path = ROOT / (f"experiment_architecture_cache_{corpus['corpus_id']}_{args.provider}"
                             f"_{arch}_{signature[:12]}.json")
        cache = rlm_eval.load_cache(cache_path, signature) if args.resume else {}
        state[arch] = {"signature": signature, "cache_path": cache_path, "cache": cache,
                       "rows": [], "reused": 0}
        if args.resume:
            print(f"resume {arch}: {len(cache)} cached rows (signature {signature[:12]})")

    quota_stop: Optional[Dict[str, Any]] = None
    started_all = time.perf_counter()
    for index, item in enumerate(queries, 1):
        question = str(item["query"])
        query_id = int(item.get("id", index))
        line = []
        for arch, spec in ARCHITECTURES.items():
            arch_state = state[arch]
            cached = arch_state["cache"].get(query_id)
            if cached is not None:
                # Scoring is deterministic; re-apply it so a scorer fix needs no new calls.
                cached["answer_evaluation"] = rlm_eval.score_answer(
                    cached["answer"], item["gold_answer"], universe, question
                )
                arch_state["rows"].append(cached)
                arch_state["reused"] += 1
                line.append(f"{arch}={'ok' if cached['answer_evaluation']['correct'] else 'x '}(c)")
                continue
            try:
                run = rlm_loop.run_loop(retrievers[spec["retriever"]], provider, question, bounds[arch])
            except QuotaExceeded as exc:
                quota_stop = {
                    "provider": exc.provider,
                    "detail": exc.detail,
                    "stopped_at_query_id": query_id,
                    "stopped_at_architecture": arch,
                    "completed_per_architecture": {key: len(value["rows"]) for key, value in state.items()},
                    "queries_requested": len(queries),
                }
                break
            row = answer_row(arch, item, index, run, universe)
            arch_state["rows"].append(row)
            arch_state["cache"][query_id] = row
            rlm_eval.save_cache(arch_state["cache_path"], arch_state["signature"], arch_state["cache"])
            line.append(f"{arch}={'ok' if row['answer_evaluation']['correct'] else 'x '}")
        print(f"[{index:02d}/{len(queries)}] {' '.join(line)} | {question[:56]}")
        if quota_stop:
            print(f"QUOTA STOP at query {query_id}, architecture {quota_stop['stopped_at_architecture']}: "
                  f"{quota_stop['detail'][:160]}")
            break
    wall_ms = (time.perf_counter() - started_all) * 1000

    # Aggregate over the queries every architecture finished, so a partial run
    # still compares the four on the same questions.
    completed_ids = [{row["id"] for row in value["rows"]} for value in state.values()]
    common_ids = set.intersection(*completed_ids) if completed_ids else set()
    complete = all(len(value["rows"]) == len(queries) for value in state.values())
    if not common_ids:
        raise SystemExit("No query was completed by every architecture, so there is nothing to compare. "
                         + (f"Quota stop: {quota_stop['detail']}" if quota_stop else ""))

    architectures: Dict[str, Any] = {}
    for arch, value in state.items():
        compared = [row for row in value["rows"] if row["id"] in common_ids]
        architectures[arch] = {
            **ARCHITECTURES[arch],
            "retriever_config": retrievers[ARCHITECTURES[arch]["retriever"]].as_record(),
            "loop_bounds": bounds[arch].as_record(),
            "run_signature": value["signature"],
            "cache_file": value["cache_path"].name,
            "queries_evaluated": len(value["rows"]),
            "queries_reused_from_cache": value["reused"],
            "aggregate": aggregate_answers(compared),
            "by_category": {
                category: aggregate_answers(group) for category, group in group_by_category(compared).items()
            },
            "per_query": value["rows"],
        }

    if not evaluative:
        status = "plumbing_only"
    else:
        status = "valid" if complete else "partial"
    return {
        "status": status,
        "evaluative": evaluative,
        "complete": complete,
        "queries_requested": len(queries),
        "queries_in_dataset": len(dataset),
        "queries_compared": len(common_ids),
        "quota_stop": quota_stop,
        "caveat": None if evaluative else (
            "Mock provider: it performs no reasoning, so answer-level numbers measure the "
            "harness, not model quality. Rerun with a verified hosted model for a quality claim."
        ),
        "partial_caveat": None if complete else (
            f"Incomplete run: aggregates cover the {len(common_ids)} of {len(queries)} requested "
            "queries that every architecture completed"
            + (f"; stopped by {quota_stop['provider']} quota" if quota_stop else "")
            + ". Re-run with --resume to continue."
        ),
        "wall_clock_ms": round(wall_ms, 3),
        "configuration": {
            "generation": {**provider.config.as_record(), "model_verification": verification},
            "prompt_template_sha256": rlm_loop.TEMPLATE_HASHES,
            "loop_version": rlm_loop.LOOP_VERSION,
            "evaluation_code_version": rlm_eval.EVALUATION_CODE_VERSION,
            "scorer_version": rlm_eval.SCORER_VERSION,
            "loop_bounds": {arch: bounds[arch].as_record() for arch in ARCHITECTURES},
        },
        "architectures": architectures,
    }


# ================================ Reporting =================================


def reproducibility_block(
    corpus: Dict[str, Any], provenance: Dict[str, Any], validation: Dict[str, Any],
    answers: Optional[Dict[str, Any]], top_k: int,
) -> Dict[str, Any]:
    configuration = (answers or {}).get("configuration") or {}
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "git_revision": rlm_eval.git_revision(),
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus_id": corpus["corpus_id"],
        "corpus_fingerprint": validation["corpus_fingerprint"],
        "dataset": provenance["dataset_path"],
        "dataset_version": provenance["dataset_version"],
        "dataset_sha256": provenance["dataset_sha256"],
        "frozen_config_file": rlm_loop.FROZEN_CONFIG_PATH.name,
        "frozen_config_sha256": rlm_eval.file_sha256(rlm_loop.FROZEN_CONFIG_PATH),
        "top_k": top_k,
        "graph_depth": bench.GRAPH_DEPTH,
        "generation": configuration.get("generation"),
        "prompt_template_sha256": rlm_loop.TEMPLATE_HASHES,
        "loop_version": rlm_loop.LOOP_VERSION,
        "loop_bounds": configuration.get("loop_bounds"),
        "platform": {"python": platform.python_version(),
                     "system": f"{platform.system()} {platform.release()}"},
        "note": ("A pinned model id and prompt hash make a run traceable. They cannot guarantee "
                 "identical output: a hosted provider may change its implementation behind the same id."),
    }


def load_previous(path: Path, reproducibility: Dict[str, Any]) -> Dict[str, Any]:
    """The other phase's block from an earlier run, if it measured the same corpus and dataset."""
    if not path.exists():
        return {}
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    earlier = previous.get("reproducibility") or {}
    same = all(earlier.get(key) == reproducibility[key]
               for key in ("corpus_fingerprint", "dataset_sha256", "frozen_config_sha256", "top_k"))
    return previous if same else {}


def print_retrieval_table(retrieval: Dict[str, Any], checks: List[Dict[str, Any]], top_k: int) -> None:
    archs = retrieval["architectures"]
    print("\n" + "=" * 80)
    print(f"RETRIEVAL (no LLM) - {archs['A']['aggregate']['queries']} queries, k={top_k}")
    print("  ".join(f"{arch}={archs[arch]['label']}" for arch in archs))
    print(f"{'metric':<20}" + "".join(f"{arch:>14}" for arch in archs))
    for name in metric_names(top_k) + ["mean_latency_ms"]:
        print(f"{name:<20}" + "".join(f"{archs[arch]['aggregate'][name]:>14.4f}" for arch in archs))
    print("-" * 80)
    print(f"{'R@' + str(top_k) + ' by category':<40}{'n':>4}" + "".join(f"{arch:>12}" for arch in archs))
    for category, bucket in archs["A"]["by_category"].items():
        print(f"{category:<40}{bucket['queries']:>4}" + "".join(
            f"{archs[arch]['by_category'][category][f'recall@{top_k}']:>12.4f}" for arch in archs))
    print("-" * 80)
    for check in checks:
        print(f"REPRODUCTION {check['status']:<11} {check['check']}")
        for name, values in (check.get("mismatches") or {}).items():
            print(f"    {name}: measured {values['measured']} expected {values['expected']}")
    link_check = retrieval["vector_prompt_link_check"]
    print(f"VECTOR PROMPTS WITHOUT LINKS: {'passed' if link_check['passed'] else 'FAILED'}")
    print("=" * 80)


def print_answer_table(answers: Dict[str, Any]) -> None:
    archs = answers["architectures"]
    print("\n" + "=" * 92)
    if not answers["evaluative"]:
        print("PLUMBING ONLY - mock provider does no reasoning; answer numbers are not quality")
    if not answers["complete"]:
        print(f"PARTIAL RUN - {answers['partial_caveat']}")
    print(f"ANSWERS - {answers['queries_compared']} queries compared")
    print("  ".join(f"{arch}={archs[arch]['label']}" for arch in archs))
    print(f"{'metric':<28}" + "".join(f"{arch:>16}" for arch in archs))
    for name in ("accuracy", "abstention_rate", "evidence_coverage", "all_relevant_present_rate",
                 "mean_input_tokens", "mean_output_tokens", "mean_retrieval_calls",
                 "mean_generation_calls", "mean_retrieval_latency_ms", "mean_generation_latency_ms"):
        print(f"{name:<28}" + "".join(f"{archs[arch]['aggregate'][name]:>16.4f}" for arch in archs))
    print("-" * 92)
    print(f"{'accuracy by category':<40}{'n':>4}" + "".join(f"{arch:>12}" for arch in archs))
    for category, bucket in archs["A"]["by_category"].items():
        print(f"{category:<40}{bucket['queries']:>4}" + "".join(
            f"{archs[arch]['by_category'][category]['accuracy']:>12.4f}" for arch in archs))
    print("=" * 92)


# ================================== Main ====================================


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--phase", choices=("retrieval", "answers", "all"), default="all")
    parser.add_argument("--provider", default="mock")
    parser.add_argument("--model", default=None)
    parser.add_argument("--corpus", default="meridian-transit-v1")
    parser.add_argument("--limit", type=int, default=0, help="answer phase: only the first N queries")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=3, help="architecture D only")
    parser.add_argument("--max-retrievals", type=int, default=3, help="architecture D only")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default=None,
                        help="reasoning models only (groq gpt-oss); omitted means the provider default")
    parser.add_argument("--resume", action="store_true",
                        help="reuse per-query answers already recorded for each exact configuration")
    args = parser.parse_args()

    if args.corpus in rlm_eval.PRIVACY_RESTRICTED_CORPORA and args.provider != "mock":
        raise SystemExit(f"Refusing to send corpus {args.corpus!r} to provider {args.provider!r}: "
                         "it contains real personal documents.")

    corpus = bench.resolve_corpus(args.corpus)
    dataset, provenance = bench.load_dataset(corpus)
    service = bench.build_service(corpus)
    validation = bench.validate_corpus_and_ground_truth(service, dataset, corpus)
    frozen = rlm_loop.load_frozen_retrieval_config()
    retrievers = build_retrievers(service, frozen)

    retrieval: Optional[Dict[str, Any]] = None
    checks: Optional[List[Dict[str, Any]]] = None
    answers: Optional[Dict[str, Any]] = None
    if args.phase in ("retrieval", "all"):
        retrieval = run_retrieval_phase(dataset, retrievers, args.top_k)
        checks = reproduction_checks(retrieval, corpus["corpus_id"], frozen, args.top_k)
    if args.phase in ("answers", "all"):
        answers = run_answer_phase(args, corpus, provenance, dataset, retrievers,
                                   rlm_eval.station_universe(service))

    out_path = ROOT / f"experiment_architecture_benchmark_{corpus['corpus_id']}_{args.provider}.json"
    reproducibility = reproducibility_block(corpus, provenance, validation, answers, args.top_k)
    previous = load_previous(out_path, reproducibility)
    if retrieval is None:
        retrieval, checks = previous.get("retrieval"), previous.get("reproduction_checks")
    if answers is None:
        answers = previous.get("answers")
        if answers:
            reproducibility = reproducibility_block(corpus, provenance, validation, answers, args.top_k)

    output = {
        "experiment": "architecture-benchmark",
        "architectures": ARCHITECTURES,
        "graph_rag_caveat": (
            "Graph RAG (B) is vector-seeded expansion with the seeds excluded, identical to "
            "benchmark.py's graph_expansion baseline. It measures what graph traversal adds, "
            "not a vector-free graph retriever."
        ),
        "retrieval": retrieval,
        "reproduction_checks": checks,
        "answers": answers,
        "reproducibility": reproducibility,
    }
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    if retrieval and checks:
        print_retrieval_table(retrieval, checks, args.top_k)
    if answers:
        print_answer_table(answers)
    print(f"Wrote {out_path.name}")

    failed = [check for check in checks or [] if check["status"] == "FAILED"]
    if failed or (retrieval and not retrieval["vector_prompt_link_check"]["passed"]):
        print("!" * 80)
        print("REPRODUCTION FAILURE: measured retrieval does not match the saved reference. "
              "Numbers were not adjusted; investigate before trusting this comparison.")
        print("!" * 80)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
