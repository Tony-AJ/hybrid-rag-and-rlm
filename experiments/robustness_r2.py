"""Robustness experiment R2: incorrect graph relationships (PREREGISTRATION_R2.md).

Each query gets its own in-memory corrupted copy of the graph
(experiments.graph_corruption.CorruptedService); retrievers are rebuilt on it,
and R2e/R2f stack R1's MaskedRetriever. Graph files, document text, the vector
store, dataset v1.1, the frozen config, loop 1.7 and earlier results are never
written.

Conditions: A, B2, C (loop 1.7 one step, grounding off), D (loop 1.7 lean
grounding) and D-gate (gate-1.0 replayed offline over D's rows).

Phases:
  validate     G1-G8 and G11 checks; no model calls
  retrieval    A, B2, C rankings on the corrupted views; no model calls
  answers      validate, then run A, B2, C, D (hosted runs need the freeze)
  freeze       validate (with Phase 2 regression), record input hashes
  replay-gate  D-gate over the D rows of an answers output
  report       comparison tables
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import benchmark as bench  # noqa: E402
from experiments import architecture_benchmark as ab  # noqa: E402
from experiments import graph_corruption as gc  # noqa: E402
from experiments import robustness_r1 as r1  # noqa: E402
from rlm import evaluate as rlm_eval  # noqa: E402
from rlm import loop as rlm_loop  # noqa: E402
from rlm import loop_evidence as ea  # noqa: E402
from rlm import robustness_scoring_r2 as r2s  # noqa: E402
from rlm.providers import ProviderUnavailable, QuotaExceeded, build_provider  # noqa: E402

RUNNER_VERSION = "r2-run-1.0"
EXPERIMENT_ID = "robustness-r2"
DATASET_PATH = ROOT / "benchmark_corpora" / "meridian" / "robustness" / "incorrect_edges.json"
FREEZE_PATH = ROOT / "experiment_robustness_r2_freeze.json"
ANSWERS_PATH = ROOT / "experiment_robustness_r2_answers_meridian-transit-v1_groq.json"
REPLAY_PATH = ROOT / "experiment_robustness_r2_gate_replay.json"
RETRIEVAL_PATH = ROOT / "experiment_robustness_r2_retrieval_meridian-transit-v1.json"
COMPARISON_PATH = ROOT / "experiment_robustness_r2_comparison.json"
ARCHS = ("A", "B2", "C", "D")
CASES = ("R2a", "R2b", "R2c", "R2d", "R2e", "R2f")

# Earlier results that must stay byte-identical.
PROTECTED_RESULTS = {
    "experiment_architecture_benchmark_meridian-transit-v1_groq_phase2_frozen.json":
        "c70e8a416b1259387c8cf47bf47a415aefa4812ca2e4bd93ad1a0687f7a31407",
    "experiment_robustness_r1_meridian-transit-v1_groq_frozen.json":
        "941f0d4a710a3af4e88b2020b578221edeee7b6a155ee90488bc848aaa417d14",
    "experiment_r1prime_replay_gate.json": "9135a61b0d02a04d7f4b687e5aaf2820dcca6190affbe2ebaee21ad1cf4a294a",
    "experiment_r1prime_r1_ea_groq.json": "41f53f88785913330380eb142821ee382cd1d7ff9e93b4f6a98b3ae1c5aef002",
    "experiment_r1prime_v11_ea_groq.json": "f4f1095868a817bb5909072c613a50b56b005d40f6c467f0a9201b766a018cc8",
    "experiment_r1prime_r1_abstain_groq.json": "8abf587446e6582152eca4fbe271f3fe9f89e6d3821c4c524fc9e73288b8ac42",
    "experiment_r1prime_v11_abstain_groq.json": "06d545d493c029c2d31c859b9fca5b0663bdce6c5f4db92b3eb41797e6e73715",
}
STORE_FILES = [
    "benchmark_corpora/meridian/corpus_graph.json",
    "benchmark_corpora/meridian/corpus_documents.json",
    "db/graph_data_bench_meridian.json",
    "db/chroma_db_bench_meridian/chroma.sqlite3",
]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(file.relative_to(path)).replace("\\", "/").encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def store_hashes() -> Dict[str, Optional[str]]:
    """Document, graph and vector-store record hashes.

    Chroma rewrites its HNSW segment files (data_level0.bin, length.bin, ...) whenever the
    store is opened, in every earlier phase too, so those bytes are not a stable identity.
    The vector store is identified by chroma.sqlite3 (documents, ids, metadata) here, by
    the corpus fingerprint (G2) and by exact reproduction of Phase 2 rankings (G7).
    """
    return {name: rlm_eval.file_sha256(ROOT / name) for name in STORE_FILES}


check = r1.check


# ================================ Context ===================================


class Context:
    def __init__(self) -> None:
        self.store_before = store_hashes()
        self.corpus = bench.resolve_corpus("meridian-transit-v1")
        self.v11, self.v11_prov = bench.load_dataset(self.corpus)
        self.service = bench.build_service(self.corpus)
        self.frozen = rlm_loop.load_frozen_retrieval_config()
        self.base = ab.build_retrievers(self.service, self.frozen)
        raw = DATASET_PATH.read_bytes()
        self.payload = json.loads(raw.decode("utf-8"))
        self.items = self.payload["queries"]
        masks = {str(item["id"]): sorted(item.get("mask") or []) for item in self.items}
        self.provenance = {
            "dataset_path": str(DATASET_PATH.relative_to(ROOT)).replace("\\", "/"),
            "dataset_version": self.payload["dataset_version"],
            "dataset_sha256": sha(raw),
            "corruption_spec_sha256": gc.spec_sha256(self.items),
            "mask_sha256": sha(json.dumps(masks, sort_keys=True).encode("utf-8")),
        }
        self.universe = rlm_eval.station_universe(self.service)
        graph = self.service.graph_db.graph
        self.texts = {node_id: graph.nodes[node_id].get("text") or "" for node_id in graph.nodes}
        self.orders = gc.line_orders(self.texts)
        self.base_edges = gc.edge_set(graph)
        self.bounds = ab.answer_bounds(5, 3, 3)

    def view(self, item: Dict[str, Any]) -> gc.CorruptedService:
        return gc.CorruptedService(self.service, item.get("corruption") or [], tag=str(item["id"]))

    def retrievers(self, item: Dict[str, Any]) -> Dict[str, Any]:
        built = ab.build_retrievers(self.view(item), self.frozen)
        return {arch: r1.MaskedRetriever(built[ab.ARCHITECTURES[arch]["retriever"]], item.get("mask") or [])
                for arch in ARCHS}

    def base_unchanged(self) -> bool:
        return gc.edge_set(self.service.graph_db.graph) == self.base_edges


# =============================== Validation =================================


def run_validation(ctx: Context, phase2_regression: bool) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    items = ctx.items

    # G2 frozen inputs and earlier results
    validation = bench.validate_corpus_and_ground_truth(ctx.service, ctx.v11, ctx.corpus)
    checks.append(check("G2 dataset v1.1 sha256 unchanged", ctx.v11_prov["dataset_sha256"] == ctx.corpus["dataset_sha256"]))
    checks.append(check("G2 corpus fingerprint matches", validation["corpus_fingerprint"] == ctx.corpus["corpus_fingerprint"]))
    checks.append(check("G2 frozen retrieval config unchanged",
                        rlm_eval.file_sha256(rlm_loop.FROZEN_CONFIG_PATH) == r1.FROZEN_CONFIG_SHA256))
    drift = {name: rlm_eval.file_sha256(ROOT / name) for name, expected in PROTECTED_RESULTS.items()
             if rlm_eval.file_sha256(ROOT / name) != expected}
    checks.append(check("G2 Phase 2, R1 and R1' result files unchanged", not drift, drift))
    checks.append(check("G2 loop 1.7 and gate-1.0 versions", rlm_loop.LOOP_VERSION == "1.7" and ea.GATE_VERSION == "gate-1.0"))

    # G3 clean graph agrees with the documents
    disagree = [(s, t, d) for s, t, d in sorted(ctx.base_edges) if t in gc.CORRUPTIBLE
                and not gc.document_supports(t, s, d, ctx.texts, ctx.orders)]
    counted = sum(1 for _, t, _ in ctx.base_edges if t in gc.CORRUPTIBLE)
    checks.append(check(f"G3 clean graph: all {counted} caused_by / maintained_by / next_station edges agree with text",
                        not disagree, disagree[:10]))

    # G4, G5, G8 per item
    g4, g5, g8 = [], [], []
    for item in items:
        view = ctx.view(item)
        graph = view.graph_db.graph
        for operation in item["corruption"]:
            relation, source, inverse = operation["relation"], operation["source"], gc.INVERSE[operation["relation"]]
            for target in operation["corrupt_targets"]:
                if gc.document_supports(relation, source, target, ctx.texts, ctx.orders):
                    g4.append({"id": item["id"], "corrupt_edge_supported_by_text": target})
                if target.split("_")[0] != (operation.get("true_targets") or gc.edges_of(ctx.service.graph_db.graph, source, relation))[0].split("_")[0]:
                    g5.append({"id": item["id"], "node_type_differs": target})
                if target not in gc.edges_of(graph, source, relation):
                    g8.append({"id": item["id"], "corrupt_edge_missing_in_view": target})
                if inverse and source not in gc.edges_of(graph, target, inverse):
                    g5.append({"id": item["id"], "inverse_missing": [target, inverse, source]})
            for target in operation.get("true_targets", []):
                if not gc.document_supports(relation, source, target, ctx.texts, ctx.orders):
                    g4.append({"id": item["id"], "true_edge_not_supported": target})
                if target in gc.edges_of(graph, source, relation):
                    g5.append({"id": item["id"], "true_edge_still_present": target})
                if inverse and source in gc.edges_of(graph, target, inverse):
                    g5.append({"id": item["id"], "true_inverse_still_present": target})
            if operation["op"] == "add_spurious":
                kept = gc.edges_of(ctx.service.graph_db.graph, source, relation)
                if not set(kept) <= set(gc.edges_of(graph, source, relation)):
                    g5.append({"id": item["id"], "spurious_dropped_true_edges": kept})
            # corrupt identifiers must not appear in the source entity's own text
            for target in operation["corrupt_targets"]:
                label = ctx.texts[target].split(" is a ")[0] if target.startswith("station_") else gc.code_of(target)
                if label in ctx.texts[source]:
                    g5.append({"id": item["id"], "corrupt_label_in_source_text": label})
        # graph readers see the corruption
        retrievers = ctx.retrievers(item)
        for operation in item["corruption"]:
            for arch in ("B2", "C"):
                seen = {t for ts in retrievers[arch].relations(operation["source"]).values() for t in ts}
                if not set(operation["corrupt_targets"]) <= seen:
                    g8.append({"id": item["id"], "arch": arch, "relations_missing_corrupt": operation["corrupt_targets"]})
                base_seen = {t for ts in ctx.base[ab.ARCHITECTURES[arch]["retriever"]].relations(operation["source"]).values() for t in ts}
                if set(operation["corrupt_targets"]) & base_seen and operation["op"] == "substitute":
                    g8.append({"id": item["id"], "arch": arch, "base_graph_shows_corruption": True})
        if not ctx.base_unchanged():
            g8.append({"id": item["id"], "base_graph_changed": True})
    checks.append(check("G4 every corrupted edge contradicts the text; every replaced true edge agrees", not g4, g4))
    checks.append(check("G5 reverse edges consistent, same node type, corrupt label absent from source text", not g5, g5))
    checks.append(check("G8 corruption visible to graph readers (B2, C relations), invisible in the base graph", not g8, g8))

    # G6 document truth readable; masks hold
    g6 = []
    for item in items:
        if item["answerable"]:
            chunk_text = " ".join(ctx.texts[node] for node in item["document_truth"]["evidence_chunks"])
            for entity in item["document_truth"]["entities"]:
                if entity not in chunk_text:
                    g6.append({"id": item["id"], "truth_not_in_evidence_text": entity})
        else:
            mask = set(item["mask"])
            for arch in ("A", "B2", "C"):
                retriever = ctx.retrievers(item)[arch]
                for probe in [item["query"], *[ctx.texts[node] for node in mask]]:
                    leaked = mask & {chunk["node_id"] for chunk in retriever.retrieve(probe, 5)}
                    if leaked:
                        g6.append({"id": item["id"], "arch": arch, "masked_returned": sorted(leaked)})
                for node in mask:
                    if retriever.fetch(node) is not None:
                        g6.append({"id": item["id"], "arch": arch, "masked_fetched": node})
            for node in mask:
                found = [other for other, text in ctx.texts.items() if other not in mask and gc.code_of(node) in text]
                if found:
                    g6.append({"id": item["id"], "masked_code_elsewhere": node, "in": found[:3]})
    checks.append(check("G6 document truth readable (R2a-d); masks unretrievable, unfetchable, not reconstructable (R2e-f)",
                        not g6, g6))

    # G7 empty spec reproduces Phase 2
    if phase2_regression:
        empty = gc.CorruptedService(ctx.service, [])
        wrapped = ab.build_retrievers(empty, ctx.frozen)
        identity = [(item["id"], name) for item in ctx.v11 for name in ("vector", "graph_entity", "hybrid")
                    if wrapped[name].ranking(item["query"], 5) != ctx.base[name].ranking(item["query"], 5)]
        checks.append(check("G7 empty corruption spec: identical rankings on all v1.1 queries", not identity, identity))
        retrieval = ab.run_retrieval_phase(ctx.v11, wrapped, 5)
        for result in ab.reproduction_checks(retrieval, ctx.corpus["corpus_id"], ctx.frozen, 5):
            checks.append(check(f"G7 Phase 2 regression through the view: {result['check']}",
                                result["status"] == "passed", result.get("mismatches")))
        phase2 = json.loads(r1.PHASE2_FROZEN_RESULT.read_text(encoding="utf-8"))
        frozen_b2 = phase2["retrieval"]["architectures"]["B2"]["aggregate"]
        measured = retrieval["architectures"]["B2"]["aggregate"]
        diff = {k: (measured[k], frozen_b2[k]) for k in ab.metric_names(5) if round(measured[k], 4) != round(frozen_b2[k], 4)}
        checks.append(check("G7 Phase 2 regression through the view: B2 == frozen Phase 2 B2", not diff, diff))

    # G1 stores unchanged so far
    checks.append(check("G1 graph, corpus and vector store files unchanged; base edge set unchanged",
                        store_hashes() == ctx.store_before and ctx.base_unchanged()))

    # G11 signatures distinct
    sigs = {arch: signature(ctx, arch, "groq", "openai/gpt-oss-120b", {"reasoning_effort": "low"}) for arch in ARCHS}
    checks.append(check("G11 per-condition signatures distinct", len(set(sigs.values())) == len(ARCHS),
                        {arch: value[:12] for arch, value in sigs.items()}))
    return checks


# ================================ Retrieval =================================


def run_retrieval(ctx: Context) -> Dict[str, Any]:
    rows = []
    for item in ctx.items:
        retrievers = ctx.retrievers(item)
        truth_chunks = set(item["document_truth"]["evidence_chunks"]) if item["answerable"] else set()
        corrupt_nodes = {node for op in item["corruption"] for node in op["corrupt_targets"]}
        entry = {"id": item["id"], "case": item["case"], "query": item["query"]}
        for arch in ("A", "B2", "C"):
            ranking = [chunk["node_id"] for chunk in retrievers[arch].retrieve(item["query"], 5)]
            entry[arch] = {
                "ranking": ranking,
                "truth_recall@5": round(len(truth_chunks & set(ranking)) / len(truth_chunks), 4) if truth_chunks else None,
                "corrupt_target_in_top5": bool(corrupt_nodes & set(ranking)),
            }
        clean = r1.MaskedRetriever(ctx.base["vector"], item.get("mask") or [])
        entry["A_identical_to_clean_graph"] = [c["node_id"] for c in clean.retrieve(item["query"], 5)] == entry["A"]["ranking"]
        rows.append(entry)
    summary = {}
    for arch in ("A", "B2", "C"):
        for case in CASES:
            group = [row for row in rows if row["case"] == case]
            recalls = [row[arch]["truth_recall@5"] for row in group if row[arch]["truth_recall@5"] is not None]
            summary.setdefault(arch, {})[case] = {
                "truth_recall@5": round(statistics.fmean(recalls), 4) if recalls else None,
                "corrupt_target_in_top5": r1.fraction(sum(row[arch]["corrupt_target_in_top5"] for row in group), len(group))["text"],
            }
    return {"per_query": rows, "summary": summary,
            "P1_A_rankings_identical_to_clean": r1.fraction(sum(row["A_identical_to_clean_graph"] for row in rows), len(rows))["text"]}


# ================================= Answers ==================================


def signature(ctx: Context, arch: str, provider: str, model: str, options: Dict[str, Any]) -> str:
    payload = json.dumps({
        "experiment": EXPERIMENT_ID, "runner_version": RUNNER_VERSION, "architecture": arch,
        "corpus_id": ctx.corpus["corpus_id"], **ctx.provenance, "corruption_version": gc.CORRUPTION_VERSION,
        "retriever": ctx.base[ab.ARCHITECTURES[arch]["retriever"]].as_record(),
        "bounds": ctx.bounds[arch].as_record(), "loop_version": rlm_loop.LOOP_VERSION,
        "prompts": rlm_loop.TEMPLATE_HASHES, "provider": provider, "model": model, "generation_options": options,
    }, sort_keys=True)
    return sha(payload.encode("utf-8"))


def coverage(item: Dict[str, Any], evidence: List[str]) -> Optional[float]:
    truth = set(item["document_truth"]["evidence_chunks"]) if item["answerable"] else set()
    return round(len(truth & set(evidence)) / len(truth), 4) if truth else None


def run_answers(args: argparse.Namespace, ctx: Context) -> Dict[str, Any]:
    try:
        inner = build_provider(args.provider, model=args.model, temperature=args.temperature,
                               max_tokens=args.max_tokens, reasoning_effort=args.reasoning_effort)
    except (ProviderUnavailable, ValueError) as exc:
        raise SystemExit(str(exc))
    provider = r1.RecordingProvider(inner)
    verification = provider.verify_model()
    options = {"reasoning_effort": args.reasoning_effort}
    state = {}
    for arch in ARCHS:
        sig = signature(ctx, arch, args.provider, provider.config.model, options)
        path = ROOT / f"experiment_robustness_r2_cache_{ctx.corpus['corpus_id']}_{args.provider}_{arch}_{sig[:12]}.json"
        state[arch] = {"signature": sig, "path": path, "rows": [],
                       "cache": rlm_eval.load_cache(path, sig) if args.resume else {}}
        print(f"resume {arch}: {len(state[arch]['cache'])} cached rows (signature {sig[:12]})")

    items = ctx.items[: args.limit] if args.limit else ctx.items
    quota_stop = None
    for index, item in enumerate(items, 1):
        retrievers = ctx.retrievers(item)
        line = []
        for arch in ARCHS:
            arch_state = state[arch]
            raw = arch_state["cache"].get(int(item["id"]))
            if raw is None:
                provider.prompts = []
                try:
                    run = rlm_loop.run_loop(retrievers[arch], provider, item["query"], ctx.bounds[arch])
                except QuotaExceeded as exc:
                    quota_stop = {"provider": exc.provider, "detail": exc.detail,
                                  "stopped_at_query_id": item["id"], "stopped_at_architecture": arch}
                    break
                raw = {"id": item["id"], "answer": run.answer, "steps": run.steps, "stop_reason": run.stop_reason,
                       "prompts": list(provider.prompts), "evidence_chunks": run.evidence_ids,
                       "cost": {"retrieval_calls": run.retrieval_calls, "generation_calls": run.generation_calls,
                                "input_tokens": run.input_tokens, "output_tokens": run.output_tokens,
                                "retrieval_latency_ms": round(run.retrieval_latency_ms, 3),
                                "generation_latency_ms": round(run.generation_latency_ms, 3),
                                "by_stage": run.stage_costs}}
                if not args.no_cache_write:
                    arch_state["cache"][int(item["id"])] = raw
                    rlm_eval.save_cache(arch_state["path"], arch_state["signature"], arch_state["cache"])
            row = score_row(arch, item, raw, ctx)
            arch_state["rows"].append(row)
            line.append(f"{arch}={row['score']['outcome'][:12]}")
        if not ctx.base_unchanged():
            raise SystemExit(f"Base graph changed during query {item['id']}; stopping.")
        print(f"[{index:02d}/{len(items)}] {item['id']} {item['case']} {' '.join(line)}")
        if quota_stop:
            print(f"QUOTA STOP: {quota_stop['detail'][:160]}")
            break

    common = set.intersection(*[{row["id"] for row in s["rows"]} for s in state.values()])
    architectures = {}
    for arch, s in state.items():
        rows = [row for row in s["rows"] if row["id"] in common]
        architectures[arch] = {**ab.ARCHITECTURES[arch], "loop_bounds": ctx.bounds[arch].as_record(),
                               "run_signature": s["signature"],
                               "cache_file": None if args.no_cache_write else s["path"].name,
                               "aggregate": aggregate(rows), "by_case": {c: aggregate([r for r in rows if r["case"] == c]) for c in CASES},
                               "per_query": rows}
    complete = all(len(s["rows"]) == len(items) for s in state.values())
    return {"status": "plumbing_only" if not provider.EVALUATIVE else ("valid" if complete else "partial"),
            "complete": complete, "queries_requested": len(items), "queries_compared": len(common),
            "quota_stop": quota_stop,
            "generation": {**provider.config.as_record(), "model_verification": verification},
            "architectures": architectures}


def score_row(arch: str, item: Dict[str, Any], raw: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
    return {**raw, "architecture": arch, "case": item["case"], "question": item["query"],
            "score": r2s.score_r2(raw["answer"], item, raw["prompts"], ctx.universe),
            "evidence_coverage": coverage(item, raw["evidence_chunks"])}


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    answerable = [row for row in rows if row["score"]["answerable"]]
    unanswerable = [row for row in rows if not row["score"]["answerable"]]

    def count(pred, subset) -> int:
        return sum(1 for row in subset if pred(row))

    def avg(values) -> Optional[float]:
        values = [v for v in values if v is not None]
        return round(statistics.fmean(values), 4) if values else None

    return {
        "queries": n,
        "outcomes": {o: r1.fraction(count(lambda row, o=o: row["score"]["outcome"] == o, rows), n)["text"] for o in r2s.OUTCOMES},
        "accuracy_answerable": r1.fraction(count(lambda row: row["score"]["outcome"] == "correct", answerable), len(answerable))["text"],
        "correct_abstention_unanswerable": r1.fraction(count(lambda row: row["score"]["outcome"] == "correct_abstention", unanswerable), len(unanswerable))["text"],
        "graph_misled_or_mixed": r1.fraction(count(lambda row: row["score"]["outcome"] in ("graph_misled", "mixed"), rows), n)["text"],
        "graph_supported": r1.fraction(count(lambda row: row["score"]["graph_supported"], rows), n)["text"],
        "corrupt_visible_in_links": r1.fraction(count(lambda row: row["score"]["corrupt_visible_in_links"], rows), n)["text"],
        "wrong_target_text_received": r1.fraction(count(lambda row: row["score"]["wrong_target_text_received"], rows), n)["text"],
        "document_supported": r1.fraction(count(lambda row: row["score"]["document_supported"], answerable), len(answerable))["text"],
        "identifier_supported": r1.fraction(count(lambda row: row["score"]["identifier_supported"], rows), n)["text"],
        "mean_evidence_coverage": avg(row["evidence_coverage"] for row in rows),
        "mean_input_tokens": avg(row["cost"]["input_tokens"] for row in rows),
        "mean_output_tokens": avg(row["cost"]["output_tokens"] for row in rows),
        "mean_generation_calls": avg(row["cost"]["generation_calls"] for row in rows),
        "mean_retrieval_calls": avg(row["cost"]["retrieval_calls"] for row in rows),
        "mean_generation_latency_ms": avg(row["cost"]["generation_latency_ms"] for row in rows),
    }


# ============================== Gate replay =================================


def replay_gate(ctx: Context, answers: Dict[str, Any]) -> Dict[str, Any]:
    items = {item["id"]: item for item in ctx.items}
    rows = []
    for row in answers["architectures"]["D"]["per_query"]:
        item = items[row["id"]]
        texts = [ctx.texts.get(node, "") for node in row["evidence_chunks"]]
        failures = [{"node_id": node, "reason": "unavailable"} for node in item.get("mask") or []]
        verdict = ea.sufficiency_gate(row["answer"], item["query"], texts, ctx.universe, failures)
        answer = ea.ABSTAIN_ANSWER if verdict["decision"] == "abstain" else row["answer"]
        rows.append({**score_row("D-gate", item, {**row, "answer": answer}, ctx),
                     "answer_control": row["answer"], "gate": verdict})
    return {"gate_version": ea.GATE_VERSION, "aggregate": aggregate(rows),
            "by_case": {c: aggregate([r for r in rows if r["case"] == c]) for c in CASES}, "per_query": rows}


# ================================ Freezing ==================================


def input_hashes(ctx: Context) -> Dict[str, Any]:
    files = {
        "preregistration": "PREREGISTRATION_R2.md", "corruption_module": "experiments/graph_corruption.py",
        "scorer_r2": "rlm/robustness_scoring_r2.py", "scorer_r1": "rlm/robustness_scoring.py",
        "runner": "experiments/robustness_r2.py", "loop_evidence_gate": "rlm/loop_evidence.py",
        "architecture_benchmark": "experiments/architecture_benchmark.py",
    }
    return {**ctx.provenance, "corruption_version": gc.CORRUPTION_VERSION, "scorer_version": r2s.R2_SCORER_VERSION,
            "runner_version": RUNNER_VERSION, "gate_version": ea.GATE_VERSION, "loop_version": rlm_loop.LOOP_VERSION,
            "prompt_template_sha256": rlm_loop.TEMPLATE_HASHES,
            **{f"{key}_sha256": rlm_eval.file_sha256(ROOT / path) for key, path in files.items()},
            "stores": ctx.store_before}


# ================================== Report ==================================


def build_report(answers: Dict[str, Any], replay: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    columns = {arch: answers["architectures"][arch] for arch in ARCHS}
    if replay:
        columns["D-gate"] = replay
    return {"overall": {name: col["aggregate"] for name, col in columns.items()},
            "by_case": {case: {name: col["by_case"][case] for name, col in columns.items()} for case in CASES}}


def print_report(report: Dict[str, Any]) -> None:
    names = list(report["overall"])
    keys = ["accuracy_answerable", "correct_abstention_unanswerable", "graph_misled_or_mixed", "graph_supported",
            "document_supported", "identifier_supported", "mean_evidence_coverage", "mean_input_tokens",
            "mean_generation_calls"]
    print("\n" + "=" * (34 + 18 * len(names)))
    print(f"{'R2 overall':<34}" + "".join(f"{n:>18}" for n in names))
    for key in keys:
        print(f"{key:<34}" + "".join(f"{str(report['overall'][n][key]):>18}" for n in names))
    for case in CASES:
        print(f"-- {case}")
        for outcome in r2s.OUTCOMES:
            print(f"  {outcome:<32}" + "".join(f"{report['by_case'][case][n]['outcomes'][outcome]:>18}" for n in names))


# =================================== Main ===================================


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--phase", choices=("validate", "retrieval", "answers", "freeze", "replay-gate", "report"),
                        default="validate")
    parser.add_argument("--provider", default="mock")
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-cache-write", action="store_true")
    parser.add_argument("--skip-phase2-regression", action="store_true")
    parser.add_argument("--answers", default=str(ANSWERS_PATH), help="answers file for replay-gate / report")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    ctx = Context()
    output: Dict[str, Any] = {"experiment": EXPERIMENT_ID, "phase": args.phase}
    out: Optional[Path] = Path(args.out) if args.out else None

    if args.phase in ("validate", "answers", "freeze"):
        checks = run_validation(ctx, phase2_regression=args.phase == "freeze" or not args.skip_phase2_regression)
        r1.print_checks(checks)
        if not r1.validation_passed(checks):
            print("Offline validation FAILED. No model calls were made.")
            return 1
        output["validation"] = checks
        if args.phase == "freeze":
            FREEZE_PATH.write_text(json.dumps({
                "experiment": EXPERIMENT_ID, "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "git_revision": rlm_eval.git_revision(), "inputs": input_hashes(ctx), "validation": checks,
            }, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Wrote {FREEZE_PATH.name}")
            return 0

    if args.phase == "retrieval":
        output["retrieval"] = run_retrieval(ctx)
        print(json.dumps(output["retrieval"]["summary"], indent=1))
        print("P1 A rankings identical to clean graph:", output["retrieval"]["P1_A_rankings_identical_to_clean"])
        out = out or RETRIEVAL_PATH
    elif args.phase == "answers":
        if args.provider != "mock":
            frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8")).get("inputs") if FREEZE_PATH.exists() else None
            current = input_hashes(ctx)
            if frozen != current:
                diff = sorted(k for k in current if not frozen or frozen.get(k) != current[k])
                raise SystemExit(f"Refusing hosted run: R2 inputs not frozen or changed since freezing: {diff}")
        output["answers"] = run_answers(args, ctx)
        print_report(build_report(output["answers"], None))
        out = out or (ANSWERS_PATH if args.provider == "groq" else
                      ROOT / f"experiment_robustness_r2_answers_meridian-transit-v1_{args.provider}.json")
    elif args.phase == "replay-gate":
        answers = json.loads(Path(args.answers).read_text(encoding="utf-8"))["answers"]
        output["replay"] = replay_gate(ctx, answers)
        out = out or REPLAY_PATH
    elif args.phase == "report":
        answers = json.loads(Path(args.answers).read_text(encoding="utf-8"))["answers"]
        replay = json.loads(REPLAY_PATH.read_text(encoding="utf-8"))["replay"] if REPLAY_PATH.exists() else None
        output["report"] = build_report(answers, replay)
        print_report(output["report"])
        out = out or COMPARISON_PATH

    if store_hashes() != ctx.store_before or not ctx.base_unchanged():
        print("G1 FAILED after the phase: a store file or the base graph changed.")
        return 1
    output["reproducibility"] = {"runner_version": RUNNER_VERSION, "git_revision": rlm_eval.git_revision(),
                                 "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                 "inputs": input_hashes(ctx)}
    if out:
        out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
