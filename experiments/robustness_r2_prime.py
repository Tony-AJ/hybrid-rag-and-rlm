"""R2': relation-level verification (PREREGISTRATION_R2_PRIME.md).

Conditions: C and D run live (loop 1.7 on the per-query corrupted, masked view);
D-gate (gate-1.0), D-relation and C-relation (relver-1.0) are exact offline replays
over the recorded prompts and answers. No condition adds a model call.

Phases:
  validate  V1 (frozen inputs), V5 (dataset), V8 (replay determinism), V9 (signatures),
            plus the V2/V3/V4 sweeps; no model calls
  sweep     V2/V3/V4 only, written to experiment_r2p_verifier_sweep.json
  answers   validate, then C and D (hosted runs need the freeze)
  freeze    validate, record input hashes
  replay    D-gate, D-relation, C-relation over an answers file
  report    comparison tables
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import benchmark as bench  # noqa: E402
from experiments import architecture_benchmark as ab  # noqa: E402
from experiments import graph_corruption as gc  # noqa: E402
from experiments import robustness_r1 as r1  # noqa: E402
from experiments import robustness_r2 as r2  # noqa: E402
from rlm import evaluate as rlm_eval  # noqa: E402
from rlm import loop as rlm_loop  # noqa: E402
from rlm import loop_evidence as ea  # noqa: E402
from rlm import relation_verifier as rv  # noqa: E402
from rlm import robustness_scoring as r1s  # noqa: E402
from rlm import robustness_scoring_r2 as r2s  # noqa: E402
from rlm import robustness_scoring_r2p as r2ps  # noqa: E402
from rlm.providers import ProviderUnavailable, QuotaExceeded, build_provider  # noqa: E402

RUNNER_VERSION = "r2p-run-1.0"
EXPERIMENT_ID = "robustness-r2-prime"
ROBUST = ROOT / "benchmark_corpora" / "meridian" / "robustness"
DATASET_PATH = ROBUST / "relation_verification.json"
SCHEMA_PATH = ROBUST / "relation_schema.json"
FREEZE_PATH = ROOT / "experiment_r2p_freeze.json"
SWEEP_PATH = ROOT / "experiment_r2p_verifier_sweep.json"
ANSWERS_PATH = ROOT / "experiment_r2p_answers_meridian-transit-v1_groq.json"
REPLAY_PATH = ROOT / "experiment_r2p_relation_replay.json"
COMPARISON_PATH = ROOT / "experiment_r2p_comparison.json"
R2_FREEZE = ROOT / "experiment_robustness_r2_freeze.json"
R2_FROZEN = ROOT / "experiment_robustness_r2_meridian-transit-v1_groq_frozen.json"
R2_DATASET = ROBUST / "incorrect_edges.json"
ARCHS = ("C", "D")
CASES = ("P", "N", "S", "U")
SWEEP_SEED = 20261113
PROTECTED = {
    **r2.PROTECTED_RESULTS,
    "experiment_robustness_r2_meridian-transit-v1_groq_frozen.json":
        "97d3e3af4b808e46030330f6d2fa985917c623d327de58fd6d411c89b56781c3",
    "experiment_robustness_r2_gate_replay.json":
        "fa738421a5d00757e08f54ee70751607fb29174b2c2dc118acb4aa8e2aacacc7",
    "experiment_robustness_r2_freeze.json":
        "8f99319284c7704e9e9ca4694543615fb9877bbdc5620ebe3350e18a94d72d4a",
}
# Sources pinned by the R2 freeze that R2' must not change.
PINNED_SOURCES = {"loop_evidence_gate_sha256": "rlm/loop_evidence.py", "corruption_module_sha256": "experiments/graph_corruption.py",
                  "scorer_r2_sha256": "rlm/robustness_scoring_r2.py", "scorer_r1_sha256": "rlm/robustness_scoring.py",
                  "runner_sha256": "experiments/robustness_r2.py", "architecture_benchmark_sha256": "experiments/architecture_benchmark.py"}

check = r1.check


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ================================ Context ===================================


class Context:
    def __init__(self) -> None:
        self.store_before = r2.store_hashes()
        self.corpus = bench.resolve_corpus("meridian-transit-v1")
        self.v11, self.v11_prov = bench.load_dataset(self.corpus)
        self.service = bench.build_service(self.corpus)
        self.frozen = rlm_loop.load_frozen_retrieval_config()
        self.base = ab.build_retrievers(self.service, self.frozen)
        raw = DATASET_PATH.read_bytes()
        self.payload = json.loads(raw.decode("utf-8"))
        self.items = self.payload["queries"]
        self.schema = rv.load_schema(SCHEMA_PATH)
        graph = self.service.graph_db.graph
        self.texts = {node: graph.nodes[node].get("text") or "" for node in graph.nodes}
        self.registry = rv.Registry(self.schema, self.texts)
        self.clean_edges = gc.edge_set(graph)
        self.universe = rlm_eval.station_universe(self.service)
        self.bounds = ab.answer_bounds(5, 3, 3)
        masks = {str(item["id"]): sorted(item.get("mask") or []) for item in self.items}
        self.provenance = {
            "dataset_path": str(DATASET_PATH.relative_to(ROOT)).replace("\\", "/"),
            "dataset_version": self.payload["dataset_version"], "dataset_sha256": sha(raw),
            "corruption_spec_sha256": gc.spec_sha256(self.items),
            "mask_sha256": sha(json.dumps(masks, sort_keys=True).encode("utf-8")),
            "schema_sha256": rlm_eval.file_sha256(SCHEMA_PATH), "schema_version": self.schema["version"],
        }

    def retriever(self, item: Dict[str, Any], arch: str = "D") -> r1.MaskedRetriever:
        built = ab.build_retrievers(gc.CorruptedService(self.service, item.get("corruption") or [], tag=str(item["id"])), self.frozen)
        return r1.MaskedRetriever(built[ab.ARCHITECTURES[arch]["retriever"]], item.get("mask") or [])

    def verifier_for(self, retriever: Any) -> rv.Verifier:
        def fetch_text(node: str) -> Optional[str]:
            chunk = retriever.fetch(node)
            return chunk.get("text") if chunk else None
        return rv.Verifier(self.schema, self.registry, fetch_text, retriever.relations)

    def clean_verifier(self) -> rv.Verifier:
        return self.verifier_for(r1.MaskedRetriever(self.base["hybrid"], []))

    def base_unchanged(self) -> bool:
        return gc.edge_set(self.service.graph_db.graph) == self.clean_edges


# ================================ Sweeps ====================================


def sweep_clean(ctx: Context) -> Dict[str, Any]:
    """V2: every clean edge must be supported (text-verifiable) or graph_only."""
    verifier = ctx.clean_verifier()
    per: Dict[str, Dict[str, int]] = {}
    problems = []
    for subject, relation, obj in sorted(ctx.clean_edges):
        verdict = verifier.verify(subject, relation, obj)
        per.setdefault(relation, {}).setdefault(verdict["status"], 0)
        per[relation][verdict["status"]] += 1
        expected = "graph_only" if ctx.schema["relations"][relation]["check"] == "not_text_verifiable" else "supported"
        if verdict["status"] != expected:
            problems.append({"claim": [subject, relation, obj], **verdict})
    false_contradictions = {r: counts.get("contradicted", 0) for r, counts in per.items()}
    return {"edges": len(ctx.clean_edges), "by_relation": per, "false_contradictions_by_relation": false_contradictions,
            "problems": problems, "passed": not problems}


def sweep_substitutions(ctx: Context) -> Dict[str, Any]:
    """V3: every same-type wrong-object substitution of every clean text-verifiable edge."""
    verifier = ctx.clean_verifier()
    by_type: Dict[str, List[str]] = {}
    for node, etype in ctx.registry.types.items():
        by_type.setdefault(etype, []).append(node)
    per: Dict[str, Dict[str, int]] = {}
    false_supports = []
    for subject, relation, obj in sorted(ctx.clean_edges):
        spec = ctx.schema["relations"][relation]
        if spec["check"] == "not_text_verifiable":
            continue
        for wrong in by_type.get(spec["object"], []):
            if wrong == obj or (subject, relation, wrong) in ctx.clean_edges:
                continue
            status = verifier.verify(subject, relation, wrong)["status"]
            per.setdefault(relation, {"contradicted": 0, "unverified": 0, "supported": 0, "graph_only": 0})[status] += 1
            if status == "supported":
                false_supports.append([subject, relation, wrong])
    summary = {}
    for relation, counts in per.items():
        total = sum(counts.values())
        summary[relation] = {**counts, "total": total,
                             "detection_rate": f"{counts['contradicted']}/{total} ({100 * counts['contradicted'] / total:.1f}%)"}
    return {"by_relation": summary, "false_supports": false_supports[:50], "false_support_count": len(false_supports)}


def sweep_random(ctx: Context, n: int = 120) -> Dict[str, Any]:
    """V4: seeded random substitutions through the corruption layer and full claim extraction."""
    rng = random.Random(SWEEP_SEED)
    test_pairs = {(op["source"], target) for item in ctx.items for op in item.get("corruption") or []
                  for target in op["corrupt_targets"]}
    candidates = sorted((s, r, o) for s, r, o in ctx.clean_edges if r in gc.CORRUPTIBLE)
    by_type: Dict[str, List[str]] = {}
    for node, etype in ctx.registry.types.items():
        by_type.setdefault(etype, []).append(node)
    outcomes: Dict[str, int] = {}
    supported_cases = []
    tried = 0
    while tried < n:
        subject, relation, obj = rng.choice(candidates)
        wrong = rng.choice(sorted(by_type[ctx.registry.type_of(obj)]))
        if wrong == obj or (subject, relation, wrong) in ctx.clean_edges or (subject, wrong) in test_pairs:
            continue
        tried += 1
        item = {"id": f"v4-{tried}", "corruption": [{"op": "substitute", "relation": relation, "source": subject,
                                                     "true_targets": gc.edges_of(ctx.service.graph_db.graph, subject, relation),
                                                     "corrupt_targets": [wrong]}]}
        retriever = ctx.retriever(item)
        chunk = retriever.fetch(subject)
        prompt = "Evidence:\n" + rlm_loop.format_evidence([chunk])
        question = f"What is related to {ctx.registry.labels[subject]}?"
        answer = f"{ctx.registry.labels[wrong]}."
        result = rv.verify_answer(answer, question, [prompt], ctx.verifier_for(retriever))
        status = result["entities"].get(wrong, {}).get("status", "not_an_answer_entity")
        outcomes[status] = outcomes.get(status, 0) + 1
        if status == "supported":
            supported_cases.append([subject, relation, wrong])
    return {"seed": SWEEP_SEED, "n": n, "entity_status": outcomes, "supported_cases": supported_cases,
            "passed": not supported_cases}


# =============================== Validation =================================


def run_validation(ctx: Context) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    # V1
    validation = bench.validate_corpus_and_ground_truth(ctx.service, ctx.v11, ctx.corpus)
    checks.append(check("V1 dataset v1.1 and corpus fingerprint unchanged",
                        ctx.v11_prov["dataset_sha256"] == ctx.corpus["dataset_sha256"]
                        and validation["corpus_fingerprint"] == ctx.corpus["corpus_fingerprint"]))
    checks.append(check("V1 frozen retrieval config unchanged",
                        rlm_eval.file_sha256(rlm_loop.FROZEN_CONFIG_PATH) == r1.FROZEN_CONFIG_SHA256))
    drift = {name: rlm_eval.file_sha256(ROOT / name) for name, expected in PROTECTED.items()
             if rlm_eval.file_sha256(ROOT / name) != expected}
    checks.append(check("V1 Phase 2, R1, R1' and R2 result files unchanged", not drift, drift))
    r2_inputs = json.loads(R2_FREEZE.read_text(encoding="utf-8"))["inputs"]
    source_drift = {path: "changed" for key, path in PINNED_SOURCES.items()
                    if rlm_eval.file_sha256(ROOT / path) != r2_inputs[key]}
    checks.append(check("V1 loop_evidence (gate-1.0), corrupt-1.0, R1/R2 scorers, R2 runner, benchmark unchanged since R2 freeze",
                        not source_drift and rlm_loop.LOOP_VERSION == "1.7"
                        and rlm_loop.TEMPLATE_HASHES == r2_inputs["prompt_template_sha256"], source_drift))
    checks.append(check("V1 store files equal the R2 freeze record", ctx.store_before == r2_inputs["stores"]))

    # V2-V4 sweeps
    sweeps = {"V2_clean": sweep_clean(ctx), "V3_substitutions": sweep_substitutions(ctx), "V4_random": sweep_random(ctx)}
    checks.append(check(f"V2 clean graph: all {sweeps['V2_clean']['edges']} edges supported or graph_only (H5)",
                        sweeps["V2_clean"]["passed"], sweeps["V2_clean"]["problems"][:10]))
    checks.append({"check": "V3 exhaustive substitution sweep (reported, not pass/fail)", "status": "info",
                   "detail": {r: v["detection_rate"] for r, v in sweeps["V3_substitutions"]["by_relation"].items()}})
    checks.append(check("V4 seeded random corruptions: no corrupted claim labelled supported",
                        sweeps["V4_random"]["passed"], sweeps["V4_random"]["supported_cases"]))

    # V5 dataset
    v5 = []
    v11_questions = {item["query"] for item in ctx.v11}
    r2_questions = {item["query"] for item in json.loads(R2_DATASET.read_text(encoding="utf-8"))["queries"]}
    for item in ctx.items:
        if item["query"] in v11_questions or item["query"] in r2_questions:
            v5.append({"id": item["id"], "question_reused": True})
        view = gc.CorruptedService(ctx.service, item.get("corruption") or [])
        for op in item.get("corruption") or []:
            for target in op["corrupt_targets"]:
                if gc.document_supports(op["relation"], op["source"], target, ctx.texts):
                    v5.append({"id": item["id"], "corrupt_edge_supported": target})
                if target not in gc.edges_of(view.graph_db.graph, op["source"], op["relation"]):
                    v5.append({"id": item["id"], "corrupt_edge_missing": target})
            for target in op.get("true_targets", []):
                if not gc.document_supports(op["relation"], op["source"], target, ctx.texts):
                    v5.append({"id": item["id"], "true_edge_not_supported": target})
        if item["answerable"] and item["family"] != "authorised_by":
            chunk_text = " ".join(ctx.texts[node] for node in item["document_truth"]["evidence_chunks"])
            v5 += [{"id": item["id"], "truth_not_readable": e} for e in item["document_truth"]["entities"] if e not in chunk_text]
        if item["mask"]:
            retriever = ctx.retriever(item)
            for probe in [item["query"], *[ctx.texts[node] for node in item["mask"]]]:
                leaked = set(item["mask"]) & {c["node_id"] for c in retriever.retrieve(probe, 5)}
                if leaked:
                    v5.append({"id": item["id"], "masked_returned": sorted(leaked)})
            for node in item["mask"]:
                if retriever.fetch(node) is not None:
                    v5.append({"id": item["id"], "masked_fetched": node})
                elsewhere = [o for o, t in ctx.texts.items() if o not in item["mask"] and gc.code_of(node) in t]
                if elsewhere:
                    v5.append({"id": item["id"], "masked_code_elsewhere": node, "in": elsewhere[:3]})
        if not ctx.base_unchanged():
            v5.append({"id": item["id"], "base_graph_changed": True})
    checks.append(check("V5 dataset: corruptions contradict text, true edges agree, truth readable, masks hold, no reused questions",
                        not v5, v5))

    # V8 replay determinism on R2 D rows (plumbing only)
    before = {name: rlm_eval.file_sha256(ROOT / name) for name in PROTECTED}
    r2_items = {item["id"]: item for item in json.loads(R2_DATASET.read_text(encoding="utf-8"))["queries"]}
    runs = []
    for _ in range(2):
        decisions = []
        for row in json.loads(R2_FROZEN.read_text(encoding="utf-8"))["answers"]["architectures"]["D"]["per_query"]:
            item = r2_items[row["id"]]
            result = rv.verify_answer(row["answer"], item["query"], row["prompts"], ctx.verifier_for(ctx.retriever(item)))
            decisions.append((row["id"], result["decision"], result["final_answer"], json.dumps(result["claims"], sort_keys=True)))
        runs.append(decisions)
    after = {name: rlm_eval.file_sha256(ROOT / name) for name in PROTECTED}
    checks.append(check("V8 relver-1.0 replay over R2 frozen D rows is deterministic; R2 files byte-identical",
                        runs[0] == runs[1] and before == after))

    # V9 signatures
    sigs = {arch: signature(ctx, arch, "groq", "openai/gpt-oss-120b", {"reasoning_effort": "low"}) for arch in ARCHS}
    checks.append(check("V9 per-condition signatures distinct and differ from R2's",
                        len(set(sigs.values())) == len(ARCHS), {a: s[:12] for a, s in sigs.items()}))
    checks.append(check("V1 store files and base graph unchanged after validation",
                        r2.store_hashes() == ctx.store_before and ctx.base_unchanged()))
    return checks, sweeps


# ================================= Answers ==================================


def signature(ctx: Context, arch: str, provider: str, model: str, options: Dict[str, Any]) -> str:
    payload = json.dumps({
        "experiment": EXPERIMENT_ID, "runner_version": RUNNER_VERSION, "architecture": arch,
        **ctx.provenance, "corruption_version": gc.CORRUPTION_VERSION,
        "retriever": ctx.base[ab.ARCHITECTURES[arch]["retriever"]].as_record(), "bounds": ctx.bounds[arch].as_record(),
        "loop_version": rlm_loop.LOOP_VERSION, "prompts": rlm_loop.TEMPLATE_HASHES,
        "provider": provider, "model": model, "generation_options": options,
    }, sort_keys=True)
    return sha(payload.encode("utf-8"))


def coverage(item: Dict[str, Any], evidence: List[str]) -> Optional[float]:
    truth = set(item["document_truth"]["evidence_chunks"]) if item["answerable"] else set()
    return round(len(truth & set(evidence)) / len(truth), 4) if truth else None


def score_row(arch: str, item: Dict[str, Any], raw: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
    return {**raw, "architecture": arch, "case": item["case"], "family": item["family"], "question": item["query"],
            "score": r2ps.score_answer(raw["answer"], item, raw["prompts"], ctx.universe),
            "evidence_coverage": coverage(item, raw["evidence_chunks"])}


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
        path = ROOT / f"experiment_r2p_cache_{ctx.corpus['corpus_id']}_{args.provider}_{arch}_{sig[:12]}.json"
        state[arch] = {"signature": sig, "path": path, "rows": [], "cache": rlm_eval.load_cache(path, sig) if args.resume else {}}
        print(f"resume {arch}: {len(state[arch]['cache'])} cached rows (signature {sig[:12]})")
    items = ctx.items[: args.limit] if args.limit else ctx.items
    quota_stop = None
    for index, item in enumerate(items, 1):
        line = []
        for arch in ARCHS:
            s = state[arch]
            raw = s["cache"].get(int(item["id"]))
            if raw is None:
                provider.prompts = []
                try:
                    run = rlm_loop.run_loop(ctx.retriever(item, arch), provider, item["query"], ctx.bounds[arch])
                except QuotaExceeded as exc:
                    quota_stop = {"provider": exc.provider, "detail": exc.detail, "stopped_at_query_id": item["id"], "stopped_at_architecture": arch}
                    break
                raw = {"id": item["id"], "answer": run.answer, "steps": run.steps, "stop_reason": run.stop_reason,
                       "prompts": list(provider.prompts), "evidence_chunks": run.evidence_ids,
                       "cost": {"retrieval_calls": run.retrieval_calls, "generation_calls": run.generation_calls,
                                "input_tokens": run.input_tokens, "output_tokens": run.output_tokens,
                                "retrieval_latency_ms": round(run.retrieval_latency_ms, 3),
                                "generation_latency_ms": round(run.generation_latency_ms, 3), "by_stage": run.stage_costs}}
                if not args.no_cache_write:
                    s["cache"][int(item["id"])] = raw
                    rlm_eval.save_cache(s["path"], s["signature"], s["cache"])
            row = score_row(arch, item, raw, ctx)
            s["rows"].append(row)
            line.append(f"{arch}={row['score']['outcome'][:12]}")
        if not ctx.base_unchanged():
            raise SystemExit(f"Base graph changed during query {item['id']}; stopping.")
        print(f"[{index:02d}/{len(items)}] {item['id']} {item['case']} {' '.join(line)}")
        if quota_stop:
            print(f"QUOTA STOP: {quota_stop['detail'][:160]}")
            break
    common = set.intersection(*[{row["id"] for row in s["rows"]} for s in state.values()])
    complete = all(len(s["rows"]) == len(items) for s in state.values())
    return {"status": "plumbing_only" if not provider.EVALUATIVE else ("valid" if complete else "partial"),
            "complete": complete, "queries_requested": len(items), "queries_compared": len(common), "quota_stop": quota_stop,
            "generation": {**provider.config.as_record(), "model_verification": verification},
            "architectures": {arch: {**ab.ARCHITECTURES[arch], "loop_bounds": ctx.bounds[arch].as_record(),
                                     "run_signature": s["signature"], "cache_file": None if args.no_cache_write else s["path"].name,
                                     "per_query": [r for r in s["rows"] if r["id"] in common]}
                              for arch, s in state.items()}}


# ================================= Replays ==================================


def replay(ctx: Context, answers: Dict[str, Any]) -> Dict[str, Any]:
    items = {item["id"]: item for item in ctx.items}
    conditions: Dict[str, List[Dict[str, Any]]] = {"D-gate": [], "D-relation": [], "C-relation": []}
    for source, targets in (("D", ("D-gate", "D-relation")), ("C", ("C-relation",))):
        for row in answers["architectures"][source]["per_query"]:
            item = items[row["id"]]
            retriever = ctx.retriever(item, source)
            for name in targets:
                started = time.perf_counter()
                if name == "D-gate":
                    received = r1s.received_evidence(row["prompts"])["text"].split("\n")
                    failures = [{"node_id": n, "reason": "unavailable"} for n in item.get("mask") or []]
                    verdict = ea.sufficiency_gate(row["answer"], item["query"], received, ctx.universe, failures)
                    final = ea.ABSTAIN_ANSWER if verdict["decision"] == "abstain" else row["answer"]
                    extra = {"gate": verdict, "decision": "abstain" if verdict["decision"] == "abstain" else "keep"}
                else:
                    result = rv.verify_answer(row["answer"], item["query"], row["prompts"], ctx.verifier_for(retriever))
                    final = result["final_answer"]
                    extra = {"verification": result, "decision": result["decision"],
                             "claims_scored": r2ps.score_claims(result["claims"], ctx.clean_edges)}
                elapsed = (time.perf_counter() - started) * 1000
                scored = score_row(name, item, {**row, "answer": final}, ctx)
                conditions[name].append({**scored, **extra, "answer_control": row["answer"],
                                         "control_outcome": row["score"]["outcome"], "check_latency_ms": round(elapsed, 3),
                                         "false_rejection": row["score"]["outcome"] == "correct" and scored["score"]["outcome"] != "correct"})
    return {"verifier_version": rv.VERIFIER_VERSION, "gate_version": ea.GATE_VERSION, "conditions": conditions}


# ================================= Report ===================================


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    agg = r2.aggregate(rows)
    n = len(rows)
    agg["graph_caused_errors"] = r1.fraction(sum(1 for r in rows if r["score"]["graph_caused_error"]), n)["text"]
    if rows and "false_rejection" in rows[0]:
        agg["false_rejections"] = r1.fraction(sum(1 for r in rows if r["false_rejection"]), n)["text"]
        agg["decisions"] = {d: sum(1 for r in rows if r.get("decision") == d) for d in ("keep", "drop", "abstain")}
        agg["mean_check_latency_ms"] = round(statistics.fmean(r["check_latency_ms"] for r in rows), 3)
    return agg


def build_report(answers: Dict[str, Any], replays: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    columns = {arch: answers["architectures"][arch]["per_query"] for arch in ARCHS}
    if replays:
        columns.update(replays["conditions"])
    report = {"overall": {name: summarise(rows) for name, rows in columns.items()},
              "by_case": {case: {name: summarise([r for r in rows if r["case"] == case]) for name, rows in columns.items()}
                          for case in CASES}}
    if replays:
        report["relation_level"] = {name: r2ps.relation_summary([c for r in rows for c in r["claims_scored"]])
                                    for name, rows in replays["conditions"].items() if name.endswith("relation")}
        report["false_rejection_on_P"] = {
            name: r1.fraction(sum(1 for r in rows if r["case"] == "P" and r["false_rejection"]),
                              sum(1 for r in rows if r["case"] == "P" and r["control_outcome"] == "correct"))["text"]
            for name, rows in replays["conditions"].items()}
    return report


def print_report(report: Dict[str, Any]) -> None:
    names = list(report["overall"])
    keys = ["accuracy_answerable", "correct_abstention_unanswerable", "graph_caused_errors", "wrong_answers",
            "identifier_supported", "mean_input_tokens", "mean_generation_calls", "mean_generation_latency_ms"]
    print("\n" + "=" * (34 + 18 * len(names)))
    print(f"{'R2prime overall':<34}" + "".join(f"{n:>18}" for n in names))
    for key in keys:
        print(f"{key:<34}" + "".join(f"{str(report['overall'][n].get(key, '')):>18}" for n in names))
    for case in CASES:
        print(f"-- {case}")
        for outcome in r2s.OUTCOMES:
            print(f"  {outcome:<32}" + "".join(f"{report['by_case'][case][n]['outcomes'][outcome]:>18}" for n in names))
    for name, rel in (report.get("relation_level") or {}).items():
        print(f"-- relation level {name}: {json.dumps({k: v for k, v in rel['overall'].items() if k != 'counts'})}")
    if report.get("false_rejection_on_P"):
        print("-- false rejection on P:", report["false_rejection_on_P"])


# ================================ Freezing ==================================


def input_hashes(ctx: Context) -> Dict[str, Any]:
    files = {"preregistration": "PREREGISTRATION_R2_PRIME.md", "verifier": "rlm/relation_verifier.py",
             "scorer_r2p": "rlm/robustness_scoring_r2p.py", "runner": "experiments/robustness_r2_prime.py",
             # Prefixed: R2's pinned "runner" must not overwrite this runner's own hash.
             **{"r2_pinned_" + key.replace("_sha256", ""): path for key, path in PINNED_SOURCES.items()}}
    return {**ctx.provenance, "verifier_version": rv.VERIFIER_VERSION, "scorer_version": r2ps.R2P_SCORER_VERSION,
            "runner_version": RUNNER_VERSION, "gate_version": ea.GATE_VERSION, "loop_version": rlm_loop.LOOP_VERSION,
            "corruption_version": gc.CORRUPTION_VERSION, "prompt_template_sha256": rlm_loop.TEMPLATE_HASHES,
            **{f"{key}_sha256": rlm_eval.file_sha256(ROOT / path) for key, path in files.items()},
            "stores": ctx.store_before}


# =================================== Main ===================================


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--phase", choices=("validate", "sweep", "answers", "freeze", "replay", "report"), default="validate")
    parser.add_argument("--provider", default="mock")
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-cache-write", action="store_true")
    parser.add_argument("--answers", default=str(ANSWERS_PATH))
    parser.add_argument("--replays", default=str(REPLAY_PATH))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    ctx = Context()
    output: Dict[str, Any] = {"experiment": EXPERIMENT_ID, "phase": args.phase}
    out: Optional[Path] = Path(args.out) if args.out else None

    if args.phase in ("validate", "answers", "freeze", "sweep"):
        checks, sweeps = run_validation(ctx)
        r1.print_checks(checks)
        output["validation"], output["sweeps"] = checks, sweeps
        if args.phase == "sweep":
            out = out or SWEEP_PATH
        if not r1.validation_passed(checks):
            print("Offline validation FAILED. No model calls were made.")
            if args.phase != "sweep":
                return 1
        if args.phase == "freeze":
            FREEZE_PATH.write_text(json.dumps({"experiment": EXPERIMENT_ID,
                                               "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                               "git_revision": rlm_eval.git_revision(), "inputs": input_hashes(ctx),
                                               "validation": checks}, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Wrote {FREEZE_PATH.name}")
            return 0
    if args.phase == "answers":
        if args.provider != "mock":
            frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8")).get("inputs") if FREEZE_PATH.exists() else None
            current = input_hashes(ctx)
            if frozen != current:
                raise SystemExit("Refusing hosted run: R2' inputs not frozen or changed since freezing: "
                                 f"{sorted(k for k in current if not frozen or frozen.get(k) != current[k])}")
        output["answers"] = run_answers(args, ctx)
        out = out or (ANSWERS_PATH if args.provider == "groq" else ROOT / f"experiment_r2p_answers_meridian-transit-v1_{args.provider}.json")
    elif args.phase == "replay":
        output["replay"] = replay(ctx, json.loads(Path(args.answers).read_text(encoding="utf-8"))["answers"])
        out = out or REPLAY_PATH
    elif args.phase == "report":
        answers = json.loads(Path(args.answers).read_text(encoding="utf-8"))["answers"]
        replays = json.loads(Path(args.replays).read_text(encoding="utf-8"))["replay"] if Path(args.replays).exists() else None
        output["report"] = build_report(answers, replays)
        print_report(output["report"])
        out = out or COMPARISON_PATH

    if r2.store_hashes() != ctx.store_before or not ctx.base_unchanged():
        print("FAILED: a store file or the base graph changed during the phase.")
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
