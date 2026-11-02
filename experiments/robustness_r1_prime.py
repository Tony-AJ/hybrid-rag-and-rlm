"""R1': evidence-aware RLM (preregistered in PREREGISTRATION_R1_PRIME.md).

Variants of architecture D on the frozen retriever, compared against frozen
controls (R1 and Phase 2 rows for A, B2, C and D, never rerun):

  D-gate     loop 1.7 + failed-fetch trace + sufficiency gate. The gate runs
             after the loop and changes nothing before it, so it is evaluated
             by exact offline replay over the recorded D rows (no model calls).
  D-abstain  rlm.loop_evidence with abstain_option=True, gate=False
  D-ea       rlm.loop_evidence with abstain_option=True, gate=True (primary)

Datasets: R1 r1-1.0 (24, abstention) and v1.1 (60, over-rejection).

Phases:
  validate  R1's offline checks, real-corpus equivalence of loop_evidence
            (gate and abstain off) with loop 1.7 under the mock provider on all
            84 queries, and the replay rebuild check. No model calls.
  replay    offline D-gate over frozen R1 and Phase 2 D rows
  freeze    validate, then record input hashes
  answers   --variant ea|abstain --dataset r1|v11 (hosted runs need the freeze)
  report    comparison tables from frozen controls and whatever variant outputs exist
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
from experiments import robustness_r1 as r1  # noqa: E402
from rlm import evaluate as rlm_eval  # noqa: E402
from rlm import loop as rlm_loop  # noqa: E402
from rlm import loop_evidence as ea  # noqa: E402
from rlm import robustness_scoring as r1s  # noqa: E402
from rlm.providers import ProviderUnavailable, QuotaExceeded, build_provider  # noqa: E402

RUNNER_VERSION = "r1prime-run-1.0"
EXPERIMENT_ID = "robustness-r1-prime"
FREEZE_PATH = ROOT / "experiment_r1prime_freeze.json"
R1_FROZEN = ROOT / "experiment_robustness_r1_meridian-transit-v1_groq_frozen.json"
R1_FROZEN_SHA256 = "941f0d4a710a3af4e88b2020b578221edeee7b6a155ee90488bc848aaa417d14"
PREREG_PATH = ROOT / "PREREGISTRATION_R1_PRIME.md"
VARIANTS = {"ea": {"gate": True, "abstain_option": True},
            "abstain": {"gate": False, "abstain_option": True}}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check(name: str, passed: bool, detail: Any = None) -> Dict[str, Any]:
    return {"check": name, "status": "passed" if passed else "FAILED", "detail": detail}


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# ================================ Context ===================================


class Context:
    def __init__(self) -> None:
        self.corpus = bench.resolve_corpus("meridian-transit-v1")
        self.v11, self.v11_prov = bench.load_dataset(self.corpus)
        self.service = bench.build_service(self.corpus)
        self.frozen = rlm_loop.load_frozen_retrieval_config()
        self.base = ab.build_retrievers(self.service, self.frozen)
        self.r1 = r1.load_r1_dataset()
        self.universe = rlm_eval.station_universe(self.service)
        self.bounds = ab.answer_bounds(5, 3, 3)["D"]
        graph = self.service.graph_db.graph
        self.texts = {node_id: graph.nodes[node_id].get("text") or "" for node_id in graph.nodes}

    def items(self, dataset: str) -> List[Dict[str, Any]]:
        return self.r1["items"] if dataset == "r1" else self.v11

    def retriever(self, item: Dict[str, Any]) -> r1.MaskedRetriever:
        return r1.MaskedRetriever(self.base["hybrid"], item.get("mask") or [])


# ============================== Validation ==================================


def equivalence_check(ctx: Context) -> Dict[str, Any]:
    """loop_evidence with gate and abstain off must equal loop 1.7 (mock provider)."""
    mismatches = []
    for dataset in ("r1", "v11"):
        for item in ctx.items(dataset):
            outcomes = []
            for runner in ("1.7", "ea"):
                provider = r1.RecordingProvider(build_provider("mock"))
                retriever = ctx.retriever(item)
                if runner == "1.7":
                    run = rlm_loop.run_loop(retriever, provider, item["query"], ctx.bounds)
                else:
                    run = ea.run_loop_ea(retriever, provider, item["query"], ctx.bounds, gate=False,
                                         abstain_option=False, universe=ctx.universe)
                outcomes.append((provider.prompts, run.answer, run.evidence_ids, run.retrieval_calls,
                                 run.generation_calls, run.input_tokens, run.output_tokens))
            if outcomes[0] != outcomes[1]:
                mismatches.append({"dataset": dataset, "id": item["id"]})
    return check("E1 loop_evidence (gate off, abstain off) equals loop 1.7 on all 84 queries (mock): "
                 "prompts, answers, evidence, calls, tokens", not mismatches, mismatches)


def received_texts_from_ids(ctx: Context, evidence_ids: List[str]) -> List[str]:
    return [ctx.texts.get(node_id, "") for node_id in evidence_ids]


def rebuild_check(ctx: Context) -> Dict[str, Any]:
    """For R1 D rows, text rebuilt from evidence ids must give the same gated identifiers
    as the recorded prompts. Phase 2 D rows have no prompts and rely on this rebuild."""
    frozen = load_json(R1_FROZEN)
    problems = []
    for row in frozen["answers"]["architectures"]["D"]["per_query"]:
        recorded = ea.gated_identifiers(r1s.received_evidence(row["prompts"])["text"], ctx.universe)
        rebuilt = set()
        for text in received_texts_from_ids(ctx, row["evidence_coverage"]["evidence_chunks"]):
            rebuilt |= ea.gated_identifiers(text, ctx.universe)
        if recorded != rebuilt:
            problems.append({"id": row["id"], "only_recorded": sorted(recorded - rebuilt),
                             "only_rebuilt": sorted(rebuilt - recorded)})
    return check("E2 replay rebuild: evidence-id text matches recorded prompt text (R1 D rows)",
                 not problems, problems)


def run_validation(ctx: Context, phase2_regression: bool) -> List[Dict[str, Any]]:
    checks = r1.run_validation(ctx.corpus, ctx.service, ctx.frozen, ctx.base, ctx.r1, 5,
                               phase2_regression=phase2_regression)
    checks.append(check("E0 R1 frozen result unchanged", rlm_eval.file_sha256(R1_FROZEN) == R1_FROZEN_SHA256))
    checks.append(check("E0 loop 1.7 unchanged", rlm_loop.LOOP_VERSION == "1.7"
                        and rlm_loop.TEMPLATE_HASHES["step"] == ea.TEMPLATE_HASHES_EA["step"]))
    checks.append(equivalence_check(ctx))
    checks.append(rebuild_check(ctx))
    return checks


# ================================ Replay ====================================


def replay_gate(ctx: Context) -> Dict[str, Any]:
    """D-gate: apply the sufficiency gate to the recorded control D rows."""
    r1_rows = load_json(R1_FROZEN)["answers"]["architectures"]["D"]["per_query"]
    r1_items = {item["id"]: item for item in ctx.r1["items"]}
    out_r1 = []
    for row in r1_rows:
        item = r1_items[row["id"]]
        texts = received_texts_from_ids(ctx, row["evidence_coverage"]["evidence_chunks"])
        failures = masked_failures(item)
        verdict = ea.sufficiency_gate(row["answer"], item["query"], texts, ctx.universe, failures)
        answer = ea.ABSTAIN_ANSWER if verdict["decision"] == "abstain" else row["answer"]
        out_r1.append({**slim(row), "answer_control": row["answer"], "answer": answer, "gate": verdict,
                       "score": r1s.score_r1(answer, item, row["prompts"], ctx.universe)})

    phase2 = load_json(r1.PHASE2_FROZEN_RESULT)["answers"]["architectures"]["D"]["per_query"]
    v11_items = {item["id"]: item for item in ctx.v11}
    out_v11 = []
    for row in phase2:
        item = v11_items[row["id"]]
        texts = received_texts_from_ids(ctx, row["evidence_coverage"]["evidence_chunks"])
        verdict = ea.sufficiency_gate(row["answer"], item["query"], texts, ctx.universe, [])
        answer = ea.ABSTAIN_ANSWER if verdict["decision"] == "abstain" else row["answer"]
        out_v11.append(v11_row("D-gate", item, answer, row, ctx, control_correct=row["answer_evaluation"]["correct"],
                               extra={"gate": verdict, "answer_control": row["answer"]}))
    return {"r1": out_r1, "v11": out_v11}


def masked_failures(item: Dict[str, Any]) -> List[Dict[str, str]]:
    return [{"node_id": node_id, "reason": "unavailable"} for node_id in item.get("mask") or []]


def slim(row: Dict[str, Any]) -> Dict[str, Any]:
    return {key: row[key] for key in ("id", "case", "question", "evidence_coverage", "cost", "stop_reason")
            if key in row}


# ================================ Scoring ===================================


def v11_row(variant: str, item: Dict[str, Any], answer: str, source: Dict[str, Any], ctx: Context,
            control_correct: bool, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    evaluation = rlm_eval.score_answer(answer, item["gold_answer"], ctx.universe, item["query"])
    relevant = set(item["relevant"])
    evidence = source["evidence_coverage"]["evidence_chunks"]
    return {
        "variant": variant, "id": item["id"], "category": item.get("category"), "question": item["query"],
        "answer": answer, "answer_evaluation": evaluation,
        "control_correct": control_correct,
        "over_rejected": control_correct and not evaluation["correct"] and evaluation["abstained"],
        "regressed": control_correct and not evaluation["correct"],
        "evidence_coverage": {"evidence_chunks": evidence,
                              "coverage": round(len(relevant & set(evidence)) / len(relevant), 4)},
        "cost": source["cost"], "stop_reason": source.get("stop_reason"), **(extra or {}),
    }


def aggregate_v11(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)

    def frac(predicate) -> Dict[str, Any]:
        return r1.fraction(sum(1 for row in rows if predicate(row)), n)

    def avg(values) -> Optional[float]:
        values = list(values)
        return round(statistics.fmean(values), 4) if values else None

    return {
        "queries": n,
        "accuracy": frac(lambda row: row["answer_evaluation"]["correct"]),
        "abstained": frac(lambda row: row["answer_evaluation"]["abstained"]),
        "over_rejected": frac(lambda row: row["over_rejected"]),
        "regressed": frac(lambda row: row["regressed"]),
        "mean_evidence_coverage": avg(row["evidence_coverage"]["coverage"] for row in rows),
        "mean_input_tokens": avg(row["cost"]["input_tokens"] for row in rows),
        "mean_output_tokens": avg(row["cost"]["output_tokens"] for row in rows),
        "mean_generation_calls": avg(row["cost"]["generation_calls"] for row in rows),
        "mean_retrieval_calls": avg(row["cost"]["retrieval_calls"] for row in rows),
        "mean_generation_latency_ms": avg(row["cost"]["generation_latency_ms"] for row in rows),
    }


def by_category(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["category"], []).append(row)
    return {category: aggregate_v11(group) for category, group in groups.items()}


def trace_counts(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    gate = {"abstain": 0, "pass": 0}
    failures: Dict[str, int] = {}
    model_abstain = 0
    for row in rows:
        for step in row.get("steps") or []:
            if step.get("step") == "gate":
                gate[step["decision"]] += 1
            if step.get("step") == "grounding":
                for failure in step.get("fetch_failures") or []:
                    failures[failure["reason"]] = failures.get(failure["reason"], 0) + 1
        if row.get("stop_reason") == "model_abstained":
            model_abstain += 1
        if "gate" in row:
            gate[row["gate"]["decision"]] += 1
    return {"gate_decisions": gate, "fetch_failures_by_reason": failures, "model_abstain_decisions": model_abstain}


# ================================ Answers ===================================


def input_hashes(ctx: Context) -> Dict[str, Any]:
    return {
        "preregistration_sha256": rlm_eval.file_sha256(PREREG_PATH),
        "r1_inputs": r1.input_hashes(ctx.r1),
        "v11_dataset_sha256": ctx.v11_prov["dataset_sha256"],
        "loop_evidence_sha256": rlm_eval.file_sha256(ROOT / "rlm" / "loop_evidence.py"),
        "loop_version_ea": ea.LOOP_VERSION_EA,
        "gate_version": ea.GATE_VERSION,
        "template_hashes_ea": ea.TEMPLATE_HASHES_EA,
        "runner_sha256": rlm_eval.file_sha256(Path(__file__).resolve()),
        "runner_version": RUNNER_VERSION,
        "r1_frozen_sha256": rlm_eval.file_sha256(R1_FROZEN),
    }


def signature(ctx: Context, variant: str, dataset: str, provider: str, model: str, options: Dict[str, Any]) -> str:
    payload = json.dumps({
        "experiment": EXPERIMENT_ID, "runner_version": RUNNER_VERSION, "variant": variant,
        "variant_flags": VARIANTS[variant], "dataset": dataset,
        "dataset_sha256": ctx.r1["provenance"]["dataset_sha256"] if dataset == "r1" else ctx.v11_prov["dataset_sha256"],
        "mask_sha256": ctx.r1["provenance"]["mask_sha256"] if dataset == "r1" else None,
        "retriever": ctx.base["hybrid"].as_record(), "bounds": ctx.bounds.as_record(),
        "loop_version": ea.LOOP_VERSION_EA, "gate_version": ea.GATE_VERSION,
        "templates": ea.TEMPLATE_HASHES_EA, "provider": provider, "model": model, "generation_options": options,
    }, sort_keys=True)
    return sha(payload.encode("utf-8"))


def run_answers(args: argparse.Namespace, ctx: Context) -> Dict[str, Any]:
    try:
        inner = build_provider(args.provider, model=args.model, temperature=args.temperature,
                               max_tokens=args.max_tokens, reasoning_effort=args.reasoning_effort)
    except (ProviderUnavailable, ValueError) as exc:
        raise SystemExit(str(exc))
    provider = r1.RecordingProvider(inner)
    verification = provider.verify_model()
    options = {"reasoning_effort": args.reasoning_effort}
    sig = signature(ctx, args.variant, args.dataset, args.provider, provider.config.model, options)
    cache_path = ROOT / f"experiment_r1prime_cache_{args.dataset}_{args.variant}_{args.provider}_{sig[:12]}.json"
    cache = rlm_eval.load_cache(cache_path, sig) if args.resume else {}
    print(f"variant {args.variant} dataset {args.dataset}: {len(cache)} cached rows (signature {sig[:12]})")
    controls = {row["id"]: row for row in load_json(r1.PHASE2_FROZEN_RESULT)["answers"]["architectures"]["D"]["per_query"]}

    items = ctx.items(args.dataset)[: args.limit] if args.limit else ctx.items(args.dataset)
    rows, quota_stop = [], None
    for index, item in enumerate(items, 1):
        query_id = int(item["id"])
        raw = cache.get(query_id)
        if raw is None:
            provider.prompts = []
            try:
                run = ea.run_loop_ea(ctx.retriever(item), provider, item["query"], ctx.bounds,
                                     universe=ctx.universe, **VARIANTS[args.variant])
            except QuotaExceeded as exc:
                quota_stop = {"provider": exc.provider, "detail": exc.detail, "stopped_at_query_id": query_id}
                print(f"QUOTA STOP at {query_id}: {exc.detail[:160]}")
                break
            raw = {"id": query_id, "answer": run.answer, "steps": run.steps, "stop_reason": run.stop_reason,
                   "prompts": list(provider.prompts),
                   "evidence_coverage": {"evidence_chunks": run.evidence_ids},
                   "cost": {"retrieval_calls": run.retrieval_calls, "generation_calls": run.generation_calls,
                            "input_tokens": run.input_tokens, "output_tokens": run.output_tokens,
                            "retrieval_latency_ms": round(run.retrieval_latency_ms, 3),
                            "generation_latency_ms": round(run.generation_latency_ms, 3),
                            "by_stage": run.stage_costs}}
            if not args.no_cache_write:
                cache[query_id] = raw
                rlm_eval.save_cache(cache_path, sig, cache)
        rows.append(score_raw(args.variant, args.dataset, item, raw, ctx, controls))
        label = rows[-1]["score"]["outcome"] if args.dataset == "r1" else (
            "ok" if rows[-1]["answer_evaluation"]["correct"] else "x")
        print(f"[{index:02d}/{len(items)}] {query_id} {label} {raw['stop_reason']}")

    aggregate = aggregate_r1_rows(rows) if args.dataset == "r1" else {
        "aggregate": aggregate_v11(rows), "by_category": by_category(rows)}
    return {
        "variant": args.variant, "variant_flags": VARIANTS[args.variant], "dataset": args.dataset,
        "status": "plumbing_only" if not provider.EVALUATIVE else ("valid" if len(rows) == len(items) else "partial"),
        "queries_requested": len(items), "queries_completed": len(rows), "quota_stop": quota_stop,
        "run_signature": sig, "cache_file": None if args.no_cache_write else cache_path.name,
        "generation": {**provider.config.as_record(), "model_verification": verification},
        **aggregate, "trace_counts": trace_counts(rows), "per_query": rows,
    }


def score_raw(variant: str, dataset: str, item: Dict[str, Any], raw: Dict[str, Any], ctx: Context,
              controls: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    if dataset == "r1":
        available = set(item.get("available_relevant") or [])
        evidence = raw["evidence_coverage"]["evidence_chunks"]
        coverage = round(len(available & set(evidence)) / len(available), 4) if available else None
        return {**raw, "case": item["case"], "question": item["query"],
                "evidence_coverage": {"evidence_chunks": evidence, "coverage": coverage},
                "score": r1s.score_r1(raw["answer"], item, raw["prompts"], ctx.universe)}
    control = controls[item["id"]]["answer_evaluation"]["correct"]
    return {**v11_row(f"D-{variant}", item, raw["answer"], raw, ctx, control),
            "steps": raw["steps"], "prompts": raw["prompts"]}


def aggregate_r1_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    shaped = [{**row, "evidence_coverage": {"coverage": (row.get("evidence_coverage") or {}).get("coverage")}}
              for row in rows]
    return {"aggregate": r1.aggregate_r1(shaped),
            "by_case": {case: r1.aggregate_r1([row for row in shaped if row["case"] == case]) for case in r1.CASES}}


# ================================ Report ====================================


def r1_summary(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    agg = aggregate_r1_rows(rows)
    by = agg["by_case"]
    tokens = [row["cost"]["input_tokens"] for row in rows if row["case"] in ("R1a", "R1b")]
    return {
        "correct_abstention": agg["aggregate"]["correct_abstention_rate"]["text"],
        "R1a": by["R1a"]["outcomes"]["correct_abstention"]["text"],
        "R1b": by["R1b"]["outcomes"]["correct_abstention"]["text"],
        "R1c": by["R1c"]["outcomes"]["correct_abstention"]["text"],
        "link_only_R1c": by["R1c"]["outcomes"]["link_only_answer"]["text"],
        "unsupported_claims": agg["aggregate"]["unsupported_claim_rate"]["text"],
        "input_tokens_all": str(agg["aggregate"]["mean_input_tokens"]),
        "input_tokens_R1a_R1b": str(round(statistics.fmean(tokens), 1)) if tokens else "n/a",
        "generation_calls": str(agg["aggregate"]["mean_generation_calls"]),
    }


def v11_summary(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    agg = aggregate_v11(rows)
    cats = by_category(rows)
    pick = lambda name: next((value["accuracy"]["text"] for key, value in cats.items() if name in key), "n/a")  # noqa: E731
    return {
        "accuracy": agg["accuracy"]["text"], "over_rejected": agg["over_rejected"]["text"],
        "regressed": agg["regressed"]["text"], "two_hop": pick("2-hop"), "adjacency": pick("adjacency"),
        "authorisation": pick("authorisation"), "coverage": str(agg["mean_evidence_coverage"]),
        "input_tokens": str(agg["mean_input_tokens"]), "generation_calls": str(agg["mean_generation_calls"]),
    }


def build_report(ctx: Context) -> Dict[str, Any]:
    r1_frozen = load_json(R1_FROZEN)["answers"]["architectures"]
    phase2 = load_json(r1.PHASE2_FROZEN_RESULT)["answers"]["architectures"]
    v11_items = {item["id"]: item for item in ctx.v11}
    report: Dict[str, Any] = {"r1": {}, "v11": {}}
    for arch in ("A", "B2", "C", "D"):
        report["r1"][f"{arch} (control)"] = r1_summary(r1_frozen[arch]["per_query"])
        rows = [v11_row(arch, v11_items[row["id"]], row["answer"], row, ctx, row["answer_evaluation"]["correct"])
                for row in phase2[arch]["per_query"]]
        report["v11"][f"{arch} (control)"] = v11_summary(rows)
    replay_path = ROOT / "experiment_r1prime_replay_gate.json"
    if replay_path.exists():
        replay = load_json(replay_path)
        report["r1"]["D-gate (replay)"] = r1_summary(replay["r1"])
        report["v11"]["D-gate (replay)"] = v11_summary(replay["v11"])
    for variant in ("abstain", "ea"):
        for dataset in ("r1", "v11"):
            path = ROOT / f"experiment_r1prime_{dataset}_{variant}_groq.json"
            if path.exists():
                rows = load_json(path)["answers"]["per_query"]
                report[dataset][f"D-{variant}"] = (r1_summary if dataset == "r1" else v11_summary)(rows)
    return report


def print_report(report: Dict[str, Any]) -> None:
    for dataset, table in report.items():
        if not table:
            continue
        columns = list(table)
        keys = list(next(iter(table.values())))
        print("\n" + "=" * (30 + 20 * len(columns)))
        print(f"R1' {dataset.upper()}")
        print(f"{'metric':<30}" + "".join(f"{column:>20}" for column in columns))
        for key in keys:
            print(f"{key:<30}" + "".join(f"{table[column].get(key, ''):>20}" for column in columns))


# ================================== Main ====================================


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--phase", choices=("validate", "replay", "freeze", "answers", "report"), default="validate")
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="ea")
    parser.add_argument("--dataset", choices=("r1", "v11"), default="r1")
    parser.add_argument("--provider", default="mock")
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-cache-write", action="store_true")
    parser.add_argument("--skip-phase2-regression", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    ctx = Context()
    output: Dict[str, Any] = {"experiment": EXPERIMENT_ID, "phase": args.phase}

    if args.phase in ("validate", "freeze", "answers"):
        regression = args.phase == "freeze" or not args.skip_phase2_regression
        checks = run_validation(ctx, phase2_regression=regression)
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

    if args.phase == "replay":
        output["replay"] = replay_gate(ctx)
        output.update(output.pop("replay"))
        out = Path(args.out) if args.out else ROOT / "experiment_r1prime_replay_gate.json"
    elif args.phase == "answers":
        if args.provider != "mock":
            frozen = load_json(FREEZE_PATH).get("inputs") if FREEZE_PATH.exists() else None
            current = input_hashes(ctx)
            if frozen != current:
                diff = {key: "changed" for key in current if not frozen or frozen.get(key) != current[key]}
                raise SystemExit(f"Refusing hosted run: R1' inputs not frozen or changed since freezing: {diff}")
        output["answers"] = run_answers(args, ctx)
        out = Path(args.out) if args.out else ROOT / (
            f"experiment_r1prime_{args.dataset}_{args.variant}_{args.provider}.json")
    elif args.phase == "report":
        output["report"] = build_report(ctx)
        print_report(output["report"])
        out = Path(args.out) if args.out else ROOT / "experiment_r1prime_comparison.json"
    else:
        out = Path(args.out) if args.out else None

    output["reproducibility"] = {
        "runner_version": RUNNER_VERSION, "git_revision": rlm_eval.git_revision(),
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "inputs": input_hashes(ctx),
    }
    if out:
        out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
