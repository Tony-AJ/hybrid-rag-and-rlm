"""Step 5: end-to-end RAG + RLM evaluation.

Reports, per query and in aggregate:
  answer correctness   from `gold_answer`: exact set match plus precision and
                       recall over extracted entities, or the fraction of
                       satisfied predicates. Deterministic, no LLM judge.
  evidence coverage    share of the chunk-level `relevant` labels that reached
                       the loop's evidence set, and whether the answer-bearing
                       chunks were present at all (an answer can only be right
                       by luck if its evidence never arrived).
  retrieval calls      count per query, against the configured bound
  latency              retrieval and generation measured separately
  token usage          input and output, summed per query

Answer-level and chunk-level metrics are kept in separate blocks. They measure
different things and a reader must not be able to confuse one for the other.

Privacy: this runs on a corpus registered in benchmark.py, and it refuses to run
`production-uploads-v1` through a hosted provider, because that store holds real
personal documents. Use the synthetic corpus for research.

Run:
    python -m rlm.evaluate --provider mock
    python -m rlm.evaluate --provider anthropic --model <verified id>
    python -m rlm.evaluate --provider mock --limit 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import benchmark as bench  # noqa: E402
from rlm import loop as rlm_loop  # noqa: E402
from rlm.providers import ProviderUnavailable, QuotaExceeded, build_provider  # noqa: E402

EVALUATION_CODE_VERSION = "rlm-eval-1.1"
HOSTED_PROVIDERS = {"anthropic", "google"}
PRIVACY_RESTRICTED_CORPORA = {"production-uploads-v1"}

ENTITY_PATTERNS = {
    "procedure_code": r"\bMP-\d{3}\b",
    "policy_code": r"\bPL-[A-L]\b",
    "equipment_code": r"\b[A-Z]{2}-\d{1,3}\b",
}


# ============================ Answer-level scoring ==========================


def extract_entities(answer: str, entity_type: str, universe: Optional[List[str]] = None) -> Set[str]:
    """Pull candidate entities of one type out of free text.

    Codes come from a strict pattern. Station names cannot be patterned, so they
    are matched against the corpus's own station list, which is the only
    authoritative universe of names.
    """
    if entity_type in ENTITY_PATTERNS:
        return set(re.findall(ENTITY_PATTERNS[entity_type], answer))
    if entity_type == "station_name":
        found = set()
        lowered = answer.lower()
        for name in universe or []:
            if name.lower() in lowered:
                found.add(name)
        return found
    raise ValueError(f"no extractor for entity type {entity_type!r}")


def score_entity_set(answer: str, gold: Dict[str, Any], universe: Optional[List[str]]) -> Dict[str, Any]:
    expected = set(gold["entities"])
    found = extract_entities(answer, gold["entity_type"], universe)
    hits = expected & found
    return {
        "kind": "entity_set",
        "entity_type": gold["entity_type"],
        "expected": sorted(expected),
        "extracted": sorted(found),
        "exact_set_match": found == expected,
        "precision": round(len(hits) / len(found), 4) if found else 0.0,
        "recall": round(len(hits) / len(expected), 4) if expected else 0.0,
    }


def score_predicates(answer: str, gold: Dict[str, Any]) -> Dict[str, Any]:
    """Each predicate is satisfied if its normalized value appears in the answer.

    Numbers are matched on word boundaries so that 2 does not match 2009, and
    strings are matched case-insensitively. This is a structured check, not
    string equality against a reference answer.
    """
    results: Dict[str, bool] = {}
    lowered = answer.lower()
    for key, value in gold["predicates"].items():
        if isinstance(value, bool):
            results[key] = str(value).lower() in lowered
        elif isinstance(value, (int, float)):
            results[key] = re.search(rf"(?<!\d){re.escape(str(value))}(?!\d)", answer) is not None
        else:
            results[key] = str(value).lower() in lowered
    satisfied = sum(results.values())
    return {
        "kind": "predicates",
        "predicates": results,
        "satisfied": satisfied,
        "total": len(results),
        "fraction_satisfied": round(satisfied / len(results), 4) if results else 0.0,
        "all_satisfied": satisfied == len(results),
    }


def score_answer(answer: str, gold: Dict[str, Any], universe: Optional[List[str]]) -> Dict[str, Any]:
    if answer.strip().upper().startswith("INSUFFICIENT EVIDENCE"):
        base = {"abstained": True}
    else:
        base = {"abstained": False}
    if gold["kind"] == "entity_set":
        base.update(score_entity_set(answer, gold, universe))
    elif gold["kind"] == "predicates":
        base.update(score_predicates(answer, gold))
    else:
        raise ValueError(f"unknown gold_answer kind {gold['kind']!r}")
    base["correct"] = bool(
        base.get("exact_set_match") if gold["kind"] == "entity_set" else base.get("all_satisfied")
    )
    return base


def station_universe(service: Any) -> List[str]:
    """Station names, read from the corpus graph rather than hard-coded."""
    names: List[str] = []
    for node_id, data in service.graph_db.graph.nodes(data=True):
        if (data.get("metadata") or {}).get("type") == "station":
            text = data.get("text") or ""
            match = re.match(r"^(.+?) is a (?:terminus|through station)", text)
            if match:
                names.append(match.group(1))
    return names


# ================================= Runner ===================================


def run_signature(
    corpus_id: str,
    dataset_sha: str,
    variant: str,
    provider_name: str,
    model: str,
    bounds_record: Dict[str, Any],
) -> str:
    """Identity of a run configuration.

    A cached per-query result may only be reused when every input that could
    change the answer is identical: corpus, dataset, retriever, provider, model,
    prompt templates, loop bounds and the evaluator version.
    """
    payload = json.dumps(
        {
            "corpus_id": corpus_id,
            "dataset_sha256": dataset_sha,
            "variant": variant,
            "provider": provider_name,
            "model": model,
            "bounds": bounds_record,
            "prompts": rlm_loop.TEMPLATE_HASHES,
            "loop_version": rlm_loop.LOOP_VERSION,
            "evaluation_code_version": EVALUATION_CODE_VERSION,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_cache(path: Path, signature: str) -> Dict[str, Any]:
    """Previously completed per-query rows for this exact configuration.

    A free-tier daily quota (20 generate requests for gemini-2.5-flash, and the
    loop spends up to 2 per query) cannot cover a 60-query set in one day, so a
    run has to be able to continue tomorrow without redoing what it already paid
    for. Rows recorded under a different signature are ignored, never reused.
    """
    if not path.exists():
        return {}
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if stored.get("run_signature") != signature:
        return {}
    return {int(key): value for key, value in (stored.get("rows") or {}).items()}


def save_cache(path: Path, signature: str, rows: Dict[int, Any]) -> None:
    path.write_text(
        json.dumps(
            {"run_signature": signature, "rows": {str(key): value for key, value in rows.items()}},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def git_revision() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10
        ).stdout.strip() or None
    except Exception:
        return None


def file_sha256(path: Path) -> Optional[str]:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="mock")
    parser.add_argument("--model", default=None)
    parser.add_argument("--corpus", default="meridian-transit-v1")
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N queries")
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--max-retrievals", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--resume", action="store_true",
                        help="reuse per-query results already recorded for this exact configuration")
    parser.add_argument("--no-cache-write", action="store_true",
                        help="do not record per-query results for later resumption")
    args = parser.parse_args()

    if args.corpus in PRIVACY_RESTRICTED_CORPORA and args.provider in HOSTED_PROVIDERS:
        raise SystemExit(
            f"Refusing to send corpus {args.corpus!r} to hosted provider {args.provider!r}: it "
            "contains real personal documents. Research experiments run on the synthetic corpus."
        )

    corpus = bench.resolve_corpus(args.corpus)
    dataset, provenance = bench.load_dataset(corpus)
    service = bench.build_service(corpus)
    validation = bench.validate_corpus_and_ground_truth(service, dataset, corpus)

    missing_gold = [item.get("id") for item in dataset if not item.get("gold_answer")]
    if missing_gold:
        raise SystemExit(
            f"Dataset has no gold_answer for queries {missing_gold[:10]}. Answer-level "
            "evaluation needs answer-level ground truth; regenerate the dataset."
        )

    frozen = rlm_loop.load_frozen_retrieval_config()
    retriever = rlm_loop.FrozenRetriever(service, frozen)

    try:
        provider = build_provider(args.provider, model=args.model, temperature=args.temperature,
                                  max_tokens=args.max_tokens)
    except ProviderUnavailable as exc:
        raise SystemExit(str(exc))
    verification = provider.verify_model()
    evaluative = args.provider != "mock"

    bounds = rlm_loop.LoopBounds(
        max_steps=args.max_steps, max_retrievals=args.max_retrievals, top_k=args.top_k
    )
    queries = dataset[: args.limit] if args.limit else dataset
    universe = station_universe(service)

    signature = run_signature(
        corpus["corpus_id"],
        provenance["dataset_sha256"],
        retriever.variant_name,
        args.provider,
        provider.config.model,
        bounds.as_record(),
    )
    cache_path = ROOT / f"experiment_rlm_cache_{corpus['corpus_id']}_{args.provider}.json"
    cache: Dict[int, Any] = load_cache(cache_path, signature) if args.resume else {}
    if args.resume:
        print(f"resume: {len(cache)} cached rows match this configuration "
              f"(signature {signature[:12]})")

    rows: List[Dict[str, Any]] = []
    quota_stop: Optional[Dict[str, Any]] = None
    reused = 0
    started_all = time.perf_counter()
    for index, item in enumerate(queries, 1):
        question = str(item["query"])
        query_id = int(item.get("id", index))

        cached = cache.get(query_id)
        if cached is not None:
            rows.append(cached)
            reused += 1
            print(f"[{index:02d}/{len(queries)}] cached  correct={cached['answer_evaluation']['correct']} "
                  f"| {question[:58]}")
            continue

        try:
            run = rlm_loop.run_loop(retriever, provider, question, bounds)
        except QuotaExceeded as exc:
            # A quota stop makes the run incomplete. Record how far it got and
            # keep the rows already paid for, rather than averaging over a
            # truncated set as if it were the whole dataset.
            quota_stop = {
                "provider": exc.provider,
                "detail": exc.detail,
                "stopped_at_query_id": query_id,
                "completed": len(rows),
                "remaining": len(queries) - len(rows),
            }
            print(f"\n[{index:02d}/{len(queries)}] QUOTA STOP after {len(rows)} of {len(queries)} queries")
            break
        answer_score = score_answer(run.answer, item["gold_answer"], universe)

        relevant = {str(value) for value in item["relevant"]}
        evidence = set(run.evidence_ids)
        rows.append({
            "id": item.get("id", index),
            "category": item.get("category"),
            "hops": item.get("hops"),
            "question": question,
            "answer": run.answer,
            "answer_evaluation": answer_score,
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
            },
            "stop_reason": run.stop_reason,
            "steps": run.steps,
        })
        cache[query_id] = rows[-1]
        if not args.no_cache_write:
            save_cache(cache_path, signature, cache)
        print(f"[{index:02d}/{len(queries)}] correct={answer_score['correct']} "
              f"coverage={rows[-1]['evidence_coverage']['coverage']:.2f} "
              f"calls={run.retrieval_calls} tokens={run.input_tokens}+{run.output_tokens} "
              f"| {question[:58]}")
    wall_ms = (time.perf_counter() - started_all) * 1000

    if not rows:
        raise SystemExit(
            "No query completed, so there is nothing to report. "
            + (f"Reason: {quota_stop['detail']}" if quota_stop else "")
        )

    def mean(path) -> float:
        return round(statistics.fmean([path(row) for row in rows]), 4) if rows else 0.0

    by_category: Dict[str, Any] = {}
    for row in rows:
        bucket = by_category.setdefault(row["category"] or "uncategorised",
                                        {"queries": 0, "correct": 0, "coverage": 0.0})
        bucket["queries"] += 1
        bucket["correct"] += int(row["answer_evaluation"]["correct"])
        bucket["coverage"] += row["evidence_coverage"]["coverage"]
    for bucket in by_category.values():
        bucket["accuracy"] = round(bucket["correct"] / bucket["queries"], 4)
        bucket["coverage"] = round(bucket["coverage"] / bucket["queries"], 4)

    aggregate = {
        "answer_level": {
            "accuracy": mean(lambda row: float(row["answer_evaluation"]["correct"])),
            "abstention_rate": mean(lambda row: float(row["answer_evaluation"]["abstained"])),
            "entity_recall": round(statistics.fmean(
                [row["answer_evaluation"]["recall"] for row in rows
                 if row["answer_evaluation"]["kind"] == "entity_set"]
            ), 4) if any(row["answer_evaluation"]["kind"] == "entity_set" for row in rows) else None,
            "predicate_fraction_satisfied": round(statistics.fmean(
                [row["answer_evaluation"]["fraction_satisfied"] for row in rows
                 if row["answer_evaluation"]["kind"] == "predicates"]
            ), 4) if any(row["answer_evaluation"]["kind"] == "predicates" for row in rows) else None,
        },
        "chunk_level": {
            "evidence_coverage": mean(lambda row: row["evidence_coverage"]["coverage"]),
            "all_relevant_present_rate": mean(
                lambda row: float(row["evidence_coverage"]["all_relevant_present"])
            ),
        },
        "cost": {
            "mean_retrieval_calls": mean(lambda row: row["cost"]["retrieval_calls"]),
            "mean_generation_calls": mean(lambda row: row["cost"]["generation_calls"]),
            "mean_input_tokens": mean(lambda row: row["cost"]["input_tokens"]),
            "mean_output_tokens": mean(lambda row: row["cost"]["output_tokens"]),
            "total_input_tokens": sum(row["cost"]["input_tokens"] for row in rows),
            "total_output_tokens": sum(row["cost"]["output_tokens"] for row in rows),
            "mean_retrieval_latency_ms": mean(lambda row: row["cost"]["retrieval_latency_ms"]),
            "mean_generation_latency_ms": mean(lambda row: row["cost"]["generation_latency_ms"]),
            "wall_clock_ms": round(wall_ms, 3),
        },
    }

    complete = len(rows) == len(queries)
    if not evaluative:
        status = "plumbing_only"
    elif complete:
        status = "valid"
    else:
        status = "partial"

    output = {
        "status": status,
        "experiment": "rag-plus-rlm-end-to-end",
        "evaluative": evaluative,
        "complete": complete,
        "queries_evaluated": len(rows),
        "queries_in_dataset": len(dataset),
        "queries_requested": len(queries),
        "queries_reused_from_cache": reused,
        "quota_stop": quota_stop,
        "caveat": None if evaluative else (
            "Run with the mock provider: it performs no reasoning, so answer-level numbers "
            "measure the harness, not model quality. Rerun with a verified hosted model for "
            "a quality claim."
        ),
        "partial_caveat": None if complete else (
            f"Incomplete run: {len(rows)} of {len(queries)} queries evaluated"
            + (f", stopped by {quota_stop['provider']} quota" if quota_stop else "")
            + ". Aggregates cover only the evaluated queries and are not comparable with a "
            "full run. Re-run with --resume to continue where this stopped."
        ),
        "reproducibility": {
            "evaluation_code_version": EVALUATION_CODE_VERSION,
            "git_revision": git_revision(),
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "corpus_id": corpus["corpus_id"],
            "corpus_fingerprint": validation["corpus_fingerprint"],
            "dataset": provenance["dataset_path"],
            "dataset_version": provenance["dataset_version"],
            "dataset_sha256": provenance["dataset_sha256"],
            "retrieval": {
                "frozen_config_file": rlm_loop.FROZEN_CONFIG_PATH.name,
                "frozen_config_sha256": file_sha256(rlm_loop.FROZEN_CONFIG_PATH),
                "variant": retriever.variant_name,
                "weights": retriever.weights,
                "seed_k": retriever.seed_k,
                "max_depth": retriever.max_depth,
                "top_k": bounds.top_k,
            },
            "generation": {
                **provider.config.as_record(),
                "model_verification": verification,
                "reported_models": sorted({row["answer_evaluation"].get("model", "") for row in []}),
            },
            "prompt_template_sha256": rlm_loop.TEMPLATE_HASHES,
            "loop_version": rlm_loop.LOOP_VERSION,
            "loop_bounds": bounds.as_record(),
            "platform": {
                "python": platform.python_version(),
                "system": f"{platform.system()} {platform.release()}",
            },
            "note": (
                "A pinned model id and prompt hash make a run traceable. They cannot "
                "guarantee identical output: a hosted provider may change its "
                "implementation behind the same id."
            ),
        },
        "aggregate": aggregate,
        "by_category": by_category,
        "per_query": rows,
    }

    out_path = ROOT / f"experiment_rlm_eval_{corpus['corpus_id']}_{args.provider}.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 80)
    if not evaluative:
        print("PLUMBING ONLY - mock provider does no reasoning; answer numbers are not quality")
    if not complete:
        print(f"PARTIAL RUN - {len(rows)} of {len(queries)} queries"
              + (f", stopped by {quota_stop['provider']} quota" if quota_stop else "")
              + "; aggregates cover evaluated queries only")
        if quota_stop:
            print(f"  {quota_stop['detail'][:160]}")
        print("  continue later with: --resume")
    print(f"answer accuracy            {aggregate['answer_level']['accuracy']:.4f}")
    print(f"abstention rate            {aggregate['answer_level']['abstention_rate']:.4f}")
    print(f"evidence coverage          {aggregate['chunk_level']['evidence_coverage']:.4f}")
    print(f"all relevant present       {aggregate['chunk_level']['all_relevant_present_rate']:.4f}")
    print(f"mean retrieval calls       {aggregate['cost']['mean_retrieval_calls']:.2f}")
    print(f"mean tokens in/out         {aggregate['cost']['mean_input_tokens']:.1f}/"
          f"{aggregate['cost']['mean_output_tokens']:.1f}")
    print(f"mean latency retrieval/gen {aggregate['cost']['mean_retrieval_latency_ms']:.1f}/"
          f"{aggregate['cost']['mean_generation_latency_ms']:.1f} ms")
    print("-" * 80)
    print(f"{'category':<38}{'n':>3}{'accuracy':>10}{'coverage':>10}")
    for category, bucket in by_category.items():
        print(f"{category:<38}{bucket['queries']:>3}{bucket['accuracy']:>10.3f}{bucket['coverage']:>10.3f}")
    print("=" * 80)
    print(f"Wrote {out_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
