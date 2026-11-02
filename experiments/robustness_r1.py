"""Robustness experiment R1: missing evidence.

Does the system decline when the evidence it can read does not answer the
question, and can it tell a visible graph relationship from text that supports
an answer?

  R1a  no supporting document (missing attribute or non-existent entity)
  R1b  partial evidence: one half of the question answerable, one not
  R1c  graph edge present, target text unavailable (target nodes masked)

Isolation from Phase 2:
  - Corpus, stores, dataset v1.1, frozen retrieval config and loop 1.7 prompts
    are used unchanged. Nothing in rlm/, app/ or experiments/architecture_benchmark.py
    is modified.
  - R1c masks nodes at retrieval time (MaskedRetriever): a masked node is never
    returned as evidence and fetch() returns nothing, but graph edges to it stay,
    so its id remains visible in `Links:` lines. No re-ingestion.
  - Own dataset (benchmark_corpora/meridian/robustness/missing_evidence.json),
    own scorer (rlm.robustness_scoring), own cache and result files.

Every generation call's prompt is recorded (RecordingProvider), and
supportedness is judged against that received text, so D's follow-up retrieval
and grounding inputs count exactly as the model saw them.

Phases:
    validate  offline checks only, no model calls (default)
    answers   validate, then run A, B2, C, D (B is excluded: it abstains by
              construction). A hosted provider additionally requires a matching
              freeze file (phase freeze).
    freeze    validate, then record dataset, scorer, mask and code hashes

Run:
    python -m experiments.robustness_r1 --phase validate
    python -m experiments.robustness_r1 --phase answers --provider mock --no-cache-write --out <path>
    python -m experiments.robustness_r1 --phase freeze
    python -m experiments.robustness_r1 --phase answers --provider groq --model openai/gpt-oss-120b --reasoning-effort low --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
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
from rlm import evaluate as rlm_eval  # noqa: E402
from rlm import loop as rlm_loop  # noqa: E402
from rlm import robustness_scoring as r1s  # noqa: E402
from rlm.providers import GenerationResult, ProviderUnavailable, QuotaExceeded, build_provider  # noqa: E402

R1_RUNNER_VERSION = "r1-run-1.0"
EXPERIMENT_ID = "robustness-r1"
DATASET_PATH = ROOT / "benchmark_corpora" / "meridian" / "robustness" / "missing_evidence.json"
FREEZE_PATH = ROOT / "experiment_robustness_r1_freeze.json"
SCORER_PATH = ROOT / "rlm" / "robustness_scoring.py"
RUNNER_PATH = Path(__file__).resolve()
R1_ARCHITECTURES = ("A", "B2", "C", "D")
CASES = ("R1a", "R1b", "R1c")

# Phase 2 artefacts that must be unchanged before and after R1.
PHASE2_FROZEN_RESULT = ROOT / "experiment_architecture_benchmark_meridian-transit-v1_groq_phase2_frozen.json"
PHASE2_FROZEN_RESULT_SHA256 = "c70e8a416b1259387c8cf47bf47a415aefa4812ca2e4bd93ad1a0687f7a31407"
FROZEN_CONFIG_SHA256 = "0e944282aa60426b9b7c358a2797c33573b29bb0574dc9b8a959b1d57334690d"

PROMPT_ABSTENTION_OPTION = {
    "A": "ANSWER_TEMPLATE: 'If evidence is insufficient, reply exactly: INSUFFICIENT EVIDENCE.'",
    "B2": "ANSWER_TEMPLATE: 'If evidence is insufficient, reply exactly: INSUFFICIENT EVIDENCE.'",
    "C": "ANSWER_TEMPLATE: 'If evidence is insufficient, reply exactly: INSUFFICIENT EVIDENCE.'",
    "D": ("STEP_TEMPLATE offers only ANSWER: or SEARCH:, and GROUND_TEMPLATE has no abstain "
          "option. D can abstain only by writing it as an answer, or through the ANSWER_TEMPLATE "
          "fallback when its step budget runs out. Loop 1.7 prompts are deliberately unchanged."),
}


# ================================ Adapters ==================================


class MaskedRetriever:
    """Any retriever adapter with some nodes made unreadable.

    Masked nodes are dropped from rankings (the base ranking is over-fetched by
    the mask size, filtered, then cut to top_k) and fetch() returns None for
    them. relations() is untouched, so edges into masked nodes stay visible.
    With an empty mask every call is passed through unchanged.
    """

    def __init__(self, base: Any, mask: List[str]):
        self.base = base
        self.mask = frozenset(mask)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if not self.mask:
            return self.base.retrieve(query, top_k)
        chunks = self.base.retrieve(query, top_k + len(self.mask))
        return [chunk for chunk in chunks if chunk["node_id"] not in self.mask][:top_k]

    def ranking(self, query: str, top_k: int) -> List[str]:
        return [chunk["node_id"] for chunk in self.retrieve(query, top_k)]

    def fetch(self, node_id: str) -> Optional[Dict[str, Any]]:
        return None if node_id in self.mask else self.base.fetch(node_id)

    def relations(self, node_id: str) -> Dict[str, List[str]]:
        return self.base.relations(node_id)

    def as_record(self) -> Dict[str, Any]:
        return {**self.base.as_record(), "mask": sorted(self.mask)}


class RecordingProvider:
    """Passes calls through and keeps every prompt the model was sent."""

    def __init__(self, provider: Any):
        self.provider = provider
        self.config = provider.config
        self.EVALUATIVE = bool(getattr(provider, "EVALUATIVE", True))
        self.prompts: List[str] = []

    def list_models(self) -> List[str]:
        return self.provider.list_models()

    def verify_model(self) -> Dict[str, Any]:
        return self.provider.verify_model()

    def generate(self, prompt: str) -> GenerationResult:
        self.prompts.append(prompt)
        return self.provider.generate(prompt)


# ================================ Dataset ===================================


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_r1_dataset(path: Path = DATASET_PATH) -> Dict[str, Any]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    masks = {str(item["id"]): sorted(item.get("mask") or []) for item in payload["queries"]}
    return {
        "payload": payload,
        "items": payload["queries"],
        "provenance": {
            "dataset_path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "dataset_version": payload["dataset_version"],
            "dataset_sha256": sha256_bytes(raw),
            "mask_sha256": sha256_bytes(json.dumps(masks, sort_keys=True).encode("utf-8")),
            "masks": masks,
        },
    }


# =============================== Validation =================================


def check(name: str, passed: bool, detail: Any = None) -> Dict[str, Any]:
    return {"check": name, "status": "passed" if passed else "FAILED", "detail": detail}


_TITLE_PATTERNS = {
    "procedure": re.compile(r"^Maintenance procedure MP-\d{3}: ([^.]+)\."),
    "policy": re.compile(r"^Authorisation policy PL-[A-L], ([^,]+),"),
}


def node_signature_terms(node_id: str, text: str) -> List[str]:
    """Strings that would let a reader reconstruct a masked node: its code and title."""
    kind, _, code = node_id.partition("_")
    terms = [code]
    pattern = _TITLE_PATTERNS.get(kind)
    match = pattern.match(text) if pattern else None
    if match:
        terms.append(match.group(1))
    return terms


def run_validation(
    corpus: Dict[str, Any], service: Any, frozen: Dict[str, Any], base: Dict[str, Any],
    r1: Dict[str, Any], top_k: int, phase2_regression: bool = True,
) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    items = r1["items"]
    graph = service.graph_db.graph
    texts = {node_id: (graph.nodes[node_id].get("text") or "") for node_id in graph.nodes}
    universe = rlm_eval.station_universe(service)

    # V1 frozen inputs ---------------------------------------------------------
    v11, v11_prov = bench.load_dataset(corpus)  # raises if dataset v1.1 changed
    validation = bench.validate_corpus_and_ground_truth(service, v11, corpus)  # raises on fingerprint drift
    checks.append(check("V1 dataset v1.1 sha256 unchanged", v11_prov["dataset_sha256"] == corpus["dataset_sha256"],
                        v11_prov["dataset_sha256"]))
    checks.append(check("V1 corpus fingerprint matches", validation["corpus_fingerprint"] == corpus["corpus_fingerprint"],
                        validation["corpus_fingerprint"]))
    frozen_sha = rlm_eval.file_sha256(rlm_loop.FROZEN_CONFIG_PATH)
    checks.append(check("V1 frozen retrieval config unchanged", frozen_sha == FROZEN_CONFIG_SHA256, frozen_sha))
    phase2_sha = rlm_eval.file_sha256(PHASE2_FROZEN_RESULT)
    checks.append(check("V1 Phase 2 frozen result unchanged", phase2_sha == PHASE2_FROZEN_RESULT_SHA256, phase2_sha))
    checks.append(check("V1 R1 dataset bound to v1.1",
                        r1["payload"].get("base_dataset_sha256") == corpus["dataset_sha256"]
                        and r1["payload"].get("corpus_id") == corpus["corpus_id"]))

    # V2 schema ------------------------------------------------------------------
    ids = [item["id"] for item in items]
    problems = []
    if len(set(ids)) != len(ids) or min(ids) < 1001:
        problems.append("ids must be unique and >= 1001")
    for item in items:
        need = ["id", "case", "query", "relevant", "available_relevant", "mask", "gold_answer"]
        need += {"R1a": ["absent_terms", "nonexistent_entities"], "R1b": ["absent_terms", "answerable_part"],
                 "R1c": ["link_source", "graph_path", "source_query_id"]}.get(item.get("case"), ["case?"])
        missing = [key for key in need if key not in item]
        if missing:
            problems.append({"id": item.get("id"), "missing": missing})
        if item.get("gold_answer", {}).get("kind") != "abstain" or item.get("relevant"):
            problems.append({"id": item.get("id"), "error": "gold kind must be abstain and relevant empty"})
        if item.get("case") == "R1c" and not item.get("mask"):
            problems.append({"id": item["id"], "error": "R1c needs a mask"})
        if item.get("case") != "R1c" and item.get("mask"):
            problems.append({"id": item["id"], "error": "only R1c may mask"})
        for node_id in item.get("available_relevant", []) + item.get("mask", []):
            if node_id not in graph:
                problems.append({"id": item["id"], "unknown_node": node_id})
    counts = {case: sum(item.get("case") == case for item in items) for case in CASES}
    checks.append(check("V2 dataset schema", not problems, {"case_counts": counts, "problems": problems}))

    # V3 masked nodes never returned as text, never fetched ----------------------
    leaks = []
    for item in [item for item in items if item["case"] == "R1c"]:
        mask = set(item["mask"])
        probes = [item["query"]]
        for node_id in mask:
            probes += [texts[node_id], *node_signature_terms(node_id, texts[node_id])]
        for arch in ("A", "B2", "C"):
            retriever = MaskedRetriever(base[ab.ARCHITECTURES[arch]["retriever"]], item["mask"])
            for probe in probes:
                chunks = retriever.retrieve(probe, top_k)
                returned = mask & {chunk["node_id"] for chunk in chunks}
                evidence = r1s.received_evidence([rlm_loop.format_evidence(chunks)])
                text_codes = {code for node_id in mask for code in [node_id.partition("_")[2]]
                              if code in evidence["text"]}
                if returned or text_codes:
                    leaks.append({"id": item["id"], "arch": arch, "probe": probe[:60],
                                  "returned": sorted(returned), "code_in_text": sorted(text_codes)})
            for node_id in mask:
                if retriever.fetch(node_id) is not None:
                    leaks.append({"id": item["id"], "arch": arch, "fetch_returned": node_id})
    checks.append(check("V3 masked nodes absent from retrieved text and fetch (A, B2, C; question, "
                        "node text, code and title probes)", not leaks, leaks))

    # V4 masked ids stay visible through the intended edge ---------------------
    missing_links = []
    visible_first_step: Dict[str, Dict[str, bool]] = {}
    for item in [item for item in items if item["case"] == "R1c"]:
        hybrid = MaskedRetriever(base["hybrid"], item["mask"])
        targets = {target for values in hybrid.relations(item["link_source"]).values() for target in values}
        absent = sorted(set(item["mask"]) - targets)
        if absent:
            missing_links.append({"id": item["id"], "link_source": item["link_source"], "not_linked": absent})
        visible_first_step[str(item["id"])] = {}
        for arch in ("B2", "C"):
            retriever = MaskedRetriever(base[ab.ARCHITECTURES[arch]["retriever"]], item["mask"])
            evidence = r1s.received_evidence([rlm_loop.format_evidence(retriever.retrieve(item["query"], top_k))])
            visible_first_step[str(item["id"])][arch] = all(
                node_id.partition("_")[2] in evidence["links"] for node_id in item["mask"])
    checks.append(check("V4 masked ids remain linked from link_source in the graph", not missing_links,
                        {"problems": missing_links,
                         "masked_ids_visible_in_first_retrieval_links (informational)": visible_first_step}))

    # V5 masked answers not reconstructable from unmasked chunks ----------------
    reconstructable = []
    for item in [item for item in items if item["case"] == "R1c"]:
        mask = set(item["mask"])
        for node_id in mask:
            for term in node_signature_terms(node_id, texts[node_id]):
                found_in = [other for other, text in texts.items()
                            if other not in mask and term.lower() in text.lower()]
                if found_in:
                    reconstructable.append({"id": item["id"], "masked": node_id, "term": term,
                                            "found_in": found_in[:5]})
    checks.append(check("V5 masked code and title appear in no unmasked chunk text", not reconstructable,
                        reconstructable))

    # V6 R1a / R1b ground truth -------------------------------------------------
    term_hits, entity_hits, answerable_fail = [], [], []
    for item in items:
        for term in item.get("absent_terms", []):
            found_in = [node_id for node_id, text in texts.items() if term.lower() in text.lower()]
            if found_in:
                term_hits.append({"id": item["id"], "term": term, "found_in": found_in[:5]})
        for entity in item.get("nonexistent_entities", []):
            if entity in universe or any(entity.lower() in text.lower() for text in texts.values()):
                entity_hits.append({"id": item["id"], "entity": entity})
        if item.get("answerable_part"):
            source = " ".join(texts[node_id] for node_id in item["available_relevant"])
            if not rlm_eval.score_predicates(source, item["answerable_part"])["all_satisfied"]:
                answerable_fail.append(item["id"])
    checks.append(check("V6 R1a/R1b absent terms occur in no chunk text", not term_hits, term_hits))
    checks.append(check("V6 R1a non-existent entities absent from corpus", not entity_hits, entity_hits))
    checks.append(check("V6 R1b answerable part satisfied by available_relevant text", not answerable_fail,
                        answerable_fail))

    # V7 Phase 2 regression through the wrapper ---------------------------------
    if phase2_regression:
        wrapped = {name: MaskedRetriever(retriever, []) for name, retriever in base.items()}
        identity = []
        for item in v11:
            for name in ("vector", "graph_entity", "hybrid"):
                if wrapped[name].ranking(item["query"], top_k) != base[name].ranking(item["query"], top_k):
                    identity.append({"id": item["id"], "retriever": name})
        checks.append(check("V7 empty-mask wrapper returns identical rankings on all v1.1 queries",
                            not identity, identity))
        retrieval = ab.run_retrieval_phase(v11, wrapped, top_k)
        repro = ab.reproduction_checks(retrieval, corpus["corpus_id"], frozen, top_k)
        for result in repro:
            checks.append(check(f"V7 Phase 2 regression: {result['check']}", result["status"] == "passed",
                                result.get("mismatches")))
        phase2 = json.loads(PHASE2_FROZEN_RESULT.read_text(encoding="utf-8"))
        names = ab.metric_names(top_k)
        frozen_b2 = phase2["retrieval"]["architectures"]["B2"]["aggregate"]
        measured_b2 = retrieval["architectures"]["B2"]["aggregate"]
        b2_diff = {name: (measured_b2[name], frozen_b2[name]) for name in names
                   if round(measured_b2[name], 4) != round(frozen_b2[name], 4)}
        checks.append(check("V7 Phase 2 regression: B2 == frozen Phase 2 B2 aggregate", not b2_diff, b2_diff))

    # Informational: does over-fetching change the order of the unmasked prefix?
    prefix = []
    for item in [item for item in items if item["case"] == "R1c"]:
        for name in ("vector", "graph_entity", "hybrid"):
            small = [chunk["node_id"] for chunk in base[name].retrieve(item["query"], top_k)]
            large = [chunk["node_id"] for chunk in base[name].retrieve(item["query"], top_k + len(item["mask"]))]
            if large[:top_k] != small:
                prefix.append({"id": item["id"], "retriever": name})
    checks.append({"check": "INFO over-fetch keeps the unmasked top-k prefix stable",
                   "status": "info", "detail": {"differences": prefix}})
    return checks


def validation_passed(checks: List[Dict[str, Any]]) -> bool:
    return all(entry["status"] != "FAILED" for entry in checks)


def print_checks(checks: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 92)
    print("R1 OFFLINE VALIDATION")
    for entry in checks:
        print(f"{entry['status']:<7} {entry['check']}")
        if entry["status"] == "FAILED":
            print(f"        {json.dumps(entry['detail'], ensure_ascii=False)[:600]}")
    print("=" * 92)


# ================================ Freezing ==================================


def input_hashes(r1: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dataset_sha256": r1["provenance"]["dataset_sha256"],
        "mask_sha256": r1["provenance"]["mask_sha256"],
        "scorer_sha256": rlm_eval.file_sha256(SCORER_PATH),
        "scorer_version": r1s.ROBUSTNESS_SCORER_VERSION,
        "lenient_patterns_sha256": sha256_bytes(
            json.dumps(r1s.LENIENT_ABSTENTION_PATTERNS).encode("utf-8")),
        "runner_sha256": rlm_eval.file_sha256(RUNNER_PATH),
        "runner_version": R1_RUNNER_VERSION,
        "architecture_benchmark_sha256": rlm_eval.file_sha256(ROOT / "experiments" / "architecture_benchmark.py"),
        "loop_version": rlm_loop.LOOP_VERSION,
        "prompt_template_sha256": rlm_loop.TEMPLATE_HASHES,
    }


def freeze_mismatches(r1: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not FREEZE_PATH.exists():
        return {"freeze_file": "missing"}
    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8")).get("inputs", {})
    current = input_hashes(r1)
    diff = {key: {"frozen": frozen.get(key), "current": value}
            for key, value in current.items() if frozen.get(key) != value}
    return diff or None


# ================================= Answers ==================================


def r1_signature(arch: str, corpus_id: str, r1: Dict[str, Any], retriever_record: Dict[str, Any],
                 provider_name: str, model: str, bounds: Dict[str, Any], options: Dict[str, Any]) -> str:
    payload = json.dumps({
        "experiment": EXPERIMENT_ID,
        "runner_version": R1_RUNNER_VERSION,
        "architecture": arch,
        "corpus_id": corpus_id,
        "dataset_sha256": r1["provenance"]["dataset_sha256"],
        "mask_sha256": r1["provenance"]["mask_sha256"],
        "retriever": retriever_record,
        "provider": provider_name,
        "model": model,
        "generation_options": options,
        "bounds": bounds,
        "prompts": rlm_loop.TEMPLATE_HASHES,
        "loop_version": rlm_loop.LOOP_VERSION,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def r1_row(arch: str, item: Dict[str, Any], run: rlm_loop.LoopRun, prompts: List[str],
           universe: List[str]) -> Dict[str, Any]:
    available = set(item.get("available_relevant") or [])
    evidence = set(run.evidence_ids)
    return {
        "architecture": arch,
        "id": item["id"],
        "case": item["case"],
        "question": item["query"],
        "mask": item.get("mask") or [],
        "answer": run.answer,
        "score": r1s.score_r1(run.answer, item, prompts, universe),
        "evidence_coverage": {
            "available_relevant": sorted(available),
            "evidence_chunks": run.evidence_ids,
            "coverage": round(len(available & evidence) / len(available), 4) if available else None,
            "masked_in_evidence": sorted(set(item.get("mask") or []) & evidence),
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
        "prompts": prompts,
    }


def fraction(count: int, n: int) -> Dict[str, Any]:
    rate = round(count / n, 4) if n else None
    return {"count": count, "n": n, "rate": rate,
            "text": f"{count}/{n} ({100 * count / n:.1f}%)" if n else f"{count}/0 (n/a)"}


def aggregate_r1(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    by_case = {case: [row for row in rows if row["case"] == case] for case in CASES}

    def count(predicate, subset) -> int:
        return sum(1 for row in subset if predicate(row))

    def avg(values: List[float]) -> Optional[float]:
        return round(statistics.fmean(values), 4) if values else None

    coverage = [row["evidence_coverage"]["coverage"] for row in rows
                if row["evidence_coverage"]["coverage"] is not None]
    return {
        "queries": n,
        "outcomes": {outcome: fraction(count(lambda row, o=outcome: row["score"]["outcome"] == o, rows), n)
                     for outcome in r1s.OUTCOMES},
        "correct_abstention_rate": fraction(count(lambda row: row["score"]["outcome"] == "correct_abstention", rows), n),
        "correct_abstention_rate_strict": fraction(count(lambda row: row["score"]["correct_abstention_strict"], rows), n),
        "safe_response_rate": fraction(count(lambda row: row["score"]["outcome"] in
                                             ("correct_abstention", "partial_correct"), rows), n),
        "unsupported_answer_rate": fraction(count(lambda row: row["score"]["outcome"] == "unsupported_answer", rows), n),
        "unsupported_claim_rate": fraction(count(lambda row: row["score"]["contains_unsupported_claim"], rows), n),
        "identifier_supported_rate": fraction(count(lambda row: row["score"]["identifier_supported"], rows), n),
        "abstention_detected_strict": fraction(count(lambda row: row["score"]["abstention_strict"], rows), n),
        "abstention_detected_lenient": fraction(count(lambda row: row["score"]["abstention_lenient"], rows), n),
        "link_only_rate_R1c": fraction(count(lambda row: row["score"]["outcome"] == "link_only_answer",
                                             by_case["R1c"]), len(by_case["R1c"])),
        "masked_ids_exposed_in_links_R1c": fraction(
            count(lambda row: (row["score"]["masked_truth"] or {}).get("exposed"), by_case["R1c"]),
            len(by_case["R1c"])),
        "link_only_rate_R1c_when_exposed": fraction(
            count(lambda row: row["score"]["outcome"] == "link_only_answer"
                  and (row["score"]["masked_truth"] or {}).get("exposed"), by_case["R1c"]),
            count(lambda row: (row["score"]["masked_truth"] or {}).get("exposed"), by_case["R1c"])),
        "named_masked_truth_R1c": fraction(count(lambda row: (row["score"]["masked_truth"] or {}).get("named_any"),
                                                 by_case["R1c"]), len(by_case["R1c"])),
        "partial_correct_rate_R1b": fraction(count(lambda row: row["score"]["outcome"] == "partial_correct",
                                                   by_case["R1b"]), len(by_case["R1b"])),
        "answerable_part_correct_R1b": fraction(count(lambda row: row["score"]["answerable_part_correct"],
                                                      by_case["R1b"]), len(by_case["R1b"])),
        "mean_evidence_coverage": avg(coverage),
        "evidence_coverage_queries": len(coverage),
        "mean_retrieval_calls": avg([row["cost"]["retrieval_calls"] for row in rows]),
        "mean_generation_calls": avg([row["cost"]["generation_calls"] for row in rows]),
        "mean_input_tokens": avg([row["cost"]["input_tokens"] for row in rows]),
        "mean_output_tokens": avg([row["cost"]["output_tokens"] for row in rows]),
        "total_input_tokens": sum(row["cost"]["input_tokens"] for row in rows),
        "total_output_tokens": sum(row["cost"]["output_tokens"] for row in rows),
        "mean_generation_latency_ms": avg([row["cost"]["generation_latency_ms"] for row in rows]),
        "mean_retrieval_latency_ms": avg([row["cost"]["retrieval_latency_ms"] for row in rows]),
    }


def run_answers(args: argparse.Namespace, corpus: Dict[str, Any], r1: Dict[str, Any],
                base: Dict[str, Any], universe: List[str]) -> Dict[str, Any]:
    try:
        inner = build_provider(args.provider, model=args.model, temperature=args.temperature,
                               max_tokens=args.max_tokens, reasoning_effort=args.reasoning_effort)
    except (ProviderUnavailable, ValueError) as exc:
        raise SystemExit(str(exc))
    provider = RecordingProvider(inner)
    verification = provider.verify_model()
    bounds = ab.answer_bounds(args.top_k, args.max_steps, args.max_retrievals)
    items = r1["items"][: args.limit] if args.limit else r1["items"]
    options = {"reasoning_effort": args.reasoning_effort}

    state: Dict[str, Dict[str, Any]] = {}
    for arch in R1_ARCHITECTURES:
        record = {**base[ab.ARCHITECTURES[arch]["retriever"]].as_record(), "masking": "per-query MaskedRetriever"}
        signature = r1_signature(arch, corpus["corpus_id"], r1, record, args.provider,
                                 provider.config.model, bounds[arch].as_record(), options)
        cache_path = ROOT / (f"experiment_robustness_r1_cache_{corpus['corpus_id']}_{args.provider}"
                             f"_{arch}_{signature[:12]}.json")
        cache = rlm_eval.load_cache(cache_path, signature) if args.resume else {}
        state[arch] = {"signature": signature, "cache_path": cache_path, "cache": cache, "rows": [],
                       "reused": 0, "retriever_record": record}
        if args.resume:
            print(f"resume {arch}: {len(cache)} cached rows (signature {signature[:12]})")

    quota_stop = None
    for index, item in enumerate(items, 1):
        line = []
        for arch in R1_ARCHITECTURES:
            arch_state = state[arch]
            cached = arch_state["cache"].get(int(item["id"]))
            if cached is not None:
                cached["score"] = r1s.score_r1(cached["answer"], item, cached["prompts"], universe)
                arch_state["rows"].append(cached)
                arch_state["reused"] += 1
                line.append(f"{arch}={cached['score']['outcome'][:12]}(c)")
                continue
            retriever = MaskedRetriever(base[ab.ARCHITECTURES[arch]["retriever"]], item.get("mask") or [])
            provider.prompts = []
            try:
                run = rlm_loop.run_loop(retriever, provider, str(item["query"]), bounds[arch])
            except QuotaExceeded as exc:
                quota_stop = {"provider": exc.provider, "detail": exc.detail, "stopped_at_query_id": item["id"],
                              "stopped_at_architecture": arch}
                break
            row = r1_row(arch, item, run, list(provider.prompts), universe)
            arch_state["rows"].append(row)
            if not args.no_cache_write:
                arch_state["cache"][int(item["id"])] = row
                rlm_eval.save_cache(arch_state["cache_path"], arch_state["signature"], arch_state["cache"])
            line.append(f"{arch}={row['score']['outcome'][:12]}")
        print(f"[{index:02d}/{len(items)}] {item['id']} {item['case']} {' '.join(line)}")
        if quota_stop:
            print(f"QUOTA STOP: {quota_stop['detail'][:160]}")
            break

    common = set.intersection(*[{row["id"] for row in value["rows"]} for value in state.values()])
    complete = all(len(value["rows"]) == len(items) for value in state.values())
    evaluative = provider.EVALUATIVE
    architectures = {}
    for arch, value in state.items():
        rows = [row for row in value["rows"] if row["id"] in common]
        architectures[arch] = {
            **ab.ARCHITECTURES[arch],
            "retriever_config": value["retriever_record"],
            "loop_bounds": bounds[arch].as_record(),
            "prompt_abstention_option": PROMPT_ABSTENTION_OPTION[arch],
            "run_signature": value["signature"],
            "cache_file": None if args.no_cache_write else value["cache_path"].name,
            "queries_reused_from_cache": value["reused"],
            "aggregate": aggregate_r1(rows),
            "by_case": {case: aggregate_r1([row for row in rows if row["case"] == case]) for case in CASES},
            "per_query": rows,
        }
    return {
        "status": ("plumbing_only" if not evaluative else ("valid" if complete else "partial")),
        "evaluative": evaluative,
        "complete": complete,
        "queries_requested": len(items),
        "queries_compared": len(common),
        "quota_stop": quota_stop,
        "caveat": None if evaluative else (
            "Mock provider: it performs no reasoning. These rows test the harness and scorer only."),
        "small_sample_caveat": ("About 8 queries per case. Raw counts are given with every rate; "
                                "differences of one or two queries are not conclusive."),
        "configuration": {
            "generation": {**provider.config.as_record(), "model_verification": verification},
            "prompt_template_sha256": rlm_loop.TEMPLATE_HASHES,
            "loop_version": rlm_loop.LOOP_VERSION,
            "scorer_version": r1s.ROBUSTNESS_SCORER_VERSION,
            "lenient_abstention_patterns": r1s.LENIENT_ABSTENTION_PATTERNS,
            "loop_bounds": {arch: bounds[arch].as_record() for arch in R1_ARCHITECTURES},
        },
        "architectures": architectures,
    }


def print_answers(answers: Dict[str, Any]) -> None:
    archs = answers["architectures"]
    print("\n" + "=" * 100)
    print(f"R1 ANSWERS ({answers['status']}) - {answers['queries_compared']} queries")
    keys = ["correct_abstention_rate", "correct_abstention_rate_strict", "safe_response_rate",
            "unsupported_answer_rate", "unsupported_claim_rate", "link_only_rate_R1c",
            "masked_ids_exposed_in_links_R1c", "link_only_rate_R1c_when_exposed",
            "partial_correct_rate_R1b", "answerable_part_correct_R1b", "abstention_detected_strict",
            "abstention_detected_lenient"]
    print(f"{'metric':<34}" + "".join(f"{arch:>16}" for arch in archs))
    for key in keys:
        print(f"{key:<34}" + "".join(f"{archs[arch]['aggregate'][key]['text']:>16}" for arch in archs))
    for key in ("mean_input_tokens", "mean_output_tokens", "mean_generation_calls", "mean_evidence_coverage"):
        print(f"{key:<34}" + "".join(f"{str(archs[arch]['aggregate'][key]):>16}" for arch in archs))
    print("-" * 100)
    for case in CASES:
        print(f"{case} outcomes")
        for outcome in r1s.OUTCOMES:
            print(f"  {outcome:<32}" + "".join(
                f"{archs[arch]['by_case'][case]['outcomes'][outcome]['text']:>16}" for arch in archs))
    print("=" * 100)


# ================================== Main ====================================


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--phase", choices=("validate", "answers", "freeze"), default="validate")
    parser.add_argument("--provider", default="mock")
    parser.add_argument("--model", default=None)
    parser.add_argument("--corpus", default="meridian-transit-v1")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=3, help="architecture D only")
    parser.add_argument("--max-retrievals", type=int, default=3, help="architecture D only")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-cache-write", action="store_true")
    parser.add_argument("--skip-phase2-regression", action="store_true",
                        help="skip the 60-query Phase 2 retrieval regression (still required for freeze)")
    parser.add_argument("--out", default=None, help="output path (default: project root, named by provider)")
    args = parser.parse_args()

    if args.corpus != "meridian-transit-v1":
        raise SystemExit("R1 is defined on meridian-transit-v1 only.")
    corpus = bench.resolve_corpus(args.corpus)
    service = bench.build_service(corpus)
    frozen = rlm_loop.load_frozen_retrieval_config()
    base = ab.build_retrievers(service, frozen)
    r1 = load_r1_dataset()
    universe = rlm_eval.station_universe(service)

    regression = not args.skip_phase2_regression or args.phase == "freeze"
    checks = run_validation(corpus, service, frozen, base, r1, args.top_k, phase2_regression=regression)
    print_checks(checks)
    if not validation_passed(checks):
        print("Offline validation FAILED. No model calls were made.")
        return 1

    if args.phase == "freeze":
        payload = {
            "experiment": EXPERIMENT_ID,
            "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_revision": rlm_eval.git_revision(),
            "inputs": input_hashes(r1),
            "validation": checks,
        }
        FREEZE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {FREEZE_PATH.name}")
        return 0

    answers = None
    if args.phase == "answers":
        if args.provider != "mock":
            diff = freeze_mismatches(r1)
            if diff:
                raise SystemExit(f"Refusing hosted run: R1 inputs are not frozen or changed since freezing: {diff}")
        answers = run_answers(args, corpus, r1, base, universe)
        print_answers(answers)

    output = {
        "experiment": EXPERIMENT_ID,
        "phase": args.phase,
        "validation": checks,
        "validation_passed": True,
        "answers": answers,
        "reproducibility": {
            "runner_version": R1_RUNNER_VERSION,
            "git_revision": rlm_eval.git_revision(),
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "corpus_id": corpus["corpus_id"],
            "corpus_fingerprint": corpus["corpus_fingerprint"],
            "base_dataset_sha256": corpus["dataset_sha256"],
            "frozen_config_sha256": rlm_eval.file_sha256(rlm_loop.FROZEN_CONFIG_PATH),
            **{key: value for key, value in r1["provenance"].items() if key != "masks"},
            "inputs": input_hashes(r1),
            "platform": {"python": platform.python_version(),
                         "system": f"{platform.system()} {platform.release()}"},
        },
    }
    out = Path(args.out) if args.out else ROOT / (
        f"experiment_robustness_r1_{args.phase}_{corpus['corpus_id']}_{args.provider}.json")
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
