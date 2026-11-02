"""Evidence-aware variant of the bounded RLM loop (R1', loop 1.8-ea).

Preregistered in PREREGISTRATION_R1_PRIME.md. Loop 1.7 (rlm/loop.py) is not
modified: this module imports its retriever interface, bounds, templates and
evidence formatting, and differs only where the preregistration says it does.

1. Failed-fetch detection (always on). Every grounding attempt writes a step
   record, even when nothing could be fetched. Each cited link target that was
   not grounded is listed with a reason:
       unavailable   fetch() returned nothing
       empty_text    the fetched chunk has no usable text
       truncated     cited beyond bounds.max_grounding_chunks
   and the unfetched targets are listed as link_only_claims.

2. Sufficiency gate (gate=True). Deterministic, no model call, after grounding.
   Endpoint-text policy: every code (procedure, policy, equipment) and station
   name the final answer names, and the question does not, must appear in the
   text of a chunk the model received (retrieved or fetched). `Links:` lines
   are not text. Numbers are not gated. A failing answer is replaced by
   INSUFFICIENT EVIDENCE and the decision is recorded.

3. Explicit abstention (abstain_option=True). STEP_TEMPLATE_EA adds an
   ABSTAIN: decision. GROUND_TEMPLATE and ANSWER_TEMPLATE are unchanged.

With gate=False and abstain_option=False the prompts, answers, evidence and
costs equal loop 1.7's; only trace records are added. tests/test_loop_evidence.py
checks this.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Set

from rlm import evaluate as rlm_eval
from rlm.loop import (
    ANSWER_TEMPLATE,
    DECIDE_TEMPLATE,
    GROUND_TEMPLATE,
    STEP_TEMPLATE,
    LoopBounds,
    LoopRun,
    cited_link_targets,
    format_evidence,
)
from rlm.providers import GenerationProvider, GenerationResult, prompt_template_sha256

LOOP_VERSION_EA = "1.8-ea"
GATE_VERSION = "gate-1.0"
ABSTAIN_ANSWER = "INSUFFICIENT EVIDENCE"

STEP_TEMPLATE_EA = """Answer using only the evidence, or ask for more.
Question: {question}
Evidence:
{evidence}
If the evidence answers the question, reply ANSWER: followed by the exact identifiers and one short sentence. Do not add unsupported facts.
If the evidence cannot answer the question and another search would not help, reply ABSTAIN: followed by one short reason.
Otherwise, reply SEARCH: followed by one short query."""

TEMPLATE_HASHES_EA = {
    "answer": prompt_template_sha256(ANSWER_TEMPLATE),
    "decide": prompt_template_sha256(DECIDE_TEMPLATE),
    "step": prompt_template_sha256(STEP_TEMPLATE),
    "step_ea": prompt_template_sha256(STEP_TEMPLATE_EA),
    "ground": prompt_template_sha256(GROUND_TEMPLATE),
}

GATED_CODE_TYPES = ("procedure_code", "policy_code", "equipment_code")


# ================================== Gate ====================================


def gated_identifiers(text: str, universe: Optional[List[str]]) -> Set[str]:
    """Codes and station names in a text. Numbers are deliberately not gated."""
    folded = rlm_eval.fold_typography(text).replace("_", " ")
    found: Set[str] = set()
    for entity_type in GATED_CODE_TYPES:
        found |= {f"code:{code}" for code in rlm_eval.extract_entities(folded, entity_type, None)}
    found |= {f"station:{name}" for name in rlm_eval.extract_entities(folded, "station_name", universe or [])}
    return found


def node_identifiers(node_id: str, universe: Optional[List[str]]) -> Set[str]:
    return gated_identifiers(node_id, universe)


def sufficiency_gate(
    answer: str, question: str, received_texts: List[str], universe: Optional[List[str]],
    fetch_failures: List[Dict[str, str]],
) -> Dict[str, Any]:
    named = gated_identifiers(answer, universe) - gated_identifiers(question, universe)
    supported: Set[str] = set()
    for text in received_texts:
        supported |= gated_identifiers(text, universe)
    unsupported = sorted(named - supported)
    reasons: Dict[str, str] = {}
    for identifier in unsupported:
        reason = "not_in_received_text"
        for failure in fetch_failures:
            if identifier in node_identifiers(failure["node_id"], universe):
                reason = f"link_target_{failure['reason']}"
                break
        reasons[identifier] = reason
    return {
        "step": "gate",
        "gate_version": GATE_VERSION,
        "decision": "abstain" if unsupported else "pass",
        "named": sorted(named),
        "unsupported": unsupported,
        "reasons": reasons,
    }


def grounding_attempt(cited: List[str], bounds: LoopBounds, retriever: Any) -> Dict[str, Any]:
    """Fetch the cited link targets and classify every one that was not grounded."""
    attempted = cited[: bounds.max_grounding_chunks]
    truncated = cited[bounds.max_grounding_chunks:]
    fetched: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = [{"node_id": node_id, "reason": "truncated"} for node_id in truncated]
    if bounds.grounding != "off":
        for node_id in attempted:
            chunk = retriever.fetch(node_id)
            if chunk is None:
                failures.append({"node_id": node_id, "reason": "unavailable"})
            elif not (chunk.get("text") or "").strip():
                failures.append({"node_id": node_id, "reason": "empty_text"})
            else:
                fetched.append(chunk)
    return {"attempted": attempted, "truncated": truncated, "fetched": fetched, "failures": failures}


# ================================== Loop ====================================


def run_loop_ea(
    retriever: Any,
    provider: GenerationProvider,
    question: str,
    bounds: LoopBounds,
    gate: bool = True,
    abstain_option: bool = True,
    universe: Optional[List[str]] = None,
) -> LoopRun:
    run = LoopRun(question=question, answer="", evidence_ids=[])
    evidence: List[Dict[str, Any]] = []
    seen: set = set()
    current_query = question
    abstained_by_model = False

    def record_generation(result: GenerationResult, stage: str) -> None:
        run.generation_calls += 1
        run.input_tokens += result.input_tokens or 0
        run.output_tokens += result.output_tokens or 0
        run.generation_latency_ms += result.latency_ms
        cost = run.stage_costs.setdefault(
            stage, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "latency_ms": 0.0}
        )
        cost["calls"] += 1
        cost["input_tokens"] += result.input_tokens or 0
        cost["output_tokens"] += result.output_tokens or 0
        cost["latency_ms"] = round(cost["latency_ms"] + result.latency_ms, 3)

    step_template = STEP_TEMPLATE_EA if abstain_option else STEP_TEMPLATE

    for step in range(1, bounds.max_steps + 1):
        if run.retrieval_calls >= bounds.max_retrievals:
            run.stop_reason = "retrieval_budget_exhausted"
            break
        started = time.perf_counter()
        try:
            chunks = retriever.retrieve(current_query, bounds.top_k)
        except Exception as exc:
            raise RuntimeError(f"Retrieval failed on step {step} for query: {current_query!r}") from exc
        run.retrieval_latency_ms += (time.perf_counter() - started) * 1000
        run.retrieval_calls += 1

        added = []
        for chunk in chunks:
            if chunk["node_id"] not in seen and len(evidence) < bounds.max_evidence_chunks:
                seen.add(chunk["node_id"])
                evidence.append(chunk)
                added.append(chunk["node_id"])
        step_record = {"step": step, "query": current_query,
                       "retrieved": [chunk["node_id"] for chunk in chunks], "new_evidence": added}

        if step == bounds.max_steps or run.retrieval_calls >= bounds.max_retrievals:
            step_record["decision"] = "bounds_reached"
            run.steps.append(step_record)
            run.stop_reason = "step_budget_exhausted" if step == bounds.max_steps else "retrieval_budget_exhausted"
            break

        template = step_template if bounds.merge_decide_answer else DECIDE_TEMPLATE
        decide_result = provider.generate(
            template.format(question=question, evidence=format_evidence(evidence, not bounds.full_links))
        )
        record_generation(decide_result, "step")
        decision = decide_result.text.strip()
        step_record["decision"] = decision[:200]
        run.steps.append(step_record)

        upper = decision.upper()
        if abstain_option and bounds.merge_decide_answer and upper.startswith("ABSTAIN"):
            run.answer = ABSTAIN_ANSWER
            run.stop_reason = "model_abstained"
            step_record["abstain_reason"] = re.sub(r"^ABSTAIN:?\s*", "", decision, flags=re.IGNORECASE)[:200]
            abstained_by_model = True
            break
        wants_search = "SEARCH:" in upper and not upper.startswith(("STOP", "ANSWER:"))
        if not wants_search:
            run.stop_reason = "model_stopped"
            if bounds.merge_decide_answer:
                run.answer = re.sub(r"^ANSWER:\s*", "", decision, flags=re.IGNORECASE).strip()
            break
        current_query = decision[upper.index("SEARCH:") + len("SEARCH:"):].strip() or question

    if not run.answer:
        answer_result = provider.generate(
            ANSWER_TEMPLATE.format(question=question, evidence=format_evidence(evidence, not bounds.full_links))
        )
        record_generation(answer_result, "answer")
        run.answer = answer_result.text.strip()

    fetch_failures: List[Dict[str, str]] = []
    if not abstained_by_model:
        cited = cited_link_targets(run.answer, evidence)
        if cited:
            attempt = grounding_attempt(cited, bounds, retriever)
            fetched = attempt["fetched"]
            fetch_failures = attempt["failures"]
            record: Dict[str, Any] = {
                "step": "grounding",
                "mode": bounds.grounding,
                "cited_from_links": sorted(attempt["attempted"]) if bounds.grounding != "off" else [],
                "cited_from_links_unverified": attempt["attempted"] if bounds.grounding == "off" else [],
                "fetched": [chunk["node_id"] for chunk in fetched],
                "fetch_failures": fetch_failures,
                "truncated": attempt["truncated"],
                "link_only_claims": [failure["node_id"] for failure in fetch_failures]
                + (attempt["attempted"] if bounds.grounding == "off" else []),
            }
            if fetched:
                draft = run.answer
                fetched_ids = {chunk["node_id"] for chunk in fetched}
                if bounds.grounding == "lean":
                    citing = [
                        chunk for chunk in evidence
                        if any(target in fetched_ids
                               for targets in (chunk.get("relations") or {}).values() for target in targets)
                    ]
                    prompt = GROUND_TEMPLATE.format(
                        question=question, draft=draft,
                        evidence=format_evidence(citing + fetched, not bounds.full_links),
                    )
                evidence.extend(fetched)
                if bounds.grounding == "full":
                    prompt = ANSWER_TEMPLATE.format(
                        question=question, evidence=format_evidence(evidence, not bounds.full_links)
                    )
                answer_result = provider.generate(prompt)
                record_generation(answer_result, "grounding")
                run.answer = answer_result.text.strip()
                record["answer_before_grounding"] = draft[:300]
                record["answer_changed"] = run.answer != draft
            run.steps.append(record)

    if gate:
        verdict = sufficiency_gate(run.answer, question, [chunk.get("text") or "" for chunk in evidence],
                                   universe, fetch_failures)
        verdict["answer_before_gate"] = run.answer[:300]
        run.steps.append(verdict)
        if verdict["decision"] == "abstain":
            run.answer = ABSTAIN_ANSWER
            run.stop_reason = "gate_abstained"

    run.evidence_ids = [chunk["node_id"] for chunk in evidence]
    if run.stop_reason == "unknown":
        run.stop_reason = "completed"
    return run
