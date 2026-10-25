"""Bounded retrieve-and-reason loop over the frozen retriever (step 4).

The loop is deliberately small and bounded. Every iteration costs a retrieval
and a generation call, so both are capped and counted, and the caps are recorded
in the run record rather than left implicit.

Retrieval is not reimplemented here. The loop calls the frozen retrieval
configuration selected in step 3, which is read from
`experiment_frozen_retrieval_config.json` and executed through
`experiments.fusion_ablation` (production candidate generation plus the frozen
fusion variant). If that file is missing the loop refuses to run, so an
experiment can never silently use an unselected retriever.

Prompt templates live in this module as module-level constants and are hashed
into the run record, so a prompt edit is visible in the results.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import fusion_ablation as ablation  # noqa: E402
from rlm.providers import (  # noqa: E402
    GenerationProvider,
    GenerationResult,
    prompt_template_sha256,
)

FROZEN_CONFIG_PATH = ROOT / "experiment_frozen_retrieval_config.json"
LOOP_VERSION = "1.2"

ANSWER_TEMPLATE = """You answer questions about a transit authority using only the evidence below.

Question: {question}

Evidence:
{evidence}

Rules:
- Use only the evidence. If the evidence does not contain the answer, reply exactly: INSUFFICIENT EVIDENCE
- Name the specific identifiers (procedure codes like MP-101, policy codes like PL-A, equipment codes like TX-440, or station names) that answer the question.
- Reply with the identifiers and one short sentence. No preamble.

Answer:"""

DECIDE_TEMPLATE = """You are deciding whether more evidence is needed.

Question: {question}

Evidence gathered so far:
{evidence}

DECIDE: If the evidence already answers the question, reply exactly STOP.
Otherwise reply with a single short search query that would find the missing piece, prefixed with SEARCH:
"""

TEMPLATE_HASHES = {
    "answer": prompt_template_sha256(ANSWER_TEMPLATE),
    "decide": prompt_template_sha256(DECIDE_TEMPLATE),
}


@dataclass
class LoopBounds:
    max_steps: int = 3
    max_retrievals: int = 3
    top_k: int = 5
    max_evidence_chunks: int = 12
    max_grounding_chunks: int = 4

    def as_record(self) -> Dict[str, Any]:
        return {
            "max_steps": self.max_steps,
            "max_retrievals": self.max_retrievals,
            "top_k": self.top_k,
            "max_evidence_chunks": self.max_evidence_chunks,
            "max_grounding_chunks": self.max_grounding_chunks,
        }


@dataclass
class LoopRun:
    question: str
    answer: str
    evidence_ids: List[str]
    steps: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_calls: int = 0
    generation_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    stop_reason: str = "unknown"

    def as_record(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "evidence_ids": self.evidence_ids,
            "steps": self.steps,
            "retrieval_calls": self.retrieval_calls,
            "generation_calls": self.generation_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "retrieval_latency_ms": round(self.retrieval_latency_ms, 3),
            "generation_latency_ms": round(self.generation_latency_ms, 3),
            "total_latency_ms": round(self.retrieval_latency_ms + self.generation_latency_ms, 3),
            "stop_reason": self.stop_reason,
        }


def load_frozen_retrieval_config() -> Dict[str, Any]:
    if not FROZEN_CONFIG_PATH.exists():
        raise RuntimeError(
            f"No frozen retrieval configuration at {FROZEN_CONFIG_PATH.name}. Run step 3 "
            "(experiments/freeze_retrieval_config.py) first: the RLM loop must run on a "
            "retriever that was selected from measured results."
        )
    config = json.loads(FROZEN_CONFIG_PATH.read_text(encoding="utf-8"))
    variant = config["selected"]["variant"]
    if variant not in ablation.VARIANTS:
        raise RuntimeError(
            f"Frozen config names variant {variant!r}, which the ablation module does not define."
        )
    return config


class FrozenRetriever:
    """The step-3 selection, callable. One implementation, reused, not copied."""

    def __init__(self, service: Any, config: Dict[str, Any]):
        self.service = service
        selected = config["selected"]
        self.variant_name = selected["variant"]
        self.scorer = ablation.VARIANTS[self.variant_name]
        self.max_depth = int(selected["max_depth"])
        self.seed_k = int(selected["seed_k"])
        self.weights = {
            "alpha": selected["alpha"],
            "beta": selected["beta"],
            "gamma": selected["gamma"],
        }

    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        candidates = ablation.collect_candidates(self.service, query, self.seed_k, self.max_depth)
        ranking = self.scorer(candidates, top_k, self.max_depth)
        return [
            {
                "node_id": node_id,
                "text": candidates[node_id]["text"],
                "hop": candidates[node_id]["hop"],
                "retrieved_by_vector": candidates[node_id]["retrieved_by_vector"],
                "relations": self.relations(node_id),
            }
            for node_id in ranking
        ]

    def relations(self, node_id: str) -> Dict[str, List[str]]:
        """Outgoing graph edges of a node, grouped by edge type.

        Chunk text alone does not say which procedures maintain a piece of
        equipment; that link exists only as a graph edge. Without it the model
        sees one procedure, cannot know a second exists, and stops early.
        """
        graph = self.service.graph_db.graph
        grouped: Dict[str, List[str]] = {}
        if node_id not in graph:
            return grouped
        for _, target, data in graph.out_edges(node_id, data=True):
            grouped.setdefault(str(data.get("type", "related_to")), []).append(str(target))
        return {edge_type: sorted(targets) for edge_type, targets in sorted(grouped.items())}

    def fetch(self, node_id: str) -> Optional[Dict[str, Any]]:
        """One node's chunk by id, read from the graph rather than searched for."""
        graph = self.service.graph_db.graph
        if node_id not in graph:
            return None
        return {
            "node_id": node_id,
            "text": graph.nodes[node_id].get("text") or "",
            "hop": None,
            "retrieved_by_vector": False,
            "relations": self.relations(node_id),
        }


TYPOGRAPHIC_DASHES = re.compile(r"[\u2010-\u2015\u2212]")


def cited_link_targets(answer: str, evidence: List[Dict[str, Any]]) -> List[str]:
    """Linked nodes the answer names that are not yet in the evidence.

    An answer may cite an identifier it only saw in a `Links:` list (for example
    MP-102 from equipment_TX-440's maintained_by edge). Such a citation rests on
    the edge alone; the grounding pass fetches the node's own text so the final
    answer is checked against it. A node counts as cited when its full id or its
    code (the part after the type prefix) appears in the answer.
    """
    text = TYPOGRAPHIC_DASHES.sub("-", answer)
    present = {chunk["node_id"] for chunk in evidence}
    cited: List[str] = []
    for chunk in evidence:
        for targets in (chunk.get("relations") or {}).values():
            for target in targets:
                if target in present or target in cited:
                    continue
                code = target.split("_", 1)[-1]
                if target in text or re.search(rf"(?<![\w-]){re.escape(code)}(?![\w-])", text):
                    cited.append(target)
    return cited


def format_evidence(chunks: List[Dict[str, Any]]) -> str:
    lines = []
    for index, chunk in enumerate(chunks, 1):
        line = f"[{index}] ({chunk['node_id']}) {chunk['text']}"
        relations = chunk.get("relations") or {}
        if relations:
            line += " Links: " + "; ".join(
                f"{edge_type} -> {', '.join(targets)}" for edge_type, targets in relations.items()
            )
        lines.append(line)
    return "\n".join(lines)


def run_loop(
    retriever: FrozenRetriever,
    provider: GenerationProvider,
    question: str,
    bounds: LoopBounds,
) -> LoopRun:
    """One bounded reasoning-and-retrieval episode.

    Retrieval and generation failures propagate. A loop that swallowed them
    would report a confident wrong answer as if evidence had been considered.
    """
    run = LoopRun(question=question, answer="", evidence_ids=[])
    evidence: List[Dict[str, Any]] = []
    seen: set = set()
    current_query = question

    def record_generation(result: GenerationResult) -> None:
        run.generation_calls += 1
        run.input_tokens += result.input_tokens or 0
        run.output_tokens += result.output_tokens or 0
        run.generation_latency_ms += result.latency_ms

    for step in range(1, bounds.max_steps + 1):
        if run.retrieval_calls >= bounds.max_retrievals:
            run.stop_reason = "retrieval_budget_exhausted"
            break

        started = time.perf_counter()
        try:
            chunks = retriever.retrieve(current_query, bounds.top_k)
        except Exception as exc:
            raise RuntimeError(
                f"Frozen retrieval failed on step {step} for query: {current_query!r}"
            ) from exc
        run.retrieval_latency_ms += (time.perf_counter() - started) * 1000
        run.retrieval_calls += 1

        added = []
        for chunk in chunks:
            if chunk["node_id"] not in seen and len(evidence) < bounds.max_evidence_chunks:
                seen.add(chunk["node_id"])
                evidence.append(chunk)
                added.append(chunk["node_id"])

        step_record = {
            "step": step,
            "query": current_query,
            "retrieved": [chunk["node_id"] for chunk in chunks],
            "new_evidence": added,
        }

        if step == bounds.max_steps or run.retrieval_calls >= bounds.max_retrievals:
            step_record["decision"] = "bounds_reached"
            run.steps.append(step_record)
            run.stop_reason = "step_budget_exhausted" if step == bounds.max_steps else "retrieval_budget_exhausted"
            break

        decide_result = provider.generate(
            DECIDE_TEMPLATE.format(question=question, evidence=format_evidence(evidence))
        )
        record_generation(decide_result)
        decision = decide_result.text.strip()
        step_record["decision"] = decision[:200]
        run.steps.append(step_record)

        if decision.upper().startswith("STOP") or "SEARCH:" not in decision.upper():
            run.stop_reason = "model_stopped"
            break
        current_query = decision.split("SEARCH:", 1)[1].strip() or question

    answer_result = provider.generate(
        ANSWER_TEMPLATE.format(question=question, evidence=format_evidence(evidence))
    )
    record_generation(answer_result)
    run.answer = answer_result.text.strip()

    # Grounding pass: an answer that cites a node seen only as a link is
    # re-answered once with that node's text in the evidence. Graph lookups,
    # not retrievals; one extra generation call, and only when needed.
    grounding = cited_link_targets(run.answer, evidence)[: bounds.max_grounding_chunks]
    fetched = [chunk for chunk in (retriever.fetch(node_id) for node_id in grounding) if chunk]
    if fetched:
        evidence.extend(fetched)
        run.steps.append({
            "step": "grounding",
            "cited_from_links": [chunk["node_id"] for chunk in fetched],
            "answer_before_grounding": run.answer[:300],
        })
        answer_result = provider.generate(
            ANSWER_TEMPLATE.format(question=question, evidence=format_evidence(evidence))
        )
        record_generation(answer_result)
        run.answer = answer_result.text.strip()

    run.evidence_ids = [chunk["node_id"] for chunk in evidence]
    if run.stop_reason == "unknown":
        run.stop_reason = "completed"
    return run
