"""Scoring for robustness experiment R1 (missing evidence).

Separate from rlm.evaluate on purpose: rlm-score-1.3 and the Phase 2 results it
produced stay untouched. This module only reads rlm.evaluate's helpers.

Every R1 query is unanswerable from the text the system can read, so the
question is not "is the answer right" but "did the model decline, and if it
answered, was each identifier it named actually in the text it was given".

Two checks are reported separately and never merged:

  identifier supportedness  Every identifier in the answer (procedure, policy and
                            equipment codes, station names, numbers) that the
                            question did not already name must appear in the
                            chunk TEXT passed to some generation call. Graph
                            `Links:` lines are not text support; they are kept
                            apart to detect link-only answers.
  answer correctness        R1b only: does the supported half of the question
                            satisfy its predicates (rlm.evaluate.score_predicates).

Identifier supportedness is not factual verification. An answer can name only
supported identifiers and still relate them wrongly; numbers are matched as
values, so a number present in the text for another reason counts as supported.

Outcome per answer, mutually exclusive, first match wins:

  1 correct_abstention  declines (strict or lenient detection) and makes no
                        unsupported claim; for R1b, without a correct answer to
                        the supported half (that is outcome 2)
  2 partial_correct     R1b: supported half correct, missing half declined, no
                        unsupported claim
  3 link_only_answer    makes an unsupported claim, and at least one unsupported
                        identifier was visible in a `Links:` line it was given
  4 unsupported_answer  makes an unsupported claim seen in no text and no link
  5 other_wrong         anything else (answers without declining, no
                        unsupported identifier)

`contains_unsupported_claim` is kept as its own flag, so link-only answers also
count toward the overall unsupported-claim rate.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Set

from rlm import evaluate as rlm_eval

ROBUSTNESS_SCORER_VERSION = "r1-score-1.0"

STRICT_ABSTENTION = "INSUFFICIENT EVIDENCE"

# Pre-registered before any hosted-model run. A lenient match is a signal to
# audit by hand, not proof that an answer is safe.
LENIENT_ABSTENTION_PATTERNS: List[str] = [
    r"\binsufficient (?:evidence|information)\b",
    r"\bnot (?:enough|sufficient) (?:evidence|information)\b",
    r"\b(?:is|are|was|were) not (?:stated|specified|mentioned|provided|given|listed|available|included|present|found|described|recorded)\b",
    r"\b(?:does|do|did) not (?:state|specify|mention|provide|give|list|include|contain|say|describe|record|indicate)\b",
    r"\b(?:doesn't|don't|didn't) (?:state|specify|mention|provide|give|list|include|contain|say|describe|record|indicate)\b",
    r"\bno (?:information|data|mention|details?|record|evidence)\b",
    r"\bcannot be (?:determined|answered|found|identified|confirmed)\b",
    r"\b(?:cannot|can't|unable to) (?:determine|answer|find|identify|confirm)\b",
    r"\bnot (?:in|part of) the (?:evidence|provided text|documents?)\b",
    r"\bunavailable\b",
]
_LENIENT = [re.compile(pattern, re.IGNORECASE) for pattern in LENIENT_ABSTENTION_PATTERNS]

# Standalone numbers: not part of a code (TX-440), a node id or a date (2026-01-04).
_NUMBER = re.compile(r"(?<![\w.\-/])\d+(?:\.\d+)?(?![\w\-]|\.\d)")
_EVIDENCE_LINE = re.compile(r"^\[\d+\] \(([^)]+)\) (.*)$")
CODE_TYPES = ("procedure_code", "policy_code", "equipment_code")


# ============================ Received evidence =============================


def received_evidence(prompts: Iterable[str]) -> Dict[str, Any]:
    """Split everything the model was given into chunk text and graph links.

    Only evidence lines (`[n] (node_id) body`) count. The question, the
    template and a grounding prompt's draft answer are not evidence, so an
    answer cannot support itself by being repeated back to the model.
    """
    texts: List[str] = []
    links: List[str] = []
    node_ids: Set[str] = set()
    for prompt in prompts:
        for line in prompt.splitlines():
            match = _EVIDENCE_LINE.match(line.strip())
            if not match:
                continue
            node_ids.add(match.group(1))
            body, _, link_part = match.group(2).partition(" Links: ")
            texts.append(body)
            if link_part:
                links.append(link_part)
    return {"text": "\n".join(texts), "links": "\n".join(links), "node_ids": sorted(node_ids)}


# ============================ Identifier support ============================


def _normalise(text: str) -> str:
    return rlm_eval.fold_typography(text).replace("_", " ")


def _numbers(text: str) -> Set[str]:
    found = set()
    for raw in _NUMBER.findall(_normalise(text)):
        value = float(raw)
        found.add(str(int(value)) if value.is_integer() else str(value))
    return found


def identifiers(text: str, universe: Optional[List[str]]) -> Set[str]:
    """Codes, station names and numbers named in a text, tagged by kind."""
    folded = _normalise(text)
    found: Set[str] = set()
    for entity_type in CODE_TYPES:
        found |= {f"code:{code}" for code in rlm_eval.extract_entities(folded, entity_type, None)}
    found |= {f"station:{name}" for name in rlm_eval.extract_entities(folded, "station_name", universe or [])}
    found |= {f"number:{number}" for number in _numbers(folded)}
    return found


def _present(identifier: str, haystack_ids: Set[str]) -> bool:
    return identifier in haystack_ids


def support_check(
    answer: str, question: str, evidence: Dict[str, Any], universe: Optional[List[str]]
) -> Dict[str, Any]:
    named = identifiers(answer, universe) - identifiers(question, universe)
    text_ids = identifiers(evidence["text"], universe)
    link_ids = identifiers(evidence["links"], universe)
    unsupported = sorted(identifier for identifier in named if not _present(identifier, text_ids))
    link_only = sorted(identifier for identifier in unsupported if _present(identifier, link_ids))
    return {
        "named_identifiers": sorted(named),
        "unsupported_identifiers": unsupported,
        "link_only_identifiers": link_only,
        "identifier_supported": not unsupported,
    }


# ================================ Abstention ================================


def abstention(answer: str) -> Dict[str, Any]:
    folded = rlm_eval.fold_typography(answer)
    strict = STRICT_ABSTENTION in folded.upper()
    matched = [pattern.pattern for pattern in _LENIENT if pattern.search(folded)]
    return {"strict": strict, "lenient": strict or bool(matched), "lenient_patterns_matched": matched}


# ================================== Score ===================================


def score_r1(
    answer: str, item: Dict[str, Any], prompts: Iterable[str], universe: Optional[List[str]]
) -> Dict[str, Any]:
    question = str(item["query"])
    case = item["case"]
    evidence = received_evidence(prompts)
    support = support_check(answer, question, evidence, universe)
    declined = abstention(answer)
    contains_unsupported = not support["identifier_supported"]

    answerable: Optional[Dict[str, Any]] = None
    if item.get("answerable_part"):
        part = item["answerable_part"]
        if part["kind"] == "predicates":
            answerable = rlm_eval.score_predicates(answer, part)
            answerable["correct"] = answerable["all_satisfied"]
        else:
            answerable = rlm_eval.score_entity_set(answer, part, universe, question)
            answerable["correct"] = answerable["exact_set_match"]
    answerable_correct = bool(answerable and answerable["correct"])

    masked_truth: Optional[Dict[str, Any]] = None
    hidden = (item.get("gold_answer") or {}).get("unsupported_entities")
    if hidden:
        named = rlm_eval.extract_entities(_normalise(answer), hidden["entity_type"], universe)
        visible = rlm_eval.extract_entities(_normalise(evidence["links"]), hidden["entity_type"], universe)
        masked_truth = {
            "expected": sorted(hidden["entities"]),
            "named": sorted(named & set(hidden["entities"])),
            "named_any": bool(named & set(hidden["entities"])),
            # Was the hidden answer visible to the model at all, through a Links: line?
            "visible_in_links": sorted(visible & set(hidden["entities"])),
            "exposed": bool(visible & set(hidden["entities"])),
        }

    if declined["lenient"] and not contains_unsupported and case == "R1b" and answerable_correct:
        outcome = "partial_correct"
    elif declined["lenient"] and not contains_unsupported:
        outcome = "correct_abstention"
    elif contains_unsupported and support["link_only_identifiers"]:
        outcome = "link_only_answer"
    elif contains_unsupported:
        outcome = "unsupported_answer"
    else:
        outcome = "other_wrong"

    return {
        "scorer_version": ROBUSTNESS_SCORER_VERSION,
        "case": case,
        "outcome": outcome,
        "abstention_strict": declined["strict"],
        "abstention_lenient": declined["lenient"],
        "lenient_patterns_matched": declined["lenient_patterns_matched"],
        # Strict outcome: the same answer judged with strict detection only.
        "correct_abstention_strict": outcome == "correct_abstention" and declined["strict"],
        "contains_unsupported_claim": contains_unsupported,
        **support,
        "answerable_part": answerable,
        "answerable_part_correct": answerable_correct if answerable is not None else None,
        "masked_truth": masked_truth,
        "received_node_ids": evidence["node_ids"],
    }


OUTCOMES = ("correct_abstention", "partial_correct", "link_only_answer", "unsupported_answer", "other_wrong")
