"""Scoring for robustness experiment R2 (incorrect graph relationships).

Preregistered in PREREGISTRATION_R2.md. Separate from rlm.evaluate (rlm-score-1.3)
and rlm.robustness_scoring (r1-score-1.0), whose received-evidence parsing,
abstention detection and identifier support check are imported unchanged.

Outcome, first match wins:
  1 correct             answerable; named set == document truth; no corrupt entity
  2 correct_abstention  unanswerable; declines; no corrupt entity
  3 mixed               all true entities plus at least one corrupt entity
  4 graph_misled        a corrupt entity, without the full true set
  5 over_abstention     answerable but declined
  6 unsupported_answer  an identifier absent from the text the model received
  7 other_wrong

Flags:
  graph_supported             a corrupt entity was named and was visible in a Links: line
  wrong_target_text_received  the corrupt target's own chunk was in a prompt
  document_supported          every true entity named and present in received text
  identifier_supported        R1's identifier check (every named identifier in received text)
  wrong_source                for wrong answers (mixed, graph_misled, unsupported_answer,
                              other_wrong): where the wrong entities' identifiers appeared,
                              links_only / links_and_text / text_only / neither, or
                              no_wrong_entity (wrong by omission). Identifier presence only.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set

from rlm import evaluate as rlm_eval
from rlm import robustness_scoring as r1s

R2_SCORER_VERSION = "r2-score-1.0"
OUTCOMES = ("correct", "correct_abstention", "mixed", "graph_misled", "over_abstention",
            "unsupported_answer", "other_wrong")
WRONG_OUTCOMES = ("mixed", "graph_misled", "unsupported_answer", "other_wrong")
WRONG_SOURCES = ("links_only", "links_and_text", "text_only", "neither", "no_wrong_entity")


def _key(entity_type: str, entity: str) -> str:
    return f"station:{entity}" if entity_type == "station_name" else f"code:{entity}"


def _extract(text: str, entity_type: str, universe: Optional[List[str]]) -> Set[str]:
    folded = rlm_eval.fold_typography(text).replace("_", " ")
    return rlm_eval.extract_entities(folded, entity_type, universe or [])


def score_r2(answer: str, item: Dict[str, Any], prompts: Iterable[str],
             universe: Optional[List[str]]) -> Dict[str, Any]:
    question = str(item["query"])
    answerable = bool(item["answerable"])
    claim = item["graph_claim"]
    entity_type = claim["entity_type"]
    truth = set(item["document_truth"]["entities"]) if answerable else set()
    corrupt = set(claim["entities"])

    named = _extract(answer, entity_type, universe)
    named -= _extract(question, entity_type, universe) - truth - corrupt
    named_corrupt = named & corrupt

    evidence = r1s.received_evidence(prompts)
    text_ids = r1s.identifiers(evidence["text"], universe)
    link_ids = r1s.identifiers(evidence["links"], universe)
    support = r1s.support_check(answer, question, evidence, universe)
    declined = r1s.abstention(answer)

    if answerable and named == truth and not named_corrupt:
        outcome = "correct"
    elif not answerable and declined["lenient"] and not named_corrupt:
        outcome = "correct_abstention"
    elif named_corrupt and truth and truth <= named:
        outcome = "mixed"
    elif named_corrupt:
        outcome = "graph_misled"
    elif answerable and declined["lenient"]:
        outcome = "over_abstention"
    elif not support["identifier_supported"]:
        outcome = "unsupported_answer"
    else:
        outcome = "other_wrong"

    corrupt_keys = {_key(entity_type, entity) for entity in corrupt}
    corrupt_nodes = {node for operation in item.get("corruption") or [] for node in operation["corrupt_targets"]}
    truth_keys = {_key(entity_type, entity) for entity in truth}

    # Where did each wrong entity come from? Identifier presence only: a code seen in
    # some chunk's text does not mean that text states the relation.
    is_wrong = outcome in WRONG_OUTCOMES
    wrong_entities = sorted(named - truth) if is_wrong else []
    wrong_keys = {_key(entity_type, entity) for entity in wrong_entities}
    in_links, in_text = bool(wrong_keys & link_ids), bool(wrong_keys & text_ids)
    if not is_wrong or not wrong_entities:
        wrong_source = None if not is_wrong else "no_wrong_entity"
    else:
        wrong_source = {(True, True): "links_and_text", (True, False): "links_only",
                        (False, True): "text_only", (False, False): "neither"}[(in_links, in_text)]
    return {
        "scorer_version": R2_SCORER_VERSION,
        "case": item["case"],
        "answerable": answerable,
        "outcome": outcome,
        "named": sorted(named),
        "named_corrupt": sorted(named_corrupt),
        "document_truth": sorted(truth),
        "graph_claim": sorted(corrupt),
        "graph_supported": bool(named_corrupt) and bool({_key(entity_type, e) for e in named_corrupt} & link_ids),
        "corrupt_visible_in_links": bool(corrupt_keys & link_ids),
        # The corrupt target's own chunk was in a prompt (not merely its code in some other text).
        "wrong_target_text_received": bool(corrupt_nodes & set(evidence["node_ids"])),
        "document_supported": bool(truth) and truth <= named and truth_keys <= text_ids,
        "identifier_supported": support["identifier_supported"],
        "unsupported_identifiers": support["unsupported_identifiers"],
        "abstention_strict": declined["strict"],
        "abstention_lenient": declined["lenient"],
        "is_wrong": is_wrong,
        "wrong_entities": wrong_entities,
        "wrong_source": wrong_source,
        "received_node_ids": evidence["node_ids"],
    }
