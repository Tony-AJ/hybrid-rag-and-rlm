"""Scoring for R2' relation-level verification (r2p-score-1.0).

Preregistered in PREREGISTRATION_R2_PRIME.md. Answer-level outcomes and flags are
R2's (rlm.robustness_scoring_r2, r2-score-1.0, imported unchanged). Relation-level
scoring is separate: every claim the verifier labels is compared with the truth,
which is membership in the clean graph's edge set.

Relation-level categories:
                 supported         contradicted              unverified        graph_only
  true claim     correct_support   false_contradiction       missed_support    permitted
  false claim    false_support     contradiction_detected    safe_miss         false_allowance
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from rlm import robustness_scoring_r2 as r2s

R2P_SCORER_VERSION = "r2p-score-1.0"
CLAIM_CATEGORIES = ("correct_support", "false_contradiction", "missed_support", "permitted",
                    "false_support", "contradiction_detected", "safe_miss", "false_allowance")
_TABLE = {
    (True, "supported"): "correct_support", (True, "contradicted"): "false_contradiction",
    (True, "unverified"): "missed_support", (True, "graph_only"): "permitted",
    (False, "supported"): "false_support", (False, "contradicted"): "contradiction_detected",
    (False, "unverified"): "safe_miss", (False, "graph_only"): "false_allowance",
}


def claim_category(true: bool, status: str) -> str:
    return _TABLE[(bool(true), status)]


def score_claims(claims: Iterable[Dict[str, Any]], clean_edges: Set[Tuple[str, str, str]]) -> List[Dict[str, Any]]:
    scored = []
    for claim in claims:
        subject, relation, obj = claim["claim"]
        true = (subject, relation, obj) in clean_edges
        scored.append({"claim": [subject, relation, obj], "relation": relation, "status": claim["status"],
                       "source": claim.get("source"), "true": true, "category": claim_category(true, claim["status"])})
    return scored


def score_answer(answer: str, item: Dict[str, Any], prompts: Iterable[str],
                 universe: Optional[List[str]]) -> Dict[str, Any]:
    """R2 answer outcome, flags and wrong-answer source (r2-score-1.0)."""
    result = r2s.score_r2(answer, item, prompts, universe)
    result["r2p_scorer_version"] = R2P_SCORER_VERSION
    result["graph_caused_error"] = result["wrong_source"] in ("links_only", "links_and_text")
    return result


def relation_summary(claims: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Counts per category, overall and per relation type, plus the preregistered rates."""
    def summarise(group: List[Dict[str, Any]]) -> Dict[str, Any]:
        counts = {category: sum(1 for c in group if c["category"] == category) for category in CLAIM_CATEGORIES}
        true_n = sum(1 for c in group if c["true"])
        false_n = len(group) - true_n
        checkable = [c for c in group if c["status"] != "graph_only"]
        correct = sum(1 for c in checkable if c["category"] in ("correct_support", "contradiction_detected"))

        def frac(k: int, n: int) -> str:
            return f"{k}/{n} ({100 * k / n:.1f}%)" if n else f"{k}/0 (n/a)"

        return {
            "claims": len(group), "true_claims": true_n, "false_claims": false_n, "counts": counts,
            "verification_accuracy": frac(correct, len(checkable)),
            "contradiction_detection": frac(counts["contradiction_detected"], false_n),
            "false_contradiction": frac(counts["false_contradiction"], true_n),
            "unverified": frac(counts["missed_support"] + counts["safe_miss"], len(group)),
            "false_support": frac(counts["false_support"], false_n),
        }

    relations = sorted({c["relation"] for c in claims})
    return {"overall": summarise(claims), "by_relation": {r: summarise([c for c in claims if c["relation"] == r]) for r in relations}}
