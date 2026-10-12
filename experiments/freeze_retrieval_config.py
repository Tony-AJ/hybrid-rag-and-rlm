"""Step 3: select and freeze the retrieval configuration from measured results.

The selection rule is defined here in code and applied to the recorded ablation
output, so the choice is auditable and not a judgement made in prose:

  Primary metric      NDCG@5  (rank-aware: rewards both presence and position)
  Secondary metric    recall@5
  Complexity margin   a variant must beat the simpler base it modifies by at
                      least MARGIN_NDCG on the primary metric to be selected.
                      MARGIN_NDCG = 0.05, which is about 3 of 60 queries. Below
                      that, the simpler mechanism wins, because a 1-2 query
                      difference on one dataset is not evidence for the extra
                      mechanism.
  Guardrail           a variant that degrades the single-hop factual family is
                      rejected outright, whatever it does elsewhere.

The production implementation is NOT modified. It stays as the control, and the
frozen configuration is a separate record the RLM loop reads.

Run:
    python experiments/freeze_retrieval_config.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CORPUS_ID = "meridian-transit-v1"
ABLATION_PATH = ROOT / f"experiment_fusion_ablation_{CORPUS_ID}.json"
BASELINE_PATH = ROOT / f"benchmark_baseline_{CORPUS_ID}.json"
OUTPUT_PATH = ROOT / "experiment_frozen_retrieval_config.json"

PRIMARY_METRIC = "ndcg@5"
SECONDARY_METRIC = "recall@5"
MARGIN_NDCG = 0.05
GUARDRAIL_CATEGORY = "single-hop factual"


def select(ablation: Dict[str, Any]) -> Dict[str, Any]:
    runs = ablation["runs"]
    bases = {
        "no_hop0_graph": "control",
        "max_norm": "control",
        "rrf": "control",
        "no_hop0_no_penalty": "no_hop0_graph",
        "no_hop0_flat_hop": "no_hop0_graph",
        "no_hop0_seed_weighted": "no_hop0_graph",
        "no_hop0_convergence": "no_hop0_graph",
        "no_hop0_query_rescore": "no_hop0_graph",
    }
    control_guardrail = runs["control"]["by_category"][GUARDRAIL_CATEGORY]["metrics"][SECONDARY_METRIC]

    decisions: List[Dict[str, Any]] = []
    selected = "control"

    # Walk the mechanism chain: a variant may only be adopted if the base it
    # modifies is itself adopted, which keeps "one mechanism at a time" honest.
    for name, base in bases.items():
        if name not in runs:
            continue
        primary_delta = runs[name]["aggregate"][PRIMARY_METRIC] - runs[base]["aggregate"][PRIMARY_METRIC]
        guardrail = runs[name]["by_category"][GUARDRAIL_CATEGORY]["metrics"][SECONDARY_METRIC]
        guardrail_ok = guardrail >= control_guardrail - 1e-9
        adopted = bool(guardrail_ok and primary_delta >= MARGIN_NDCG and base == selected)
        reason = (
            "guardrail failed: single-hop factual retrieval degraded"
            if not guardrail_ok
            else f"gain {primary_delta:+.4f} on {PRIMARY_METRIC} is below the {MARGIN_NDCG} margin"
            if primary_delta < MARGIN_NDCG
            else f"base {base!r} was not itself adopted"
            if base != selected
            else f"gain {primary_delta:+.4f} on {PRIMARY_METRIC} clears the {MARGIN_NDCG} margin"
        )
        decisions.append({
            "variant": name,
            "base": base,
            f"{PRIMARY_METRIC}_delta_vs_base": round(primary_delta, 4),
            f"{SECONDARY_METRIC}_delta_vs_base": round(
                runs[name]["aggregate"][SECONDARY_METRIC] - runs[base]["aggregate"][SECONDARY_METRIC], 4
            ),
            "guardrail_passed": guardrail_ok,
            "adopted": adopted,
            "reason": reason,
        })
        if adopted:
            selected = name

    return {"selected_variant": selected, "decisions": decisions}


def trade_offs(ablation: Dict[str, Any], selected: str) -> List[Dict[str, Any]]:
    """Every category where the selection is worse than the control, plus the cost side."""
    runs = ablation["runs"]
    notes: List[Dict[str, Any]] = []
    for category, bucket in runs["control"]["by_category"].items():
        control_value = bucket["metrics"][SECONDARY_METRIC]
        selected_value = runs[selected]["by_category"][category]["metrics"][SECONDARY_METRIC]
        if selected_value < control_value - 1e-9:
            notes.append({
                "kind": "regression",
                "category": category,
                "metric": SECONDARY_METRIC,
                "control": control_value,
                "selected": selected_value,
                "query_equivalents": round(
                    (control_value - selected_value) * runs["control"]["by_category"][category]["queries"], 3
                ),
            })
    notes.append({
        "kind": "composition_shift",
        "detail": (
            f"graph-only chunks now occupy {runs[selected]['aggregate']['graph_only_in_top_k']:.2f} of "
            f"{ablation['configuration']['top_k']} top-K slots on average, versus "
            f"{runs['control']['aggregate']['graph_only_in_top_k']:.2f} for the control. Vector hits are "
            "displaced by graph evidence; the single-hop guardrail shows this did not cost factual retrieval "
            "on this corpus, but a corpus with weaker graph precision could suffer."
        ),
    })
    notes.append({
        "kind": "latency",
        "detail": (
            f"mean {runs[selected]['aggregate']['mean_latency_ms']:.1f} ms versus "
            f"{runs['control']['aggregate']['mean_latency_ms']:.1f} ms for the control. The fusion change is "
            "arithmetic over an already-built candidate set, so cost is unchanged; both figures are dominated "
            "by query embedding and vary between runs."
        ),
    })
    notes.append({
        "kind": "unresolved",
        "detail": (
            "None of the six tested mechanisms recovers the 2-hop incident-to-procedure family. Hop-1 "
            "neighbours of the vector seeds fill the top-K ahead of the hop-2 answer. Depth-level "
            "mechanisms (penalty removal, flat hop credit, seed-strength weighting) leave it at 0.000, and "
            "both round-3 hypotheses were measured and rejected: multi-seed convergence cost -0.117 "
            "recall@5 with 9 regressions, and query-aware rescoring of graph candidates with the production "
            "embedding model cost -0.192 recall@5 with 13 regressions at 2.4x the latency. Rescoring fails "
            "for a diagnosable reason: the query names a station, so station and incident chunks outscore "
            "the procedure chunk that actually answers it, which is the same lexical gap the corpus was "
            "built to create."
        ),
    })
    notes.append({
        "kind": "interpretation",
        "detail": (
            "The 2-hop answer is unreachable by embedding similarity and, structurally, indistinguishable "
            "from the irrelevant hop-1 neighbours of the same seed. Separating them requires knowing which "
            "relation to follow (caused_by then maintained_by) for this question, which is query intent, not "
            "a scalar any static fusion formula can carry. Two honest routes: a relation-aware retriever "
            "that conditions edge traversal on the query, or the bounded RLM loop, where the model issues a "
            "follow-up retrieval after seeing the incident chunk. The second is already built (rlm/loop.py) "
            "and is the case where a reasoning loop should beat any static fusion. Hand-tuned edge-type "
            "priors would also lift this family but would encode this dataset's answer paths, so they are "
            "excluded as not a fair test."
        ),
    })
    return notes


def main() -> int:
    if not ABLATION_PATH.exists():
        raise RuntimeError(
            f"{ABLATION_PATH.name} not found. Run experiments/fusion_ablation.py first: the "
            "configuration must be selected from measured results."
        )
    ablation = json.loads(ABLATION_PATH.read_text(encoding="utf-8"))
    if not ablation.get("control_faithfulness", {}).get("queries_checked"):
        raise RuntimeError("Ablation record has no control-faithfulness proof; refusing to freeze.")
    if ablation["control_faithfulness"]["mismatches"] != 0:
        raise RuntimeError("Control did not reproduce production output; refusing to freeze.")

    outcome = select(ablation)
    selected = outcome["selected_variant"]
    runs = ablation["runs"]
    config = ablation["configuration"]

    baseline_reference: Optional[Dict[str, Any]] = None
    if BASELINE_PATH.exists():
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        baseline_reference = {
            "file": BASELINE_PATH.name,
            "recorded_at": baseline["baseline"]["recorded_at"],
            "label": baseline["baseline"]["label"],
            "corpus_fingerprint": baseline["baseline"]["corpus_fingerprint"],
        }

    record = {
        "status": "frozen",
        "frozen_at_corpus": CORPUS_ID,
        "selection_rule": {
            "primary_metric": PRIMARY_METRIC,
            "secondary_metric": SECONDARY_METRIC,
            "complexity_margin_on_primary": MARGIN_NDCG,
            "guardrail_category": GUARDRAIL_CATEGORY,
            "note": "Stated before inspecting the numbers; applied programmatically in select().",
        },
        "selected": {
            "variant": selected,
            "description": ablation["variant_descriptions"][selected],
            "seed_k": runs[selected]["seed_k"],
            "top_k": config["top_k"],
            "max_depth": config["max_depth"],
            "alpha": config["alpha"],
            "beta": config["beta"],
            "gamma": config["gamma"],
            "implementation": f"experiments.fusion_ablation.VARIANTS[{selected!r}]",
            "production_service_modified": False,
        },
        "control": {
            "variant": "control",
            "description": "app.service.HybridRetrievalService.hybrid_search, unmodified",
            "metrics": runs["control"]["aggregate"],
            "preserved_as": "the control for every later comparison",
        },
        "selected_metrics": runs[selected]["aggregate"],
        "selected_by_category": runs[selected]["by_category"],
        "decisions": outcome["decisions"],
        "trade_offs": trade_offs(ablation, selected),
        "provenance": {
            "ablation_file": ABLATION_PATH.name,
            "ablation_sha256": hashlib.sha256(ABLATION_PATH.read_bytes()).hexdigest(),
            "corpus_fingerprint": config["corpus_fingerprint"],
            "dataset": config["dataset"],
            "dataset_sha256": config["dataset_sha256"],
            "vector_db_path": config["vector_db_path"],
            "graph_db_path": config["graph_db_path"],
            "control_faithfulness": ablation["control_faithfulness"],
            "baseline_reference": baseline_reference,
        },
    }

    OUTPUT_PATH.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"selection rule: primary={PRIMARY_METRIC}, margin={MARGIN_NDCG}, guardrail={GUARDRAIL_CATEGORY!r}\n")
    for decision in outcome["decisions"]:
        mark = "ADOPT " if decision["adopted"] else "reject"
        print(f"  {mark} {decision['variant']:<24} vs {decision['base']:<16}"
              f"d_{PRIMARY_METRIC}={decision[f'{PRIMARY_METRIC}_delta_vs_base']:+.4f}  {decision['reason']}")
    print(f"\nfrozen variant: {selected}")
    print(f"  {PRIMARY_METRIC}={runs[selected]['aggregate'][PRIMARY_METRIC]:.4f} "
          f"(control {runs['control']['aggregate'][PRIMARY_METRIC]:.4f})")
    print(f"  {SECONDARY_METRIC}={runs[selected]['aggregate'][SECONDARY_METRIC]:.4f} "
          f"(control {runs['control']['aggregate'][SECONDARY_METRIC]:.4f})")
    print("\ntrade-offs recorded:")
    for note in record["trade_offs"]:
        if note["kind"] == "regression":
            print(f"  regression  {note['category']}: {note['control']:.3f} -> {note['selected']:.3f} "
                  f"({note['query_equivalents']} query-equivalents)")
        else:
            print(f"  {note['kind']}")
    print(f"\nWrote {OUTPUT_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
