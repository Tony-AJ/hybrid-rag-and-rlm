"""R2'b: targeted test of contradicted maintained_by links (PREREGISTRATION_R2_PRIME_B.md).

A thin runner. It reuses every function of the frozen R2' runner
(experiments/robustness_r2_prime.py, unchanged) with this experiment's dataset,
ids, signatures and output paths, and adds checks W1-W5. relver-1.0, the schema,
the scorers, loop 1.7 and gate-1.0 are used unchanged.

Run:
    python -m experiments.robustness_r2_prime_b --phase validate
    python -m experiments.robustness_r2_prime_b --phase answers --provider mock --no-cache-write --out <path>
    python -m experiments.robustness_r2_prime_b --phase freeze
    python -m experiments.robustness_r2_prime_b --phase answers --provider groq --model openai/gpt-oss-120b --reasoning-effort low --resume
    python -m experiments.robustness_r2_prime_b --phase replay
    python -m experiments.robustness_r2_prime_b --phase report
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import architecture_benchmark as ab  # noqa: E402
from experiments import graph_corruption as gc  # noqa: E402
from experiments import robustness_r1 as r1  # noqa: E402
from experiments import robustness_r2_prime as base  # noqa: E402
from rlm import evaluate as rlm_eval  # noqa: E402

RUNNER_VERSION = "r2pb-run-1.0"
R2P_FREEZE = ROOT / "experiment_r2p_freeze.json"
R2P_DATASET = base.ROBUST / "relation_verification.json"
R2P_PROTECTED = {
    "experiment_r2p_meridian-transit-v1_groq_frozen.json": "eb633e0cdae55fba1dd86fab1812147eaf42ec0ff35002235c2ab98b10734f9e",
    "experiment_r2p_relation_replay.json": "2bdc058867c31687b9ae4e38cfe867881b215011e1b078f777c4427992016232",
    "experiment_r2p_verifier_sweep.json": "fe9a5347f9a7544a560c18adcfccd60e376dca5cfbc5bc7851336af4b31aa350",
    "experiment_r2p_freeze.json": "c9bf1fa6ffaa2debb45754d5bf2c1529da0adb7b551d00a7ac105955a717bdbb",
}
# R2' frozen inputs that R2'b must reuse unchanged.
R2P_PINNED = {"verifier_sha256": "rlm/relation_verifier.py", "schema_sha256": "benchmark_corpora/meridian/robustness/relation_schema.json",
              "scorer_r2p_sha256": "rlm/robustness_scoring_r2p.py", "runner_sha256": "experiments/robustness_r2_prime.py"}

# ---- Point the R2' runner at this experiment -------------------------------
base.EXPERIMENT_ID = "robustness-r2-prime-b"
base.RUNNER_VERSION = RUNNER_VERSION
base.DATASET_PATH = base.ROBUST / "relation_verification_b.json"
base.FREEZE_PATH = ROOT / "experiment_r2pb_freeze.json"
base.SWEEP_PATH = ROOT / "experiment_r2pb_verifier_sweep_rerun.json"
base.ANSWERS_PATH = ROOT / "experiment_r2pb_answers_meridian-transit-v1_groq.json"
base.REPLAY_PATH = ROOT / "experiment_r2pb_relation_replay.json"
base.COMPARISON_PATH = ROOT / "experiment_r2pb_comparison.json"
base.CASES = ("P", "N")
base.PROTECTED = {**base.PROTECTED, **R2P_PROTECTED}

_base_validation = base.run_validation
_base_input_hashes = base.input_hashes


def run_validation(ctx: base.Context) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    checks, sweeps = _base_validation(ctx)
    check = r1.check

    # W1 R2' inputs reused unchanged
    r2p_inputs = json.loads(R2P_FREEZE.read_text(encoding="utf-8"))["inputs"]
    drift = {path: "changed" for key, path in R2P_PINNED.items() if rlm_eval.file_sha256(ROOT / path) != r2p_inputs[key]}
    checks.append(check("W1 relver-1.0, relschema-1.0, r2p scorer and R2' runner unchanged since the R2' freeze", not drift, drift))

    # W2 dataset pairing and no reuse of R2' questions
    r2p_questions = {item["query"] for item in json.loads(R2P_DATASET.read_text(encoding="utf-8"))["queries"]}
    by_station: Dict[str, Dict[str, Any]] = {}
    w2 = []
    for item in ctx.items:
        if item["query"] in r2p_questions:
            w2.append({"id": item["id"], "reuses_r2prime_question": True})
        by_station.setdefault(item["pair_key"], {})[item["case"]] = item
    for station, pair in by_station.items():
        if set(pair) != {"P", "N"} or pair["P"]["query"] != pair["N"]["query"] or pair["P"]["id"] == pair["N"]["id"]:
            w2.append({"station": station, "bad_pair": sorted(pair)})
    checks.append(check("W2 paired P/N items (same question, distinct ids); no R2' question reused", not w2, w2))

    # W3 retrieval precondition
    w3 = []
    for item in ctx.items:
        inc, eq = item["document_truth"]["evidence_chunks"][:2]
        procs = set(item["document_truth"]["evidence_chunks"][2:])
        ids = [c["node_id"] for c in ctx.retriever(item, "C").retrieve(item["query"], 5)]
        if inc not in ids or eq not in ids or procs & set(ids):
            w3.append({"id": item["id"], "top5": ids})
    checks.append(check("W3 retrieval precondition: incident and equipment in C top 5, no true procedure (all 14 items)", not w3, w3))

    # W4 sweeps identical to the committed R2' sweep
    committed = json.loads((ROOT / "experiment_r2p_verifier_sweep.json").read_text(encoding="utf-8"))["sweeps"]
    same = all(sweeps[key] == committed[key] for key in ("V2_clean", "V3_substitutions"))
    checks.append(check("W4 V2/V3 sweeps with relver-1.0 identical to the committed R2' sweep", same))
    return checks, sweeps


def input_hashes(ctx: base.Context) -> Dict[str, Any]:
    hashes = _base_input_hashes(ctx)
    hashes["r2pb_runner_sha256"] = rlm_eval.file_sha256(Path(__file__).resolve())
    hashes["r2pb_preregistration_sha256"] = rlm_eval.file_sha256(ROOT / "PREREGISTRATION_R2_PRIME_B.md")
    return hashes


base.run_validation = run_validation
base.input_hashes = input_hashes

if __name__ == "__main__":
    raise SystemExit(base.main())
