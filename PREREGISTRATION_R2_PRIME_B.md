# Preregistration: R2′b, a targeted test of contradicted maintenance links

Status: **fixed before item generation, implementation and any model call.**
Branch `phase3-r2-prime-b` from `phase3-r2-prime` (`0737d709`).

## Why

In R2′, H1 was not met (5/8). All three misses were masked-text claims, labelled
unverified by design. No corrupted `maintained_by` claim with **readable** text reached D's
answers, because the new question wording often failed to retrieve the anchor. This was
R2's main failure (D named the wrong procedure after reading its contradicting text,
22/24), and on live answers it is still untested. R2′b tests it, and nothing else.

## Unchanged and frozen

- Verifier `relver-1.0` (`rlm/relation_verifier.py`, sha256 `17567a01…`)
- Schema `relschema-1.0`
- Scorers `r2p-score-1.0` and `r2-score-1.0`
- Corruption layer `corrupt-1.0`
- Loop 1.7, `gate-1.0` and the model settings
- The R2′ runner and results

**No verifier, schema or policy change.**

New code is only a thin runner (`experiments/robustness_r2_prime_b.py`). It reuses the R2′
runner's functions with this experiment's dataset, ids and output paths.

## Items (dataset `r2pb-1.0`, ids 4001–4014)

**Question template:** "Which maintenance procedures cover the fault that stopped service at <station>?"

**Stations:** the incident stations 21–28, which no v1.1, R2 or R2′ question has asked about with this template.

**Retrieval precondition** (offline, no model, checked on 2026-11-13 before this file):
- The station's incident chunk and equipment chunk must be in C's first top-5 retrieval.
- The `maintained_by` link must therefore be visible.
- No true procedure chunk may be in that top 5, so the answer must come through the link.
- Corrupting `maintained_by` does not change rankings (R2 retrieval control: 14/14 identical).

Seven of eight stations qualify. **Vantage Point (TC-6) is excluded** because its equipment
chunk is not in the top 5.

**Paired design:** each qualifying station appears twice, with identical question text.

| Station | Equipment | P (clean) truth | N (`maintained_by` substituted) wrong targets |
|---|---|---|---|
| Saltmarsh | CP-55 | MP-113, MP-114 | MP-103, MP-104 |
| Thistlebank | DR-8 | MP-115, MP-116 | MP-109, MP-110 |
| Underbank | PS-19 | MP-117 | MP-120 |
| Westmere | VE-31 | MP-120 | MP-117 |
| Yarrow Cross | ES-14 | MP-121 | MP-122 |
| Ashfen | LT-77 | MP-122 | MP-121 |
| Beacon Rise | WS-5 | MP-123, MP-124 | MP-107, MP-108 |

P items are 4001–4007 and N items are 4008–4014, in table order. All targets' texts are readable: no mask.

## Conditions

- **C and D:** run live (loop 1.7). Groq `openai/gpt-oss-120b`, temperature 0, `reasoning_effort: low`.
- **D-gate, D-relation and C-relation:** exact replays. No extra model calls.
- **Repair:** remains deferred.

## Hypotheses

| Id | Hypothesis | Criterion |
|---|---|---|
| H1b | Contradicted readable relations are detected | ≥ 90% of **false claims on D's answer paths** labelled contradicted. If fewer than 6 false claims surface on D's paths, H1b is **inconclusive**, not met. |
| H2b | No over-rejection on the clean pairs | D-relation false rejection ≤ 1 of the P items D answered correctly |
| H3b | Graph-caused errors removed | D-relation graph-caused wrong answers on N ≤ half of D's (if D has none, H3b is inconclusive) |

Also reported for C-relation and D-gate:
- relation-level categories per claim (truth = clean graph);
- answer outcomes;
- paired P-versus-N comparison per station;
- tokens, calls, latency and check time.

**Conclusion rule.** Relation-level verification is judged reliable enough to move toward
production integration only if H1b, H2b and H3b hold. R2′ H2–H5 already held. Production
promotion remains the owner's decision.

## Offline validation (before any model call)

| Check | Requirement |
|---|---|
| W1 | Phase 2, R1, R1′, R2 and R2′ frozen and replay files unchanged; the verifier, schema, R2′ runner and scorers unchanged from the R2′ freeze; store files unchanged |
| W2 | Dataset: each N corruption contradicts the text (V5 logic); truth readable; no question reused from v1.1, R2 or R2′; P and N pairs have identical question text and distinct ids |
| W3 | The retrieval precondition holds for all 14 items, on the clean view (P) and the corrupted view (N) |
| W4 | V2 and V3 sweeps rerun with `relver-1.0` and identical to the committed R2′ sweep |
| W5 | Mock end-to-end run with no cache writes; distinct signatures from R2′; the hosted run refuses without the R2′b freeze |
