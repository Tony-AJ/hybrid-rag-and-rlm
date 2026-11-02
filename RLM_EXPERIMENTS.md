# RLM Experiments: Token Optimization on `meridian-transit-v1`

Research log for the bounded retrieve-and-reason loop (`rlm/loop.py`), evaluated by
`rlm/evaluate.py`. Each change below was measured on its own, against the same
queries, before the next one was made. Every number comes from a saved result file
listed in [Result files](#result-files).

## Setup (shared by every run below)

| Item | Value |
|---|---|
| Corpus | `meridian-transit-v1`, 160 chunks, 412 typed edges, fingerprint `63cf21ec4c8c…` |
| Dataset | 60 queries, version 1.1, sha256 `8f61e760d6ec…` (20 single-hop factual, 10 cause, 6 authorisation, 4 adjacency, 20 two-hop incident-to-procedure) |
| Retrieval | Frozen config `experiment_frozen_retrieval_config.json` (sha256 `0e944282aa60…`): variant `no_hop0_graph`, α 0.6, β 0.4, γ 0.1, seed_k 5, max_depth 2, top_k 5 |
| Model | Groq `openai/gpt-oss-120b`, temperature 0.0, max_tokens 512, verified against the provider's model list |
| Scoring | Deterministic, no LLM judge. Scorer `rlm-score-1.3`, evaluator `rlm-eval-1.1` |
| Loop bounds | max_steps 3, max_retrievals 3, top_k 5, max_evidence_chunks 12, max_grounding_chunks 4 |

Metrics: **accuracy** is an exact match of the extracted identifiers (procedure,
policy and equipment codes, station names) or of all answer predicates.
**Evidence coverage** is the share of ground-truth chunks that were in the
evidence the model read. Token counts are the provider's reported usage, summed
over all generation calls of a query.

## Headline result (all 60 queries)

| Configuration | Accuracy | Coverage | Input/query | Output/query | Gen. calls | Gen. latency |
|---|---|---|---|---|---|---|
| Loop 1.2: separate decide and answer calls, full grounding | 100% | 1.000 | 1,721 | 316 | 2.38 | 13.2 s |
| **Loop 1.7: merged calls, compact links, lean grounding** | **100%** | **1.000** | **869** (−50%) | **218** (−31%) | **1.4** | **6.0 s** |
| Loop 1.7 + `reasoning_effort: low` | **100%** | **1.000** | 870 | **103** (−67%) | 1.4 | 6.3 s |
| Loop 1.7, grounding off (diagnostic only) | 100% | 0.658 | 712 | 146 | 1.1 | 5.8 s |

Targets set at the start: input under 1,300 and output under 200 tokens per query,
with accuracy held at 100%. Lean grounding plus low reasoning effort meets all
three on the full set.

### By category (all 60 queries)

| Category | n | Lean: in / out | Lean + low effort: in / out | Grounding off: coverage |
|---|---|---|---|---|
| single-hop factual | 20 | 653 / 180 | 653 / 62 | 1.00 |
| 1-hop cause | 10 | 704 / 123 | 705 / 73 | 0.90 |
| 1-hop authorisation | 6 | 758 / 147 | 757 / 86 | 0.83 |
| 1-hop adjacency | 4 | 650 / 123 | 650 / 65 | 1.00 |
| 2-hop incident to procedure | 20 | 1,244 / 342 | 1,246 / 171 | 0.07 |

Accuracy was 100% in every category for all three loop 1.7 runs.

## What each change did (20 two-hop queries)

Two-hop questions ("Which maintenance procedures cover the fault that stopped
service at X?") cost the most and broke most often, so each step was gated on them.

| Run | Change from the previous row | Accuracy | Input | Output | Calls |
|---|---|---|---|---|---|
| Separate calls | (baseline for this table) | 100% | 2,253 | 419 | 3.0 |
| Merged, compact links | One call either answers or asks to search | 100% | 1,667 | 310 | 2.1 |
| Merged, full links | Same, with every edge listed | 100% | 1,691 | 310 | 2.0 |
| One step only | No follow-up search possible | **85%** | 1,376 | 263 | 1.9 |
| Merged, compact, lean grounding | Grounding re-checks against cited chunks only | 100% | 1,263 | 355 | 2.1 |

Findings:

1. **Re-sending evidence costs more than evidence size.** Each generation call
   carried the whole evidence block. Merging the decide and answer calls removed one
   call on most queries and cut input by 26%.
2. **Compact links barely matter.** Showing each connected pair of nodes once,
   instead of every edge and its inverse, saved 24 input tokens per query (1.4%).
   An earlier version that hid every link between chunks already in evidence
   dropped two-hop accuracy to 85%. It hid `equipment maintained_by MP-x, MP-y`,
   which is the one fact saying both procedures apply.
3. **Follow-up search is necessary.** Limiting the loop to one retrieval lost 3
   of 20 two-hop answers.
4. **The grounding call was the remaining duplication.** It fired on 19 of 20
   two-hop queries and re-sent the full evidence (about 815 tokens) to add one or
   two chunks. Lean grounding sends only the chunks whose links cited the new nodes
   plus the fetched chunks (about 456 tokens). The re-checked answer kept the same
   procedure codes in all 19 cases; only wording changed.
5. **Reasoning effort drives output tokens.** Most output tokens were gpt-oss
   reasoning, not answer text. `reasoning_effort: low` halved output again with no
   accuracy loss.

## What grounding contributes

With grounding off, accuracy stays at 100% but evidence coverage falls from 1.000 to
0.658, and to 0.07 on two-hop questions. The model answers from graph edges listed
in the evidence (`Links: maintained_by -> procedure_MP-101, procedure_MP-102`)
without reading the procedures themselves. Grounding is what makes the answer rest
on retrieved text rather than on edge labels alone. That is why grounding-off is
kept as a diagnostic and not as a candidate configuration.

## Scorer fixes made along the way

Several early "wrong" answers were correct answers the scorer could not read. They
were fixed before the experiments above, and every result file was re-scored with
the same scorer (`rlm-score-1.3`):

- Typographic dashes and spaces (`PL‑A` with U+2011, `Series 400` with U+202F)
  are folded to ASCII before matching.
- Chunk ids count as their entity: `procedure_MP-107` is MP-107,
  `station_kestrel_green` is Kestrel Green.
- An entity the question itself names ("either side of Alder Vale") is not counted
  as a wrong pick.

## Reproducing

```powershell
# Loop 1.7, lean grounding (headline configuration)
python -m rlm.evaluate --provider groq --model openai/gpt-oss-120b --corpus meridian-transit-v1 --grounding lean --resume
# + low reasoning effort
python -m rlm.evaluate --provider groq --model openai/gpt-oss-120b --corpus meridian-transit-v1 --grounding lean --reasoning-effort low --resume
# grounding off (diagnostic)
python -m rlm.evaluate --provider groq --model openai/gpt-oss-120b --corpus meridian-transit-v1 --grounding off --resume
# Two-hop comparisons: add --category 2-hop --no-cache-write, and --separate-decide, --full-links or --max-steps 1
```

`--resume` reuses per-query results stored in a cache file named by the run's
configuration signature, so a run stopped by the provider's daily quota continues
where it stopped. The signature covers corpus, dataset, retriever, provider, model,
prompt template hashes, loop version and bounds, evaluator version, and
`reasoning_effort` when set.

## Result files

| File | Run | Signature | Recorded (UTC) |
|---|---|---|---|
| `experiment_rlm_eval_meridian-transit-v1_groq_loop12_full60.json` | Loop 1.2, 60 queries | n/a (pre-signature cache) | 2026-11-02 04:06 |
| `experiment_rlm_eval_meridian-transit-v1_groq_lean_full60.json` | Loop 1.7 lean, 60 queries | `9d198654f6d2` | 2026-11-02 06:56 |
| `experiment_rlm_eval_meridian-transit-v1_groq_lean_low_full60.json` | Loop 1.7 lean + low effort | `0c7795b8eaf8` | 2026-11-02 07:10 |
| `experiment_rlm_eval_meridian-transit-v1_groq_off_full60.json` | Loop 1.7 grounding off | `c1809adbc506` | 2026-11-02 07:16 |
| `exp_baseline_2hop.json`, `exp_C_2hop.json`, `exp_Bfull_2hop.json`, `exp_A_2hop.json`, `exp_C_lean_2hop.json` | Two-hop comparisons above | n/a (`--no-cache-write`) | 2026-11-02 |

Each file carries a `reproducibility` block: git revision, corpus fingerprint,
dataset hash, frozen retrieval config hash, generation settings, prompt template
hashes, loop version and bounds.

## Limitations

- **One synthetic corpus, 60 queries, one model.** These results show the loop's
  behaviour on `meridian-transit-v1` with gpt-oss-120b. They are not evidence of
  general performance on other corpora or models.
- **Single runs.** Each configuration ran once at temperature 0. Hosted inference
  is not guaranteed to be deterministic, and latency in particular varied with
  provider load and per-minute rate-limit waits.
- **Partly cached rows.** In the loop 1.2 run, 20 of 60 rows came from an
  earlier same-configuration run; in the lean run, 16 of 60 did. Cached rows were
  produced under the identical signature.
- **Signature gap.** The run signature does not yet include `temperature` or
  `max_tokens`. Changing either with `--resume` would wrongly reuse cached rows.
  All runs above used the defaults (0.0 and 512).
- **`answer_changed`** in grounding steps compares raw text, so it flags rewording
  as a change. Identifier-level comparison is still to do.
