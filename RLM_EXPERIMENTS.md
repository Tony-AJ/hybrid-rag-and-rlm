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

---

# Phase 2: Five-architecture comparison

Same corpus, dataset, frozen retrieval configuration, model and scorer as above;
only the retrieval architecture and the answer loop change. Code:
`experiments/architecture_benchmark.py` (`arch-bench-1.1`).

## Architectures

| Id | Name | Retrieval | Answer generation |
|---|---|---|---|
| A | Vector RAG | Production vector search, top 5 (`benchmark.extract_vector_ids`). No graph links in the prompt. | One call, no grounding |
| B | Graph RAG (vector-seeded) | Vector top 5 as seeds, graph expansion to depth 2, **seeds excluded** (`benchmark.graph_expansion_ids`). Measures what traversal adds, not a vector-free graph retriever. | One call, no grounding |
| B2 | Graph-only RAG | Query entity linking against 156 node aliases (station, line, procedure and policy names, incident numbers, codes), then outgoing traversal to depth 2. Ranked by hop, seed convergence, keyword overlap, node id. **No embeddings.** | One call, no grounding |
| C | Hybrid RAG | Frozen `no_hop0_graph` (α 0.6, β 0.4, γ 0.1, seed_k 5, depth 2, top 5) | One call, no grounding |
| D | Hybrid RAG + RLM | C's retriever | Loop 1.7: merged step, follow-up search, lean grounding (max_steps 3) |

A, B, B2 and C share one answer prompt and one call, so their answer differences
come from retrieval alone. D differs from C only in the loop.

B2 was designed and frozen (`experiment_architecture_freeze_b2.json`) before any
answer run and was not tuned on answer results. A depth-3 variant of B2 was
measured as a sensitivity check only; its retrieval metrics were identical to depth 2.

## Retrieval only (no LLM, 60 queries, k = 5)

| Metric | A | B | B2 | C |
|---|---|---|---|---|
| Precision@5 | 0.077 | 0.047 | 0.113 | **0.140** |
| Recall@5 | 0.375 | 0.233 | 0.500 | **0.633** |
| MRR | 0.348 | 0.131 | 0.375 | **0.417** |
| NDCG@5 | 0.354 | 0.157 | 0.410 | **0.472** |
| Mean latency | 15.7 ms | 9.7 ms | **0.5 ms** | 10.3 ms |

Recall@5 by category:

| Category | n | A | B | B2 | C |
|---|---|---|---|---|---|
| single-hop factual | 20 | 1.00 | 0.00 | 1.00 | 1.00 |
| 1-hop cause (see limitation below) | 10 | 0.00 | 1.00 | 0.00 | 0.90 |
| 2-hop incident to procedure | 20 | 0.03 | 0.00 | 0.00 | 0.00 |
| 1-hop authorisation | 6 | 0.33 | 0.67 | 1.00 | 0.83 |
| 1-hop adjacency | 4 | 0.00 | 0.00 | 1.00 | 1.00 |

Reproduction checks passed: A equals the saved vector baseline, B equals the saved
graph-expansion baseline, C equals the frozen config's `selected_metrics` (all four
metrics to 4 dp). A's prompts were checked to contain no `Links:` lines.

## Answers (60 queries)

| Metric | A | B | B2 | C | D |
|---|---|---|---|---|---|
| **Accuracy** | 0.550 | 0.033 | 0.700 | 0.933 | **1.000** |
| Abstention rate | 0.433 | 0.917 | 0.300 | 0.067 | **0.000** |
| Evidence coverage | 0.375 | 0.233 | 0.500 | 0.633 | **1.000** |
| Input tokens / query | **535** | 664 | 638 | 653 | 883 |
| Output tokens / query | 67 | **48** | 67 | 76 | 104 |
| Generation calls / query | 1.00 | 1.00 | 1.00 | 1.00 | 1.40 |
| Retrieval calls / query | 1.00 | 1.00 | 1.00 | 1.00 | 1.07 |
| Generation latency / query | **4.2 s** | 5.0 s | 4.8 s | 5.1 s | 7.2 s |

Accuracy by category:

| Category | n | A | B | B2 | C | D |
|---|---|---|---|---|---|---|
| single-hop factual | 20 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| 1-hop cause (see limitation below) | 10 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| 2-hop incident to procedure | 20 | 0.00 | 0.00 | 0.10 | 0.85 | **1.00** |
| 1-hop authorisation | 6 | 0.33 | 0.00 | 1.00 | 0.83 | 1.00 |
| 1-hop adjacency | 4 | 0.25 | 0.50 | 1.00 | 1.00 | 1.00 |

## Interpretation

1. **Two-hop questions separate the architectures.** No single retrieval puts the
   procedure chunks in its top 5. C still answers 17 of 20 because its evidence
   carries graph `Links:` lines naming the procedures. D answers 20 of 20 because it
   searches again and its grounding step reads the procedure chunks themselves
   (coverage 1.000).
2. **The RLM loop costs about 35% more input tokens than C** (883 vs 653), 0.4
   extra generation calls and about 2 s per query, for +6.7 points of accuracy and
   full evidence coverage.
3. **B2 is strong whenever the question names an entity** (100% on authorisation
   and adjacency) with sub-millisecond retrieval and no embedding model, but it
   cannot reach the procedures on two-hop questions within its top 5.
4. **B fails by construction.** Excluding the vector seeds removes the chunk that
   holds the answer to most questions; it abstains on 92% of them.

## Limitation: answer accuracy and retrieval recall measure different things

On the 10 cause questions ("Which component failed when trains were held at X?"),
A and B2 answer **10 of 10 correctly with Recall@5 = 0.00**.

Checked against the corpus: each question's only labelled relevant chunk is the
equipment chunk (for example `equipment_TX-440`, path
`incident_2026_01 -caused_by-> equipment_TX-440`). But the incident chunk itself
names the code: *"…trains were held at Harbour Row on the Azure Line for 11 minutes
after a TX-440 traction inverter suffered a phase imbalance trip."* This holds for
all 10 questions, and A and B2 retrieve the incident chunk for all 10. The answer is
available one hop earlier than the labels assume.

Consequences:

- **Recall@5 measures retrieval of the annotated chunk; accuracy measures whether
  the answer could be produced.** They diverge whenever the answer appears in a
  chunk that is not labelled relevant. Neither metric is wrong; they measure
  different things.
- **The cause category does not test 1-hop graph retrieval as intended.** Its
  Recall@5 figures (A 0.00, B 1.00, B2 0.00, C 0.90) say whether the equipment
  chunk was ranked, not whether graph traversal was needed to answer.
- **Do not use the cause category for claims about retrieval quality** until its
  relevance labels are reviewed: either also label the incident chunk relevant, or
  rewrite the incident text so the code appears only in the equipment chunk.
  Either change makes a new dataset version and must not be applied retroactively
  to this frozen comparison.
- Evidence coverage uses the same labels, so A's and B2's coverage (0.375, 0.500)
  understates what their evidence contained on this category.

## Other limitations

- **One synthetic corpus, 60 queries, one model, one run at temperature 0.** No
  confidence intervals. With n = 4 (adjacency) or 6 (authorisation), one question
  moves category accuracy by 17 to 25 points.
- **The 60 queries are not an independent test set.** The frozen hybrid
  configuration and the loop were developed on them. B2 was frozen before its answer
  run, but its alias rules were written with knowledge of this corpus. A second
  dataset is needed for an unbiased comparison.
- **Two sessions.** The first 13 queries (14 for A and B) were answered in one
  session and the rest in a resumed session after a provider daily-quota stop, under
  the same per-architecture cache signatures and the same benchmark code (file
  sha256 `0ff7f3f355bf…`, committed in `d2a7588f`).
- **`git_revision` does not record uncommitted changes.** The working tree for
  `experiments/`, `rlm/` and `app/` matched `d2a7588f` when the result was written
  (checked with `git diff HEAD`).
- **Latency** depends on Groq load and per-minute rate-limit waits; compare it only
  within this run.

## Provenance (validated)

Each hash below was recomputed from the files on disk after the run and matched the
value recorded in the result file.

| Item | Value |
|---|---|
| Git revision | `d2a7588f85732bbdced922f5f02247a311ec49d5` (HEAD; no uncommitted code changes) |
| Recorded | 2026-11-02 09:20:52 UTC |
| Benchmark version | `arch-bench-1.1`, `experiments/architecture_benchmark.py` sha256 `0ff7f3f355bf91e735098a81921e01a97ee815b6b2a63d8065463e30c3a6d862` |
| Corpus | `meridian-transit-v1`, fingerprint `63cf21ec4c8c04ad83977ebb9c029b6a1c26ea19c345a74587175aded99e97f1` (vector, graph and hybrid reproduction checks passed against it) |
| Dataset | `benchmark_corpora/meridian/dataset.json` v1.1, sha256 `8f61e760d6ec9b1ca95f8bf955bfb33698851caa8e61cf7ff0d3b1f758a54fb2` |
| Frozen retrieval config | `experiment_frozen_retrieval_config.json`, sha256 `0e944282aa60426b9b7c358a2797c33573b29bb0574dc9b8a959b1d57334690d` |
| Model | Groq `openai/gpt-oss-120b`, verified against the provider model list; temperature 0.0, max_tokens 512, `reasoning_effort: low` |
| Prompt template sha256 | answer `2fd1b101…`, decide `29895476…`, step `0d0a2745…`, ground `23b0ea16…` (match current `rlm/loop.py`) |
| Loop | version 1.7; A/B/B2/C `max_steps 1, grounding off`; D `max_steps 3, grounding lean`, merged decide and answer |
| Scorer / evaluator | `rlm-score-1.3` / `rlm-eval-1.1` |
| Platform | Python 3.12.10, Windows 11 |

## Phase 2 result files

| File | Content |
|---|---|
| `experiment_architecture_benchmark_meridian-transit-v1_groq_phase2_frozen.json` | **Frozen Phase 2 result, read-only. Cite this one.** sha256 `c70e8a416b1259387c8cf47bf47a415aefa4812ca2e4bd93ad1a0687f7a31407` |
| `experiment_architecture_benchmark_meridian-transit-v1_groq.json` | Identical at freeze time; any later run of the script rewrites it |
| `experiment_architecture_cache_meridian-transit-v1_groq_{A_a132400acaea,B_d3b35eb3e60c,B2_2bbdb456ec50,C_07c5e314a62b,D_8713432e7d64}.json` | Per-architecture answer caches; signature prefix in the name |
| `experiment_architecture_retrieval_frozen_b2.json`, `experiment_architecture_freeze_b2.json` | B2 retrieval results and freeze record, taken before the answer run |

Reproduce (with the caches present, cached rows are re-scored without model calls):

```powershell
python -m experiments.architecture_benchmark --provider groq --model openai/gpt-oss-120b --reasoning-effort low --resume
```
