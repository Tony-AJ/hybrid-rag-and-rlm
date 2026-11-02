# Preregistration: R1′, evidence-aware RLM

Status: **draft for review. No implementation, no model calls.**
Written after R1 was frozen (`experiment_robustness_r1_meridian-transit-v1_groq_frozen.json`,
sha256 `941f0d4a…`, commit `77081267`). R1's results and interpretation stay as recorded.
Whether graph relationships are acceptable evidence is treated here as a separate,
explicitly named evidence policy, not as a reason to reinterpret R1.

## 1. Problem, from the loop 1.7 code

The grounding path ([rlm/loop.py:384-422](rlm/loop.py#L384-L422)):

```python
grounding = cited_link_targets(run.answer, evidence)[: bounds.max_grounding_chunks]
fetched = [chunk for chunk in (retriever.fetch(node_id) for node_id in grounding) if chunk]
if fetched:
    ...  # one grounding call, then a step record
```

| Defect | Effect seen in R1 |
|---|---|
| A fetch that returns `None` is silently dropped | All 8 D answers on R1c kept their link-only codes, with no trace entry |
| If every fetch fails, no grounding step is recorded at all | The D trace is indistinguishable from an ordinary answer |
| A fetch with empty text (`text == ""`) counts as success | The model would be "grounded" on an empty chunk (not hit in R1, but possible) |
| Targets beyond `max_grounding_chunks` (4) are dropped | Silently unverified citations |
| When some fetches fail and others succeed, grounding runs on the successful ones only | The final answer can still name the unfetched targets |
| `STEP_TEMPLATE` offers only `ANSWER:` or `SEARCH:` | On R1a/R1b, D searches until the step budget runs out (1,748 input tokens per query vs C's 657) |

## 2. Evidence policy under test

**Endpoint-text policy.** A graph edge may support a relation (for example
`next_station`, `maintained_by`) only if the document text of every cited endpoint
is available and was read by the model. Concretely: every identifier (procedure,
policy or equipment code, station name) that the final answer names, and that the
question does not, must appear in chunk text the model received, counting text
fetched during grounding. `Links:` lines alone are not sufficient.

- This permits adjacency answers: neighbouring station names appear in their own
  station chunks, which grounding fetches.
- This rejects R1c answers: the masked procedure or policy text cannot be fetched.
- It does not verify that the text *states* the relation. That stricter policy
  (text must state the relation) is out of scope; it cannot be checked
  deterministically here.
- Numbers are excluded from the gate. Number matching by value is too loose to
  act on; numbers are still reported by the R1 scorer.

## 3. Variants

All are new code. `rlm/loop.py` (loop 1.7) is not modified.

| Variant | Failed-fetch trace | Sufficiency gate | `ABSTAIN:` in step prompt | Model calls |
|---|---|---|---|---|
| D (control) | no | no | no | none: frozen R1 / Phase 2 rows |
| D-gate | yes | yes | no | **none**: exact offline replay (see 5) |
| D-abstain | yes | no | yes | yes |
| **D-ea (primary)** | yes | yes | yes | yes |

A, B2 and C are controls only: their frozen R1 and Phase 2 rows are reused, not rerun.

### 3.1 Failed-fetch detection (all new variants)

Every grounding attempt writes one step record, even when nothing could be fetched:

```json
{"step": "grounding", "mode": "lean",
 "cited_from_links": ["procedure_MP-101", "procedure_MP-102"],
 "fetched": [],
 "fetch_failures": [{"node_id": "procedure_MP-101", "reason": "unavailable"},
                    {"node_id": "procedure_MP-102", "reason": "unavailable"}],
 "truncated": [],
 "link_only_claims": ["procedure_MP-101", "procedure_MP-102"]}
```

The reasons are:
- `unavailable`: `fetch` returned nothing.
- `empty_text`: the fetched chunk has no usable text, i.e. its text is empty after stripping whitespace.
- `truncated`: the target was cited beyond `max_grounding_chunks`.

### 3.2 Sufficiency gate (D-gate, D-ea)

The gate is deterministic, with no LLM call, and runs after grounding.

1. `named` = identifiers in the final answer, minus those in the question.
2. `supported` = the identifiers found in the text of all chunks the model received, including fetched chunks. `Links:` lines are excluded.
3. If `named − supported` is not empty, the gate replaces the answer with `INSUFFICIENT EVIDENCE` and records the result:
   `{"step": "gate", "decision": "abstain", "unsupported": [...], "reasons": {...}, "answer_before_gate": "..."}`.
4. Otherwise the gate passes the answer through unchanged and records `"decision": "pass"`.

### 3.3 Explicit abstention (D-abstain, D-ea)

A new step template (hashed separately) adds one line:

> If the evidence cannot answer the question and another search would not help,
> reply ABSTAIN: followed by one short reason.

When the model replies `ABSTAIN:`, the final answer becomes `INSUFFICIENT EVIDENCE` and the reason is stored in the trace. `GROUND_TEMPLATE` and `ANSWER_TEMPLATE` are unchanged, which keeps the change to a single prompt.

## 4. Datasets

| Set | n | Purpose |
|---|---|---|
| R1 `r1-1.0` (frozen) | 24 | Abstention when evidence is missing |
| v1.1 (frozen) | 60 | **Over-rejection**: answerable questions, including the 20 two-hop, 6 authorisation and 4 adjacency questions, which depend on graph structure |

There are no new queries and no changes to either dataset.

## 5. Offline replay of D-gate (exact, no model calls)

The gate runs after the loop and changes nothing before it. Applying it to the recorded D traces is therefore identical to running D-gate. The inputs it needs are:
- the final answer;
- the prompts the model received (recorded in the R1 caches);
- fetch outcomes, which are deterministic and taken from the local graph and mask.

**Phase 2 D rows did not record prompts.** For those rows, the received text is rebuilt from the recorded evidence ids and fetched ids using the same `format_evidence`. Because retrieval is deterministic (reproduction checks passed), the rebuilt text matches what the model received. The rebuild is checked first: on R1 rows, the rebuilt text must equal the recorded prompts' evidence text.

## 6. Hypotheses and decision rules (fixed before any run)

These rules are compared on raw counts. With n = 8 to 24 per comparison, they are decision thresholds, not significance tests.

| Id | Hypothesis | Success criterion |
|---|---|---|
| H1 | D-ea abstains on R1c | ≥ 7/8 correct abstentions (control D: 0/8) |
| H2 | D-ea keeps R1a/R1b abstention | ≥ 15/16 (control D: 15/16) |
| H3 | D-ea does not over-reject answerable questions | v1.1 accuracy ≥ 58/60 (control D: 60/60), **and** adjacency 4/4, **and** two-hop ≥ 19/20 |
| H4 | `ABSTAIN:` reduces D's cost on unanswerable questions | D-ea mean input tokens on R1a+R1b below control D's (reported with the ratio) |
| H5 | The gate alone fixes R1c | D-gate R1c ≥ 7/8 (offline) |

**Adoption rule.** D-ea is recommended only if H1, H2 and H3 all hold. If H5 holds and D-abstain adds nothing beyond it, the cheaper D-gate is recommended instead. The cost trade-off is reported as Δ input tokens per query on v1.1 and on R1, and as Δ tokens per additional correct abstention. Adoption remains your decision.

**Expected limitation, stated in advance.** The gate checks identifiers, not relations. The R1 failure on 1008 (D answered about PL-K when asked about PL-Z) passes the gate, because PL-K is in the text. That miss is expected and is not counted against the gate.

## 7. Metrics

These are reported per variant, per dataset and per case, as raw counts with rates:

**Answer quality**
- correct abstention (strict and lenient)
- unsupported answers
- link-only answers (overall, and when the masked id was exposed)
- accuracy on v1.1
- over-rejection: v1.1 queries that control D answered correctly but the variant declined

**Trace counts**
- gate triggers (abstain / pass)
- fetch failures by reason
- `ABSTAIN:` decisions

**Cost**
- evidence coverage
- input and output tokens
- retrieval and generation calls
- latency

## 8. Implementation (after approval)

| File | Content |
|---|---|
| `rlm/loop_evidence.py` | `run_loop_ea(..., gate: bool, abstain_option: bool)`. Loop version `1.8-ea`, new template hashes, gate version `gate-1.0`. Reuses `loop.py` helpers by import; does not edit `loop.py`. |
| `experiments/robustness_r1_prime.py` | Runs D-abstain and D-ea on R1 and v1.1, runs the offline D-gate replay, and builds the comparison tables against the frozen controls. |
| `tests/test_loop_evidence.py` | Mock retriever and provider tests (see the list after this table). |

The tests check that:
- a masked fetch is recorded as `unavailable`;
- empty text is recorded as `empty_text`;
- a target past the limit is recorded as `truncated`;
- the gate abstains when a cited link target is unfetched;
- the gate passes when the endpoint text was fetched (the adjacency case);
- `ABSTAIN:` is parsed;
- **equivalence**: with `gate=False` and `abstain_option=False`, the trace and answer equal loop 1.7 on the mock for all 60 v1.1 and 24 R1 queries, apart from the added trace fields.

**Cache isolation.** Signatures include the experiment id `robustness-r1-prime`, the loop version, the template hashes, the gate version, the variant, and the dataset and mask hashes.

**Freeze.** Before any model call, run the offline checks (R1's 17 checks, the equivalence test and the replay rebuild check), then record the hashes of the dataset, loop, templates, gate, scorer and runner. The hosted-run guard refuses to start without the freeze record, as in R1.

## 9. Budget and order (Groq free tier: 200k tokens/day per organisation)

| Run | Estimated tokens |
|---|---|
| D-gate offline replay on R1 and v1.1 | 0 |
| D-ea on R1 (24) | ~35k |
| D-ea on v1.1 (60) | ~60k |
| D-abstain on R1 (24) | ~35k |
| D-abstain on v1.1 (60) | ~60k |

The runs go in the order listed. The offline replay comes first, because its result decides whether the prompt ablation is worth the tokens.
