# Preregistration: R2′, relation-level verification

Status: **fixed before implementation, before any verifier sweep and before any model call.**
Branch `phase3-r2-prime` from `phase3-r2` (`07a8c99f`). Phase 2, R1, R1′ and R2 code,
data and results, `rlm/loop.py` (1.7), `rlm/loop_evidence.py` (`gate-1.0`),
`experiments/graph_corruption.py` (`corrupt-1.0`) and production files are not modified.

## Question

R2 showed that `gate-1.0` passes answers whose identifiers appear in received text even
when that text contradicts the relation (22 of 22). Can a deterministic
subject–relation–object verifier tell supported, contradicted and unverified relations
apart, using only document text, without rejecting valid answers?

## Verifier (`rlm/relation_verifier.py`, `relver-1.0`; deterministic, no model calls)

**Schema-driven.** `benchmark_corpora/meridian/robustness/relation_schema.json` (`relschema-1.0`) declares:
- **Entity types and labels.** Each type has an id prefix and a label rule: code from the id; station name from the station text's lead; line name; incident number `YYYY/NN`.
- **For each relation:** its subject and object types, a check kind, and the witness endpoint.
- **An exhaustiveness flag:** whether the witness text names every entity of the counterpart type it relates to.

No relation phrase appears in verifier code.

| Check kind | Relations | Rule |
|---|---|---|
| `endpoint_mention` | `caused_by`, `occurred_at`, `affected_line` (witness: incident); `maintained_by` (witness: procedure) / `maintains`; `serves` / `served_by` (witness: line); `site_of_incident` (witness: incident); `supersedes` / `superseded_by` (witness: superseding procedure, **non-exhaustive**) | **supported**: the witness text mentions the counterpart's label. **contradicted**: only if the witness is exhaustive for that slot, its text is available, and it names at least one *other* entity of the counterpart type but not this one. **unverified**: otherwise, including when the witness text is unavailable (masked) or names no entity of that type. |
| `sequence` | `next_station` (forward), `previous_station` (backward) | The container is the line reached through `served_by`. The container text's order comes from a schema-declared extractor; this is a data-format adapter, not verifier code. **supported**: the object is immediately after (forward) or before (backward) the subject in the subject's line order. **contradicted**: the subject's line text is available and lists the subject, but the adjacent position holds a different station or none. **unverified**: the line text is unavailable. |
| `not_text_verifiable` | `authorised_by`, `authorises`, `performed_by`, `responsible_for` | **graph_only**: permitted, with provenance recorded |

Witness text is fetched through the same per-query retriever D used (corruption and masks
applied). It is used for checking only and is never shown to the model.

**Claim extraction.**
1. *Anchors* are the entities the question names.
2. *Answer entities* are the entities the answer names, minus the anchors.
3. *Hops* come from two sources:
   - every `Links:` edge in the prompts the model received, with its verification status;
   - text-derived relations between pool entities (anchors, received nodes, link targets, answer entities, entities mentioned in received text) that the verifier finds **supported**.
4. For each answer entity, enumerate simple paths from an anchor, at most 3 hops, ignoring edge direction for connectivity and verifying each edge in its stored direction. **Every edge on a path is verified.**
5. Entity status:
   - **supported** if some path has all hops supported or graph_only; it is labelled graph_only if any hop is graph_only;
   - otherwise **contradicted** if some path contains a contradicted hop;
   - otherwise **unverified**, which includes having no path.

**Decision policy** (approved):
1. The answer abstains, or has no answer entities: **keep**.
2. Any answer entity is unverified: **abstain** (`INSUFFICIENT EVIDENCE`).
3. Every answer entity is supported or graph_only: **keep**.
4. Some are contradicted:
   - The *terminal* entities that remain must be non-empty and complete. Terminal means not an intermediate on another answer entity's path. Complete means no other entity has a supported or graph_only relation of the dropped claim's type to the same subject, among received link targets and text-derived hops, and is missing from the answer.
   - If so, **drop**, and the answer becomes `Verified: <labels>.`
   - Otherwise **abstain**.

Every decision records its claims, statuses, witness ids, paths and action.

## Conditions

| Condition | Run | Extra model calls |
|---|---|---|
| C | loop 1.7, one step, grounding off | — |
| D | loop 1.7, three steps, lean grounding | — |
| D-gate | `gate-1.0` replayed on D's recorded prompts | 0 |
| **D-relation** (primary) | `relver-1.0` replayed on D's recorded prompts and answers. Exact: it runs after the loop and changes nothing before it. | 0 |
| C-relation | `relver-1.0` replayed on C | 0 |

D-relation-repair is deferred. Model: Groq `openai/gpt-oss-120b`, temperature 0,
max_tokens 512, `reasoning_effort: low`.

## Dataset `r2p-1.0` (42 queries, ids 3001–3042)

These are new questions. None repeats an R2 item or a v1.1 question's wording. Corruption
uses `corrupt-1.0`, unchanged; masking uses R1's `MaskedRetriever`.

| Case | n | Items (anchor: truth, corruption) |
|---|---|---|
| **P** valid, clean | 16 | cause, "Which component failed in service incident 2026/NN?": 21 CP-55, 22 DR-8, 23 PS-19, 24 TC-6 · one-hop, "Which maintenance procedures apply to the <code> <name>?": AX-3 (111, 112), CP-55 (113, 114), DR-8 (115, 116), WS-5 (123, 124) · adjacency, "On its own line, which station comes immediately before <X> and which comes immediately after it?": Copper Lane, Yarrow Cross, Hartsmere · authorisation (graph_only), "Which authorisation policy covers maintenance procedure <MP>?": MP-115 (PL-G), MP-117 (PL-H), MP-122 (PL-J) · two-hop, "Which maintenance procedures cover the equipment that failed in service incident 2026/NN?": 19 (SG-12: 109, 110), 20 (AX-3: 111, 112) |
| **N** contradicted | 16 | cause: 25 VE-31→BR-7, 26 ES-14→TX-440, 27 LT-77→CP-55, 28 WS-5→PD-90 · one-hop maintenance: TC-6 (118, 119→103, 104), PS-19 (117→122), LT-77 (122→117), HV-21 (103, 104→109, 110) · adjacency (next_station substituted): Kestrel Green (Alder Vale→Ivory Dock), Foundry Bridge (Garnet Hill→Carrow End), Beacon Rise (Carrow End→Larkspur), Thistlebank (Underbank→Elmsworth) · two-hop, second hop corrupt: 21 (CP-55 maintained_by→MP-107, MP-108), 22 (DR-8→MP-120) · two-hop, **first hop corrupt** (caused_by): 23 (PS-19→VE-31; the path then reaches MP-120, which is truly VE-31's), 24 (TC-6→ES-14; reaches MP-121) |
| **S** extra edge | 4 | one-hop maintenance + one spurious `maintained_by`: SG-12 (+MP-104), TX-440 (+MP-116), VE-31 (+MP-108), PD-90 (+MP-122) |
| **U** insufficient | 6 | one-hop maintenance, unanswerable. Edge to a masked target, true targets masked: ES-14 (121→110), BR-7 (105, 106→124), TX-440 (101, 102→114). True edge, target masked: VE-31 (120), LT-77 (122), PS-19 (117). |

Answerable: P, N and S (36). Unanswerable: U (6). If an item fails offline validation
(V5), it is replaced before freezing and the replacement is recorded.

## Scoring (`rlm/robustness_scoring_r2p.py`, `r2p-score-1.0`)

**Answer level.** R2 outcome rules (`r2-score-1.0`, imported unchanged): correct, correct_abstention, mixed, graph_misled, over_abstention, unsupported_answer, other_wrong. Also:
- **graph-caused error rate:** wrong answers with `wrong_source` links_only or links_and_text, over all answers.
- **false rejection:** D (or C) correct, but the variant did not keep a correct answer; reported for P and overall.

**Relation level.** Reported separately from the answer level. For every claim the verifier labels, truth is membership in the clean graph's edge set. The clean graph agrees with document text for every text-verifiable relation (offline check V2).

| | Supported | Contradicted | Unverified | graph_only |
|---|---|---|---|---|
| True claim | correct | false contradiction | missed support | permitted |
| False claim | false support | contradiction detected | safe miss | false allowance |

Reported, with raw counts and per relation type:
- verification accuracy over text-verifiable claims;
- contradiction detection rate (contradicted over false claims);
- false contradiction rate (contradicted over true claims);
- unverified rate.

**Also reported:** correct abstention on U; evidence coverage; input and output tokens, model calls and latency (verifier time separately).

## Hypotheses and adoption rule

| Id | Hypothesis | Criterion |
|---|---|---|
| H1 | D-relation detects false relations D relied on | ≥ 90% of false claims on D's answer paths labelled contradicted |
| H2 | No over-rejection | false rejection ≤ 1 of the P items D answered correctly |
| H3 | Fewer graph-caused errors | D-relation graph-caused wrong answers on N+S ≤ half of D's |
| H4 | Abstains when evidence is missing | correct abstention ≥ 5/6 on U |
| H5 | No false contradiction of valid edges | 0 false contradictions over all clean-graph edges (offline V2) |

**Adoption:** D-relation is recommended only if H1–H5 all hold. Production promotion remains the owner's decision.

**Predicted risks:**
- **Adjacency may be over-rejected.** If D never received the line text and the witness fetch fails, the claim is unverified and the answer abstains. This counts against H2.
- **Authorisation is graph_only by policy.** It is permitted, not text-verified.
- **Identifier-based mentions cannot detect a wrong *kind* of relation** between two entities that the witness text co-mentions.

## Offline validation (all before any hosted call)

| Check | Requirement |
|---|---|
| V1 | Phase 2, R1, R1′ and R2 frozen and replay files unchanged; corpus fingerprint, dataset v1.1, frozen config, loop 1.7, `gate-1.0` and `corrupt-1.0` sources unchanged; store files unchanged after every phase |
| V2 | Clean-graph sweep: every edge of a text-verifiable relation labelled supported; every not-text-verifiable edge labelled graph_only (H5) |
| V3 | Exhaustive sweep: every single same-type wrong-object substitution of every clean edge, labelled and counted per relation type (detected, unverified, false support) |
| V4 | Seeded random corruptions (seed 20261113), excluding test-item pairs, through the corruption layer and full claim extraction: a corrupted claim is never labelled supported |
| V5 | Dataset: corruptions contradict the text and replaced edges agree (R2 G3–G5 logic); masks unretrievable, unfetchable and not reconstructable; no overlap with R2 or v1.1 questions; truth readable for P, N and S |
| V6 | Claim-extraction unit tests: link paths, text-derived hops, anchors, two-hop chains, every edge verified |
| V7 | Unit tests: every label, every decision (keep / drop / abstain, completeness), every scoring outcome |
| V8 | Replay determinism: `relver-1.0` over R2's frozen D rows twice gives identical decisions, and R2 files are byte-identical afterwards (plumbing only; not reported as an R2 result) |
| V9 | Mock-provider end-to-end run with no cache writes; distinct signatures and cache names; the hosted run refuses without a matching freeze record |

**No verifier, schema or decision-policy change after the V2/V3 sweeps are computed.** A defect found later is registered as a new version.
