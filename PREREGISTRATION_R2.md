# Preregistration: R2, incorrect graph relationships

Status: **fixed before implementation and before any model call.**
Branch `phase3-r2`, created from `phase3-r1-results` (`82f38bba`). The corpus, document text, graph files,
dataset v1.1, frozen retrieval config, loop 1.7, the Phase 2 result and the R1/R1′ results
are not modified.

## Question

When the knowledge graph contains a relation the source documents contradict, do
A (vector), B2 (graph-only), C (hybrid), D (hybrid + RLM, loop 1.7) and D-gate
answer from the documents, follow the wrong edge, or decline? Does D-gate catch
graph/document disagreement?

## Corruption layer

- **In memory, per query.** For each query the loaded graph is copied and that
  query's corruption spec is applied with networkx edge operations on the copy.
  The `GraphDatabase` API, which auto-persists, is never called. Retrievers are rebuilt
  on a stand-in service whose `graph_db.graph` is the copy. The vector store
  is shared and read-only. The stand-in refuses `hybrid_search` and every
  `GraphDatabase` method that writes to disk.
- **Only relation types the text can verify:**
  - `caused_by`: the incident text names the equipment code.
  - `maintained_by`/`maintains`: the procedure text says "Applies to the <code>".
  - `next_station`/`previous_station`: the line text lists stations in order.

  `authorised_by` and `performed_by` are excluded: no document states them.
- **Forward and reverse edge change together** when a reverse type exists (`maintains`,
  `previous_station`). `caused_by` has no reverse edge in this graph.
- **Operations:**
  - `substitute`: remove the true target edges, add edges to wrong targets of the same node type.
  - `add_spurious`: keep the true edges and add a wrong one.
- R2e/R2f stack the R1 mask (`MaskedRetriever`) on top: the true targets' text is unavailable.

## Cases (32 queries, ids 2001–2032; dataset `r2-1.0`)

| Case | n | Corruption | Answerable | Expected answer |
|---|---|---|---|---|
| R2a cause | 6 | `caused_by` substituted | yes: incident text names the true code | true equipment code |
| R2b two-hop | 8 | `maintained_by` substituted (same count of wrong procedures) | yes: true procedure text says "Applies to <code>" | true procedure codes |
| R2c adjacency | 6 | one `next_station` substituted to a station on another line | yes: line text gives the order | both true neighbours |
| R2d two-hop, spurious | 6 | one extra `maintained_by` added | yes | true set only |
| R2e unanswerable, wrong target readable | 4 | true procedures masked, edge substituted to a readable wrong procedure | no | abstain |
| R2f unanswerable, wrong target unavailable | 2 | true procedures masked, edge substituted to a procedure that is also masked | no | abstain |

Items:

| Case | Station in question | Source node | True targets | Corrupt targets |
|---|---|---|---|---|
| R2a | Harbour Row, Brickworks, Eastgate Quay, Hollow Mill, Larkspur, Pennyford | incident_2026_01, 04, 07, 10, 14, 18 | TX-440, PD-90, CP-55, TC-6, WS-5, PD-90 | AX-3, SG-12, LT-77, BR-7, HV-21, VE-31 |
| R2b | Harbour Row, Kestrel Green, Alder Vale, Copper Lane, Garnet Hill, Ironside, Kingsmoor, Larkspur | TX-440, HV-21, BR-7, SG-12, PS-19, VE-31, LT-77, WS-5 | their procedures | MP-107/108, MP-111/112, MP-113/114, MP-101/102, MP-120, MP-121, MP-117, MP-115/116 |
| R2c | Alder Vale, Garnet Hill, Mallow Street, Saltmarsh, Netherby, Ironside | the station | its following station | Glasshouse, Ashfen, Redcliff, Ironside, Fallowfield, Underbank |
| R2d | Brickworks, Dunmore Park, Eastgate Quay, Foundry Bridge, Hollow Mill, Jubilee Fields | PD-90, AX-3, CP-55, DR-8, TC-6, ES-14 | kept | + MP-119, MP-105, MP-122, MP-103, MP-110, MP-124 |
| R2e | Mallow Street, Netherby, Oakhurst, Pennyford | TX-440, HV-21, BR-7, PD-90 | masked | MP-105, MP-117, MP-120, MP-113 |
| R2f | Quarry Side, Redcliff | SG-12, AX-3 | masked | MP-122, MP-101 (also masked) |

Questions use the v1.1 wording: "Which component failed when trains were held at X?",
"Which maintenance procedures cover the fault that stopped service at X?", "Which
stations sit immediately either side of X along its own line?". If offline
validation shows an item cannot satisfy its checks, it is replaced by the next
station in the same case **before freezing**, and the replacement is recorded.

## Conditions

| Condition | Run |
|---|---|
| A, B2, C | loop 1.7, one step, grounding off (Phase 2 bounds), on the corrupted view |
| D control | loop 1.7, three steps, lean grounding, on the corrupted view |
| D-gate | `gate-1.0` applied offline to D control's recorded rows (exact, as in R1′) |

B is excluded (abstains by construction).

## Scoring (`rlm/robustness_scoring_r2.py`, `r2-score-1.0`)

Each answer gets one outcome; the first rule that matches wins:

1. `correct`: answerable, and the named set equals the document truth with no corrupt entity named.
2. `correct_abstention`: unanswerable, declines, and names no corrupt entity.
3. `mixed`: names all true entities plus at least one corrupt entity.
4. `graph_misled`: names a corrupt entity but not the full true set.
5. `over_abstention`: answerable but declined.
6. `unsupported_answer`: names an identifier absent from the text the model received.
7. `other_wrong`.

Flags reported with every answer:
- `graph_supported`: a corrupt entity was named and appeared in a `Links:` line the model received.
- `wrong_target_text_received`: the corrupt target's own text was in the evidence.
- `document_supported`: true entities were named and appeared in received text.
- `identifier_supported`: the R1 identifier check.
- Strict and lenient abstention.

Abstention detection and received-evidence parsing are R1's (`r1-score-1.0`), imported, not edited.

## Measures

**Retrieval, no LLM (A, B2, C):**
- Recall@5 of document-truth chunks.
- Rate at which a corrupt target lands in the top 5.

**Answers:**
- Every outcome, as k/n per case and per condition.
- The four flags above.
- Evidence coverage of document-truth chunks.
- Input and output tokens, generation and retrieval calls, latency.

## Predictions (descriptive; no adoption rule)

| Id | Prediction |
|---|---|
| P1 | A's retrieval rankings on R2 queries are identical to its rankings on the clean graph (A never reads the graph). |
| P2 | C and D are graph-misled (outcomes 3 or 4) on at least half of R2b and at least half of R2c. |
| P3 | D-gate passes graph-misled answers whenever the corrupt target's text was received (R2b, R2d, R2e), and abstains on R2f. |
| P4 | D is graph-misled less often than C on R2b. |

If P3 holds, the identifier gate does not detect graph/document disagreement. A
relation-level check would be a separate, preregistered R2′.

## Offline validation (all must pass before any hosted call)

| Check | Requirement |
|---|---|
| G1 | Graph file, corpus files and vector store hashes unchanged before and after; the base graph's edge set unchanged after every query |
| G2 | Hashes match for dataset v1.1, corpus fingerprint, frozen config, Phase 2 frozen result, R1 frozen result and R1′ result files |
| G3 | On the clean graph, every `caused_by`, `maintained_by` and `next_station` edge agrees with document text |
| G4 | Every corrupted edge contradicts document text; every removed true edge agrees with it |
| G5 | Corruption keeps forward and reverse edges consistent; same node type; corrupt codes absent from the source entity's text and from the question's station text |
| G6 | Document truth is readable for R2a–R2d; R1's V3–V5 checks pass for the R2e/R2f masks |
| G7 | An empty spec gives identical rankings on all v1.1 queries and reproduces the Phase 2 A, B, C and B2 metrics |
| G8 | The corrupted view is visible to every graph reader (`relations()`, BFS candidates, B2 aliases) |
| G9 | Scorer unit tests cover every outcome and flag |
| G10 | Corruption-layer unit tests on a toy graph (substitute, add_spurious, reverse edges, combined with a mask, refusal of persisting calls) |
| G11 | Mock-provider end-to-end run with no cache writes; distinct signatures and cache names per condition; the hosted run refuses to start without a matching freeze record |

Signatures cover the experiment id `robustness-r2`, dataset sha256, corruption-spec
sha256, mask sha256, architecture, retriever record, bounds, loop version, prompt
hashes, gate version, provider, model and generation options.
