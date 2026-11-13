# Amendment 1 to the R2′ preregistration

Date: 2026-11-13. Amends [PREREGISTRATION_R2_PRIME.md](PREREGISTRATION_R2_PRIME.md)
(commit `78116bc5`, sha256 `15eff8b7…`). The original file is unchanged and remains the
record of what was registered first. This amendment records two changes to the
verifier's decision procedure. It also fixes the V4 denominator for reporting.

## Timeline (all times UTC)

| Event | Commit / record | Time |
|---|---|---|
| Original preregistration committed | `78116bc5` | 2026-11-13 06:28:18 |
| Verifier, schema, dataset, scorer and runner committed, **including D1 and D2** | `dd9caf5d` | 2026-11-13 06:35:59 |
| V2–V4 sweeps computed, once | `experiment_r2p_verifier_sweep.json`, `recorded_at` 06:36:39, `git_revision` `dd9caf5d` | 2026-11-13 06:36:39 |
| Sweep results committed | `ad12f843` | 2026-11-13 06:39:25 |
| Hosted model calls | none before this amendment | — |

D1 and D2 therefore preceded the V2–V4 sweeps and every model call. Both came from
design review of the verifier code, not from data: no sweep, replay or model output
existed when they were made. The verifier has not changed since `dd9caf5d`.

## D1 — Stricter completeness check

**Registered:** a contradicted entity may be dropped only if no other entity has a
supported or graph_only relation of the dropped claim's type to the same subject,
among received link targets and text-derived hops, and is missing from the answer.

**Amended:** the same condition applies, oriented the way the path walks from the anchor. In addition,
two kinds of entity count as missing fillers for the contradicted slot:
- the entities named by the contradicted hop's checked text, when that text belongs to the path's fixed side (for example, the true equipment an incident text names);
- the true neighbour that a station-order container gives.

**Effect:** drops can only become stricter, meaning more abstentions and never fewer.
Without D1, an adjacency answer holding one true neighbour and one wrong one would have been reduced to the
true neighbour alone, an incomplete answer, whenever the other true neighbour was absent from the prompts.

## D2 — Shortest-path-only claim resolution

**Registered:** enumerate simple paths of up to 3 hops. An entity is supported if any path has all hops
supported or graph_only, otherwise contradicted if any path has a contradicted hop,
otherwise unverified.

**Amended:** the same rules apply, but only to the **shortest** anchor-to-entity paths found.

**Effect:** paths ignore edge direction when testing connectivity. Without D2, a longer chain through
unrelated shared entities could have labelled a wrong entity supported and hidden a direct contradicted link.
An example is two incidents on the same line.

## V4 reporting denominator

The V4 sampler (seed 20261113, n = 120) drew one degenerate substitution. Draw 106
replaced Oakhurst's `next_station` target with Oakhurst itself. Because the question
names Oakhurst, it is an anchor and not an answer entity, so that draw tests nothing.
V4 is therefore reported over **119 valid corruptions: 119/119 contradicted, 0 supported**,
with the degenerate draw listed separately. The sweep is not rerun or changed.
No V4b has been performed. Any later V4b would be reported separately and would not
replace this result.

## Unchanged

Dataset `r2p-1.0`, schema `relschema-1.0`, scorer `r2p-score-1.0`, hypotheses H1–H5,
adoption rule, conditions and the no-tuning rule after the sweeps all stand as
registered.
