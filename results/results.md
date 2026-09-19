# Results: Complex PCT Attention vs Real Sigmoid Control

Single seed per variant (spec §5/§6.2). Kernel v11 (full 150M-token run at pre-flight LR 3e-4), 2×T4, COMPLETE in ~5.7h total.

## Setup (measured)

| | A (control) | B (PCT) |
|---|---|---|
| Params | 51.47M | 51.32M (0.29% mismatch) |
| LR (probe-selected) | 3e-4 | same, no disagreement |
| Steps × batch × seq | 2289 × 128 × 512 ≈ 150.0M tokens | same |
| Tokens/epoch (measured) | 48.5M train (94,746 rows); val 2.4M (4,740 rows) → ~3.1 epochs | same |
| Wall-clock train | ~1.7h | ~2.5h (parallel bottleneck) |
| Eval (sequential) | probes ~1.2h + samples | probes ~1.9h + samples |

LR probe (500 steps × 2 LR, spec §5 rule):

| LR | A val ELBO | B val ELBO |
|---|---|---|
| 3e-4 | 63.749 | 64.932 |
| 1e-3 | 64.093 | 64.980 |

Both variants prefer 3e-4 → use it, no disagreement caveat (B's margin is thin at 0.05, noted).

Throughput probe (50 steps): A 4.7 s/step, B 7.7 s/step.

## 6.1 Held-out ELBO

| Variant | val ELBO |
|---|---|
| A | 59.210 |
| B | 61.038 |

Both models learned (step-50 ELBO ~82.3/82.4 → ~59/61). A leads by ~1.8 — a first-order read only, under the single-seed caveat.

Gate-sum histograms (no starvation check): n/a — the `is_compiling` guard excludes gate logging from the compiled graph, so the histogram is sacrificed under compile. No starvation signal available either way.

## 6.2 Probes (n≈200/cell, 95% Wilson CI)

Chance: Copy 6.25%, NIAH 1.6%. Void: ≤2× chance. Discriminates: CIs do not overlap.

### Copy Memory (per-token acc / exact-match 0/200 everywhere)

| K | win | A | B | call |
|---|---|---|---|---|
| 8 | 512 | 0.00% [0.00, 0.24] | 0.00% [0.00, 0.24] | void, within noise |
| 16 | 512 | 0.00% [0.00, 0.12] | 0.00% [0.00, 0.12] | void, within noise |
| 32 | 512 | 0.00% [0.00, 0.06] | 0.00% [0.00, 0.06] | void, within noise |
| 8 | 1024 | 0.00% [0.00, 0.24] | 0.00% [0.00, 0.24] | void, within noise |
| 16 | 1024 | 0.00% [0.00, 0.12] | 0.00% [0.00, 0.12] | void, within noise |
| 32 | 1024 | 0.00% [0.00, 0.06] | 0.00% [0.00, 0.06] | void, within noise |

### NIAH (retrieval acc)

| Depth | win | A | B | call |
|---|---|---|---|---|
| 0.1 | 512 | 0.00% [0.00, 1.88] | 0.50% [0.09, 2.78] (1/200) | void, within noise |
| 0.25 | 512 | 0.00% [0.00, 1.88] | 0.50% [0.09, 2.78] (1/200) | void, within noise |
| 0.5 | 512 | 0.00% [0.00, 1.88] | 0.50% [0.09, 2.78] (1/200) | void, within noise |
| 0.75 | 512 | 0.00% [0.00, 1.88] | 0.50% [0.09, 2.78] (1/200) | void, within noise |
| 0.9 | 512 | 0.00% [0.00, 1.88] | 0.50% [0.09, 2.78] (1/200) | void, within noise |
| 0.1–0.9 | 1024 | 0.00% [0.00, 1.88] | 0.00% [0.00, 1.88] | void, within noise |

Every cell is void (both variants ≤ 2× chance) → "no resolving power", never "no difference". B's single hit per 512-cell is floor noise.

## 6.3 Generation samples

20 fixed prompts, 32 steps, 128 tokens: `results/samples/A.txt`, `results/samples/B.txt` (in kernel v11 output bundle; not yet pulled locally).
Eyeballed read: wash — both variants produce degenerate loops (`the the the…`, `and and…`, `her her…`). No qualitative difference at this scale; human judgment has nothing to discriminate.

## Sensitivity rerun (§2 rule)

Not triggered: the rule fires only if B *loses* on 6.2, and 6.2 is void, not a loss. B′ (dc=368, h=832) not run.

## Second seed (§6.2 trigger)

Hit: headline 512-window cells landed within noise, so a second full seed for both variants is the pre-approved follow-up. Not yet run.

## Limitations (pre-declared confounds)

Componentwise Re/Im SiLU distorts phase; B's budget sits in embedding+attention vs
A's FFN-heavy split; single seed; shared LR (no probe disagreement to report).
No claim generalizes beyond this scale.
