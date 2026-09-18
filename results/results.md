# Results: Complex PCT Attention vs Real Sigmoid Control

Single seed per variant (spec §5/§6.2). Second-seed follow-up only if headline cells land within noise.

## Setup (measured)

| | A (control) | B (PCT) |
|---|---|---|
| Params | 51.47M | 51.32M (0.29% mismatch) |
| LR (probe-selected) | TBD | same |
| Steps × batch × seq | TBD × 128 × 512 | same |
| Tokens/epoch (measured) | TBD | same |
| Wall-clock train | TBD | TBD |

## 6.1 Held-out ELBO

| Variant | val ELBO |
|---|---|
| A | TBD |
| B | TBD |

Gate-sum histograms (no starvation check): TBD.

## 6.2 Probes (n≈200/cell, 95% Wilson CI)

Chance: Copy 6.25%, NIAH 1.6%. Void: ≤2× chance. Discriminates: CIs do not overlap.

### Copy Memory (per-token acc / exact-match)

| K | win | A | B | call |
|---|---|---|---|---|
| 8 | 512 | TBD | TBD | TBD |
| 16 | 512 | TBD | TBD | TBD |
| 32 | 512 | TBD | TBD | TBD |
| 8 | 1024 | TBD | TBD | TBD |
| 16 | 1024 | TBD | TBD | TBD |
| 32 | 1024 | TBD | TBD | TBD |

### NIAH (retrieval acc)

| Depth | win | A | B | call |
|---|---|---|---|---|
| 0.1–0.9 | 512 | TBD | TBD | TBD |
| 0.1–0.9 | 1024 | TBD | TBD | TBD |

(Full per-cell tables in `results/probes_A.txt`, `results/probes_B.txt`.)

## 6.3 Generation samples

20 fixed prompts, 32 steps, 128 tokens: `results/samples/A.txt`, `results/samples/B.txt`.
Eyeballed read: TBD.

## Sensitivity rerun (§2 rule)

Triggered only if B loses on 6.2: B′ (dc=368, h=832 complex, ~60.4M). Outcome: TBD / not triggered.

## Limitations (pre-declared confounds)

Componentwise Re/Im SiLU distorts phase; B's budget sits in embedding+attention vs
A's FFN-heavy split; single seed; shared LR (probe disagreement reported if any).
No claim generalizes beyond this scale.
