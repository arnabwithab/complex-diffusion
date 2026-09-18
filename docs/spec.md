# Spec: Complex-Attention Masked Diffusion LM — A/B Test

## 1. Goal

Test whether replacing real-valued attention with **complex-valued, phase-coherent (PCT-style, non-competing-gate) attention** changes behavior in a small **masked diffusion language model**, relative to a real-valued control of matched parameter count, trained identically.

This is an architecture-isolation experiment, not an attempt to build a useful chatbot. Success = a clean, reproducible signal (or lack of one) on the specific tasks below — not chat quality.

## 2. Two model variants

Both variants share the same block skeleton (RMSNorm → attention → residual; SwiGLU FFN; RoPE; non-causal) and **tied input/output embeddings**. They differ in attention substrate + gate, complex-valued norm/FFN, widths (512 real vs 368 complex), head width, and the output-head [Re; Im] concat — the isolation is at the recipe level, not at token-identical layers (confounds named in §10).

| Component | Control (A) | Experimental (B) |
|---|---|---|
| Attention substrate | real-valued | complex-valued (Q,K,V,output projections complex) |
| Gate | Apple sigmoid attention: α = σ(QKᵀ/√d_head + b), b init −log N; no softmax, no row-norm | PCT gate (Hioki §3.3.1): q̄ = q/‖q‖₂, k̄ = k/‖k‖₂ (Hermitian L2 norm over complex head dims); s = Re⟨q̄ᵢ, k̄ⱼ⟩ · √d_head ∈ [−√d, √d]; α = σ(s + b), b init −log N; no row-normalization; out = W_o Σⱼ αᵢⱼ vⱼ (real gate × complex value, complex-linear W_o) |
| Positional encoding | RoPE (base 10,000) | RoPE on the **interleaved** (Re, Im) view of complex Q/K: RoPE's pair (x_{2d}, x_{2d+1}) = (Re_d, Im_d) of the *same* complex dim, so each complex dim is rotated by e^{iθ_d}. Layout pinned as interleaved — NOT [Re; Im]-concat, which would rotate Re_d with Re_{d+1} and scramble phase. Rotation-identity asserted in smoke_test.py |
| Norm | RMSNorm | complex RMSNorm: z / sqrt(mean(|z|²) + ε) × complex gain |
| FFN | SwiGLU, h = 1408 | SwiGLU, complex-linear, componentwise SiLU on Re/Im, h = 320 complex |
| Output head | real hidden (512) → logits via tied embedding | complex hidden → [Re; Im] concat (736 real) → logits via tied embedding |
| Attention masking | non-causal (bidirectional), doc-blocked over packed sequences | same |

### Locked configs (parameter-matched)

| | A | B |
|---|---|---|
| Width | d = 512 (real) | dc = 368 complex (= 736 real = 512 × 1.41) |
| Layers / heads | 8 / 8 (head_dim 64 real) | 8 / 8 (head_dim 46 complex = 92 real) |
| FFN hidden | 1408 | 320 complex (= 640 real) |
| Embedding (tied) | 50,258 × 512 | 50,258 × 736 |
| Total params | 51.4M | 51.3M (0.2% mismatch) |

**Parameter fairness:** complex dim × 1.41 ≈ real dim, per the PCT paper's convention, with **totals matched to 0.2%** by tuning B's FFN width. At GPT-2 vocab size this forces a tradeoff — B's 1.41×-wider tied embedding absorbs parameter budget that A spends on FFN — so we keep dc×1.41 + exact totals + equal depth/heads, which leaves B's FFN proportionally smaller.

**Pre-declared decision rule:** if B loses on the probes (§6.2), rerun B as a *single-variable* sensitivity check — dc = 368 unchanged, h raised to 832 complex (FFN ratio 2.26 vs A's 2.75; real-cost ≈ A's). Only FFN allocation varies; total grows to ~60.4M (+17% vs A) and is reported alongside. Interpretation: if B′ still loses, the loss is robust to FFN allocation; if B′ wins, the original B loss was allocation-driven.

**Complex-op implementation:** complex tensors are stored as real tensors with a trailing `(..., 2)` axis; complex matmul via the Re/Im block form. This avoids fp16-autocast/complex-dtype CUDA pitfalls entirely.

**Gate bias note (pinned):** b init = −log 512 in both variants (N = packed row length, per the PCT/Apple convention). With doc-blocked packing a query typically sees fewer than 512 keys, so effective gate sums can sit below 1 on short docs — identical for both variants and paper-faithful, so no per-variant correction. train.py logs gate-sum histograms on real packed batches so any systematic starvation is visible before the full run.

**Complex initialization (pinned):** every complex linear (QKV, W_o, FFN) initializes Re and Im independently ~ N(0, 1/√(2·fan_in)) — complex weights then have E|w|² = 1/fan_in, the complex match of a real linear's N(0, 1/√fan_in) (Trabelsi-style variance matching). Complex RMSNorm gain init = 1 + 0i, norm ε = 1e-6 (inside the mean |z|²). Tied embedding: each stored real component ~ N(0, 0.02) in both variants. smoke_test.py asserts per-layer output RMS ≈ 1 at init for both variants.

## 3. Backbone architecture (both variants)

- LLaMA/Qwen-style block (RMSNorm, SwiGLU, RoPE) with causal mask removed — the same recipe LLaDA and Dream use, scaled down.
- Size: **51M params each** (locked; see §2 config table).
- Sequence length: 512 packed. Probes additionally evaluated at 1024 (free long-range extrapolation read; both variants will degrade — RoPE does not extrapolate natively — so the 1024 read is *relative*).
- Tokenizer: GPT-2 BPE + `[MASK]` added (id 50,257) → vocab 50,258.
- Training objective: masked diffusion — absorbing-state forward process, 1/t-weighted ELBO loss (MDLM parameterization, **ported verbatim** from kuleshov-group/mdlm `compute_loss`): t = (1−ε)·u + ε with u ~ U(0,1) and **ε = 1e-3** (verified against MDLM's master config `sampling_eps`; bounds the 1/t weight at 1000); MDLM-style antithetic pairing within each 32-row micro-batch; mask prob 1−t; cross-entropy on masked positions only, with **SUBS** — the [MASK]-id logit is set to −∞ where the likelihood is evaluated, killing the trivial predict-the-mask short-circuit (exact placement ported from MDLM's code, which is ground truth where this spec is ambiguous). All t/mask RNG derives from (seed, step, micro-batch) — resume-safe. Loss and 1/t weight computed in fp32 outside autocast.
- Inference: LLaDA-style confidence-based iterative unmask-and-remask sampling.

## 4. Hardware / venue (locked)

- **Primary venue: Kaggle 2×T4 (16GB each). Variants A and B train in parallel, one per GPU, in a single session (~3–5h training + ~30min eval).** Budget ~12–16h of the 30h weekly GPU allowance (worst-case metering).
- **Pre-flight before the full run commits:** a 50-step throughput probe validates the wall-clock estimate, then a 500-step × 2-LR (3e-4 / 1e-3) × both-variants probe (~1h, parallel) selects the training LR per the §5 rule.
- T4 is Turing — **no bf16** → fp16 AMP + gradient scaler; the 1/t-weighted loss stays in fp32 (overflow guard, since the weight can reach 1000× at small t).
- `--amp bf16` flag retained for the 8GB laptop fallback (sequential runs, ~3–4h each at this scale).
- Gradient checkpointing: on. Micro-batch 32 × grad-accum 4 = effective batch 128.
- Checkpoint every 500 steps + auto-resume (session-death insurance).
- 8-bit Adam dropped — 51M params fit fp32 AdamW states in <1GB; no bitsandbytes dependency.

## 5. Training data (locked)

- Source: **`HuggingFaceH4/ultrachat_200k`** (ungated; splits pinned): **train = `train_sft`** subsampled to **40,000 conversations (seed 42)**; **validation = 2,000 conversations from `test_sft`** (seed 42 — disjoint from train by split); the 20 generation prompts (§6.3) are the first 20 validation conversations.
- **Token yield measured after tokenization** (ultrachat averages ~1.5k tokens/conversation → expect ~55–65M tokens/epoch → ~5 epochs at 300M tokens, not the naive ~3). Step count derives from the measured count; measured tokens/epoch is reported in results.md.
- Preprocessing: format turns as plain text ("User: …\nAssistant: …"), strip markup artifacts, tokenize, split long conversations into ≤512-token chunks, greedy-pack into 512-token rows with EOS joins, no padding. **EOS (`<|endoftext|>`, id 50256) is never masked** — always a visible structural boundary. **RoPE positions are doc-relative** (reset at each packed document) so packed neighbors don't leak false distance cues; probes mirror both rules.
- **Doc-boundary block attention mask:** each token attends only within its own packed document. Without this, the model denoises tokens conditioned on unrelated neighboring conversations and train/generate mismatch appears.
- Fairness mechanics: identical data order, step count, and seeds for A and B; all training RNGs (data order, t, mask draws) derive from (seed, step). Single seed per variant this round; the second-seed trigger is pre-declared in §6.2.

### Locked training config (identical for A and B)

- ~2,300 steps (exact count = ceil(150M / (128 × 512)), set after the tokenized data is measured) × effective batch 128 × seq 512 ≈ 150M tokens (~3.8 epochs of the measured 39.5M-token train split).
- AdamW β = (0.9, 0.95), weight decay **1e-5** (aligned to PCT §3.6's optimizer protocol), grad clip 1.0.
- LR ∈ {3e-4, 1e-3}, selected by the §4 pre-flight probe: if both variants prefer the same LR, use it; if they disagree, use the LR with the lower mean val ELBO and report the disagreement as a caveat. Cosine decay to LR/10, 230-step linear warmup (protocol follows PCT §3.6: cosine + linear warmup + clip 1.0).

## 6. Evaluation suite

Three evals, run identically for A and B:

### 6.1 Held-out loss / ELBO
- Masked-diffusion ELBO (1/t-weighted loss) on the ultrachat_200k validation split, fixed t-seed.
- Primary sanity check: confirms both models are actually learning and gives a first-order comparison point. A model that fails here invalidates the 6.2/6.3 reads.

### 6.2 Synthetic long-range / positional probes: Copy Memory + NIAH
Self-authored generators (task *shapes* follow PCT §3.5; no code borrowed), deterministic seeds, ~200 samples per cell, windows 512 and 1024 (window includes demos).

**Shared format (pinned):** every probe sequence = **2 solved in-context demos + 1 test instance** joined by `<|endoftext|>` (id 50256 — the same structural boundary token the model sees in training; no new special tokens needed). Demos make the zero-shot format learnable in-context — the PCT paper *trained* on these formats, we don't; identical demos for both variants keep the comparison fair. Demos are fully visible; only the test instance's answer slots are `[MASK]`ed.

- **Copy Memory**: source alphabet = 16 single-char tokens "A"–"P" (ids 32–47; printable-ASCII byte b maps to id b−33 in GPT-2's ordering). Instance = K random source tokens, `<|endoftext|>`, filler "~" (id 93) × D, `<|endoftext|>`, answer region = the K source tokens repeated (visible in demos, `[MASK]`×K in the test instance). Sweep K ∈ {8, 16, 32}; D auto-fills the window. Score: per-token accuracy over answer slots + exact-sequence-match rate.
- **NIAH**: distractor alphabet = 64 fixed ids (0–63 excluding 30, plus 64). Needle value drawn from that alphabet, distractors from the remaining 63 values (needle value occurs exactly once); needle inserted at depth ∈ {0.1, 0.25, 0.5, 0.75, 0.9} of the test instance's haystack. Cue = "?" (id 30 — never an alphabet member), then 1 `[MASK]`. Demo haystacks are short (64 tokens) so the test haystack gets the window. Score: retrieval accuracy (exact needle id).

**Inference protocol (pinned, identical for both variants):** the same 32-step confidence-remasking procedure as §6.3, restricted to the answer slots, with **argmax (temperature-0) decoding** for determinism; early-stops once all answer slots are filled. The decode + score path is covered in smoke_test.py.

**Pre-declared decision rules (fixed before any GPU time):**
- Chance floors: Copy per-token = 1/16 = 6.25%; Copy exact-match = 16^−K (negligible); NIAH = 1/64 ≈ 1.6%.
- **Void rule:** a cell where *both* variants score ≤ 2× chance (Copy ≤ 12.5% per-token; NIAH ≤ 3.1%) is void — reported as "no resolving power", never as "no difference".
- **Discrimination rule:** a cell discriminates only if the 95% Wilson CIs (n ≈ 200) of A and B do not overlap; otherwise the gap is "within noise". All cells reported with Wilson CIs in results.md.
- **Second-seed trigger:** if the headline cells (512-window Copy/NIAH) land within noise, a second full seed for both variants is the pre-approved follow-up.

- This is the task category where PCT showed its largest published gap over both softmax and vanilla complex attention — the most direct test of whether the port carries over any of that advantage. (The PCT paper's own tables predict sigmoid-attention Control A collapses on NIAH and long Copy; reproducing that contrast at our scale is the headline read.)

### 6.3 Eyeballed generation samples
- Fixed set of 20 held-out ultrachat prompts, 32 denoising steps, 128 generated tokens, LLaDA-style confidence remasking, identical settings for both variants.
- Manually reviewed for fluency, coherence, and qualitative difference — human judgment is more informative than automated text metrics at this scale.

### Explicitly out of scope for this round
- **HumanEval / GSM8K / MMLU / other standard benchmarks**: at 51M params trained on a small chat subset, these will land at or near floor-noise for both variants and won't discriminate between them. Deferred to a future, larger-scale run.
- A dedicated multi-token-consistency probe (the "Prefix-A/Suffix-B" test): deferred for this round per current scope.

## 7. Deliverables

1. Model definitions for variant A (real, Apple sigmoid-gated attention) and variant B (complex, PCT-gated attention), parameter-matched (asserted < 0.5% in smoke test).
2. MDLM-style training loop (masked diffusion objective, fp16/bf16 AMP, gradient checkpointing, checkpoint/resume).
3. ultrachat_200k data pipeline (download/subsample/tokenize/pack + doc masks).
4. Copy Memory + NIAH synthetic eval generators and eval script.
5. Generation/sampling script for both variants on the fixed 20-prompt set.
6. `results.md` comparing A vs. B across 6.1–6.3, including param counts, wall-clock, single-seed limitation, and the §2 sensitivity-rule outcome if triggered.

### Repo layout (fresh minimal repo — MDLM loss ported, not forked)

```
complex-diffusion/
  src/
    model.py      # both variants, one shared backbone, attention block swapped
    diffusion.py  # MDLM loss (ported) + confidence unmask/remask sampler
    data.py       # ultrachat_200k: download/subsample/tokenize/pack + doc masks
    train.py      # AMP fp16/bf16, grad ckpt, cosine schedule, ckpt/resume, --variant --device
    probes.py     # Copy Memory + NIAH generators + eval
    generate.py   # fixed-prompt sampling for both checkpoints
    smoke_test.py # CPU: param counts (<0.5% A/B), init activation RMS ≈ 1 both variants, RoPE rotation identity, loss finite + 20-step overfit, probe generators + decode/score path, sampler
  kaggle/         # kernel metadata + push script (parallel A/B subprocesses, then joint eval)
  results/        # results.md, samples/
```

## 8. Explicit non-goals

- Not attempting to match or approach LLaDA/Dream-scale quality.
- Not claiming any result here generalizes to 7B+ scale without further testing.
- Not testing the consistency/self-contradiction hypothesis this round (deferred).

## 9. Sources / references

What each piece of the spec is drawn from, so it's clear what's being reimplemented vs. reused as-is.

### Masked diffusion objective & training recipe
- Sahoo, Arriola, Schiff, Gokaslan, Marroquin, Chiu, Rush, Kuleshov. **"Simple and Effective Masked Diffusion Language Models" (MDLM)**, NeurIPS 2024. arXiv: https://arxiv.org/abs/2406.07524
  Code (loss **ported** from `compute_loss`, not forked): https://github.com/kuleshov-group/mdlm
  — Source of the SUBS-parameterized, 1/t-weighted ELBO loss used in §3/§6.1.
- Nie et al., **LLaDA** ("Large Language Diffusion Models"). arXiv: https://arxiv.org/abs/2502.09992
  Code: https://github.com/ML-GSAI/LLaDA
  — Source of the confidence-based iterative unmask/remask sampling procedure (§3) and the "LLaMA-style backbone with causal mask removed" pattern (§3).
- Ye et al., **Dream 7B**. Project / code: https://github.com/HKUNLP/Dream
  — Informs backbone hyperparameter choices only (training from scratch here).
- ashishk1331, **MDLM-TinyShakespeare** (from-scratch small reference implementation, ~11.4M params, RoPE + RMSNorm + SwiGLU + sigmoid-gated non-causal attention). https://huggingface.co/ashishk1331/MDLM-TinyShakespeare
  — Direct structural template for the real-valued backbone in §2/§3.

### Complex-valued / phase-coherent attention
- Hioki, **"Complex-Valued Phase-Coherent Transformer" (PCT)**, 2026. arXiv: https://arxiv.org/abs/2605.10123
  — Source of the PCT gate used verbatim in §2 (**exact equations: §3.3.1** — L2-normalised complex Q/K, s = Re⟨q̄,k̄⟩·√d, sigmoid + bias init −log N, no row normalisation, real gate × complex value); source of the parameter-fairness convention (§2) and optimizer protocol (§5); source of the Copy Memory / NIAH task shapes (§3.5, §6.2).
- Eilers & Jiang, **"Building Blocks for a Complex-Valued Transformer Architecture"**, ICASSP 2023. arXiv: https://arxiv.org/abs/2306.09827
  — Source of complex-valued attention / complex normalization primitives underlying PCT's "vanilla complex" baseline; our complex-linear projections / complex RMSNorm in §2 draw on the same building blocks.
- Trabelsi et al., **"Deep Complex Networks"**, ICLR 2018. arXiv: https://arxiv.org/abs/1705.09792
  — Foundational complex-valued NN building blocks (complex conv/norm/init) that the above papers build on.
- Ramapuram et al., **"Theory, Analysis, and Best Practices for Sigmoid Self-Attention"**, ICLR 2025. arXiv: https://arxiv.org/abs/2409.04431
  Code: https://github.com/apple/ml-sigmoid-attention
  — Source of the real-valued sigmoid-gated attention used as-is in variant A (§2), including the −log N bias init (same init PCT uses). Before freezing model.py, cross-check the 1/√d scale, bias placement, and any norm placement against the apple/ml-sigmoid-attention code — it is ground truth for Control A.

### Positional encoding
- Su et al., **RoFormer / RoPE**, Neurocomputing 2024 (orig. 2021). arXiv: https://arxiv.org/abs/2104.09864
  — Positional encoding used unchanged in both variants; applied to the (Re, Im) real view of complex Q/K for B, which reduces to per-dim rotation by e^{iθ}.

### Data
- **`HuggingFaceH4/ultrachat_200k`** (HF Hub) — ShareGPT-style multi-turn conversational dataset, cleaned/dedup'd, chosen over raw ShareGPT scrapes for reliable hosting and cleanliness (§5).
- GPT-2 BPE tokenizer (50,257 vocab) — standard HF `gpt2` tokenizer as-is, plus one added `[MASK]` token for the absorbing-state diffusion process (§3).

## 10. Decision log (implementation-round decisions)

- **Scale 51M, not 95M/123M:** fits one Kaggle session with margin; both variants + eval in a single run.
- **Kaggle 2×T4 over 8GB laptop:** A/B train in parallel (halves wall-clock), 16GB/GPU headroom, laptop stays free. Costs fp16-only on T4 (no bf16) → fp32 loss guard, `--amp bf16` fallback for laptop.
- **~150M tokens/variant:** ~3.8 epochs of the 40k-conversation subsample (39.5M measured); fits one Kaggle session at measured 5–8 s/step. Downscoped from 300M after the pre-flight measured throughput (7–11.5 s/step with checkpointing).
- **Tied embeddings:** vocab matrix counted once; makes A/B total-param matching clean at GPT-2 vocab size.
- **Fresh minimal repo, ported MDLM loss:** the loss is ~20 lines to port; forking inherits MDLM's Lightning/data scaffolding wired to their datasets and tokenizers.
- **dc=368/h=320 for B (convention-faithful, totals exact) with pre-declared dc=320/h=832 sensitivity rerun if B loses:** the param-fairness tradeoff at large vocab is unavoidable (§2); the rule makes the outcome interpretable either way.
- **Single seed per variant:** first round; both variants share data/order/t-step seeds. Multi-seed deferred unless the result is within noise.
- **Probe zero-shot fix (adversarial-review blocker):** 2 in-context demos baked into every probe sequence + pre-declared void (≤2× chance), discrimination (Wilson CI non-overlap), and second-seed rules — a both-at-floor outcome reads as "no resolving power", never as "no difference". Every probe token id is pinned (EOT separator, A–P source alphabet, "~" filler, byte-level NIAH alphabet, "?" cue); no undefined tokens remain.
- **Loss port pinned (blocker):** SUBS mechanism + ε = 1e-3 t-floor (1/t weight bounded at 1000) + fp32 loss. Verified against MDLM's master config (`sampling_eps: 1e-3`, `antithetic_sampling: True`); MDLM's code stays ground truth where this spec is ambiguous.
- **Complex init pinned (blocker):** Re/Im ~ N(0, 1/√(2·fan_in)) per complex linear, gain 1 + 0i, norm ε 1e-6, embedding components N(0, 0.02); init activation RMS asserted in smoke test.
- **RoPE layout pinned (blocker-adjacent):** interleaved (Re, Im) pairing = per-dim e^{iθ} rotation; the concat layout would scramble phase. Rotation-identity test in smoke test.
- **wd 1e-5 + pre-flight LR probe:** aligns the optimizer with PCT §3.6; the 500-step × 2-LR × both-variants probe resolves the same-LR fairness challenge without a per-variant sweep (identical hypers preserved).
- **Sensitivity rerun is single-variable:** h-only change (dc stays 368); the earlier dc = 320 / h = 832 fallback varied two things at once.
- **EOS never masked; doc-relative RoPE positions:** removes the boundary/packing ambiguity for both training and probes.
- **Named confounds (for results.md limitations):** componentwise Re/Im SiLU is a phase-distorting choice (vs modSiLU-type alternatives) — partially covered by the FFN sensitivity rerun; B's params sit more in embedding+attention vs A's FFN-heavy allocation; single seed in round one; same LR for both variants (probe-reported if they disagree).
