"""Copy Memory + NIAH probe generators, scorers, and eval runner.

Pinned ids: EOT 50256 (separator), MASK 50257, Copy A-P 32-47, filler '~' 93,
NIAH alphabet 0-63 excl 30 plus 64, cue '?' 30. Every sequence = 2 solved
demos + 1 test, joined by EOT. Demos visible; only test answers masked.
"""

import argparse
import logging
import math

import torch

from diffusion import MASK_ID, EOS_ID, sample

logger = logging.getLogger(__name__)

EOT, MASK = EOS_ID, MASK_ID
COPY_IDS = list(range(32, 48))  # 'A'-'P'
FILLER = 93  # '~'
NIAH_IDS = [i for i in range(64) if i != 30] + [64]
CUE = 30  # '?'

COPY_KS = (8, 16, 32)
NIAH_DEPTHS = (0.1, 0.25, 0.5, 0.75, 0.9)
WINDOWS = (512, 1024)
N_SAMPLES = 200


def join(parts):
    """Join demo/test instances with EOT separators (no trailing EOT)."""
    out = []
    for i, p in enumerate(parts):
        out += list(p)
        if i < len(parts) - 1:
            out += [EOT]
    return torch.tensor(out, dtype=torch.long)


def gen_copy(rng, k, window=512, n_demos=2, demo_fill=32):
    """Returns (ids, answer_mask, answer_ids). Test answer = MASK x k."""
    def instance(src, fill, solved):
        ans = list(src) if solved else [MASK] * k
        return list(src) + [EOT] + [FILLER] * fill + [EOT] + ans

    demo_len = 2 * k + demo_fill + 2
    fill = window - n_demos * demo_len - n_demos - (2 * k + 2)
    assert fill >= 0, f"window {window} too small for K={k}"
    parts, answers, key = [], [], None
    for d in range(n_demos + 1):
        src = [COPY_IDS[i] for i in torch.randint(0, 16, (k,), generator=rng)]
        solved = d < n_demos
        parts.append(instance(src, demo_fill if solved else fill, solved))
        if not solved:
            key = torch.tensor(src)
    ids = join(parts)
    assert len(ids) <= window, (len(ids), window)
    pad = window - len(ids)
    ids = torch.cat([ids, torch.full((pad,), EOT)])  # neutral pad, own doc
    mask = torch.zeros(window, dtype=torch.bool)
    ans_start = n_demos * (demo_len + 1) + (k + 1 + fill + 1)
    mask[ans_start : ans_start + k] = True
    assert (ids[mask] == MASK).all()
    return ids, mask, key


def gen_niah(rng, depth, window=512, demo_h=64):
    """Returns (ids, answer_mask, needle_id)."""
    others = lambda needle, n: [i for i in NIAH_IDS if i != needle]

    def instance(hay, needle, solved):
        return list(hay) + [CUE] + ([needle] if solved else [MASK])

    demo_len = demo_h + 2
    h = window - 2 * demo_len - 2 - 2
    assert h > 0
    parts, key = [], None
    for d in range(3):
        if d < 2:
            needle = NIAH_IDS[torch.randint(0, 64, (1,), generator=rng).item()]
            pool = others(needle, demo_h)
            hay = [pool[i] for i in torch.randint(0, 63, (demo_h,), generator=rng)]
            parts.append(instance(hay, needle, True))
        else:
            needle = NIAH_IDS[torch.randint(0, 64, (1,), generator=rng).item()]
            pool = others(needle, h)
            hay = [pool[i] for i in torch.randint(0, 63, (h - 1,), generator=rng)]
            at = min(h - 1, int(depth * h))
            hay.insert(at, needle)
            parts.append(instance(hay, needle, False))
            key = needle
    ids = join(parts)
    assert len(ids) <= window, (len(ids), window)
    ids = torch.cat([ids, torch.full((window - len(ids),), EOT)])
    mask = torch.zeros(window, dtype=torch.bool)
    ans_pos = 2 * (demo_len + 1) + (h + 1)
    mask[ans_pos] = True
    assert ids[mask].item() == MASK
    return ids, mask, key


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


def chance(cell):
    return 1 / 16 if cell[0] == "copy" else 1 / 64


def run_cell(model, cell, n=N_SAMPLES, seed=0, device="cpu", steps=32):
    kind, param, window = cell
    rng = torch.Generator().manual_seed(seed)
    tok_hit, seq_hit, tot = 0, 0, 0
    for _ in range(n):
        if kind == "copy":
            ids, mask, key = gen_copy(rng, param, window)
        else:
            ids, mask, key = gen_niah(rng, param, window)
        out = sample(model, ids.unsqueeze(0).to(device), mask.unsqueeze(0).to(device),
                     steps=steps)
        pred = out[0].cpu()[mask]
        if kind == "copy":
            tok_hit += int((pred == key).sum())
            seq_hit += int(bool((pred == key).all()))
            tot += len(key)
        else:
            seq_hit += int(pred.item() == key)
            tot += 1
    acc = (tok_hit if kind == "copy" else seq_hit) / tot
    lo, hi = wilson(acc, tot)  # per-token trials for copy, n for NIAH
    ch = chance(cell)
    return {"acc": acc, "exact": seq_hit / n, "n": n, "ci": (lo, hi),
            "chance": ch, "below_floor": acc <= 2 * ch}


def cells():
    return ([("copy", k, w) for k in COPY_KS for w in WINDOWS]
            + [("niah", d, w) for d in NIAH_DEPTHS for w in WINDOWS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", default="A", choices=["A", "B"])
    ap.add_argument("--n", type=int, default=N_SAMPLES)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    from model import build_model

    device = torch.device(args.device)
    model = build_model(args.variant).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    print(f"{'cell':22s} {'acc':>7s} {'ci95':>17s} {'floor':>5s}")
    for c in cells():
        r = run_cell(model, c, n=args.n, device=device)
        print(f"{str(c):22s} {r['acc']*100:6.2f}% "
              f"[{r['ci'][0]*100:5.2f},{r['ci'][1]*100:5.2f}] {str(r['below_floor']):>5s}")
    print("floor = per-variant acc <= 2x chance; cell void iff BOTH variants floor")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
