"""CPU smoke suite: param match, init RMS, RoPE identity, loss/overfit, probes, sampler."""

import math

import torch

from diffusion import MASK_ID, compute_loss, doc_ids_positions, sample
from model import _rope_apply, build_model, count_params
from probes import gen_copy, gen_niah, run_cell

torch.manual_seed(0)
N, B = 32, 2


def _batch():
    ids = torch.randint(0, 50200, (B, N))
    return ids, *doc_ids_positions(ids)


def test_param_match():
    pa, pb = count_params(build_model("A")), count_params(build_model("B"))
    print(f"A {pa/1e6:.2f}M B {pb/1e6:.2f}M mismatch {abs(pa-pb)/pa*100:.2f}%")
    assert abs(pa - pb) / pa < 0.005


def test_init_rms():
    ids, doc, pos = _batch()
    for v in ("A", "B"):
        m = build_model(v).eval()
        h = m.embed(ids) if v == "A" else m.embed(ids).view(B, N, 368, 2)
        with torch.no_grad():
            for b in m.blocks:
                h = b(h, doc, pos)
        rms = float(h.pow(2).mean().sqrt()) if v == "A" else float(
            h.pow(2).sum(-1).mean().sqrt())
        print(v, "final residual RMS", round(rms, 3))
        assert 0.05 < rms < 10  # order-unity at init, neither vanished nor exploded


def test_rope_identity_and_rotation():
    x = torch.randn(1, 2, 4, 8)
    pos0 = torch.zeros(1, 4, dtype=torch.long)
    assert torch.allclose(_rope_apply(x, pos0, 4), x, atol=1e-5)
    # B layout: interleaved (Re_d, Im_d) rotation == per-dim complex e^{iθ}
    re, im = torch.randn(2), torch.randn(2)
    z = torch.stack([re, im], -1).reshape(1, 1, 1, 4)  # hdc=2 flattened
    p = torch.tensor([[3]])
    got = _rope_apply(z, p, 2).reshape(2, 2)
    for d in range(2):
        th = 3 * 10000.0 ** (-d / 2)
        er, ei = re[d], im[d]
        assert abs(got[d, 0] - (er * math.cos(th) - ei * math.sin(th))) < 1e-5
        assert abs(got[d, 1] - (er * math.sin(th) + ei * math.cos(th))) < 1e-5


def test_loss_finite_and_overfit():
    ids, doc, pos = _batch()
    for v in ("A", "B"):
        m = build_model(v)
        opt = torch.optim.Adam(m.parameters(), lr=3e-4)
        l0 = float(compute_loss(m, ids, doc, pos, 0, 0, 0))
        assert math.isfinite(l0)
        for s in range(20):
            opt.zero_grad()
            compute_loss(m, ids, doc, pos, 0, s, 0).backward()
            opt.step()
        l1 = float(compute_loss(m, ids, doc, pos, 0, 0, 0))  # same mask as l0
        print(v, "loss", round(l0, 2), "->", round(l1, 2))
        assert l1 < l0


def test_probe_generators_and_decode():
    rng = torch.Generator().manual_seed(0)
    ids, mask, key = gen_copy(rng, 8, 512)
    assert len(ids) == 512 and mask.sum() == 8 and (ids[mask] == MASK_ID).all()
    ids, mask, key = gen_niah(rng, 0.5, 512)
    assert len(ids) == 512 and mask.sum() == 1 and ids[mask].item() == MASK_ID
    m = build_model("A").eval()
    r = run_cell(m, ("niah", 0.5, 512), n=2, steps=2)
    assert 0 <= r["acc"] <= 1 and len(r["ci"]) == 2


def test_sampler():
    m = build_model("A").eval()
    ids, doc, pos = _batch()
    tm = torch.zeros_like(ids, dtype=torch.bool)
    tm[:, -4:] = True
    noisy = ids.clone()
    noisy[tm] = MASK_ID
    out = sample(m, noisy, tm, steps=4)
    assert bool((out[tm] != MASK_ID).all()) and bool((out[~tm] == ids[~tm]).all())
