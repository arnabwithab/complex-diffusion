"""MDLM-style masked-diffusion loss (ported logic) + LLaDA confidence sampler."""

import logging
import math

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

MASK_ID = 50257
EOS_ID = 50256
EPS = 1e-3  # t floor; 1/t weight bounded at 1000


def _rng(seed, step, mb):
    g = torch.Generator()
    g.manual_seed((seed * 1_000_003 + step * 1_009 + mb * 917) % 2**63)
    return g


def sample_t(n, eps=EPS, rng=None):
    """t in [eps, 1], antithetic pairing: second half mirrors first."""
    u = torch.rand(n, generator=rng)
    half = n // 2
    u[half:] = 1 - u[: n - half]
    return (1 - eps) * u + eps


def forward_diffuse(ids, t, rng, eos_id=EOS_ID, mask_id=MASK_ID):
    """Mask each row with prob 1-t; EOS never masked."""
    p = (1 - t).unsqueeze(-1)  # [B, 1]
    rand = torch.rand(ids.shape, generator=rng).to(ids.device)
    mask = (rand < p) & (ids != eos_id)
    return torch.where(mask, torch.tensor(mask_id, device=ids.device), ids), mask


def compute_loss(model, ids, doc_ids, pos, seed=0, step=0, mb=0):
    """1/t-weighted CE on masked positions only, SUBS, fp32."""
    dev = ids.device
    rng = _rng(seed, step, mb)
    t = sample_t(ids.shape[0], rng=rng).to(dev)
    with torch.no_grad():
        noisy, mask = forward_diffuse(ids, t, rng)
    logits = model(noisy, doc_ids, pos).float()  # fp32 loss guard
    logits[..., MASK_ID] = -float("inf")  # SUBS: kill predict-the-mask shortcut
    if not mask.any():
        return torch.zeros((), device=dev)
    nll = F.cross_entropy(logits[mask], ids[mask], reduction="none")
    w = (1.0 / t).unsqueeze(-1).expand_as(ids)[mask]
    return (w * nll).mean()


def doc_ids_positions(ids, eot_id=EOS_ID):
    """Doc-block ids + doc-relative RoPE positions; boundaries at EOT."""
    B, N = ids.shape
    is_eot = ids == eot_id
    doc = torch.cumsum(is_eot.int(), 1)
    doc = doc - doc[:, :1]  # start at 0
    pos = torch.zeros_like(ids)
    for b in range(B):
        p, last = 0, 0
        for i in range(N):
            pos[b, i] = p
            p = 0 if is_eot[b, i] else p + 1
    return doc, pos


@torch.no_grad()
def sample(model, ids, target_mask, steps=32):
    """Confidence unmask/remask (LLaDA-style), argmax for determinism.

    ids: [B, N] with MASK at fillable slots; target_mask: slots we may fill.
    Fills ~1/steps of remaining slots per step, most-confident first.
    """
    cur = ids.clone()
    doc, pos = doc_ids_positions(cur)
    remaining = target_mask & (cur == MASK_ID)
    total = int(remaining.sum())
    if total == 0:
        return cur
    per_step = max(1, math.ceil(total / steps))
    for _ in range(steps):
        open_slots = target_mask & (cur == MASK_ID)
        if not open_slots.any():
            break
        logits = model(cur, doc, pos).float()
        logits[..., MASK_ID] = -float("inf")
        conf, pred = logits.max(-1)
        conf = torch.where(open_slots, conf, torch.full_like(conf, -float("inf")))
        k = min(per_step, int(open_slots.sum()))
        _, idx = torch.topk(conf.view(-1), k)
        flat_cur, flat_pred = cur.view(-1), pred.view(-1)
        flat_cur[idx] = flat_pred[idx]
    return cur
