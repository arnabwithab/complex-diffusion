"""Both model variants: shared LLaMA-style backbone, attention block swapped.

Variant A: real-valued, Apple sigmoid attention.
Variant B: complex-valued (trailing (..., 2) Re/Im axis), PCT gate.
Tied input/output embeddings. Non-causal, doc-blocked attention.
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

VOCAB = 50258
SEQ = 512
BIAS_INIT = -math.log(512)  # -log N, N = packed row length (spec section 2)
ROPE_BASE = 10000.0
NORM_EPS = 1e-6

CONFIGS = {
    # d/hd: real dims; dc/hdc/ffn_c: complex dims for B
    "A": {"d": 512, "layers": 8, "heads": 8, "hd": 64, "ffn": 1408},
    "B": {"dc": 368, "layers": 8, "heads": 8, "hdc": 46, "ffn_c": 320},
}


def count_params(model):
    return sum(p.numel() for p in model.parameters())


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + NORM_EPS) * self.gain


class ComplexRMSNorm(nn.Module):
    """z / sqrt(mean(|z|^2) + eps) * complex gain, gain init 1+0i."""

    def __init__(self, d):
        super().__init__()
        self.gain = nn.Parameter(torch.zeros(d, 2))
        self.gain.data[:, 0] = 1.0

    def forward(self, z):
        # z: [..., d, 2]
        mag2 = z.pow(2).sum(-1, keepdim=True)  # |z|^2 per dim -> [..., d, 1]
        norm = torch.sqrt(mag2.mean(-2, keepdim=True) + NORM_EPS)
        zn = z / norm
        gr, gi = self.gain[:, 0], self.gain[:, 1]
        out_re = zn[..., 0] * gr - zn[..., 1] * gi
        out_im = zn[..., 0] * gi + zn[..., 1] * gr
        return torch.stack([out_re, out_im], -1)


class ComplexLinear(nn.Module):
    """Complex y = Wx, Re/Im block form. Re/Im ~ N(0, 1/sqrt(2*fan_in))."""

    def __init__(self, in_f, out_f):
        super().__init__()
        std = 1.0 / math.sqrt(2 * in_f)
        self.wr = nn.Parameter(torch.randn(out_f, in_f) * std)
        self.wi = nn.Parameter(torch.randn(out_f, in_f) * std)

    def forward(self, z):
        # z: [..., in_f, 2] -> [..., out_f, 2]
        xr, xi = z[..., 0], z[..., 1]
        return torch.stack(
            [xr @ self.wr.T - xi @ self.wi.T, xr @ self.wi.T + xi @ self.wr.T], -1
        )


_ROPE_CACHE = {}
ROPE_MAX = 2048  # covers 512 train / 1024 probe doc-relative positions


def _rope_tables(n_pairs, device=None):
    # fp32 tables, cached per (pairs, device); indexed by pos (no rebuilds,
    # compile-safe: static shapes after first trace).
    key = (n_pairs, str(device))
    t = _ROPE_CACHE.get(key)
    if t is None:
        inv = 1.0 / (ROPE_BASE ** (torch.arange(0, n_pairs) / n_pairs))
        fr = torch.outer(torch.arange(ROPE_MAX).float(), inv)
        t = (fr.cos(), fr.sin())
        _ROPE_CACHE[key] = t
    c, s = t
    return c.to(device), s.to(device)


def _rope_apply(x, pos, n_pairs):
    # x: [B, H, N, hd]; pairs (2d, 2d+1); pos: [B, N] doc-relative
    cos, sin = _rope_tables(n_pairs, x.device)
    c, s = cos[pos].to(x.dtype), sin[pos].to(x.dtype)  # [B, N, pairs]
    c, s = c[:, None, :, :], s[:, None, :, :]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


def _doc_allowed(doc_ids):
    return (doc_ids[:, None, :] == doc_ids[:, :, None]).float()  # [B, N, N]


class SigmoidAttention(nn.Module):
    """Apple sigmoid attention: alpha = sigmoid(QK^T/sqrt(dh) + b). No row-norm."""

    def __init__(self, d, heads, hd):
        super().__init__()
        self.heads, self.hd = heads, hd
        self.qkv = nn.Linear(d, 3 * heads * hd)
        self.wo = nn.Linear(heads * hd, d)
        self.bias = nn.Parameter(torch.full((heads,), BIAS_INIT))
        self.last_gate_sum = None

    def forward(self, x, doc_ids, pos):
        B, N, _ = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = _rope_apply(q, pos, self.hd // 2), _rope_apply(k, pos, self.hd // 2)
        s = q @ k.transpose(-1, -2) / math.sqrt(self.hd)
        alpha = torch.sigmoid(s + self.bias[None, :, None, None])
        alpha = alpha * _doc_allowed(doc_ids)[:, None]
        if not torch.compiler.is_compiling():
            self.last_gate_sum = alpha.sum(-1).detach().float().mean()
        out = (alpha @ v).transpose(1, 2).reshape(B, N, -1)
        return self.wo(out)


class PCTAttention(nn.Module):
    """PCT gate section 3.3.1: unit-norm q/k, s = Re<qbar,kbar>*sqrt(d), sigmoid+b."""

    def __init__(self, dc, heads, hdc):
        super().__init__()
        self.heads, self.hdc = heads, hdc
        self.qkv = ComplexLinear(dc, 3 * heads * hdc)
        self.wo = ComplexLinear(heads * hdc, dc)
        self.bias = nn.Parameter(torch.full((heads,), BIAS_INIT))
        self.last_gate_sum = None

    def forward(self, z, doc_ids, pos):
        # z: [B, N, dc, 2]
        B, N, _, _ = z.shape
        qkv = self.qkv(z).view(B, N, 3, self.heads, self.hdc, 2).permute(2, 0, 3, 1, 4, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, N, hdc, 2]
        # RoPE on interleaved (Re, Im) view: pair = (Re_d, Im_d) of same complex dim
        qr = _rope_apply(q.flatten(-2), pos, self.hdc).unflatten(-1, (self.hdc, 2))
        kr = _rope_apply(k.flatten(-2), pos, self.hdc).unflatten(-1, (self.hdc, 2))
        qn = qr / torch.sqrt(qr.pow(2).sum((-1, -2), keepdim=True) + NORM_EPS)
        kn = kr / torch.sqrt(kr.pow(2).sum((-1, -2), keepdim=True) + NORM_EPS)
        # Re<qbar_i, kbar_j> via contracted einsum (never materialize N x N x hdc)
        s = torch.einsum("bhic,bhjc->bhij", qn.reshape(B, self.heads, N, -1),
                         kn.reshape(B, self.heads, N, -1))
        s = s * math.sqrt(self.hdc)  # [B, H, N, N]
        alpha = torch.sigmoid(s + self.bias[None, :, None, None])
        alpha = alpha * _doc_allowed(doc_ids)[:, None]
        if not torch.compiler.is_compiling():
            self.last_gate_sum = alpha.sum(-1).detach().float().mean()
        # real gate x complex value; einsum over keys
        out = torch.stack(
            [
                torch.einsum("bhin,bhnc->bhic", alpha, v[..., 0]),
                torch.einsum("bhin,bhnc->bhic", alpha, v[..., 1]),
            ],
            -1,
        )  # [B, H, N, hdc, 2]
        return self.wo(out.permute(0, 2, 1, 3, 4).reshape(B, N, -1, 2))


class BlockA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1, self.n2 = RMSNorm(cfg["d"]), RMSNorm(cfg["d"])
        self.attn = SigmoidAttention(cfg["d"], cfg["heads"], cfg["hd"])
        d, f = cfg["d"], cfg["ffn"]
        self.gate, self.up, self.down = nn.Linear(d, f), nn.Linear(d, f), nn.Linear(f, d)

    def forward(self, x, doc_ids, pos):
        x = x + self.attn(self.n1(x), doc_ids, pos)
        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class BlockB(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dc, f = cfg["dc"], cfg["ffn_c"]
        self.n1, self.n2 = ComplexRMSNorm(dc), ComplexRMSNorm(dc)
        self.attn = PCTAttention(dc, cfg["heads"], cfg["hdc"])
        self.gate, self.up, self.down = (
            ComplexLinear(dc, f),
            ComplexLinear(dc, f),
            ComplexLinear(f, dc),
        )

    @staticmethod
    def _csilu(z):
        return torch.stack([F.silu(z[..., 0]), F.silu(z[..., 1])], -1)

    def forward(self, z, doc_ids, pos):
        z = z + self.attn(self.n1(z), doc_ids, pos)
        h = self.n2(z)
        g, u = self._csilu(self.gate(h)), self.up(h)
        # complex product g*u
        act = torch.stack(
            [g[..., 0] * u[..., 0] - g[..., 1] * u[..., 1],
             g[..., 0] * u[..., 1] + g[..., 1] * u[..., 0]],
            -1,
        )
        return z + self.down(act)


class DiffusionLM(nn.Module):
    def __init__(self, variant="A"):
        super().__init__()
        assert variant in ("A", "B")
        self.variant = variant
        cfg = CONFIGS[variant]
        self.cfg = cfg
        if variant == "A":
            d = cfg["d"]
            self.embed = nn.Embedding(VOCAB, d)
            nn.init.normal_(self.embed.weight, std=0.02)
            self.blocks = nn.ModuleList([BlockA(cfg) for _ in range(cfg["layers"])])
            self.norm = RMSNorm(d)
        else:
            dc = cfg["dc"]
            self.embed = nn.Embedding(VOCAB, 2 * dc)  # [Re; Im] storage, N(0, 0.02) each
            nn.init.normal_(self.embed.weight, std=0.02)
            self.blocks = nn.ModuleList([BlockB(cfg) for _ in range(cfg["layers"])])
            self.norm = ComplexRMSNorm(dc)

    def forward(self, ids, doc_ids, pos):
        if self.variant == "A":
            h = self.embed(ids)
            for b in self.blocks:
                h = b(h, doc_ids, pos)
            return self.norm(h) @ self.embed.weight.T
        dc = self.cfg["dc"]
        z = self.embed(ids).view(*ids.shape, dc, 2)
        for b in self.blocks:
            z = b(z, doc_ids, pos)
        z = self.norm(z)
        h = z.reshape(*ids.shape, 2 * dc)  # [Re; Im] concat
        return h @ self.embed.weight.T


def build_model(variant="A"):
    m = DiffusionLM(variant)
    logger.info("%s params: %.2fM", variant, count_params(m) / 1e6)
    return m
