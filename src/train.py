"""Identical-hypers A/B training loop: MDLM loss, AMP, grad ckpt, cosine schedule."""

import argparse
import logging
import math
import os
import time
from contextlib import nullcontext

import torch
from torch.utils.checkpoint import checkpoint

from diffusion import compute_loss, doc_ids_positions
from model import build_model

logger = logging.getLogger(__name__)

MICRO, ACCUM, BATCH = 32, 4, 128  # effective batch 128
SEQ = 512
TARGET_TOKENS = 300_000_000
WARMUP = 230
CKPT_EVERY = 500
SEED = 42


def schedule(step, total, lr):
    if step < WARMUP:
        return lr * (step + 1) / WARMUP
    p = (step - WARMUP) / max(1, total - WARMUP)
    return lr / 10 + (lr - lr / 10) * 0.5 * (1 + math.cos(math.pi * p))


def ckpt_path(d, variant):
    return os.path.join(d, f"ckpt_{variant}.pt")


def save(d, variant, model, opt, scaler, step):
    os.makedirs(d, exist_ok=True)
    torch.save(
        {"step": step, "model": model.state_dict(), "opt": opt.state_dict(),
         "scaler": scaler.state_dict() if scaler else None},
        ckpt_path(d, variant),
    )


def load(d, variant, model, opt, scaler):
    p = ckpt_path(d, variant)
    if not os.path.exists(p):
        return 0
    ck = torch.load(p, map_location="cpu")
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    if scaler and ck["scaler"]:
        scaler.load_state_dict(ck["scaler"])
    logger.info("resumed %s at step %d", variant, ck["step"])
    return ck["step"]


def run_block(block, h, doc, pos):
    return checkpoint(block, h, doc, pos, use_reentrant=False)


def forward_ckpt(model, ids, doc, pos):
    if model.variant == "A":
        h = model.embed(ids)
        for b in model.blocks:
            h = run_block(b, h, doc, pos)
        return model.norm(h) @ model.embed.weight.T
    dc = model.cfg["dc"]
    h = model.embed(ids).view(*ids.shape, dc, 2)
    for b in model.blocks:
        h = run_block(b, h, doc, pos)
    h = model.norm(h)
    return h.reshape(*ids.shape, 2 * dc) @ model.embed.weight.T


def val_elbo(model, val, device, n_batches=8):
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(min(n_batches, len(val) // MICRO)):
            ids = val[i * MICRO : (i + 1) * MICRO].to(device)
            doc, pos = doc_ids_positions(ids)
            losses.append(float(compute_loss(model, ids, doc, pos, SEED, 10**6 + i, 0)))
    model.train()
    return sum(losses) / max(1, len(losses))


def train(args):
    torch.manual_seed(SEED)
    device = torch.device(args.device)
    train_rows = torch.load(os.path.join(args.data, "train.pt"))
    val_rows = torch.load(os.path.join(args.data, "val.pt"))
    total = args.steps or math.ceil(TARGET_TOKENS / (BATCH * SEQ))
    order = torch.randperm(len(train_rows), generator=torch.Generator().manual_seed(SEED))

    model = build_model(args.variant).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=1e-5)
    use_amp = args.amp in ("fp16", "bf16") and device.type == "cuda"
    dtype = torch.float16 if args.amp == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda") if use_amp and dtype == torch.float16 else None
    step = load(args.ckpt, args.variant, model, opt, scaler)
    model.train()

    t0, tokens = time.time(), 0
    while step < total:
        opt.zero_grad()
        for mb in range(ACCUM):
            idx = order[((step * BATCH + mb * MICRO) % len(order)) :][:MICRO]
            if len(idx) < MICRO:  # wrap
                idx = torch.cat([idx, order[: MICRO - len(idx)]])
            ids = train_rows[idx].to(device)
            doc, pos = doc_ids_positions(ids)
            for g in opt.param_groups:
                g["lr"] = schedule(step, total, args.lr)
            ctx = torch.amp.autocast("cuda", dtype) if use_amp else nullcontext()
            with ctx:
                from diffusion import forward_diffuse, sample_t, _rng, MASK_ID, EOS_ID

                rng = _rng(SEED, step, mb)
                t = sample_t(MICRO, rng=rng).to(device)
                with torch.no_grad():
                    noisy, mask = forward_diffuse(ids, t, rng)
                import torch.nn.functional as F

                logits = forward_ckpt(model, noisy, doc, pos).float()
                logits[..., MASK_ID] = -float("inf")
                if mask.any():
                    nll = F.cross_entropy(logits[mask], ids[mask], reduction="none")
                    w = (1.0 / t).unsqueeze(-1).expand_as(ids)[mask]
                    loss = (w * nll).mean() / ACCUM
                else:
                    loss = torch.zeros((), device=device)
            (scaler.scale(loss) if scaler else loss).backward()
        if scaler:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        step += 1
        tokens += BATCH * SEQ
        if step % 100 == 0:
            gates = [b.attn.last_gate_sum for b in model.blocks]
            logger.info("step %d loss %.3f tok/s %d gatesum %.2f",
                        step, float(loss) * ACCUM, int(tokens / (time.time() - t0)),
                        sum(gates) / len(gates))
        if step % CKPT_EVERY == 0:
            save(args.ckpt, args.variant, model, opt, scaler, step)
    save(args.ckpt, args.variant, model, opt, scaler, step)
    elbo = val_elbo(model, val_rows, device)
    logger.info("done %s steps=%d val_elbo=%.3f", args.variant, total, elbo)
    return elbo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="A", choices=["A", "B"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--ckpt", default="checkpoints")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=0)  # 0 = derive from 300M tokens
    ap.add_argument("--amp", default="fp16", choices=["fp16", "bf16", "none"])
    ap.add_argument("--mode", default="full", choices=["full", "throughput", "lrs"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.mode == "throughput":  # 50-step wall-clock probe
        args.steps = 50
        t0 = time.time()
        train(args)
        print(f"THROUGHPUT {(time.time()-t0)/50:.1f}s/step")
    elif args.mode == "lrs":  # 500-step x 2-LR probe
        for lr in (3e-4, 1e-3):
            args.lr, args.steps, args.ckpt = lr, 500, f"checkpoints_probe_{lr}"
            elbo = train(args)
            print(f"LR {lr} val_elbo {elbo:.3f}")
    else:
        train(args)


if __name__ == "__main__":
    main()
