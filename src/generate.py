"""Fixed-prompt sampling: first 20 val conversations, 32 steps, 128 tokens."""

import argparse
import logging
import os

import torch

from data import N_VAL, SEED, download_split, format_conversation, get_tokenizer
from diffusion import MASK_ID, sample
from model import build_model

logger = logging.getLogger(__name__)

N_PROMPTS, GEN_LEN, STEPS, CTX = 20, 128, 32, 256


def build_prompts(tok, n=N_PROMPTS, ctx=CTX):
    convs = download_split("test_sft", N_VAL, SEED)[:n]
    prompts = []
    for c in convs:
        ids = tok(format_conversation(c))["input_ids"][:ctx]
        ids = ids + [MASK_ID] * GEN_LEN
        prompts.append(torch.tensor(ids, dtype=torch.long))
    return prompts, convs


def generate(model, prompts, device, steps=STEPS):
    model.eval()
    outs = []
    with torch.no_grad():
        for p in prompts:
            ids = p.unsqueeze(0).to(device)
            mask = torch.zeros_like(ids, dtype=torch.bool)
            mask[0, -GEN_LEN:] = True
            ids[0, -GEN_LEN:] = MASK_ID
            outs.append(sample(model, ids, mask, steps=steps)[0].cpu())
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", default="A", choices=["A", "B"])
    ap.add_argument("--out", default="results/samples")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    device = torch.device(args.device)
    model = build_model(args.variant).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    tok = get_tokenizer()
    prompts, convs = build_prompts(tok)
    outs = generate(model, prompts, device)
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/{args.variant}.txt", "w") as f:
        for i, (c, o) in enumerate(zip(convs, outs)):
            gen = tok.decode(o[-GEN_LEN:].tolist())
            f.write(f"=== prompt {i} ===\n{format_conversation(c)[:500]}\n"
                    f"--- {args.variant} ---\n{gen}\n\n")
    print(f"wrote {args.out}/{args.variant}.txt")


if __name__ == "__main__":
    main()
