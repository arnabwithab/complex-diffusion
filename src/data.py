"""ultrachat_200k pipeline: download/subsample/tokenize/chunk/greedy-pack.

Packed rows are plain token ids; doc structure is recoverable from EOS joins
at train time (diffusion.doc_ids_positions). Deterministic under seed 42, so
A/B see identical order.
"""

import argparse
import logging
import re

import torch

logger = logging.getLogger(__name__)

DATASET = "HuggingFaceH4/ultrachat_200k"
TRAIN_SPLIT, VAL_SPLIT = "train_sft", "test_sft"
N_TRAIN, N_VAL, SEED = 40000, 2000, 42
SEQ = 512
EOS_ID, MASK_ID = 50256, 50257

_ROLE = {"user": "User", "assistant": "Assistant", "system": "System"}
_WS = re.compile(r"[ \t]+")


def format_conversation(messages):
    turns = []
    for m in messages:
        role = _ROLE.get(m.get("role", "user"), "User")
        text = _WS.sub(" ", m.get("content", "")).strip()
        if text:
            turns.append(f"{role}: {text}")
    return "\n".join(turns)


def chunk_ids(ids, seq=SEQ):
    return [ids[i : i + seq] for i in range(0, len(ids), seq)]


def pack_rows(pieces, seq=SEQ):
    """Greedy-pack pre-terminated pieces into exact-seq rows. No padding.

    Pieces carry their own EOS (conv-final only); continuations concatenate
    with no separator. Overflow splits across rows, never truncates.
    """
    rows, cur = [], []
    for p in pieces:
        p = list(p)
        while p:
            take = seq - len(cur)
            cur += p[:take]
            p = p[take:]
            if len(cur) == seq:
                rows.append(cur)
                cur = []
    return torch.tensor(rows, dtype=torch.long)  # tail dropped, no padding


def build_rows(tokenizer, conversations, seq=SEQ):
    pieces = []
    for conv in conversations:
        full = tokenizer(format_conversation(conv))["input_ids"] + [EOS_ID]
        pieces.extend(chunk_ids(full, seq - 1))  # EOS only at conv end
    packed = pack_rows(pieces, seq)
    logger.info("packed %d rows (~%.1fM tokens)", len(packed), packed.numel() / 1e6)
    return packed


def download_split(split, n, seed=SEED):
    from datasets import load_dataset

    ds = load_dataset(DATASET, split=split)
    return ds.shuffle(seed=seed).select(range(min(n, len(ds))))["messages"]


def get_tokenizer():
    from transformers import GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.add_special_tokens({"mask_token": "[MASK]"})
    assert tok.mask_token_id == MASK_ID, tok.mask_token_id
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    ap.add_argument("--n-val", type=int, default=N_VAL)
    args = ap.parse_args()
    import os

    os.makedirs(args.out, exist_ok=True)
    tok = get_tokenizer()
    for name, split, n in (("train", TRAIN_SPLIT, args.n_train), ("val", VAL_SPLIT, args.n_val)):
        convs = download_split(split, n)
        rows = build_rows(tok, convs)
        torch.save(rows, f"{args.out}/{name}.pt")
        print(name, tuple(rows.shape), f"{rows.numel()/1e6:.1f}M tokens")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
