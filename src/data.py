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


def pack_rows(chunks, seq=SEQ, eos=EOS_ID):
    """Greedy-pack chunks into exact-seq rows joined by EOS. Drops tail."""
    rows, cur = [], []
    for ch in chunks:
        piece = ch + [eos]
        if len(cur) + len(piece) > seq:
            if len(cur) == seq:
                rows.append(cur)
                cur = []
            else:
                cur = (cur + piece)[:seq]  # rare >seq chunk truncates
                if len(cur) == seq:
                    rows.append(cur)
                    cur = []
                continue
        cur = cur + piece
        if len(cur) == seq:
            rows.append(cur)
            cur = []
    # merge leftover into full rows only; no padding
    return torch.tensor(rows, dtype=torch.long)


def build_rows(tokenizer, conversations, seq=SEQ):
    chunks = []
    for conv in conversations:
        ids = tokenizer(format_conversation(conv))["input_ids"]
        chunks.extend(chunk_ids(ids, seq - 1))  # room for EOS joins
    packed = pack_rows(chunks, seq)
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
