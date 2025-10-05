"""Inspect the POS-augmented Shakespeare character stream."""

import argparse
import os
import pickle
from typing import List

import numpy as np

BASE_DIR = os.path.dirname(__file__)


def load_meta():
    with open(os.path.join(BASE_DIR, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    with open(os.path.join(BASE_DIR, "meta_pos.pkl"), "rb") as f:
        meta_pos = pickle.load(f)
    return meta, meta_pos


def decode_slice(ids: np.ndarray, meta, meta_pos, start: int, length: int) -> List[str]:
    stoi = meta["stoi"]
    itos = meta["itos"]
    vocab_size = meta["vocab_size"]
    pos_vocab = meta_pos["program_pos_vocab"]
    offset = meta_pos["program_offset"]
    decoded = []
    for idx in ids[start : start + length]:
        idx = int(idx)
        if idx < vocab_size:
            decoded.append(itos[idx])
        else:
            pos_idx = idx - offset
            if 0 <= pos_idx < len(pos_vocab):
                decoded.append(f"<{pos_vocab[pos_idx]}>")
            elif idx == meta_pos["program_nil_id"]:
                decoded.append("<NIL>")
            else:
                decoded.append(f"<UNK:{idx}>")
    return decoded


def main():
    parser = argparse.ArgumentParser(description="Peek at POS-augmented tiny Shakespeare stream")
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index into the augmented stream (default: 0)",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=120,
        help="Number of tokens to display (default: 120)",
    )
    parser.add_argument(
        "--val",
        action="store_true",
        help="Peek into the validation stream instead of train.",
    )
    args = parser.parse_args()

    file_name = "val_with_pos.bin" if args.val else "train_with_pos.bin"
    path = os.path.join(BASE_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{file_name} not found. Run prepare.py --with-pos first to generate augmented binaries."
        )

    meta, meta_pos = load_meta()
    ids = np.memmap(path, dtype=np.uint32, mode="r")
    start = max(0, min(args.start, len(ids)))
    length = max(0, min(args.length, len(ids) - start))
    decoded = decode_slice(ids, meta, meta_pos, start, length)

    print("file:", file_name)
    print("start:", start, "length:", length)
    print("tokens:")
    print("".join(decoded))


if __name__ == "__main__":
    main()
