#!/usr/bin/env python3
"""
Semicat training wrapper with simplified dataset interface.

Usage:
    python train.py                                # Text8 (default)
    python train.py -d /path/to/my_corpus.txt      # Custom text file (char-level, 27 vocab)
    python train.py -d lm1b                        # Google 1B Word (BERT tokenizer, 30K vocab)
    python train.py -d openwebtext                 # OpenWebText (GPT2 tokenizer, 50K vocab)
    python train.py -d openwebtext --tokenizer bert # OWT with BERT tokenizer

Extra arguments are passed as Hydra overrides to semicat.train.
"""
import argparse
import hashlib
import os
import pickle
import sys
import urllib.request
import zipfile

import numpy as np


CACHE_DIR = os.environ.get("CVL_CACHE_DIR", "/cache")


# ---------------------------------------------------------------------------
# Char-level data preparation
# ---------------------------------------------------------------------------

def prepare_text8(cache_dir):
    """Download and prepare the standard Text8 dataset."""
    data_dir = os.path.join(cache_dir, "semicat_text8")
    meta_path = os.path.join(data_dir, "meta.pkl")
    if os.path.exists(meta_path):
        print(f"Text8 data already prepared at {data_dir}")
        return data_dir

    os.makedirs(data_dir, exist_ok=True)
    url = "http://mattmahoney.net/dc/text8.zip"
    zip_path = os.path.join(data_dir, "text8.zip")

    print(f"Downloading Text8 from {url} ...")
    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        raw = zf.read("text8").decode("utf-8")
    os.remove(zip_path)

    _save_char_data(raw, data_dir, train_split_chars=90_000_000)
    return data_dir


def prepare_char_data(text_path, cache_dir):
    """Prepare char-level data from a user text file (27-char vocab: a-z + space)."""
    path_hash = hashlib.md5(os.path.abspath(text_path).encode()).hexdigest()[:12]
    data_dir = os.path.join(cache_dir, f"semicat_char_{path_hash}")
    meta_path = os.path.join(data_dir, "meta.pkl")

    if os.path.exists(meta_path):
        print(f"Using cached char data at {data_dir}")
        return data_dir

    os.makedirs(data_dir, exist_ok=True)

    print(f"Reading {text_path} ...")
    with open(text_path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    # Normalize to 27-char vocab (a-z + space)
    raw = raw.lower()
    raw = "".join(c if c in "abcdefghijklmnopqrstuvwxyz " else " " for c in raw)
    raw = " ".join(raw.split())  # collapse whitespace

    if len(raw) < 1000:
        print(f"WARNING: text very short ({len(raw)} chars). Training quality may be poor.")

    split_idx = int(len(raw) * 0.9)
    _save_char_data(raw, data_dir, train_split_chars=split_idx)
    return data_dir


def _save_char_data(raw, data_dir, train_split_chars):
    """Encode text as uint16 binaries + meta.pkl."""
    chars = sorted(set(raw))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    encoded = np.array([stoi[ch] for ch in raw], dtype=np.uint16)
    train_data = encoded[:train_split_chars]
    val_data = encoded[train_split_chars:]

    train_data.tofile(os.path.join(data_dir, "train.bin"))
    val_data.tofile(os.path.join(data_dir, "val.bin"))

    meta = {"vocab_size": vocab_size, "stoi": stoi, "itos": itos}
    with open(os.path.join(data_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"Prepared: {len(raw)} chars, vocab={vocab_size}, "
          f"train={len(train_data)}, val={len(val_data)}")


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_file_path(path):
    """Resolve a file path, checking CVL workspace mount."""
    if os.path.isfile(path):
        return path
    # Try as relative path under CVL workspace
    cvl_path = os.path.join("/mnt/cvl/workspace", path)
    if os.path.isfile(cvl_path):
        return cvl_path
    # Try basename only (handles absolute host paths like /tmp/foo/file.txt
    # when user's CWD containing file.txt is mounted at /mnt/cvl/workspace)
    basename_path = os.path.join("/mnt/cvl/workspace", os.path.basename(path))
    if os.path.isfile(basename_path):
        return basename_path
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train semicat categorical flow model",
        epilog="Extra arguments are passed as Hydra overrides to semicat.train",
    )
    parser.add_argument(
        "--dataset", "-d", default="text8",
        help="'text8' (default), 'lm1b', 'openwebtext', or path to a .txt file",
    )
    parser.add_argument(
        "--tokenizer", default=None,
        help="Tokenizer for HF datasets: 'gpt2' (default) or 'bert'. Auto-set for lm1b.",
    )
    parser.add_argument("--seq-len", type=int, default=None, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--steps", type=int, default=None, help="Training steps")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")

    args, extra_hydra_args = parser.parse_known_args()

    overrides = [
        "trainer=gpu",
        "extras.enforce_tags=false",
    ]

    file_path = resolve_file_path(args.dataset)

    if file_path is not None:
        # ── Custom text file → char-level ──
        data_dir = prepare_char_data(file_path, CACHE_DIR)
        seq_len = args.seq_len or 256
        batch_size = args.batch_size or 128
        steps = args.steps or 100_000

        # Read back vocab_size from prepared data
        with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
            vocab_size = pickle.load(f)["vocab_size"]

        overrides += [
            "experiment=text8_dit_sm",
            f"+data.data_dir={data_dir}",
            f"data.batch_size={batch_size}",
            f"data.k={seq_len}",
            f"model.in_shape=[{seq_len},{vocab_size}]",
            f"model.net.vocab_size={vocab_size}",
            f"model.net.length={seq_len}",
            f"trainer.min_steps={steps}",
            f"++trainer.max_steps={steps}",
            f"tags=[semicat,custom,cvl]",
        ]

    elif args.dataset == "text8":
        # ── Text8 (default) ──
        data_dir = prepare_text8(CACHE_DIR)
        seq_len = args.seq_len or 256
        batch_size = args.batch_size or 128
        steps = args.steps or 100_000

        overrides += [
            "experiment=text8_dit_sm",
            f"+data.data_dir={data_dir}",
            f"data.batch_size={batch_size}",
            f"trainer.min_steps={steps}",
            f"tags=[semicat,text8,cvl]",
        ]
        if args.seq_len:
            overrides += [
                f"data.k={seq_len}",
                f"model.net.length={seq_len}",
                f"model.in_shape=[{seq_len},27]",
            ]

    elif args.dataset == "lm1b":
        # ── LM1B with BERT tokenizer ──
        batch_size = args.batch_size or 400
        steps = args.steps or 250_000

        hf_cache = os.path.join(CACHE_DIR, "semicat_hf")
        os.makedirs(hf_cache, exist_ok=True)
        os.environ["DATASET_CACHE_DIR"] = hf_cache

        overrides += [
            "experiment=lm1b_dit",
            f"data.batch_size={batch_size}",
            f"trainer.min_steps={steps}",
            f"tags=[semicat,lm1b,cvl]",
        ]

    elif args.dataset in ("openwebtext", "owt"):
        # ── OpenWebText with GPT2 tokenizer (or user-chosen) ──
        tokenizer = args.tokenizer or "gpt2"
        if tokenizer == "bert":
            vocab_size = 30_522
            default_seq_len = 128
        else:
            vocab_size = 50_257
            default_seq_len = 1024

        seq_len = args.seq_len or default_seq_len
        batch_size = args.batch_size or 32
        steps = args.steps or 250_000

        hf_cache = os.path.join(CACHE_DIR, "semicat_hf")
        os.makedirs(hf_cache, exist_ok=True)
        os.environ["DATASET_CACHE_DIR"] = hf_cache

        overrides += [
            "experiment=owt_dit",
            f"data.batch_size={batch_size}",
            f"data.max_length={seq_len}",
            f"model.in_shape=[{seq_len},{vocab_size}]",
            f"model.net.vocab_size={vocab_size}",
            f"model.net.length={seq_len}",
            f"trainer.min_steps={steps}",
            f"tags=[semicat,openwebtext,cvl]",
        ]

    else:
        print(f"Unknown dataset: '{args.dataset}'")
        print("Supported: text8, lm1b, openwebtext, or a path to a .txt file")
        sys.exit(1)

    if not args.compile:
        overrides.append("model.compile=false")

    overrides += extra_hydra_args

    cmd = [sys.executable, "-m", "semicat.train"] + overrides
    print(f"Running: {' '.join(cmd)}")
    print()
    os.execvp(sys.executable, cmd)


if __name__ == "__main__":
    main()
