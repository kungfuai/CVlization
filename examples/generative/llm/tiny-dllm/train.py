#!/usr/bin/env python3
"""Train a tiny diffusion language model on Tiny Shakespeare.

Character-level diffusion transformer (~10.7M params) that learns to denoise
masked text. Training takes ~20min on an A100 or ~5min on modern GPUs with
compile enabled.

Based on: https://github.com/nathan-barry/tiny-diffusion
"""

import argparse
import os
import sys
import time

import torch

from model import DiffusionLM, CharTokenizer, generate, BLOCK_SIZE

try:
    from cvlization.paths import get_output_dir, resolve_output_path
    CVL_AVAILABLE = True
except ImportError:
    CVL_AVAILABLE = False

    def get_output_dir():
        d = os.path.join(os.getcwd(), "outputs")
        os.makedirs(d, exist_ok=True)
        return d

    def resolve_output_path(path, base):
        return path if os.path.isabs(path) else os.path.join(base, path)


DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_data(path):
    """Download Tiny Shakespeare if not present."""
    if os.path.exists(path):
        return
    print(f"Downloading Tiny Shakespeare to {path}")
    import urllib.request
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    urllib.request.urlretrieve(DATA_URL, path)


def get_batch(train_data, val_data, split, batch_size, mask_token_id, device):
    data = train_data if split == "train" else val_data
    idx = torch.randint(len(data) - BLOCK_SIZE, (batch_size,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in idx])
    y = x.clone()
    mask_probs = torch.rand(batch_size, 1)
    mask = torch.rand(batch_size, BLOCK_SIZE) < mask_probs
    x[mask] = mask_token_id
    return x.to(device), y.to(device), mask.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, mask_token_id,
                  device, eval_iters=200):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, M = get_batch(train_data, val_data, split, batch_size,
                                mask_token_id, device)
            _, loss = model(X, Y, M)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Train tiny diffusion language model on Tiny Shakespeare",
    )
    parser.add_argument("--data", default="data.txt",
                        help="Path to training text (default: data.txt)")
    parser.add_argument("--iters", type=int, default=10000,
                        help="Training iterations (default: 10000)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--eval-interval", type=int, default=500,
                        help="Evaluate every N steps (default: 500)")
    parser.add_argument("--seed", type=int, default=1337,
                        help="Random seed (default: 1337)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for weights and tokenizer")
    args = parser.parse_args()

    device = (
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Device: {device}")
    torch.manual_seed(args.seed)

    # Data
    download_data(args.data)
    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Dataset: {len(text):,} characters")

    tokenizer = CharTokenizer(text)
    print(f"Vocabulary: {tokenizer.vocab_size} characters")

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    # Model
    model = DiffusionLM(tokenizer.vocab_size).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count / 1e6:.1f}M")

    # Output paths
    out_dir = args.output_dir or get_output_dir()
    os.makedirs(out_dir, exist_ok=True)
    weights_path = os.path.join(out_dir, "diffusion.pt")
    tokenizer_path = os.path.join(out_dir, "tokenizer.pt")

    # Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start = time.time()

    for step in range(args.iters):
        if step % args.eval_interval == 0 or step == args.iters - 1:
            losses = estimate_loss(model, train_data, val_data,
                                   args.batch_size, tokenizer.mask_token_id,
                                   device)
            elapsed = time.time() - start
            print(f"step {step}: train {losses['train']:.4f}, "
                  f"val {losses['val']:.4f}, time {elapsed:.1f}s")

            if step > 0:
                sample = generate(model, tokenizer, device,
                                  seed_text=text[:16], max_new_tokens=240)
                print(f"Sample: {sample[:200]}...")

        xb, yb, mb = get_batch(train_data, val_data, "train",
                                args.batch_size, tokenizer.mask_token_id, device)
        _, loss = model(xb, yb, mb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Saving weights to {weights_path}")
    torch.save(model.state_dict(), weights_path)
    tokenizer.save(tokenizer_path)
    print(f"Saved tokenizer to {tokenizer_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
