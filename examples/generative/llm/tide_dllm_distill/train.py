#!/usr/bin/env python3
"""TIDE: Cross-Architecture Distillation for Diffusion LLMs.

Implements TIDE-Shared distillation (Pipeline B): distill knowledge from a
frozen teacher dLLM into a smaller student dLLM that shares the same tokenizer.

Key components:
- TIDAL: Dual-axis interpolation that gradually shifts from student
  self-supervision to teacher-guided learning via cosine schedule.
- CompDemo: Complementary demonstration that improves teacher signal quality
  at high masking ratios by splitting masked positions into two subsets and
  merging predictions from two forward passes.

Reference: "Turning the Tide: Cross-Architecture Distillation for Diffusion
Large Language Models" (arXiv 2604.26951)
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

try:
    from cvlization.paths import get_output_dir
    CVL_AVAILABLE = True
except ImportError:
    CVL_AVAILABLE = False

    def get_output_dir():
        d = os.path.join(os.getcwd(), "outputs")
        os.makedirs(d, exist_ok=True)
        return d


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TokenizedSFTDataset(Dataset):
    """Tokenize an instruction-following dataset for masked diffusion training."""

    def __init__(self, tokenizer, dataset_name="tatsu-lab/alpaca",
                 max_length=512, max_samples=None):
        from datasets import load_dataset

        print(f"Loading dataset: {dataset_name}")
        ds = load_dataset(dataset_name, split="train")
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))

        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.sequences = []

        for item in ds:
            text = _extract_text(item)
            if not text:
                continue

            tokens = tokenizer.encode(
                text, max_length=max_length, truncation=True,
                add_special_tokens=True,
            )
            if len(tokens) < 16:
                continue

            # Pad to max_length
            if len(tokens) < max_length:
                tokens = tokens + [pad_token_id] * (max_length - len(tokens))

            self.sequences.append(torch.tensor(tokens, dtype=torch.long))

        print(f"Loaded {len(self.sequences)} sequences (max_length={max_length})")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def _extract_text(item):
    """Pull a training string from common HF dataset formats."""
    if "instruction" in item:
        parts = [item["instruction"]]
        if item.get("input"):
            parts.append(item["input"])
        if item.get("output"):
            parts.append(item["output"])
        return "\n".join(parts)
    if "text" in item:
        return item["text"]
    return None


# ---------------------------------------------------------------------------
# TIDAL scheduler
# ---------------------------------------------------------------------------

class TIDALScheduler:
    """Dual-axis interpolation for distillation strength.

    Lambda increases from lambda_init to lambda_max via cosine schedule:
      early training -> lambda ~ lambda_init  (student learns from itself)
      late training  -> lambda ~ lambda_max   (student learns from teacher)
    """

    def __init__(self, lambda_init=0.1, lambda_max=0.9, total_steps=10000):
        self.lambda_init = lambda_init
        self.lambda_max = lambda_max
        self.total_steps = total_steps

    def get_lambda(self, step):
        progress = min(step / max(self.total_steps, 1), 1.0)
        return (
            self.lambda_init
            + (self.lambda_max - self.lambda_init)
            * (1 - math.cos(math.pi * progress)) / 2
        )

    @staticmethod
    def timestep_weight(mask_ratio, sigma=0.15):
        """Midrange timestep weighting: emphasise mask_ratio ~ 0.5."""
        return math.exp(-((mask_ratio - 0.5) ** 2) / (2 * sigma ** 2))


# ---------------------------------------------------------------------------
# TIDAL loss
# ---------------------------------------------------------------------------

def compute_tidal_loss(student_logits, teacher_logits, mask,
                       lambda_t, temperature=2.0):
    """KL-based distillation loss with interpolated soft targets.

    Args:
        student_logits: (B, L, V)
        teacher_logits: (B, L, V)
        mask:           (B, L) bool – True for masked positions
        lambda_t:       interpolation weight  (0=student, 1=teacher)
        temperature:    softmax temperature
    """
    if mask.sum() == 0:
        return torch.tensor(0.0, device=student_logits.device)

    s = student_logits[mask]   # (N, V)
    t = teacher_logits[mask]   # (N, V)

    # Interpolated target  (detach student to avoid second-order gradients)
    interp = ((1 - lambda_t) * s.detach() + lambda_t * t) / temperature
    target = F.softmax(interp, dim=-1)

    log_probs = F.log_softmax(s / temperature, dim=-1)
    return F.kl_div(log_probs, target, reduction="batchmean") * (temperature ** 2)


# ---------------------------------------------------------------------------
# CompDemo – Complementary Demonstration
# ---------------------------------------------------------------------------

def comp_demo_teacher_forward(teacher, input_ids, original_ids, mask,
                              mask_token_id):
    """Two-pass teacher inference with complementary mask splits.

    Splits masked positions into A/B subsets.  Pass 1 reveals A as context and
    collects predictions for B.  Pass 2 reveals B and collects for A.  Merged
    logits give each position ~50 % extra revealed context.
    """
    if mask.sum() == 0:
        with torch.no_grad():
            return teacher(input_ids).logits

    device = input_ids.device
    mask_idx = mask.nonzero(as_tuple=False)  # (N, 2)
    n = mask_idx.shape[0]
    perm = torch.randperm(n, device=device)
    half = n // 2

    # Build subset masks
    subset_a = torch.zeros_like(mask)
    subset_b = torch.zeros_like(mask)
    if half > 0:
        a_idx = mask_idx[perm[:half]]
        subset_a[a_idx[:, 0], a_idx[:, 1]] = True
    if n - half > 0:
        b_idx = mask_idx[perm[half:]]
        subset_b[b_idx[:, 0], b_idx[:, 1]] = True

    # Pass 1: reveal A, keep B masked  →  good logits for B
    inp1 = original_ids.clone()
    inp1[subset_b] = mask_token_id

    # Pass 2: reveal B, keep A masked  →  good logits for A
    inp2 = original_ids.clone()
    inp2[subset_a] = mask_token_id

    with torch.no_grad():
        logits1 = teacher(inp1).logits
        logits2 = teacher(inp2).logits

    merged = logits1.clone()
    merged[subset_a] = logits2[subset_a]
    return merged


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def apply_random_masking(input_ids, mask_token_id, pad_token_id):
    """Token-level masking with uniform mask-ratio sampling."""
    B, L = input_ids.shape
    device = input_ids.device

    non_pad = input_ids != pad_token_id
    ratios = torch.rand(B, 1, device=device)
    mask = (torch.rand(B, L, device=device) < ratios) & non_pad

    masked = input_ids.clone()
    masked[mask] = mask_token_id
    return masked, mask, ratios.mean().item()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    torch.manual_seed(args.seed)

    # Tokenizer (shared between teacher & student in Pipeline B)
    print(f"Loading tokenizer from: {args.student_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.student_model, trust_remote_code=True,
    )
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    print(f"  mask_token={mask_token_id}  pad_token={pad_token_id}")

    # Student (trainable)
    print(f"Loading student: {args.student_model}")
    student = AutoModelForMaskedLM.from_pretrained(
        args.student_model, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    student.train()

    # Teacher (frozen)
    teacher_id = args.teacher_model or args.student_model
    print(f"Loading teacher: {teacher_id}")
    if teacher_id == args.student_model:
        print("  (self-distillation mode – student used as its own teacher)")
    teacher = AutoModelForMaskedLM.from_pretrained(
        teacher_id, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Data
    dataset = TokenizedSFTDataset(
        tokenizer, dataset_name=args.dataset,
        max_length=args.max_length, max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=0,
    )

    # TIDAL
    tidal = TIDALScheduler(
        lambda_init=args.lambda_init, lambda_max=args.lambda_max,
        total_steps=args.steps,
    )

    # Optimiser & scheduler
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.1,
    )

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nStarting TIDE distillation")
    print(f"  steps={args.steps}  batch_size={args.batch_size}  lr={args.lr}")
    print(f"  TIDAL lambda: {args.lambda_init} -> {args.lambda_max}")
    print(f"  CompDemo: {args.use_comp_demo}")
    print(f"  temperature={args.temperature}  "
          f"ce_weight={args.ce_weight}  tidal_weight={args.tidal_weight}")
    print()

    step = 0
    t0 = time.time()
    data_iter = iter(loader)
    acc_loss = acc_ce = acc_td = 0.0

    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        input_ids = batch.to(device)
        masked_ids, mask, mask_ratio = apply_random_masking(
            input_ids, mask_token_id, pad_token_id,
        )

        # Teacher logits
        if args.use_comp_demo:
            t_logits = comp_demo_teacher_forward(
                teacher, masked_ids, input_ids, mask, mask_token_id,
            )
        else:
            with torch.no_grad():
                t_logits = teacher(masked_ids).logits

        # Student forward
        s_logits = student(masked_ids).logits

        # CE loss on masked positions
        if mask.sum() > 0:
            ce_loss = F.cross_entropy(s_logits[mask], input_ids[mask])
        else:
            ce_loss = torch.tensor(0.0, device=device)

        # TIDAL loss
        lam = tidal.get_lambda(step)
        tw = tidal.timestep_weight(mask_ratio)
        td_loss = compute_tidal_loss(
            s_logits, t_logits, mask, lam, temperature=args.temperature,
        )

        loss = args.ce_weight * ce_loss + args.tidal_weight * tw * td_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(),
                                           args.max_grad_norm)
        optimizer.step()
        lr_sched.step()

        acc_loss += loss.item()
        acc_ce += ce_loss.item()
        acc_td += td_loss.item()
        step += 1

        if step % args.log_interval == 0 or step == args.steps:
            n = args.log_interval
            lr = lr_sched.get_last_lr()[0]
            print(
                f"step {step}/{args.steps} | "
                f"loss {acc_loss/n:.4f} "
                f"(CE {acc_ce/n:.4f}  TIDAL {acc_td/n:.4f}) | "
                f"lambda {lam:.3f} | lr {lr:.2e} | "
                f"{time.time()-t0:.1f}s"
            )
            acc_loss = acc_ce = acc_td = 0.0

        if step % args.save_interval == 0 or step == args.steps:
            ckpt = os.path.join(out_dir, f"checkpoint-{step}")
            print(f"  saving -> {ckpt}")
            student.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            with open(os.path.join(ckpt, "training_state.json"), "w") as f:
                json.dump({"step": step, "lambda": lam,
                           "args": {k: str(v) for k, v in vars(args).items()}},
                          f, indent=2)

    print(f"\nTraining complete in {time.time()-t0:.1f}s")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="TIDE distillation for dLLMs")

    # Models
    p.add_argument("--student-model", default="dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1",
                   help="Student model HF ID")
    p.add_argument("--teacher-model", default=None,
                   help="Teacher model HF ID (default: same as student)")

    # Data
    p.add_argument("--dataset", default="tatsu-lab/alpaca",
                   help="HuggingFace dataset name")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap on training samples (None=all)")

    # Training
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    # TIDAL
    p.add_argument("--lambda-init", type=float, default=0.1)
    p.add_argument("--lambda-max", type=float, default=0.9)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--ce-weight", type=float, default=1.0)
    p.add_argument("--tidal-weight", type=float, default=1.0)

    # CompDemo
    p.add_argument("--use-comp-demo", action="store_true", default=True)
    p.add_argument("--no-comp-demo", dest="use_comp_demo",
                   action="store_false")

    # Output
    p.add_argument("--output-dir", default=None)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--save-interval", type=int, default=1000)

    args = p.parse_args()
    if args.output_dir is None:
        args.output_dir = get_output_dir()
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
