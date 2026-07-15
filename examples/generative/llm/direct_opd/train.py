"""Direct-OPD: Weak-to-Strong On-Policy Distillation Training.

Implements the core Direct-OPD objective from:
  "Weak-to-Strong Generalization via Direct On-Policy Distillation"
  (Feng, Gao, Chi et al., Tsinghua AIR / ByteDance Seed, 2026)
  https://github.com/BytedTsinghua-SIA/Direct-OPD

The method transfers RL-induced policy shifts from a weak teacher to a
stronger student by computing per-token log-ratio between post-RL and
pre-RL teacher checkpoints as a dense implicit reward, applied to the
student's on-policy rollouts.

Algorithm:
  1. Student generates responses on-policy
  2. At each response token position, select the student's top-K tokens
  3. Compute delta = log pi_teacher(v) - log pi_teacher_ref(v) for top-K
  4. Weight by student's (renormalized) probability over top-K
  5. Use weighted delta as advantage for policy gradient update
  6. Regularize with KL penalty against the student's initial policy
"""

import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_prompts(path: str) -> list[dict]:
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def load_model(name: str, device: str, trainable: bool = False):
    """Load a causal LM in bfloat16. Freeze if not trainable."""
    model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.to(device)
    if not trainable:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    else:
        model.train()
        if True:  # gradient checkpointing
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
    return model


def tokenize_prompt(tokenizer, prompt_text: str, max_len: int, device: str):
    """Apply chat template and tokenize a single prompt."""
    messages = [{"role": "user", "content": prompt_text}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        padding=False,
    )
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def generate_response(model, input_ids, attention_mask, tokenizer, config):
    """Generate a single on-policy response from the student."""
    gen_cfg = config["generation"]
    max_new = config["training"]["max_response_len"]

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new,
        temperature=gen_cfg["temperature"],
        do_sample=gen_cfg["do_sample"],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return outputs


def forward_logprobs(model, input_ids, attention_mask):
    """Compute per-position log probabilities over the full vocabulary."""
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    return F.log_softmax(out.logits, dim=-1)


def compute_opd_loss(
    student_logprobs,  # [1, T, V] with grad
    teacher_logprobs,  # [1, T, V] no grad
    teacher_ref_logprobs,  # [1, T, V] no grad
    response_mask,  # [1, R] bool — which positions are response tokens
    prompt_len: int,
    top_k: int,
):
    """Compute the Direct-OPD policy gradient loss.

    At each response position t:
      1. Select student's top-K tokens
      2. Compute delta = teacher_lp - teacher_ref_lp for those tokens
      3. Advantage = stop_grad(softmax(student_lp[top_k]) * delta)
      4. Loss = -sum_k(advantage_k * student_lp[top_k])

    Returns:
        loss: scalar tensor with grad
        metrics: dict of floats for logging
    """
    # Logits at position t predict token at t+1, so use positions
    # [prompt_len-1 .. T-2] to correspond to response tokens [prompt_len .. T-1]
    T = student_logprobs.shape[1]
    start = prompt_len - 1
    end = T - 1

    stu_lp = student_logprobs[:, start:end, :]  # [1, R, V]
    tea_lp = teacher_logprobs[:, start:end, :]
    tea_ref_lp = teacher_ref_logprobs[:, start:end, :]

    R = stu_lp.shape[1]
    if R == 0:
        zero = torch.tensor(0.0, device=stu_lp.device, requires_grad=True)
        return zero, {"pg_loss": 0.0, "mean_delta": 0.0, "mean_adv": 0.0}

    # Truncate response_mask to match
    mask = response_mask[:, :R].float()  # [1, R]

    # Clamp vocab to minimum across models to handle tokenizer differences
    V_min = min(stu_lp.shape[-1], tea_lp.shape[-1], tea_ref_lp.shape[-1])
    stu_lp = stu_lp[:, :, :V_min]
    tea_lp = tea_lp[:, :, :V_min]
    tea_ref_lp = tea_ref_lp[:, :, :V_min]

    # Top-K from student (indices are not differentiable)
    _, top_idx = stu_lp.detach().topk(top_k, dim=-1)  # [1, R, K]

    # Gather log probs for top-K tokens
    stu_topk = stu_lp.gather(-1, top_idx)  # [1, R, K] — carries grad
    tea_topk = tea_lp.gather(-1, top_idx)  # [1, R, K]
    tea_ref_topk = tea_ref_lp.gather(-1, top_idx)  # [1, R, K]

    # Teacher delta: implicit reward from RL
    delta = tea_topk - tea_ref_topk  # [1, R, K]

    # Student probability renormalized over top-K
    p_bar = F.softmax(stu_topk.detach(), dim=-1)  # [1, R, K]

    # Weighted advantage (Rao-Blackwellized)
    advantage = (p_bar * delta).detach()  # [1, R, K]

    # Policy gradient loss: push student toward tokens RL preferred
    token_loss = -(advantage * stu_topk).sum(-1)  # [1, R]
    num_tokens = mask.sum().clamp(min=1)
    pg_loss = (token_loss * mask).sum() / num_tokens

    metrics = {
        "pg_loss": pg_loss.item(),
        "mean_delta": delta.mean().item(),
        "mean_adv": advantage.sum(-1).mean().item(),
        "delta_pos_frac": (delta > 0).float().mean().item(),
    }
    return pg_loss, metrics


def compute_kl_penalty(
    student_logprobs,  # [1, T, V] current student
    ref_logprobs,  # [1, T] log probs of generated tokens at generation time
    generated_ids,  # [1, R] generated token ids
    prompt_len: int,
    response_mask,  # [1, R]
):
    """Approximate KL(pi_current || pi_ref) on the generated tokens.

    Uses the low-variance KL estimator:
      KL ~= exp(log_pi_ref - log_pi_cur) - (log_pi_ref - log_pi_cur) - 1
    """
    T = student_logprobs.shape[1]
    start = prompt_len - 1
    end = T - 1
    R = end - start

    if R <= 0:
        return torch.tensor(0.0, device=student_logprobs.device, requires_grad=True)

    mask = response_mask[:, :R].float()

    # Current student log prob of the generated token at each position
    V_min = student_logprobs.shape[-1]
    stu_lp = student_logprobs[:, start:end, :]  # [1, R, V]
    gen_ids = generated_ids[:, :R].unsqueeze(-1).clamp(max=V_min - 1)  # [1, R, 1]
    cur_lp = stu_lp.gather(-1, gen_ids).squeeze(-1)  # [1, R]

    # Reference log prob (from generation time)
    old_lp = ref_logprobs[:, :R]  # [1, R]

    # Low-variance KL estimator
    ratio = old_lp - cur_lp  # log(pi_ref / pi_cur)
    kl = (ratio.exp() - ratio - 1.0)

    num_tokens = mask.sum().clamp(min=1)
    return (kl * mask).sum() / num_tokens


def train(config_path: str):
    config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = config["training"]["seed"]
    random.seed(seed)
    torch.manual_seed(seed)

    # --- Load tokenizer (from student model) ---
    print(f"Loading tokenizer from {config['models']['student']}...")
    tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["student"], trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Prepare data ---
    data_path = "data/train.jsonl"
    if not os.path.exists(data_path):
        print("Preparing training data...")
        data_cfg = config["data"]
        split = data_cfg.get("split", "math")
        os.system(
            f"python prepare_data.py"
            f" --dataset '{data_cfg['dataset']}'"
            f" --split {split}"
            f" --max-prompts {data_cfg['max_prompts']}"
            f" --output {data_path}"
            f" --seed {seed}"
        )

    prompts = load_prompts(data_path)
    print(f"Loaded {len(prompts)} prompts")

    # --- Load models ---
    print(f"Loading student: {config['models']['student']}...")
    student = load_model(config["models"]["student"], device, trainable=True)

    print(f"Loading teacher: {config['models']['teacher']}...")
    teacher = load_model(config["models"]["teacher"], device, trainable=False)

    print(f"Loading teacher_ref: {config['models']['teacher_ref']}...")
    teacher_ref = load_model(config["models"]["teacher_ref"], device, trainable=False)

    # --- Optimizer ---
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            student.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
        print("Using 8-bit AdamW optimizer")
    except ImportError:
        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
        print("Using standard AdamW optimizer (bitsandbytes not available)")

    # Linear warmup scheduler
    warmup_steps = config["training"]["warmup_steps"]
    max_steps = config["training"]["max_steps"]

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, 1.0 - (step - warmup_steps) / max(1, max_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Adaptive KL coefficient
    opd_cfg = config["opd"]
    kl_coef = opd_cfg["kl_coef"]

    # --- Output directory ---
    out_dir = config["training"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    log_file = open(os.path.join(out_dir, "train_log.jsonl"), "w")

    # --- Training loop ---
    top_k = opd_cfg["top_k"]
    max_prompt_len = config["training"]["max_prompt_len"]

    print(f"\n{'='*60}")
    print("Starting Direct-OPD training")
    print(f"  Student:     {config['models']['student']}")
    print(f"  Teacher:     {config['models']['teacher']}")
    print(f"  Teacher Ref: {config['models']['teacher_ref']}")
    print(f"  Top-K:       {top_k}")
    print(f"  KL coef:     {kl_coef}")
    print(f"  Max steps:   {max_steps}")
    print(f"  Device:      {device}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU:         {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"{'='*60}\n")

    global_step = 0
    prompt_idx = 0

    for step in range(max_steps):
        step_start = time.time()

        # Sample a prompt
        prompt_data = prompts[prompt_idx % len(prompts)]
        prompt_idx += 1
        prompt_text = prompt_data["prompt"]

        # Tokenize prompt
        enc = tokenize_prompt(tokenizer, prompt_text, max_prompt_len, device)
        prompt_ids = enc["input_ids"]
        prompt_mask = enc["attention_mask"]
        prompt_len = prompt_ids.shape[1]

        # 1. Generate on-policy response from student
        student.eval()
        full_ids = generate_response(
            student, prompt_ids, prompt_mask, tokenizer, config
        )
        student.train()

        response_len = full_ids.shape[1] - prompt_len
        if response_len <= 0:
            print(f"Step {step}: empty response, skipping")
            continue

        # Build attention mask for full sequence
        full_mask = torch.ones_like(full_ids)

        # Response mask (which positions are response tokens)
        response_ids = full_ids[:, prompt_len:]
        response_mask = (response_ids != tokenizer.pad_token_id)

        # Save reference log probs (from generation-time student) for KL
        with torch.no_grad():
            ref_out = student(input_ids=full_ids, attention_mask=full_mask)
            ref_lp = F.log_softmax(ref_out.logits, dim=-1)
            # Log prob of each generated token
            V_ref = ref_lp.shape[-1]
            shifted_ids = full_ids[:, prompt_len:].unsqueeze(-1).clamp(max=V_ref - 1)
            ref_token_lp = ref_lp[:, prompt_len - 1:-1, :].gather(-1, shifted_ids).squeeze(-1)

        # 2. Forward pass: student (with grad)
        student_out = student(input_ids=full_ids, attention_mask=full_mask)
        student_logprobs = F.log_softmax(student_out.logits, dim=-1)

        # 3. Forward pass: teacher and teacher_ref (no grad)
        with torch.no_grad():
            teacher_logprobs = forward_logprobs(teacher, full_ids, full_mask)
            teacher_ref_logprobs = forward_logprobs(teacher_ref, full_ids, full_mask)

        # 4. Compute Direct-OPD loss
        pg_loss, opd_metrics = compute_opd_loss(
            student_logprobs,
            teacher_logprobs,
            teacher_ref_logprobs,
            response_mask,
            prompt_len,
            top_k,
        )

        # 5. KL penalty
        kl_loss = compute_kl_penalty(
            student_logprobs,
            ref_token_lp,
            response_ids,
            prompt_len,
            response_mask,
        )

        # 6. Total loss
        loss = pg_loss + kl_coef * kl_loss

        # 7. Backward and update
        optimizer.zero_grad()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # 8. Adaptive KL coefficient
        if opd_cfg["adaptive_kl"]:
            mean_adv = opd_metrics["mean_adv"]
            direction = 1.0 if mean_adv >= 0 else -1.0
            kl_coef = max(
                opd_cfg["kl_coef_min"],
                min(
                    opd_cfg["kl_coef_max"],
                    kl_coef * (1.0 + opd_cfg["adaptive_kl_step"] * direction),
                ),
            )

        step_time = time.time() - step_start

        # Log metrics
        metrics = {
            "step": step,
            "loss": loss.item(),
            "pg_loss": opd_metrics["pg_loss"],
            "kl_loss": kl_loss.item(),
            "kl_coef": kl_coef,
            "mean_delta": opd_metrics["mean_delta"],
            "mean_adv": opd_metrics["mean_adv"],
            "delta_pos_frac": opd_metrics["delta_pos_frac"],
            "grad_norm": grad_norm.item(),
            "lr": scheduler.get_last_lr()[0],
            "response_len": response_len,
            "step_time": step_time,
        }

        if device == "cuda":
            metrics["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()

        log_file.write(json.dumps(metrics) + "\n")
        log_file.flush()

        if step % config["training"]["logging_steps"] == 0:
            print(
                f"Step {step:4d} | loss {loss.item():+8.4f} | "
                f"pg {opd_metrics['pg_loss']:+8.4f} | "
                f"kl {kl_loss.item():.4f} | "
                f"delta {opd_metrics['mean_delta']:+.4f} | "
                f"adv {opd_metrics['mean_adv']:+.4f} | "
                f"d+ {opd_metrics['delta_pos_frac']:.2f} | "
                f"gnorm {grad_norm.item():.2f} | "
                f"rlen {response_len:4d} | "
                f"{step_time:.1f}s"
            )

        # Save checkpoint
        if (step + 1) % config["training"]["save_steps"] == 0:
            ckpt_dir = os.path.join(out_dir, f"checkpoint-{step + 1}")
            print(f"Saving checkpoint to {ckpt_dir}...")
            student.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

        global_step += 1

    # Save final model
    final_dir = os.path.join(out_dir, "final_model")
    print(f"Saving final model to {final_dir}...")
    student.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    log_file.close()
    print("\nTraining complete!")
    print(f"Logs: {out_dir}/train_log.jsonl")
    print(f"Final model: {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direct-OPD Training")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()
    train(args.config)
