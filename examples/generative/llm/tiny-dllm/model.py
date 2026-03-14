"""Tiny diffusion and GPT language model architectures.

Character-level transformers (~10.7M params) trained on Tiny Shakespeare.
Two modes:
- diffusion: bidirectional attention, generates via iterative denoising
- gpt: causal attention, generates autoregressively

Based on: https://github.com/nathan-barry/tiny-diffusion
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


# Architecture hyperparameters
BLOCK_SIZE = 256
N_EMBD = 384
N_HEAD = 6
N_LAYER = 6
HEAD_DIM = N_EMBD // N_HEAD


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).to(x.dtype)


class MultiHeadAttention(nn.Module):
    def __init__(self, is_causal=False):
        super().__init__()
        self.is_causal = is_causal
        self.c_q = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.c_k = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.c_v = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD, bias=False)

    def forward(self, x, cos_sin):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, N_HEAD, HEAD_DIM)
        k = self.c_k(x).view(B, T, N_HEAD, HEAD_DIM)
        v = self.c_v(x).view(B, T, N_HEAD, HEAD_DIM)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(N_EMBD, 4 * N_EMBD, bias=False)
        self.c_proj = nn.Linear(4 * N_EMBD, N_EMBD, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, is_causal=False):
        super().__init__()
        self.attn = MultiHeadAttention(is_causal=is_causal)
        self.mlp = MLP()

    def forward(self, x, cos_sin):
        x = x + self.attn(norm(x), cos_sin)
        x = x + self.mlp(norm(x))
        return x


class TinyLM(nn.Module):
    """Shared transformer for both diffusion and GPT modes."""

    def __init__(self, vocab_size, is_causal=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.is_causal = is_causal
        self.token_emb = nn.Embedding(vocab_size, N_EMBD)
        self.rotary_seq_len = BLOCK_SIZE * 2
        cos, sin = self._precompute_rotary(self.rotary_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.blocks = nn.ModuleList(
            [Block(is_causal=is_causal) for _ in range(N_LAYER)]
        )
        self.lm_head = nn.Linear(N_EMBD, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _precompute_rotary(self, seq_len, base=10000, device=None):
        if device is None:
            device = self.token_emb.weight.device
        channel_range = torch.arange(0, HEAD_DIM, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / HEAD_DIM))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        return cos[None, :, None, :], sin[None, :, None, :]

    def forward(self, idx, targets=None, mask=None):
        B, T = idx.size()
        x = self.token_emb(idx)
        x = norm(x)
        assert T <= self.cos.size(1)
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        for block in self.blocks:
            x = block(x, cos_sin)
        x = norm(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        logits_flat = logits.view(B * T, C)
        targets_flat = targets.view(B * T)
        if mask is not None:
            mask_flat = mask.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
            loss = (loss * mask_flat).sum() / mask_flat.sum()
        else:
            loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss


# Keep backward compat alias
DiffusionLM = TinyLM


class CharTokenizer:
    """Character-level tokenizer. Optionally includes a mask token."""

    def __init__(self, text, add_mask_token=True):
        chars = sorted(list(set(text)))
        if add_mask_token:
            self.chars = ["_"] + chars  # underscore as mask token
        else:
            self.chars = chars
        self.has_mask_token = add_mask_token
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.mask_token_id = self.stoi.get("_", None)

    def encode(self, s):
        return [self.stoi[ch] for ch in s]

    def decode(self, ids):
        return "".join([self.itos[i] for i in ids])

    def save(self, path):
        torch.save({"chars": self.chars, "has_mask_token": self.has_mask_token}, path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, weights_only=True)
        tok = cls.__new__(cls)
        tok.chars = data["chars"]
        tok.has_mask_token = data.get("has_mask_token", True)
        tok.vocab_size = len(tok.chars)
        tok.stoi = {ch: i for i, ch in enumerate(tok.chars)}
        tok.itos = {i: ch for i, ch in enumerate(tok.chars)}
        tok.mask_token_id = tok.stoi.get("_", None)
        return tok


# ---------------------------------------------------------------------------
# Generation: diffusion (parallel confidence-based decoding)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_diffusion(model, tokenizer, device, seed_text=None,
                       max_new_tokens=2000, prompt_len=16, temp=0.8,
                       confidence_threshold=0.95, top_k=2):
    """Generate text using confidence-based parallel decoding."""
    model.eval()

    if seed_text is not None:
        all_tokens = tokenizer.encode(seed_text[:prompt_len])
        prompt_len = len(all_tokens)
    else:
        all_tokens = [tokenizer.mask_token_id] * prompt_len

    total_steps = 0
    while len(all_tokens) - prompt_len < max_new_tokens:
        block_len = min(240, prompt_len + max_new_tokens - len(all_tokens))
        x = torch.full((1, BLOCK_SIZE), tokenizer.mask_token_id,
                        dtype=torch.long, device=device)
        x[0, :prompt_len] = torch.tensor(all_tokens[-prompt_len:], device=device)
        masked = torch.zeros(1, BLOCK_SIZE, dtype=torch.bool, device=device)
        masked[0, prompt_len:prompt_len + block_len] = True

        while masked.any():
            total_steps += 1
            logits, _ = model(x)
            probs = F.softmax(logits / temp, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
            confidences = top_k_probs.sum(dim=-1)

            decode_mask = (confidences >= confidence_threshold) & masked
            if not decode_mask.any():
                masked_conf = torch.where(masked, confidences,
                                          torch.tensor(-float("inf")))
                decode_mask.view(-1)[masked_conf.argmax()] = True

            top_k_probs_norm = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            sampled_k = torch.multinomial(
                top_k_probs_norm.view(-1, top_k), 1
            ).view(1, BLOCK_SIZE)
            sampled_tokens = torch.gather(
                top_k_indices, -1, sampled_k.unsqueeze(-1)
            ).squeeze(-1)

            x = torch.where(decode_mask, sampled_tokens, x)
            masked = masked & ~decode_mask

        all_tokens.extend(x[0, prompt_len:prompt_len + block_len].tolist())

    tokens_generated = len(all_tokens) - prompt_len
    print(f"Steps: {total_steps} for {tokens_generated} tokens "
          f"({tokens_generated / total_steps:.1f} tokens/step)")
    return tokenizer.decode(all_tokens)


# Keep backward compat alias
generate = generate_diffusion


# ---------------------------------------------------------------------------
# Generation: GPT (autoregressive)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_gpt(model, tokenizer, device, seed_text=None,
                 max_new_tokens=2000, prompt_len=16, temp=0.8):
    """Generate text autoregressively, one token at a time."""
    model.eval()

    if seed_text is not None:
        tokens = tokenizer.encode(seed_text[:prompt_len])
    else:
        tokens = [0] * prompt_len  # fallback

    x = torch.tensor([tokens], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        ctx = x[:, -BLOCK_SIZE:]
        logits, _ = model(ctx)
        logits = logits[:, -1, :]
        if temp == 0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / temp, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, next_token), dim=1)

    print(f"Steps: {max_new_tokens} for {max_new_tokens} tokens "
          f"(1.0 tokens/step)")
    return tokenizer.decode(x[0].tolist())
