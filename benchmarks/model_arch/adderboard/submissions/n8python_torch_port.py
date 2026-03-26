"""
Torch port of N8python's hand-coded MLX Qwen3 adder gist.

This submission parses the weight literals from `n8python_02e41d15.py`
and runs an equivalent tiny decoder-only forward pass in PyTorch.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

MODEL_LAYERS = 2
MODEL_DIM = 5
ATTENTION_HEADS = 2
KEY_VALUE_HEADS = 1
HEAD_DIM = 2
INTERMEDIATE_SIZE = 3
VOCAB_SIZE = 10
OUTPUT_DIGITS = 11
MAX_ADDEND = 10**10 - 1
RMS_NORM_EPS = 1e-6
ROPE_THETA = 10000.0


def _extract_subscript_path(node: ast.Subscript) -> List[str | int]:
    path: List[str | int] = []
    cur = node
    while isinstance(cur, ast.Subscript):
        slc = cur.slice
        if isinstance(slc, ast.Constant):
            key = slc.value
        else:
            raise ValueError(f"Unsupported subscript slice node: {ast.dump(slc)}")
        if not isinstance(key, (str, int)):
            raise ValueError(f"Unsupported key type: {type(key)}")
        path.append(key)
        cur = cur.value
    if not isinstance(cur, ast.Name) or cur.id != "params":
        raise ValueError("Expected assignment target rooted at `params`")
    path.reverse()
    return path


def _parse_weights_from_gist_source(gist_path: Path) -> Dict[Tuple[str | int, ...], torch.Tensor]:
    src = gist_path.read_text(encoding="utf-8")
    mod = ast.parse(src)

    fn = None
    for node in mod.body:
        if isinstance(node, ast.FunctionDef) and node.name == "hand_set_weights_magic":
            fn = node
            break
    if fn is None:
        raise ValueError("Could not find hand_set_weights_magic in source")

    weights: Dict[Tuple[str | int, ...], torch.Tensor] = {}
    for stmt in fn.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        if not isinstance(target, ast.Subscript):
            continue
        path = _extract_subscript_path(target)

        value = stmt.value
        if not isinstance(value, ast.Call):
            continue
        if not isinstance(value.func, ast.Attribute):
            continue
        if not (isinstance(value.func.value, ast.Name) and value.func.value.id == "mx"):
            continue
        if value.func.attr != "array":
            continue
        if not value.args:
            continue

        literal = ast.literal_eval(value.args[0])
        tensor = torch.tensor(literal, dtype=torch.float32)
        weights[tuple(path)] = tensor

    if ("lm_head", "weight") not in weights:
        raise ValueError("Failed to parse expected lm_head weights")
    return weights


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = RMS_NORM_EPS) -> torch.Tensor:
    denom = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * denom * weight


def _apply_rope(x: torch.Tensor, base: float = ROPE_THETA) -> torch.Tensor:
    # x: [B, H, L, D], D=2 in this model.
    bsz, n_heads, seq_len, dim = x.shape
    half = dim // 2
    device = x.device
    dtype = x.dtype
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    pos = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(pos, inv_freq)  # [L, half]
    cos = freqs.cos().view(1, 1, seq_len, half)
    sin = freqs.sin().view(1, 1, seq_len, half)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    out = torch.empty_like(x)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out


class TinyQwen3Torch:
    def __init__(self, weights: Dict[Tuple[str | int, ...], torch.Tensor]):
        self.w = weights

    def _get(self, *path: str | int) -> torch.Tensor:
        return self.w[tuple(path)]

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, L]
        x = self._get("model", "embed_tokens", "weight")[input_ids]  # [B, L, D]

        for layer_idx in range(MODEL_LAYERS):
            x_ln = _rms_norm(
                x, self._get("model", "layers", layer_idx, "input_layernorm", "weight")
            )
            attn = self._attention(layer_idx, x_ln)
            h = x + attn
            h_ln = _rms_norm(
                h,
                self._get("model", "layers", layer_idx, "post_attention_layernorm", "weight"),
            )
            mlp = self._mlp(layer_idx, h_ln)
            x = h + mlp

        x = _rms_norm(x, self._get("model", "norm", "weight"))
        logits = torch.matmul(x, self._get("lm_head", "weight").T)
        return logits

    def _attention(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = torch.matmul(x, self._get("model", "layers", layer_idx, "self_attn", "q_proj", "weight").T)
        k = torch.matmul(x, self._get("model", "layers", layer_idx, "self_attn", "k_proj", "weight").T)
        v = torch.matmul(x, self._get("model", "layers", layer_idx, "self_attn", "v_proj", "weight").T)

        q = q.view(bsz, seq_len, ATTENTION_HEADS, HEAD_DIM)
        k = k.view(bsz, seq_len, KEY_VALUE_HEADS, HEAD_DIM)
        v = v.view(bsz, seq_len, KEY_VALUE_HEADS, HEAD_DIM)

        q = _rms_norm(
            q, self._get("model", "layers", layer_idx, "self_attn", "q_norm", "weight")
        ).permute(0, 2, 1, 3)
        k = _rms_norm(
            k, self._get("model", "layers", layer_idx, "self_attn", "k_norm", "weight")
        ).permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        q = _apply_rope(q)
        k = _apply_rope(k)

        # Grouped-query attention (repeat K/V heads across Q heads).
        kv_repeat = ATTENTION_HEADS // KEY_VALUE_HEADS
        if kv_repeat > 1:
            k = k.repeat_interleave(kv_repeat, dim=1)
            v = v.repeat_interleave(kv_repeat, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * (HEAD_DIM ** -0.5)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(causal_mask.view(1, 1, seq_len, seq_len), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # [B, H, L, hd]
        out = out.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, ATTENTION_HEADS * HEAD_DIM)
        out = torch.matmul(
            out, self._get("model", "layers", layer_idx, "self_attn", "o_proj", "weight").T
        )
        return out

    def _mlp(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        gate = torch.matmul(x, self._get("model", "layers", layer_idx, "mlp", "gate_proj", "weight").T)
        up = torch.matmul(x, self._get("model", "layers", layer_idx, "mlp", "up_proj", "weight").T)
        hidden = F.silu(gate) * up
        down = torch.matmul(hidden, self._get("model", "layers", layer_idx, "mlp", "down_proj", "weight").T)
        return down


def _validate_addends(a: int, b: int) -> None:
    if not isinstance(a, int) or not isinstance(b, int):
        raise ValueError("a and b must be ints")
    if a < 0 or a > MAX_ADDEND or b < 0 or b > MAX_ADDEND:
        raise ValueError(f"a and b must be in [0, {MAX_ADDEND}]")


def _encode_addends_internal(a: int, b: int) -> List[int]:
    _validate_addends(a, b)
    prompt = f"{a:010d}{b:010d}"
    ad = [int(c) for c in prompt[:10]]
    bd = [int(c) for c in prompt[10:]]
    return [0] + list(reversed(ad)) + [0] + [0] + list(reversed(bd)) + [0]


def build_model():
    gist_path = Path(__file__).with_name("n8python_02e41d15.py")
    weights = _parse_weights_from_gist_source(gist_path)
    model = TinyQwen3Torch(weights)
    metadata = {
        "name": "N8python Torch Port",
        "author": "N8python + CVlization port",
        "params": 241,
        "architecture": "2L decoder, d=5, 2h/1kv, hd=2",
        "tricks": ["hand-coded weights", "torch port of MLX gist"],
    }
    return model, metadata


def add(model, a: int, b: int) -> int:
    seq = _encode_addends_internal(a, b)
    with torch.no_grad():
        for _ in range(OUTPUT_DIGITS):
            x = torch.tensor([seq], dtype=torch.long)
            logits = model(x)
            next_digit = int(torch.argmax(logits[0, -1]).item())
            seq.append(next_digit)
    pred_reversed = "".join(str(d) for d in seq[-OUTPUT_DIGITS:])
    return int(pred_reversed[::-1])
