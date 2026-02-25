"""
Explicit Torch encoding of xangma gist compact variant:
rank1+embed2+sparse_gate0+no_norm_weight
"""

from __future__ import annotations

from typing import List

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


def _rms_norm_no_weight(x: torch.Tensor, scale: float, eps: float = RMS_NORM_EPS) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * scale


def _apply_rope(x: torch.Tensor, base: float = ROPE_THETA) -> torch.Tensor:
    # x: [B, H, L, D], D=2 in this model.
    _, _, seq_len, dim = x.shape
    half = dim // 2
    device = x.device
    dtype = x.dtype
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    pos = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(pos, inv_freq)
    cos = freqs.cos().view(1, 1, seq_len, half)
    sin = freqs.sin().view(1, 1, seq_len, half)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


class Rank1Linear:
    def __init__(self, u: List[float], v: List[float]):
        self.u = torch.tensor(u, dtype=torch.float32)
        self.v = torch.tensor(v, dtype=torch.float32)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        s = (x * self.v).sum(dim=-1, keepdim=True)
        return s * self.u


class FactorizedEmbedding:
    def __init__(self):
        self.A = torch.tensor([[1.0, float(i)] for i in range(VOCAB_SIZE)], dtype=torch.float32)
        self.B = torch.tensor(
            [[100.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32
        )

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        return self.A[ids] @ self.B


class SparseGateProj0:
    def __init__(self):
        self.W23 = torch.tensor(
            [[-3.3532020e-01, -1.3412670e03, 6.0353305e04], [-1.3743691e01, -1.3418693e03, 6.0353277e04]],
            dtype=torch.float32,
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y2 = x[..., :3] @ self.W23.T
        pad = torch.zeros((*y2.shape[:-1], 1), dtype=y2.dtype, device=y2.device)
        return torch.cat([y2, pad], dim=-1)


class XangmaCompactTorch:
    def __init__(self):
        self.embed = FactorizedEmbedding()
        self.gate0 = SparseGateProj0()
        self.lm_head = torch.tensor(
            [
                [5.5779090e00, 3.1322198e00, -4.0438358e02, 6.2589108e01, 9.9358273e-01],
                [5.0814748e00, 2.4687927e00, -3.1444955e02, 4.8671352e01, 7.7272820e-01],
                [3.6916721e00, 1.7657869e00, -2.2455742e02, 3.4757641e01, 5.5075526e-01],
                [1.4084998e00, 1.0232025e00, -1.3470717e02, 2.0847967e01, 3.2766387e-01],
                [-1.7680415e00, 2.4103954e-01, -4.4898785e01, 6.9423370e00, 1.0345399e-01],
                [-5.8379521e00, -5.8070201e-01, 4.4867714e01, -6.9592528e00, -1.2187435e-01],
                [-1.0801232e01, -1.4420221e00, 1.3459233e02, -2.0856800e01, -3.4832114e-01],
                [-1.6657881e01, -2.3429208e00, 2.2427509e02, -3.4750309e01, -5.7588643e-01],
                [-2.3407900e01, -3.2833982e00, 3.1391595e02, -4.8639774e01, -8.0457014e-01],
                [-3.1051287e01, -4.2634540e00, 4.0351492e02, -6.2525200e01, -1.0343723e00],
            ],
            dtype=torch.float32,
        )
        self.gate1_w = torch.tensor(
            [
                [-4.3951669e-01, 5.6323919e00, 4.9838150e-01, 1.3435575e03, 6.0357680e04],
                [-1.2112466e02, 3.2923722e-01, -5.0313854e00, 1.3449166e03, 6.0357438e04],
                [-1.3453412e02, -2.6000220e-01, -5.6458039e00, 1.3450677e03, 6.0357410e04],
            ],
            dtype=torch.float32,
        )

        # layer 0 rank-1
        self.q0 = Rank1Linear([0.98502123, 0.17243294, 0.96630472, -0.25740093], [1.0, 0.0, 0.0, 0.0, 0.0])
        self.k0 = Rank1Linear([-0.31672141, -0.94851863], [1.0, 0.0, 0.0, 0.0, 0.0])
        self.v0 = Rank1Linear([1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0])
        self.o0 = Rank1Linear([0.0, 0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0])
        self.up0 = Rank1Linear([1.0, 1.0, 0.0], [1.4898191e-02, 6.6922739e-04, 2.9977213e-05, 0.0, 0.0])
        self.down0 = Rank1Linear([0.0, 0.0, 0.0, 1.0, 0.0], [1.0, -1.0, 0.0])

        # layer 1 rank-1
        self.q1 = Rank1Linear([-0.25507239, 0.96692199, 0.17478994, 0.98460573], [1.0, 0.0, 0.0, 0.0, 0.0])
        self.k1 = Rank1Linear([0.32702553, -0.94501549], [1.0, 0.0, 0.0, 0.0, 0.0])
        self.v1 = Rank1Linear([1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0])
        self.o1 = Rank1Linear([0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0])
        self.up1 = Rank1Linear(
            [1.0, 1.0, 1.0],
            [1.4899401e-02, 6.5471046e-04, 6.8268733e-04, -1.6779384e-04, 2.9817384e-05],
        )
        self.down1 = Rank1Linear([0.0, 0.0, 1.0, 0.0, 0.0], [1.0, -10.0, 10.0])

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        x = self._block0(x)
        x = self._block1(x)
        x = _rms_norm_no_weight(x, scale=1.0)
        return x @ self.lm_head.T

    def _attention(self, x: torch.Tensor, qf, kf, vf, of) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = qf(x).view(bsz, seq_len, ATTENTION_HEADS, HEAD_DIM)
        k = kf(x).view(bsz, seq_len, KEY_VALUE_HEADS, HEAD_DIM)
        v = vf(x).view(bsz, seq_len, KEY_VALUE_HEADS, HEAD_DIM)
        q = _rms_norm_no_weight(q, scale=16.0).permute(0, 2, 1, 3)
        k = _rms_norm_no_weight(k, scale=16.0).permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        q = _apply_rope(q)
        k = _apply_rope(k)
        k = k.repeat_interleave(ATTENTION_HEADS // KEY_VALUE_HEADS, dim=1)
        v = v.repeat_interleave(ATTENTION_HEADS // KEY_VALUE_HEADS, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (HEAD_DIM**-0.5)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(causal_mask.view(1, 1, seq_len, seq_len), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, ATTENTION_HEADS * HEAD_DIM)
        return of(out)

    def _mlp0(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate0(x)
        up = self.up0(x)
        return self.down0(F.silu(gate) * up)

    def _mlp1(self, x: torch.Tensor) -> torch.Tensor:
        gate = x @ self.gate1_w.T
        up = self.up1(x)
        return self.down1(F.silu(gate) * up)

    def _block0(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self._attention(_rms_norm_no_weight(x, scale=1.0), self.q0, self.k0, self.v0, self.o0)
        return h + self._mlp0(_rms_norm_no_weight(h, scale=1.0))

    def _block1(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self._attention(_rms_norm_no_weight(x, scale=1.0), self.q1, self.k1, self.v1, self.o1)
        return h + self._mlp1(_rms_norm_no_weight(h, scale=1.0))


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
    model = XangmaCompactTorch()
    metadata = {
        "name": "xangma Torch Port (explicit compact)",
        "author": "xangma + CVlization port",
        "params": 197,
        "architecture": "2L decoder, d=5, 2h/1kv, hd=2",
        "tricks": [
            "rank-1 linear",
            "factorized embedding",
            "sparse gate",
            "param-free norm",
            "explicit decomposed modules in torch",
        ],
    }
    return model, metadata


def add(model, a: int, b: int) -> int:
    seq = _encode_addends_internal(a, b)
    with torch.no_grad():
        for _ in range(OUTPUT_DIGITS):
            x = torch.tensor([seq], dtype=torch.long)
            logits = model(x)
            seq.append(int(torch.argmax(logits[0, -1]).item()))
    pred_reversed = "".join(str(d) for d in seq[-OUTPUT_DIGITS:])
    return int(pred_reversed[::-1])
