#!/usr/bin/env python3
"""
TinyAdder: 36-parameter hand-crafted transformer for 10-digit addition.

Parameter counting:
- Identity mappings (direct copy): 0 params
- Broadcast (1 value to N outputs): 1 param
- Distinct values: count each
"""
import torch
import torch.nn.functional as F
from math import log, exp

# === Constants ===
NUM_DIGITS = 10
TOKENS = [str(i) for i in range(NUM_DIGITS)] + ["=", "<bos>", "<eos>", "+"]

POS_ANS_OUTPUT_START = 22
POS_ANS_OUTPUT_END = 33

DIGIT_EMBED_SCALE = 10
V_SCALE = 1e4
DIGIT_SCALE = 1e10
FINAL_SCALE = 100
DIGIT_OFFSET = 0.5
GATE_BIAS_SHIFT = 15.0
ALIBI_CONSTANT = log(10)

EQ_DIM, SPECIAL_DIM, DIGIT_DIM, COUNT_DIM, SCALE_DIM = 0, 1, 2, 3, 4
EMBEDDING_DIM = 5
LAYER0_HEADS = 5
ADJUSTMENT_HEAD = 3
SCALE_HEAD = 4
CANDIDATES_START = 5
DIGIT_POS_DIM = 15
LAYER1_D_MODEL = 16

K_DIGIT_SCORE = -1000.0
K_SPECIAL_SCORE = -40.0
V_PROJ_SPECIAL = 0.1
V_PROJ_NEG_DOUBLE = -1.1
V_PROJ_SCALE = exp(K_SPECIAL_SCORE - log(10))


def softmax1(x, dim=-1):
    exp_x = x.exp()
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))


def apply_alibi(seq_len, n_heads):
    pos = torch.arange(seq_len)
    rel_pos = pos.unsqueeze(0) - pos.unsqueeze(1)
    slopes = torch.zeros(n_heads, dtype=torch.float64)
    slopes[ADJUSTMENT_HEAD] = ALIBI_CONSTANT
    return slopes.unsqueeze(1).unsqueeze(2) * rel_pos.unsqueeze(0)


def pad_to(x, d):
    if x.size(-1) >= d:
        return x[..., :d]
    return torch.cat([x, torch.zeros(*x.shape[:-1], d - x.size(-1), dtype=x.dtype)], dim=-1)


class TinyAdder:
    """
    36-parameter transformer for 10-digit addition.

    Params: 13 emb + 6 L0-attn + 12 L0-ffn + 2 L1-attn + 3 L1-ffn = 36
    """

    def __init__(self):
        d = torch.float64

        # === EMBEDDING (13 params) ===
        # 9 digit values (1-9) + 4 special flags
        emb_idx = [[i, DIGIT_DIM] for i in range(1, 10)]
        emb_idx += [[10, EQ_DIM], [10, SPECIAL_DIM], [11, SPECIAL_DIM], [13, SPECIAL_DIM]]
        emb_val = [float(i * DIGIT_EMBED_SCALE) for i in range(1, 10)] + [1.0, 1.0, 1.0, 1.0]
        self.embedding = torch.sparse_coo_tensor(
            torch.tensor(emb_idx).T, torch.tensor(emb_val, dtype=d), (14, 5)
        ).to_dense()

        # === L0 ATTENTION (6 params) ===
        # q: bias=1 broadcast (1), k: weight+bias (2), v: 3 weights (3)
        self.k0_weight = torch.tensor(K_SPECIAL_SCORE - K_DIGIT_SCORE, dtype=d)
        self.k0_bias = torch.tensor(K_DIGIT_SCORE, dtype=d)
        self.v0_w1 = torch.tensor(V_PROJ_SPECIAL / V_PROJ_SCALE, dtype=d)
        self.v0_w2 = torch.tensor(V_PROJ_NEG_DOUBLE / V_PROJ_SCALE, dtype=d)
        self.v0_w3 = torch.tensor(1.0, dtype=d)

        # === L0 FFN (12 params) ===
        # gate: weight=1 broadcast (1), up: 11 values (11), down: identity (0)
        pv = [(i + DIGIT_OFFSET) * DIGIT_SCALE * FINAL_SCALE for i in range(NUM_DIGITS)]
        self.up0_vals = torch.tensor(pv + [DIGIT_SCALE], dtype=d)

        # === L1 ATTENTION (2 params) ===
        # v: weight (1) + bias (1)

        # === L1 FFN (3 params) ===
        # gate: +V_SCALE (1), -V_SCALE (1), up: FINAL_SCALE broadcast (1), down: identity (0)

    @torch.inference_mode()
    def forward(self, x):
        batch_size, seq_len = x.shape
        d = torch.float64
        h = self.embedding[x]

        # === LAYER 0 ===
        h = pad_to(h, EMBEDDING_DIM)

        # Q = 1 broadcast
        q = torch.ones(batch_size, seq_len, LAYER0_HEADS, dtype=d)

        # K: only ADJUSTMENT_HEAD reads SPECIAL_DIM
        k = torch.zeros(batch_size, seq_len, LAYER0_HEADS, dtype=d)
        k[..., ADJUSTMENT_HEAD] = h[..., SPECIAL_DIM] * self.k0_weight + self.k0_bias

        # V: sparse reads
        v = torch.zeros(batch_size, seq_len, LAYER0_HEADS, dtype=d)
        v[..., ADJUSTMENT_HEAD] = h[..., SPECIAL_DIM] * self.v0_w1 + h[..., EQ_DIM] * self.v0_w2
        v[..., SCALE_HEAD] = h[..., EQ_DIM] * self.v0_w3

        q = q.view(batch_size, seq_len, LAYER0_HEADS, 1).transpose(1, 2)
        k = k.view(batch_size, seq_len, LAYER0_HEADS, 1).transpose(1, 2)
        v = v.view(batch_size, seq_len, LAYER0_HEADS, 1).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) + apply_alibi(seq_len, LAYER0_HEADS).unsqueeze(0)
        scores = scores.masked_fill(torch.triu(torch.ones(seq_len, seq_len), 1).bool(), float('-inf'))
        attn = softmax1(scores, dim=-1).double()
        h = h + torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # FFN: gate=1 broadcast, up=distinct values, down=identity
        gate_in = torch.zeros(batch_size, seq_len, 11, dtype=d)
        gate_in[..., :NUM_DIGITS] = h[..., SCALE_DIM:SCALE_DIM+1]  # broadcast
        gate_in[..., NUM_DIGITS] = h[..., DIGIT_DIM]
        gate_out = F.relu(gate_in)
        up_out = h[..., COUNT_DIM:COUNT_DIM+1] * self.up0_vals
        ffn_hidden = gate_out * up_out

        h = pad_to(h, LAYER1_D_MODEL)
        h[..., 5:16] = h[..., 5:16] + ffn_hidden  # identity down projection

        # === LAYER 1: Attention ===
        # Explicit Q,K,V (with Q,K = 0) to make the attention operation clear.
        q = torch.zeros(batch_size, seq_len, 1, dtype=d)
        k = torch.zeros(batch_size, seq_len, 1, dtype=d)
        # V projection: dot with a fixed vector selecting DIGIT_POS_DIM, then add bias.
        v_weight = torch.zeros(LAYER1_D_MODEL, dtype=d)
        v_weight[DIGIT_POS_DIM] = FINAL_SCALE
        v = (h * v_weight).sum(dim=-1, keepdim=True) + GATE_BIAS_SHIFT

        q = q.view(batch_size, seq_len, 1, 1).transpose(1, 2)
        k = k.view(batch_size, seq_len, 1, 1).transpose(1, 2)
        v = v.view(batch_size, seq_len, 1, 1).transpose(1, 2)

        # Uniform causal attention (Q=0, K=0) via masked softmax.
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores.masked_fill(torch.triu(torch.ones(seq_len, seq_len), 1).bool(), float('-inf'))
        attn = softmax1(scores, dim=-1).double()
        h = h + torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # FFN: V-shape via relu(+x) + relu(-x), down=identity sum
        candidates = h[..., CANDIDATES_START:CANDIDATES_START+NUM_DIGITS]
        gate_pos = F.relu(candidates * V_SCALE)
        gate_neg = F.relu(candidates * -V_SCALE)
        ffn_out = (gate_pos + gate_neg) * FINAL_SCALE

        h = pad_to(h, NUM_DIGITS)
        h = h + ffn_out

        return h.argmin(dim=-1)


def build_model():
    """Build and return the model with metadata."""
    model = TinyAdder()
    metadata = {
        "name": "TinyAdder",
        "author": "Alex Litzenberger",
        "params": 36,
        "architecture": "2-layer transformer with ALiBi, ReGLU FFN",
        "tricks": [
            "ALiBi positional encoding",
            "softmax1",
            "Identity mappings (0 params)",
            "Broadcast parameters",
            "Double Precision"
        ],
    }
    return model, metadata


def add(model, a: int, b: int) -> int:
    """Add two integers using the model, autoregressively generating the sum digits."""
    S = f"{a:010d}+{b:010d}="
    generated = []
    for i in range(11):
        # Build a full-length sequence with placeholders for yet-unknown digits.
        toks = [TOKENS.index(t) for t in ["<bos>"] + list(S)]
        x = torch.tensor(toks).unsqueeze(0)
        pred = model.forward(x)
        next_digit = TOKENS[int(pred[0, -1].item())]
        S += next_digit
        generated.append(next_digit)
    return int("".join(generated))


if __name__ == "__main__":
    model, meta = build_model()
    print(f"Model: {meta['name']}")
    print(f"Parameters: {meta['params']}")
    print("\nBreakdown:")
    print("  Embedding:  13 (9 digits + 4 flags)")
    print("  L0 Attn:     6 (q bias, k w+b, v×3)")
    print("  L0 FFN:     12 (gate bcast, up×11)")
    print("  L1 Attn:     2 (v w+b)")
    print("  L1 FFN:      3 (±V_SCALE, up bcast)")
    print("  ───────────────")
    print("  Total:      36")

    import random
    random.seed(42)
    correct = 0
    for _ in range(100):
        a = random.randint(0, 9_999_999_999)
        b = random.randint(0, 9_999_999_999)
        if add(model, a, b) == a + b:
            correct += 1
    print(f"\nSelf-test: {correct}/100")

