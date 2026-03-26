import argparse
import random

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_map
from mlx_lm.models.qwen3 import Model, ModelArgs

MODEL_LAYERS = 2
MODEL_DIM = 5
ATTENTION_HEADS = 2
KEY_VALUE_HEADS = 1
HEAD_DIM = 2
INTERMEDIATE_SIZE = 3
VOCAB_SIZE = 10
OUTPUT_DIGITS = 11
MAX_ADDEND = 10**10 - 1
VARIANT_CHOICES = (
    "baseline",
    "rank1",
    "rank1+embed2",
    "rank1+embed2+sparse_gate0",
    "rank1+embed2+sparse_gate0+no_norm_weight",
)
VARIANT_FEATURES = {
    "baseline": frozenset(),
    "rank1": frozenset({"rank1"}),
    "rank1+embed2": frozenset({"rank1", "embed2"}),
    "rank1+embed2+sparse_gate0": frozenset({"rank1", "embed2", "sparse_gate0"}),
    "rank1+embed2+sparse_gate0+no_norm_weight": frozenset(
        {"rank1", "embed2", "sparse_gate0", "no_norm_weight"}
    ),
}


def build_model_args() -> ModelArgs:
    return ModelArgs(
        model_type="qwen3",
        hidden_size=MODEL_DIM,
        num_hidden_layers=MODEL_LAYERS,
        intermediate_size=INTERMEDIATE_SIZE,
        num_attention_heads=ATTENTION_HEADS,
        rms_norm_eps=1e-6,
        vocab_size=VOCAB_SIZE,
        tie_word_embeddings=False,
        num_key_value_heads=KEY_VALUE_HEADS,
        max_position_embeddings=2048,
        rope_theta=10000,
        head_dim=HEAD_DIM,
    )


def _validate_addends(a: int, b: int) -> None:
    if not isinstance(a, int) or not isinstance(b, int):
        raise ValueError("a and b must be ints")
    if a < 0 or a > MAX_ADDEND or b < 0 or b > MAX_ADDEND:
        raise ValueError(f"a and b must be in [0, {MAX_ADDEND}]")


def _encode_addends_internal(a: int, b: int) -> list[int]:
    _validate_addends(a, b)
    prompt = f"{a:010d}{b:010d}"
    a = [int(c) for c in prompt[:10]]
    b = [int(c) for c in prompt[10:]]
    return [0] + list(reversed(a)) + [0] + [0] + list(reversed(b)) + [0]


def _expected_output(a: int, b: int) -> str:
    _validate_addends(a, b)
    return str(a + b)[::-1].ljust(OUTPUT_DIGITS, "0")


class Rank1Linear(nn.Module):
    def __init__(self, out_features: int, in_features: int):
        super().__init__()
        self.u = mx.zeros((out_features,), dtype=mx.float32)
        self.v = mx.zeros((in_features,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        s = mx.sum(x * self.v, axis=-1, keepdims=True)
        return s * self.u


class FactorizedEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.A = mx.zeros((vocab_size, 2), dtype=mx.float32)
        self.B = mx.zeros((2, dim), dtype=mx.float32)

    def __call__(self, ids: mx.array) -> mx.array:
        return self.A[ids] @ self.B


class SparseGateProj0(nn.Module):
    def __init__(self):
        super().__init__()
        self.W23 = mx.zeros((2, 3), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        x3 = x[..., :3]
        y2 = x3 @ self.W23.T
        pad = mx.zeros((*y2.shape[:-1], 1), dtype=y2.dtype)
        return mx.concatenate([y2, pad], axis=-1)


class RMSNormNoWeight(nn.Module):
    def __init__(self, dims: int, eps: float, scale: float):
        super().__init__()
        self.dims = dims
        self.eps = eps
        self.scale = scale

    def __call__(self, x: mx.array) -> mx.array:
        weight = mx.full((self.dims,), self.scale, dtype=x.dtype)
        return mx.fast.rms_norm(x, weight, self.eps)


def _set_param(params, path: list, value) -> None:
    node = params
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = mx.array(value, dtype=mx.float32)


def _set_rank1(
    params,
    layer_idx: int,
    module_path: str,
    u,
    v,
    w_original,
    verify_rank1: bool,
) -> None:
    node = params["model"]["layers"][layer_idx]
    for key in module_path.split("."):
        node = node[key]
    u_arr = mx.array(u, dtype=mx.float32)
    v_arr = mx.array(v, dtype=mx.float32)
    node["u"] = u_arr
    node["v"] = v_arr
    if verify_rank1:
        w_hat = u_arr[:, None] * v_arr[None, :]
        w_orig_arr = mx.array(w_original, dtype=mx.float32)
        if not bool(np.array(mx.allclose(w_hat, w_orig_arr)).item()):
            raise AssertionError(
                f"Rank-1 reconstruction mismatch for layer {layer_idx} {module_path}"
            )


def apply_variant_monkeypatches(model: Model, variant: str) -> None:
    features = VARIANT_FEATURES[variant]

    if "rank1" in features:
        for layer in model.model.layers:
            layer.self_attn.q_proj = Rank1Linear(4, 5)
            layer.self_attn.k_proj = Rank1Linear(2, 5)
            layer.self_attn.v_proj = Rank1Linear(2, 5)
            layer.self_attn.o_proj = Rank1Linear(5, 4)
            layer.mlp.up_proj = Rank1Linear(3, 5)
            layer.mlp.down_proj = Rank1Linear(5, 3)

    if "embed2" in features:
        model.model.embed_tokens = FactorizedEmbedding(VOCAB_SIZE, MODEL_DIM)

    if "sparse_gate0" in features:
        model.model.layers[0].mlp.gate_proj = SparseGateProj0()

    if "no_norm_weight" in features:
        for layer in model.model.layers:
            layer.input_layernorm = RMSNormNoWeight(
                MODEL_DIM, eps=model.args.rms_norm_eps, scale=1.0
            )
            layer.post_attention_layernorm = RMSNormNoWeight(
                MODEL_DIM, eps=model.args.rms_norm_eps, scale=1.0
            )
            layer.self_attn.q_norm = RMSNormNoWeight(
                HEAD_DIM, eps=model.args.rms_norm_eps, scale=16.0
            )
            layer.self_attn.k_norm = RMSNormNoWeight(
                HEAD_DIM, eps=model.args.rms_norm_eps, scale=16.0
            )
        model.model.norm = RMSNormNoWeight(
            MODEL_DIM, eps=model.args.rms_norm_eps, scale=1.0
        )


def hand_set_weights_magic(
    model: Model, variant: str, verify_rank1: bool = False
) -> None:
    features = VARIANT_FEATURES[variant]
    use_rank1 = "rank1" in features
    use_embed2 = "embed2" in features
    use_sparse_gate0 = "sparse_gate0" in features
    use_no_norm = "no_norm_weight" in features

    params = tree_map(lambda x: mx.zeros_like(x), model.parameters())
    _set_param(
        params,
        ["lm_head", "weight"],
        [
            [5.5779090e00, 3.1322198e00, -4.0438358e02, 6.2589108e01, 9.9358273e-01],
            [5.0814748e00, 2.4687927e00, -3.1444955e02, 4.8671352e01, 7.7272820e-01],
            [3.6916721e00, 1.7657869e00, -2.2455742e02, 3.4757641e01, 5.5075526e-01],
            [1.4084998e00, 1.0232025e00, -1.3470717e02, 2.0847967e01, 3.2766387e-01],
            [-1.7680415e00, 2.4103954e-01, -4.4898785e01, 6.9423370e00, 1.0345399e-01],
            [
                -5.8379521e00,
                -5.8070201e-01,
                4.4867714e01,
                -6.9592528e00,
                -1.2187435e-01,
            ],
            [-1.0801232e01, -1.4420221e00, 1.3459233e02, -2.0856800e01, -3.4832114e-01],
            [-1.6657881e01, -2.3429208e00, 2.2427509e02, -3.4750309e01, -5.7588643e-01],
            [-2.3407900e01, -3.2833982e00, 3.1391595e02, -4.8639774e01, -8.0457014e-01],
            [-3.1051287e01, -4.2634540e00, 4.0351492e02, -6.2525200e01, -1.0343723e00],
        ],
    )

    if use_embed2:
        _set_param(
            params,
            ["model", "embed_tokens", "A"],
            [[1.0, float(i)] for i in range(VOCAB_SIZE)],
        )
        _set_param(
            params,
            ["model", "embed_tokens", "B"],
            [[100.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]],
        )
    else:
        _set_param(
            params,
            ["model", "embed_tokens", "weight"],
            [[100.0, float(i), 0.0, 0.0, 0.0] for i in range(VOCAB_SIZE)],
        )

    if not use_no_norm:
        for layer_idx in range(MODEL_LAYERS):
            _set_param(
                params,
                ["model", "layers", layer_idx, "input_layernorm", "weight"],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            )
            _set_param(
                params,
                ["model", "layers", layer_idx, "post_attention_layernorm", "weight"],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            )
            _set_param(
                params,
                ["model", "layers", layer_idx, "self_attn", "q_norm", "weight"],
                [16.0, 16.0],
            )
            _set_param(
                params,
                ["model", "layers", layer_idx, "self_attn", "k_norm", "weight"],
                [16.0, 16.0],
            )
        _set_param(params, ["model", "norm", "weight"], [1.0, 1.0, 1.0, 1.0, 1.0])

    if use_sparse_gate0:
        _set_param(
            params,
            ["model", "layers", 0, "mlp", "gate_proj", "W23"],
            [
                [-3.3532020e-01, -1.3412670e03, 6.0353305e04],
                [-1.3743691e01, -1.3418693e03, 6.0353277e04],
            ],
        )
    else:
        _set_param(
            params,
            ["model", "layers", 0, "mlp", "gate_proj", "weight"],
            [
                [-3.3532020e-01, -1.3412670e03, 6.0353305e04, 0.0, 0.0],
                [-1.3743691e01, -1.3418693e03, 6.0353277e04, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        )

    _set_param(
        params,
        ["model", "layers", 1, "mlp", "gate_proj", "weight"],
        [
            [-4.3951669e-01, 5.6323919e00, 4.9838150e-01, 1.3435575e03, 6.0357680e04],
            [-1.2112466e02, 3.2923722e-01, -5.0313854e00, 1.3449166e03, 6.0357438e04],
            [-1.3453412e02, -2.6000220e-01, -5.6458039e00, 1.3450677e03, 6.0357410e04],
        ],
    )

    if use_rank1:
        _set_rank1(
            params,
            0,
            "self_attn.q_proj",
            [0.98502123, 0.17243294, 0.96630472, -0.25740093],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [
                [0.98502123, 0.0, 0.0, 0.0, 0.0],
                [0.17243294, 0.0, 0.0, 0.0, 0.0],
                [0.96630472, 0.0, 0.0, 0.0, 0.0],
                [-0.25740093, 0.0, 0.0, 0.0, 0.0],
            ],
            verify_rank1,
        )
        _set_rank1(
            params,
            0,
            "self_attn.k_proj",
            [-0.31672141, -0.94851863],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [
                [-0.31672141, 0.0, 0.0, 0.0, 0.0],
                [-0.94851863, 0.0, 0.0, 0.0, 0.0],
            ],
            verify_rank1,
        )
        _set_rank1(
            params,
            0,
            "self_attn.v_proj",
            [1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
            verify_rank1,
        )
        _set_rank1(
            params,
            0,
            "self_attn.o_proj",
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            verify_rank1,
        )
        _set_rank1(
            params,
            0,
            "mlp.up_proj",
            [1.0, 1.0, 0.0],
            [1.4898191e-02, 6.6922739e-04, 2.9977213e-05, 0.0, 0.0],
            [
                [1.4898191e-02, 6.6922739e-04, 2.9977213e-05, 0.0, 0.0],
                [1.4898191e-02, 6.6922739e-04, 2.9977213e-05, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            verify_rank1,
        )
        _set_rank1(
            params,
            0,
            "mlp.down_proj",
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, -1.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            verify_rank1,
        )
        _set_rank1(
            params,
            1,
            "self_attn.q_proj",
            [-0.25507239, 0.96692199, 0.17478994, 0.98460573],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [
                [-0.25507239, 0.0, 0.0, 0.0, 0.0],
                [0.96692199, 0.0, 0.0, 0.0, 0.0],
                [0.17478994, 0.0, 0.0, 0.0, 0.0],
                [0.98460573, 0.0, 0.0, 0.0, 0.0],
            ],
            verify_rank1,
        )
        _set_rank1(
            params,
            1,
            "self_attn.k_proj",
            [0.32702553, -0.94501549],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [[0.32702553, 0.0, 0.0, 0.0, 0.0], [-0.94501549, 0.0, 0.0, 0.0, 0.0]],
            verify_rank1,
        )
        _set_rank1(
            params,
            1,
            "self_attn.v_proj",
            [1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
            verify_rank1,
        )
        _set_rank1(
            params,
            1,
            "self_attn.o_proj",
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
            ],
            verify_rank1,
        )
        _set_rank1(
            params,
            1,
            "mlp.up_proj",
            [1.0, 1.0, 1.0],
            [1.4899401e-02, 6.5471046e-04, 6.8268733e-04, -1.6779384e-04, 2.9817384e-05],
            [
                [1.4899401e-02, 6.5471046e-04, 6.8268733e-04, -1.6779384e-04, 2.9817384e-05],
                [1.4899401e-02, 6.5471046e-04, 6.8268733e-04, -1.6779384e-04, 2.9817384e-05],
                [1.4899401e-02, 6.5471046e-04, 6.8268733e-04, -1.6779384e-04, 2.9817384e-05],
            ],
            verify_rank1,
        )
        _set_rank1(
            params,
            1,
            "mlp.down_proj",
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, -10.0, 10.0],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, -10.0, 10.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            verify_rank1,
        )
    else:
        _set_param(
            params,
            ["model", "layers", 0, "self_attn", "q_proj", "weight"],
            [
                [0.98502123, 0.0, 0.0, 0.0, 0.0],
                [0.17243294, 0.0, 0.0, 0.0, 0.0],
                [0.96630472, 0.0, 0.0, 0.0, 0.0],
                [-0.25740093, 0.0, 0.0, 0.0, 0.0],
            ],
        )
        _set_param(
            params,
            ["model", "layers", 0, "self_attn", "k_proj", "weight"],
            [[-0.31672141, 0.0, 0.0, 0.0, 0.0], [-0.94851863, 0.0, 0.0, 0.0, 0.0]],
        )
        _set_param(
            params,
            ["model", "layers", 0, "self_attn", "v_proj", "weight"],
            [[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
        )
        _set_param(
            params,
            ["model", "layers", 0, "self_attn", "o_proj", "weight"],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        )
        _set_param(
            params,
            ["model", "layers", 0, "mlp", "up_proj", "weight"],
            [
                [1.4898191e-02, 6.6922739e-04, 2.9977213e-05, 0.0, 0.0],
                [1.4898191e-02, 6.6922739e-04, 2.9977213e-05, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        )
        _set_param(
            params,
            ["model", "layers", 0, "mlp", "down_proj", "weight"],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, -1.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        )
        _set_param(
            params,
            ["model", "layers", 1, "self_attn", "q_proj", "weight"],
            [
                [-0.25507239, 0.0, 0.0, 0.0, 0.0],
                [0.96692199, 0.0, 0.0, 0.0, 0.0],
                [0.17478994, 0.0, 0.0, 0.0, 0.0],
                [0.98460573, 0.0, 0.0, 0.0, 0.0],
            ],
        )
        _set_param(
            params,
            ["model", "layers", 1, "self_attn", "k_proj", "weight"],
            [[0.32702553, 0.0, 0.0, 0.0, 0.0], [-0.94501549, 0.0, 0.0, 0.0, 0.0]],
        )
        _set_param(
            params,
            ["model", "layers", 1, "self_attn", "v_proj", "weight"],
            [[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
        )
        _set_param(
            params,
            ["model", "layers", 1, "self_attn", "o_proj", "weight"],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
            ],
        )
        _set_param(
            params,
            ["model", "layers", 1, "mlp", "up_proj", "weight"],
            [
                [1.4899401e-02, 6.5471046e-04, 6.8268733e-04, -1.6779384e-04, 2.9817384e-05],
                [1.4899401e-02, 6.5471046e-04, 6.8268733e-04, -1.6779384e-04, 2.9817384e-05],
                [1.4899401e-02, 6.5471046e-04, 6.8268733e-04, -1.6779384e-04, 2.9817384e-05],
            ],
        )
        _set_param(
            params,
            ["model", "layers", 1, "mlp", "down_proj", "weight"],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, -10.0, 10.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        )

    model.update(params)
    mx.eval(model.parameters())


def build_magic_model(variant: str, verify_rank1: bool = False) -> Model:
    model = Model(build_model_args())
    apply_variant_monkeypatches(model, variant)
    hand_set_weights_magic(model, variant, verify_rank1=verify_rank1)
    return model


def _generate_output_batch(model: Model, addends: list[tuple[int, int]]) -> list[str]:
    internal = [_encode_addends_internal(a, b) for a, b in addends]
    for _ in range(OUTPUT_DIGITS):
        x = mx.array(internal, dtype=mx.int32)
        logits = model(x)
        next_digits = np.array(mx.argmax(logits[:, -1, :], axis=-1), dtype=np.int32)
        for seq, next_digit in zip(internal, next_digits):
            seq.append(int(next_digit))
    return ["".join(str(d) for d in seq[-OUTPUT_DIGITS:]) for seq in internal]


def run_self_test_batched(model: Model, num_tests: int, batch_size: int) -> None:
    rng = random.Random(123)
    tested = 0
    while tested < num_tests:
        cur_batch_size = min(batch_size, num_tests - tested)
        addends = []
        expected = []
        for _ in range(cur_batch_size):
            a = rng.randint(0, 10**10 - 1)
            b = rng.randint(0, 10**10 - 1)
            addends.append((a, b))
            expected.append(_expected_output(a, b))
        actual = _generate_output_batch(model, addends)
        for (a, b), exp, act in zip(addends, expected, actual):
            if act != exp:
                raise AssertionError(
                    f"Mismatch for a={a:010d}, b={b:010d}: expected {exp}, got {act}"
                )
        tested += cur_batch_size
        print(f"self-test progress: {tested}/{num_tests}")


def count_parameters(node) -> int:
    if isinstance(node, dict):
        return sum(count_parameters(v) for v in node.values())
    if isinstance(node, (list, tuple)):
        return sum(count_parameters(v) for v in node)
    if hasattr(node, "shape"):
        n = 1
        for dim in node.shape:
            n *= int(dim)
        return n
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tests", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--variant", choices=VARIANT_CHOICES, default="baseline")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.num_tests = 1024
        args.batch_size = 256
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.num_tests < 0:
        raise ValueError("--num-tests must be >= 0")
    model = build_magic_model(args.variant, verify_rank1=args.quick)
    print(f"variant: {args.variant}")
    print(f"parameter count: {count_parameters(model.parameters())}")
    try:
        run_self_test_batched(model, args.num_tests, args.batch_size)
    except AssertionError as e:
        print(f"self-test: FAIL ({e})")
        raise SystemExit(1)
    print(f"self-test: PASS ({args.num_tests} random cases, batch size {args.batch_size})")


if __name__ == "__main__":
    main()

# Builds on @N8Programs' code here: https://gist.github.com/N8python/02e41d156ec615328cde2e1e5c0e9d53
