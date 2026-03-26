import argparse
import random

import mlx.core as mx
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


def hand_set_weights_magic(model: Model) -> None:
    params = tree_map(lambda x: mx.zeros_like(x), model.parameters())
    params['lm_head']['weight'] = mx.array([[ 5.5779090e+00,  3.1322198e+00, -4.0438358e+02,  6.2589108e+01,  9.9358273e-01],
 [ 5.0814748e+00,  2.4687927e+00, -3.1444955e+02,  4.8671352e+01,  7.7272820e-01],
 [ 3.6916721e+00,  1.7657869e+00, -2.2455742e+02,  3.4757641e+01,  5.5075526e-01],
 [ 1.4084998e+00,  1.0232025e+00, -1.3470717e+02,  2.0847967e+01,  3.2766387e-01],
 [-1.7680415e+00,  2.4103954e-01, -4.4898785e+01,  6.9423370e+00,  1.0345399e-01],
 [-5.8379521e+00, -5.8070201e-01,  4.4867714e+01, -6.9592528e+00, -1.2187435e-01],
 [-1.0801232e+01, -1.4420221e+00,  1.3459233e+02, -2.0856800e+01, -3.4832114e-01],
 [-1.6657881e+01, -2.3429208e+00,  2.2427509e+02, -3.4750309e+01, -5.7588643e-01],
 [-2.3407900e+01, -3.2833982e+00,  3.1391595e+02, -4.8639774e+01, -8.0457014e-01],
 [-3.1051287e+01, -4.2634540e+00,  4.0351492e+02, -6.2525200e+01, -1.0343723e+00]], dtype=mx.float32)
    params['model']['embed_tokens']['weight'] = mx.array([[100.,   0.,   0.,   0.,   0.],
 [100.,   1.,   0.,   0.,   0.],
 [100.,   2.,   0.,   0.,   0.],
 [100.,   3.,   0.,   0.,   0.],
 [100.,   4.,   0.,   0.,   0.],
 [100.,   5.,   0.,   0.,   0.],
 [100.,   6.,   0.,   0.,   0.],
 [100.,   7.,   0.,   0.,   0.],
 [100.,   8.,   0.,   0.,   0.],
 [100.,   9.,   0.,   0.,   0.]], dtype=mx.float32)
    params['model']['layers'][0]['input_layernorm']['weight'] = mx.array([1., 1., 1., 1., 1.], dtype=mx.float32)
    params['model']['layers'][0]['mlp']['down_proj']['weight'] = mx.array([[ 0.,  0.,  0.],
 [ 0.,  0.,  0.],
 [ 0.,  0.,  0.],
 [ 1., -1.,  0.],
 [ 0.,  0.,  0.]], dtype=mx.float32)
    params['model']['layers'][0]['mlp']['gate_proj']['weight'] = mx.array([[-3.3532020e-01, -1.3412670e+03,  6.0353305e+04,  0.0000000e+00,  0.0000000e+00],
 [-1.3743691e+01, -1.3418693e+03,  6.0353277e+04,  0.0000000e+00,  0.0000000e+00],
 [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00]], dtype=mx.float32)
    params['model']['layers'][0]['mlp']['up_proj']['weight'] = mx.array([[1.4898191e-02, 6.6922739e-04, 2.9977213e-05, 0.0000000e+00, 0.0000000e+00],
 [1.4898191e-02, 6.6922739e-04, 2.9977213e-05, 0.0000000e+00, 0.0000000e+00],
 [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00]], dtype=mx.float32)
    params['model']['layers'][0]['post_attention_layernorm']['weight'] = mx.array([1., 1., 1., 1., 1.], dtype=mx.float32)
    params['model']['layers'][0]['self_attn']['k_norm']['weight'] = mx.array([16., 16.], dtype=mx.float32)
    params['model']['layers'][0]['self_attn']['k_proj']['weight'] = mx.array([[-0.31672141,  0.00000000,  0.00000000,  0.00000000,  0.00000000],
 [-0.94851863,  0.00000000,  0.00000000,  0.00000000,  0.00000000]], dtype=mx.float32)
    params['model']['layers'][0]['self_attn']['o_proj']['weight'] = mx.array([[0., 0., 0., 0.],
 [0., 0., 0., 0.],
 [1., 0., 1., 0.],
 [0., 0., 0., 0.],
 [0., 0., 0., 0.]], dtype=mx.float32)
    params['model']['layers'][0]['self_attn']['q_norm']['weight'] = mx.array([16., 16.], dtype=mx.float32)
    params['model']['layers'][0]['self_attn']['q_proj']['weight'] = mx.array([[ 0.98502123,  0.00000000,  0.00000000,  0.00000000,  0.00000000],
 [ 0.17243294,  0.00000000,  0.00000000,  0.00000000,  0.00000000],
 [ 0.96630472,  0.00000000,  0.00000000,  0.00000000,  0.00000000],
 [-0.25740093,  0.00000000,  0.00000000,  0.00000000,  0.00000000]], dtype=mx.float32)
    params['model']['layers'][0]['self_attn']['v_proj']['weight'] = mx.array([[0., 1., 0., 0., 0.],
 [0., 0., 0., 0., 0.]], dtype=mx.float32)
    params['model']['layers'][1]['input_layernorm']['weight'] = mx.array([1., 1., 1., 1., 1.], dtype=mx.float32)
    params['model']['layers'][1]['mlp']['down_proj']['weight'] = mx.array([[  0.,   0.,   0.],
 [  0.,   0.,   0.],
 [  1., -10.,  10.],
 [  0.,   0.,   0.],
 [  0.,   0.,   0.]], dtype=mx.float32)
    params['model']['layers'][1]['mlp']['gate_proj']['weight'] = mx.array([[-4.3951669e-01,  5.6323919e+00,  4.9838150e-01,  1.3435575e+03,  6.0357680e+04],
 [-1.2112466e+02,  3.2923722e-01, -5.0313854e+00,  1.3449166e+03,  6.0357438e+04],
 [-1.3453412e+02, -2.6000220e-01, -5.6458039e+00,  1.3450677e+03,  6.0357410e+04]], dtype=mx.float32)
    params['model']['layers'][1]['mlp']['up_proj']['weight'] = mx.array([[ 1.4899401e-02,  6.5471046e-04,  6.8268733e-04, -1.6779384e-04,  2.9817384e-05],
 [ 1.4899401e-02,  6.5471046e-04,  6.8268733e-04, -1.6779384e-04,  2.9817384e-05],
 [ 1.4899401e-02,  6.5471046e-04,  6.8268733e-04, -1.6779384e-04,  2.9817384e-05]], dtype=mx.float32)
    params['model']['layers'][1]['post_attention_layernorm']['weight'] = mx.array([1., 1., 1., 1., 1.], dtype=mx.float32)
    params['model']['layers'][1]['self_attn']['k_norm']['weight'] = mx.array([16., 16.], dtype=mx.float32)
    params['model']['layers'][1]['self_attn']['k_proj']['weight'] = mx.array([[ 0.32702553,  0.00000000,  0.00000000,  0.00000000,  0.00000000],
 [-0.94501549,  0.00000000,  0.00000000,  0.00000000,  0.00000000]], dtype=mx.float32)
    params['model']['layers'][1]['self_attn']['o_proj']['weight'] = mx.array([[0., 0., 0., 0.],
 [0., 0., 0., 0.],
 [0., 0., 0., 0.],
 [0., 0., 0., 0.],
 [1., 0., 1., 0.]], dtype=mx.float32)
    params['model']['layers'][1]['self_attn']['q_norm']['weight'] = mx.array([16., 16.], dtype=mx.float32)
    params['model']['layers'][1]['self_attn']['q_proj']['weight'] = mx.array([[-0.25507239,  0.00000000,  0.00000000,  0.00000000,  0.00000000],
 [ 0.96692199,  0.00000000,  0.00000000,  0.00000000,  0.00000000],
 [ 0.17478994,  0.00000000,  0.00000000,  0.00000000,  0.00000000],
 [ 0.98460573,  0.00000000,  0.00000000,  0.00000000,  0.00000000]], dtype=mx.float32)
    params['model']['layers'][1]['self_attn']['v_proj']['weight'] = mx.array([[0., 1., 0., 0., 0.],
 [0., 0., 0., 0., 0.]], dtype=mx.float32)
    params['model']['norm']['weight'] = mx.array([1., 1., 1., 1., 1.], dtype=mx.float32)
    model.update(params)
    mx.eval(model.parameters())


def build_magic_model() -> Model:
    model = Model(build_model_args())
    hand_set_weights_magic(model)
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
                raise AssertionError(f"Mismatch for a={a:010d}, b={b:010d}: expected {exp}, got {act}")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.num_tests < 0:
        raise ValueError("--num-tests must be >= 0")
    model = build_magic_model()
    print(f"parameter count: {count_parameters(model.parameters())}")
    run_self_test_batched(model, args.num_tests, args.batch_size)
    print(f"self-test passed ({args.num_tests} random cases, batch size {args.batch_size})")


if __name__ == "__main__":
    main()
