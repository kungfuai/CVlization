"""
Adapter for gist: https://gist.github.com/xangma/c538a7a9d415f16e61f7bb26ae5cf6b0
"""

from pathlib import Path
import importlib.util


_SRC = Path(__file__).with_name("xangma_c538a7a.py")


def _load_src_module():
    spec = importlib.util.spec_from_file_location("xangma_c538a7a_src", _SRC)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load source module at {_SRC}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_model():
    mod = _load_src_module()
    variant = "rank1+embed2+sparse_gate0+no_norm_weight"
    model = mod.build_magic_model(variant=variant, verify_rank1=False)
    metadata = {
        "name": "xangma c538a7a adapter",
        "author": "xangma",
        "params": 197,
        "architecture": "2L qwen3 d=5, 2h/1kv, hd=2",
        "tricks": [
            "rank-1 linear",
            "factorized embedding",
            "sparse gate",
            "param-free norm",
        ],
    }
    return (mod, model), metadata


def add(model, a: int, b: int) -> int:
    mod, core_model = model
    pred_reversed = mod._generate_output_batch(core_model, [(a, b)])[0]
    return int(pred_reversed[::-1])
