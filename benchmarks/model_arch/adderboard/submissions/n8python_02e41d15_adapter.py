"""
Adapter for gist: https://gist.github.com/N8python/02e41d156ec615328cde2e1e5c0e9d53
"""

from pathlib import Path
import importlib.util


_SRC = Path(__file__).with_name("n8python_02e41d15.py")


def _load_src_module():
    spec = importlib.util.spec_from_file_location("n8python_02e41d15_src", _SRC)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load source module at {_SRC}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_model():
    mod = _load_src_module()
    core_model = mod.build_magic_model()
    metadata = {
        "name": "N8python 02e41d15 adapter",
        "author": "N8python",
        "params": 241,
        "architecture": "2L qwen3 d=5, 2h/1kv, hd=2",
        "tricks": ["hand-coded weights"],
    }
    return (mod, core_model), metadata


def add(model, a: int, b: int) -> int:
    mod, core_model = model
    pred_reversed = mod._generate_output_batch(core_model, [(a, b)])[0]
    return int(pred_reversed[::-1])
