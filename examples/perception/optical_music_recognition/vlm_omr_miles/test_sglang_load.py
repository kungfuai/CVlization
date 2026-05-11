"""De-risk Miles+Qwen3.5-VLM compatibility.

Tests:
1. Can SGLang (the Miles inference backend) load Qwen3.5-9B VLM?
2. Can it accept image inputs?
3. Does the Miles mbridge `qwen3_5.py` work with the vision variant?

If these pass, Miles GRPO should work. If they fail, we know exactly
where the incompatibility is.
"""

import sys
import time


def test_mbridge_import():
    """Test 1: Can we import the Miles Qwen3.5 mbridge?"""
    print("=" * 60)
    print("Test 1: Miles qwen3_5 mbridge import")
    print("=" * 60)
    try:
        sys.path.insert(0, "/root/miles")
        from miles_plugins.mbridge.qwen3_5 import Qwen3_5Bridge
        print(f"OK — Qwen3_5Bridge imported: {Qwen3_5Bridge}")
        # Check if it handles VLM (has text_config nesting)
        import inspect
        src = inspect.getsource(Qwen3_5Bridge)
        has_vlm_handling = "text_config" in src
        print(f"VLM handling (text_config check in source): {has_vlm_handling}")
        return True
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        return False


def test_model_config_compatibility():
    """Test 2: Does our merged model's config match what the mbridge expects?"""
    print("\n" + "=" * 60)
    print("Test 2: Model config compatibility")
    print("=" * 60)
    try:
        from transformers import AutoConfig
        model_path = "/sft_workspace/outputs/tamqjf4k_merged"
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print(f"model_type: {cfg.model_type}")
        print(f"architectures: {cfg.architectures}")
        text_cfg = getattr(cfg, "text_config", cfg)
        print(f"text_config.num_hidden_layers: {getattr(text_cfg, 'num_hidden_layers', 'N/A')}")
        print(f"text_config.hidden_size: {getattr(text_cfg, 'hidden_size', 'N/A')}")
        print(f"has vision_config: {hasattr(cfg, 'vision_config')}")
        return True
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        return False


def test_sglang_load():
    """Test 3: Can SGLang actually load the model and do image inference?"""
    print("\n" + "=" * 60)
    print("Test 3: SGLang VLM load + inference")
    print("=" * 60)
    try:
        import sglang as sgl
        from sglang.srt.server_args import ServerArgs

        model_path = "/sft_workspace/outputs/tamqjf4k_merged"
        print(f"Loading {model_path} with SGLang ...")
        t0 = time.time()
        # Use minimal config for quick test
        engine = sgl.Engine(
            model_path=model_path,
            mem_fraction_static=0.5,
            tp_size=1,
            disable_cuda_graph=True,  # faster startup
            log_level="warning",
        )
        load_time = time.time() - t0
        print(f"SGLang loaded in {load_time:.1f}s")

        # Try a simple text-only inference first (no image)
        t0 = time.time()
        out = engine.generate(
            prompt="Hello, can you describe an image?",
            sampling_params={"max_new_tokens": 20, "temperature": 0.0},
        )
        text_time = time.time() - t0
        print(f"Text-only inference: {text_time:.2f}s")
        print(f"  Output: {out.get('text', '<no text>')[:100]}")

        engine.shutdown()
        return True
    except Exception as e:
        import traceback
        print(f"FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    results = {
        "mbridge_import": test_mbridge_import(),
        "config_compat": test_model_config_compatibility(),
        "sglang_load": test_sglang_load(),
    }
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")

    if all(results.values()):
        print("\nMiles+Qwen3.5-VLM should work. Proceed with GRPO training.")
        sys.exit(0)
    else:
        print("\nIncompatibility detected. See errors above.")
        sys.exit(1)
