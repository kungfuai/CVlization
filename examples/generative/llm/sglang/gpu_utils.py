"""GPU utilities for SGLang on different architectures."""

import torch


# SSM/Mamba-based models have no attention layers and are rejected by the
# triton attention backend at startup (SGLang raises AssertionError).
# Mapped to the backend that works for them.
_MODEL_ATTENTION_BACKEND_OVERRIDES: dict[str, str] = {
    "lfm2": "torch_native",  # LiquidAI/LFM2-* (pure Mamba SSM)
}

_MODEL_EXTRA_SERVER_ARGS: dict[str, list[str]] = {}


def get_model_attention_backend_override(model_id: str) -> str | None:
    """Return a required attention backend for known SSM/Mamba models, or None."""
    lower = model_id.lower()
    for pattern, backend in _MODEL_ATTENTION_BACKEND_OVERRIDES.items():
        if pattern in lower:
            return backend
    return None


def get_model_extra_server_args(model_id: str) -> list[str]:
    """Return extra launch_server args required for specific model families."""
    lower = model_id.lower()
    for pattern, args in _MODEL_EXTRA_SERVER_ARGS.items():
        if pattern in lower:
            return args
    return []


def get_optimal_attention_backend() -> str:
    """Auto-detect optimal attention backend based on GPU architecture.

    - SM 80-90 (Ampere/Hopper): flashinfer works well
    - SM 100 (B200/B100 data center Blackwell): flashinfer works
    - SM 120+ (RTX 5090/PRO 6000 consumer Blackwell): triton only
      - flashinfer: CUTLASS kernel failures on SM 120
      - fa3: requires SM <= 90
      - trtllm_mha: requires TensorRT-LLM installation
    """
    if not torch.cuda.is_available():
        return "triton"  # Safe fallback

    major, minor = torch.cuda.get_device_capability(0)
    sm_version = major * 10 + minor

    if sm_version > 100:
        # Consumer Blackwell (SM 120: RTX 5090, RTX PRO 6000) - flashinfer broken
        return "triton"
    else:
        # Ampere (SM 80-89), Hopper (SM 90), Data center Blackwell (SM 100)
        return "flashinfer"
