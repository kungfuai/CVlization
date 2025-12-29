"""GPU utilities for SGLang on different architectures."""

import torch


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
