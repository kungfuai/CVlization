"""GPU utilities for vLLM on different architectures."""

import os

import torch


def configure_flash_attn_for_gpu():
    """Set VLLM_FLASH_ATTN_VERSION=2 for SM120+ GPUs (Blackwell consumer).

    FA3 doesn't work on consumer Blackwell (RTX 5090, RTX PRO 6000).
    See: https://github.com/vllm-project/vllm/issues/14452

    Must be called before importing vLLM.
    """
    if not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability(0)
    sm_version = major * 10 + minor
    if sm_version > 100 and "VLLM_FLASH_ATTN_VERSION" not in os.environ:
        os.environ["VLLM_FLASH_ATTN_VERSION"] = "2"
        print(f"Auto-configured VLLM_FLASH_ATTN_VERSION=2 for SM {sm_version}")
