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
    # vLLM 0.21+ defaults to FlashInfer for sampler and MoE. FlashInfer
    # 0.6.8's CUDA arch detection is broken on SM120 (consumer Blackwell):
    # TARGET_CUDA_ARCHS comes back empty / major=None, so downstream code
    # fails with "No supported CUDA architectures found for major versions
    # None". Disable FlashInfer for the sampler; vLLM falls back to its
    # native PyTorch sampler.
    if sm_version > 100:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        # FlashInfer's TARGET_CUDA_ARCHS auto-detection via
        # torch.cuda.get_device_capability fails silently on SM12.x, leaving
        # it empty; downstream FP8 GEMM JIT then raises "No supported CUDA
        # architectures for major versions [12]". Populate it directly with
        # the SM12.0 'f' suffix (the consumer Blackwell variant per FlashInfer
        # _normalize_cuda_arch). Requires CUDA >= 12.9 (we ship with 13.0).
        os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", f"{major}.{minor}f")
        os.environ.setdefault("VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER", "0")
        print(
            f"Auto-configured TORCH_CUDA_ARCH_LIST={major}.{minor}, "
            f"FLASHINFER_CUDA_ARCH_LIST={major}.{minor}f, "
            "VLLM_USE_FLASHINFER_SAMPLER=0, "
            "VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0"
        )
