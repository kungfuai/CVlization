"""
Patched LiveTalk inference script with CPU VAE decode support.

This script monkey-patches the CausalInferencePipeline to support:
- VAE_DEVICE=cpu environment variable for CPU VAE decode

Usage:
    VAE_DEVICE=cpu python /workspace/patches/inference_patched.py --config config.yaml
"""
import os
import sys

# Get VAE device from environment
VAE_DEVICE = os.environ.get('VAE_DEVICE', 'cuda')

# Import the original module - this must be done after setting PYTHONPATH
sys.path.insert(0, '/workspace/LiveTalk')
sys.path.insert(0, '/workspace/LiveTalk/OmniAvatar')

import torch

# Store original methods before any modifications
_original_decode = None


def patch_vae_for_cpu_decode(vae, device='cpu'):
    """Patch VAE to decode on CPU."""
    global _original_decode

    if _original_decode is None:
        _original_decode = vae.decode_to_pixel

    original_vae_device = next(vae.parameters()).device

    def cpu_decode_to_pixel(latent, use_cache=False):
        print(f"[CVL] Moving VAE to {device} for decode...")

        # Move VAE to CPU
        vae.to(device)
        vae.model.to(device)
        vae.mean = vae.mean.to(device)
        vae.std = vae.std.to(device)

        # Move latent to CPU
        latent_cpu = latent.to(device)

        # Decode on CPU (may be slow but avoids OOM)
        with torch.no_grad():
            result = _original_decode(latent_cpu, use_cache=use_cache)

        print(f"[CVL] VAE decode complete on {device}")

        # Keep on CPU - video saving doesn't need GPU
        return result

    vae.decode_to_pixel = cpu_decode_to_pixel
    return vae


def main():
    # Import the original inference module
    from scripts.inference_example import CausalInferencePipeline, main as original_main

    if VAE_DEVICE == 'cpu':
        print("[CVL] CPU VAE decode enabled - will decode video on CPU to save VRAM")

        # Monkey-patch the pipeline's __call__ method
        original_call = CausalInferencePipeline.__call__

        def patched_call(self, *args, **kwargs):
            # Patch VAE before inference runs
            patch_vae_for_cpu_decode(self.vae, device='cpu')
            return original_call(self, *args, **kwargs)

        CausalInferencePipeline.__call__ = patched_call

    # Run original main
    original_main()


if __name__ == '__main__':
    main()
