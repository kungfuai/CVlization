"""
Monkey-patch for LiveTalk to enable VAE decode on CPU.

This patch intercepts the VAE decode call and runs it on CPU to avoid OOM
on GPUs with less than 24GB VRAM.

Usage:
  import cpu_vae_decode
  cpu_vae_decode.patch_pipeline(pipeline)
"""

import torch


def patch_pipeline(pipeline, vae_device='cpu'):
    """
    Patch a CausalInferencePipeline to decode VAE on CPU.

    Args:
        pipeline: CausalInferencePipeline instance
        vae_device: Device to run VAE decode on ('cpu' or 'cuda:X')
    """
    original_call = pipeline.__call__

    def patched_call(*args, **kwargs):
        # Run the normal inference but intercept at the end
        # Store original device
        original_device = pipeline.device
        original_dtype = pipeline.dtype

        # Temporarily disable VAE to avoid decode during forward
        original_vae = pipeline.vae

        # Run inference with a modified return_latents=True
        kwargs['return_latents'] = True
        video, latents = original_call(*args, **kwargs)

        # If video was already decoded (shouldn't happen), return it
        if video is not None:
            return video

        # Move VAE to CPU and decode there
        print(f"[CPU VAE] Moving VAE to {vae_device} for decode...")
        pipeline.vae.to(vae_device)
        latents_cpu = latents.to(vae_device)

        with torch.no_grad():
            video = pipeline.vae.decode_to_pixel(latents_cpu)
            video = (video * 0.5 + 0.5).clamp(0, 1)

        # Move VAE back to original device if needed
        pipeline.vae.to(original_device)

        return video.to(original_device)

    pipeline.__call__ = patched_call
    return pipeline


def create_cpu_vae_wrapper(vae, decode_device='cpu'):
    """
    Create a wrapper that decodes on CPU.

    Args:
        vae: WanVAEWrapper instance
        decode_device: Device for decode ('cpu')

    Returns:
        Wrapped VAE with CPU decode
    """
    original_decode = vae.decode_to_pixel
    original_device = next(vae.parameters()).device

    def cpu_decode_to_pixel(latent, use_cache=False):
        # Move VAE and latent to CPU
        vae.to(decode_device)
        latent_cpu = latent.to(decode_device)

        # Decode on CPU
        with torch.no_grad():
            result = original_decode(latent_cpu, use_cache=use_cache)

        # Move VAE back to GPU (for any subsequent operations)
        vae.to(original_device)

        return result.to(original_device)

    vae.decode_to_pixel = cpu_decode_to_pixel
    return vae
