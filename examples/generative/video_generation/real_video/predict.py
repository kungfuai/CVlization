#!/usr/bin/env python3
"""
RealVideo Batch Inference Script

Generates lip-synced video from audio and reference image using Self-Forcing diffusion.
Based on the RealVideo architecture but simplified for batch (non-realtime) inference.

Architecture (multi-GPU mode):
- Rank 0: VAE + Audio Encoder + Text Encoder (encoding/decoding)
- Rank 1: DiT model (video latent generation)

Single GPU mode runs both components on the same device.

Usage:
    # Single GPU
    python predict.py --audio input.wav --image reference.jpg --output output.mp4

    # Multi-GPU (2 GPUs)
    torchrun --standalone --nproc_per_node=2 predict.py \
        --audio input.wav --image reference.jpg --output output.mp4
"""
import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

from cvlization.paths import resolve_input_path, resolve_output_path

import torch
import torch.distributed as dist
import torchvision.transforms as TT
from einops import rearrange
from PIL import Image

# Add RealVideo modules to path
sys.path.insert(0, "/workspace/RealVideo")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


def get_rank_prefix():
    """Get rank prefix for logging."""
    if dist.is_initialized():
        return f"[Rank {dist.get_rank()}]"
    return ""


def parse_args():
    parser = argparse.ArgumentParser(description="RealVideo Batch Inference")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file (WAV)")
    parser.add_argument("--image", type=str, required=True, help="Path to reference image")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video path")
    parser.add_argument("--prompt", type=str, default="A person is talking.", help="Text prompt")
    parser.add_argument("--fps", type=int, default=16, help="Output video FPS")
    parser.add_argument("--width", type=int, default=480, help="Output width")
    parser.add_argument("--height", type=int, default=640, help="Output height")
    parser.add_argument("--num_denoising_steps", type=int, default=4, help="Number of denoising steps per block")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to RealVideo checkpoint")
    parser.add_argument("--wan_model_path", type=str, default="wan_models/Wan2.2-S2V-14B", help="Path to Wan model")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for speedup")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def download_models_if_needed(checkpoint_path: Optional[str], wan_model_path: str):
    """Download models from HuggingFace using built-in caching.

    Uses HuggingFace's cache system (HF_HOME) which is mounted from host.
    snapshot_download() is idempotent - returns cached path if already downloaded.
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    # Download Wan2.2-S2V-14B (uses HF cache, skips if exists)
    logger.info("Loading Wan2.2-S2V-14B (downloading if not cached)...")
    cached_wan_path = snapshot_download(repo_id="Wan-AI/Wan2.2-S2V-14B")
    logger.info(f"Wan model path: {cached_wan_path}")

    # Download Wan2.1-T2V-1.3B for VAE
    logger.info("Loading Wan2.1-T2V-1.3B VAE (downloading if not cached)...")
    cached_vae_path = snapshot_download(repo_id="Wan-AI/Wan2.1-T2V-1.3B")
    logger.info(f"VAE model path: {cached_vae_path}")

    # Create symlinks for code that expects wan_models/ path
    wan_models_dir = "wan_models"
    os.makedirs(wan_models_dir, exist_ok=True)

    symlink_path = os.path.join(wan_models_dir, "Wan2.2-S2V-14B")
    if not os.path.exists(symlink_path):
        os.symlink(cached_wan_path, symlink_path)

    vae_symlink_path = os.path.join(wan_models_dir, "Wan2.1-T2V-1.3B")
    if not os.path.exists(vae_symlink_path):
        os.symlink(cached_vae_path, vae_symlink_path)

    # Download RealVideo checkpoint (fine-tuned lip-sync weights)
    logger.info("Loading RealVideo checkpoint (downloading if not cached)...")
    checkpoint = hf_hub_download(
        repo_id="zai-org/RealVideo",
        filename="model.pt",
    )
    logger.info(f"RealVideo checkpoint path: {checkpoint}")

    return checkpoint


def nearest_multiple_of_64(n):
    lower = (n // 64) * 64
    upper = (n // 64 + 1) * 64
    return lower if abs(n - lower) < abs(n - upper) else upper


def read_image(image_path: str, target_height: int = 640, target_width: int = 480):
    """Read and preprocess reference image."""
    image = Image.open(image_path).convert("RGB")

    # Resize to target dimensions
    target_h = nearest_multiple_of_64(target_height)
    target_w = nearest_multiple_of_64(target_width)

    transforms = TT.Compose([
        TT.Resize((target_h, target_w), interpolation=TT.InterpolationMode.BICUBIC),
        TT.ToTensor(),
    ])
    image = transforms(image).unsqueeze(0)  # [1, 3, H, W]
    image = image * 2.0 - 1.0  # Normalize to [-1, 1]

    # Determine split dimension for sequence parallelism
    sp_dim = "h" if target_h < target_w else "w"

    return image, sp_dim


def read_audio(audio_path: str, sample_rate: int = 16000):
    """Read audio file and convert to tensor."""
    import librosa

    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    audio = torch.from_numpy(audio).float()
    return audio


def save_video(frames: torch.Tensor, output_path: str, fps: int = 16):
    """Save frames as video."""
    import av

    frames = frames.cpu().numpy()  # [T, H, W, C]
    frames = (frames * 255).clip(0, 255).astype("uint8")

    container = av.open(output_path, mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = frames.shape[2]
    stream.height = frames.shape[1]
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "fast"}

    for frame_data in frames:
        frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()

    logger.info(f"Saved video to: {output_path}")


def remap_latent_to_image(img: torch.Tensor):
    """Convert latent-decoded image from [-1,1] to [0,1] RGB."""
    img = torch.clamp((img + 1) / 2, min=0, max=1)
    return img


class VAEEncoder:
    """VAE + Text Encoder + Audio Encoder for Rank 0."""

    def __init__(self, device: torch.device, wan_model_path: str):
        from self_forcing.utils.wan_wrapper import WanTextEncoder, WanVAEWrapper
        from self_forcing.wan.modules.audio_encoder import AudioEncoder

        logger.info("Loading VAE...")
        self.vae = WanVAEWrapper().to(dtype=torch.bfloat16, device=device)

        logger.info("Loading Text Encoder...")
        self.text_encoder = WanTextEncoder().to(dtype=torch.bfloat16, device=device)

        logger.info("Loading Audio Encoder...")
        self.audio_encoder = AudioEncoder(device=device)

        self.device = device

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to latent."""
        image = image.to(self.device).unsqueeze(2)  # [B, C, 1, H, W]
        latent = self.vae.encode_to_latent(image.to(torch.bfloat16))
        latent = latent.permute(0, 2, 1, 3, 4)  # [B, 1, C, H, W]
        return latent.to(torch.bfloat16)

    @torch.no_grad()
    def encode_text(self, prompt: str) -> torch.Tensor:
        """Encode text prompt."""
        result = self.text_encoder(text_prompts=[prompt])
        return result["prompt_embeds"]

    @torch.no_grad()
    def encode_audio(self, audio: torch.Tensor, fps: int = 16) -> tuple:
        """Encode audio to features."""
        z = self.audio_encoder.extract_audio_feat(
            audio_input=audio, return_all_layers=True
        )
        audio_embed, num_blocks = self.audio_encoder.get_audio_embed_bucket_fps(
            z, fps=fps, batch_frames=(len(audio) // 1000), m=0
        )
        audio_embed = audio_embed.to(self.device).to(torch.bfloat16)
        audio_embed = audio_embed.unsqueeze(0)  # [1, T, F, D]
        if len(audio_embed.shape) == 3:
            audio_embed = audio_embed.permute(0, 2, 1)
        elif len(audio_embed.shape) == 4:
            audio_embed = audio_embed.permute(0, 2, 3, 1)
        return audio_embed, num_blocks

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to video frames."""
        frames = self.vae.decode_to_pixel(latent, use_cache=True)
        frames = rearrange(frames, "b t c h w -> b t h w c").squeeze(0)
        frames = remap_latent_to_image(frames)
        return frames


class DiTPipeline:
    """DiT inference pipeline for Rank 1."""

    def __init__(
        self,
        device: torch.device,
        wan_model_path: str,
        checkpoint_path: Optional[str],
        num_denoising_steps: int = 4,
        compile_model: bool = False,
    ):
        from omegaconf import OmegaConf

        from self_forcing.utils.wan_wrapper import WanDiffusionWrapper

        # Load config
        config_path = "/workspace/RealVideo/self_forcing/configs/sample_14B_s2v_sparse_nfb2.yaml"
        default_config_path = "/workspace/RealVideo/self_forcing/configs/default_config.yaml"

        config = OmegaConf.load(config_path)
        default_config = OmegaConf.load(default_config_path)
        self.config = OmegaConf.merge(default_config, config)

        # Model kwargs
        model_kwargs = dict(getattr(self.config, "model_kwargs", {}))
        num_frame_per_block = getattr(self.config, "num_frame_per_block", 2)
        is_sparse_causal = getattr(self.config, "is_sparse_causal", True)
        local_attn_size = getattr(self.config, "local_attn_size", 2)

        logger.info(f"Loading DiT model: {model_kwargs.get('model_name', 'Wan2.2-S2V-14B')}...")

        causal_kwargs = {
            "num_frame_per_block": num_frame_per_block,
            "sink_size": 1 if is_sparse_causal else 0,
            "independent_first_frame": False,
            "is_sparse_causal": is_sparse_causal,
            "local_attn_size": local_attn_size,
        }

        self.generator = WanDiffusionWrapper(
            **model_kwargs,
            is_causal=True,
            skip_init_model=True,
            **causal_kwargs,
        )

        # Load checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.generator.load_state_dict(state_dict.get("generator", state_dict), strict=False)

        self.generator = self.generator.to(device=device, dtype=torch.bfloat16)
        self.generator.eval()

        if compile_model:
            logger.info("Compiling model with torch.compile...")
            self.generator.model = torch.compile(self.generator.model, mode="reduce-overhead")

        self.device = device
        self.num_frame_per_block = num_frame_per_block
        self.scheduler = self.generator.get_scheduler()

        # Setup denoising steps (after scheduler is available)
        self._setup_denoising_steps(num_denoising_steps)

        # Initialize caches
        self.kv_cache = None
        self.crossattn_cache = None

    def _setup_denoising_steps(self, num_steps: int):
        """Setup denoising step schedule."""
        if num_steps == 2:
            self.denoising_step_list = torch.tensor([1000, 500], dtype=torch.long)
        elif num_steps == 4:
            self.denoising_step_list = torch.tensor([1000, 750, 500, 250], dtype=torch.long)
        else:
            # Linear spacing
            steps = torch.linspace(1000, 1000 // num_steps, num_steps, dtype=torch.long)
            self.denoising_step_list = steps

        # Warp steps if needed
        timesteps = torch.cat([self.scheduler.timesteps.cpu(), torch.tensor([0.0])])
        self.denoising_step_list = timesteps[1000 - self.denoising_step_list].long()

    def _initialize_kv_cache(self, batch_size: int, height: int, width: int):
        """Initialize KV cache for causal generation."""
        num_layers = self.generator.model.num_layers
        num_heads = self.generator.model.num_heads
        dim = self.generator.model.dim
        patch_size = self.generator.model.patch_size

        frame_seq_length = height * width // patch_size[1] // patch_size[2]
        kv_cache_size = round((2.5 + self.num_frame_per_block * 2) * frame_seq_length)

        self.kv_cache = []
        for _ in range(num_layers):
            self.kv_cache.append({
                "k": torch.empty([batch_size, kv_cache_size, num_heads, dim // num_heads],
                                dtype=torch.bfloat16, device=self.device),
                "v": torch.empty([batch_size, kv_cache_size, num_heads, dim // num_heads],
                                dtype=torch.bfloat16, device=self.device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=self.device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=self.device),
            })

        # Cross-attention cache
        text_seq_len = 512
        self.crossattn_cache = []
        for _ in range(num_layers):
            self.crossattn_cache.append({
                "k": torch.empty([batch_size, text_seq_len, num_heads, dim // num_heads],
                                dtype=torch.bfloat16, device=self.device),
                "v": torch.empty([batch_size, text_seq_len, num_heads, dim // num_heads],
                                dtype=torch.bfloat16, device=self.device),
                "is_init": False,
            })

        self.frame_seq_length = frame_seq_length
        self.current_start_token = 0

    def _reset_crossattn_cache(self):
        """Reset cross-attention cache."""
        if self.crossattn_cache:
            for cache in self.crossattn_cache:
                cache["is_init"] = False

    @torch.no_grad()
    def prefill_reference(self, conditional_dict: dict, sp_dim: str):
        """Prefill KV cache with reference image."""
        batch_size = 1
        timestep = torch.zeros([batch_size, 1], device=self.device, dtype=torch.int64)

        ref_length = self.generator(
            noisy_image_or_video=conditional_dict["ref_latents"],
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=self.kv_cache,
            crossattn_cache=self.crossattn_cache,
            current_start=self.current_start_token,
            sp_dim=sp_dim,
            initial_ref=True,
            sink_size=2.5,
            disable_float_conversion=True,
        )
        self.current_start_token += ref_length

    @torch.no_grad()
    def generate_block(
        self,
        conditional_dict: dict,
        sp_dim: str,
        audio_ptr: int,
    ) -> torch.Tensor:
        """Generate one video block."""
        height, width = conditional_dict["ref_latents"].shape[-2:]
        batch_size = 1

        noisy_input = torch.randn(
            [1, self.num_frame_per_block, 16, height, width],
            device=self.device,
            dtype=torch.bfloat16,
        )

        slice_index = [audio_ptr, audio_ptr + self.num_frame_per_block]

        # Denoising loop
        for i, current_timestep in enumerate(self.denoising_step_list):
            timestep = torch.ones(
                [batch_size, self.num_frame_per_block],
                device=self.device,
                dtype=torch.int64,
            ) * current_timestep

            _, denoised = self.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=self.kv_cache,
                crossattn_cache=self.crossattn_cache,
                current_start=self.current_start_token,
                sp_dim=sp_dim,
                slice_index=slice_index,
                sink_size=2.5,
                disable_float_conversion=True,
            )

            if i < len(self.denoising_step_list) - 1:
                next_timestep = self.denoising_step_list[i + 1]
                noisy_input = self.scheduler.add_noise(
                    denoised.flatten(0, 1),
                    torch.randn_like(denoised.flatten(0, 1)),
                    next_timestep * torch.ones([batch_size * self.num_frame_per_block],
                                               device=self.device, dtype=torch.long),
                ).unflatten(0, denoised.shape[:2]).to(torch.bfloat16)

        output = denoised.to(torch.bfloat16)

        # Update KV cache with clean context
        context_timestep = torch.zeros_like(timestep)
        self.generator(
            noisy_image_or_video=denoised,
            conditional_dict=conditional_dict,
            timestep=context_timestep,
            kv_cache=self.kv_cache,
            crossattn_cache=self.crossattn_cache,
            current_start=self.current_start_token,
            sp_dim=sp_dim,
            slice_index=slice_index,
            sink_size=2.5,
            disable_float_conversion=True,
        )

        # Update token position
        self.current_start_token += self.frame_seq_length * self.num_frame_per_block

        return output


def run_single_gpu(args, device, checkpoint):
    """Run inference on a single GPU with both VAE and DiT."""
    import gc

    logger.info("Running in single GPU mode...")

    # Initialize VAE encoder
    encoder = VAEEncoder(device=device, wan_model_path=args.wan_model_path)

    # Read and encode inputs
    logger.info("Encoding inputs...")
    image, sp_dim = read_image(args.image, args.height, args.width)
    audio = read_audio(args.audio)

    ref_latent = encoder.encode_image(image)
    prompt_embeds = encoder.encode_text(args.prompt)
    audio_embed, num_audio_blocks = encoder.encode_audio(audio, fps=args.fps)

    logger.info(f"ref_latent shape: {ref_latent.shape}")
    logger.info(f"prompt_embeds shape: {prompt_embeds.shape}")
    logger.info(f"audio_embed shape: {audio_embed.shape}")
    logger.info(f"num_audio_blocks: {num_audio_blocks}")

    conditional_dict = {
        "ref_latents": ref_latent,
        "prompt_embeds": prompt_embeds,
        "audio_input": audio_embed,
        "motion_latents": None,
        "motion_frames": [73, 19],
    }

    # Free encoder memory before loading DiT (except VAE which we need for decoding)
    del encoder.text_encoder
    del encoder.audio_encoder
    gc.collect()
    torch.cuda.empty_cache()

    # Initialize DiT pipeline
    logger.info("Loading DiT model...")
    pipeline = DiTPipeline(
        device=device,
        wan_model_path=args.wan_model_path,
        checkpoint_path=checkpoint,
        num_denoising_steps=args.num_denoising_steps,
        compile_model=args.compile,
    )
    t_model_ready = time.perf_counter()

    # Initialize KV cache
    height, width = ref_latent.shape[-2:]
    pipeline._initialize_kv_cache(batch_size=1, height=height, width=width)
    pipeline.prefill_reference(conditional_dict, sp_dim)

    # Generate blocks
    total_blocks = max(1, num_audio_blocks)
    all_latents = []

    t_gen_start = time.perf_counter()

    for block_idx in range(total_blocks):
        logger.info(f"Generating block {block_idx + 1}/{total_blocks}...")
        start_time = time.time()

        audio_ptr = min(block_idx * pipeline.num_frame_per_block,
                       audio_embed.shape[1] - pipeline.num_frame_per_block)

        latent_block = pipeline.generate_block(
            conditional_dict=conditional_dict,
            sp_dim=sp_dim,
            audio_ptr=audio_ptr,
        )

        elapsed = time.time() - start_time
        logger.info(f"Block {block_idx + 1} generated in {elapsed:.2f}s")
        all_latents.append(latent_block)

    t_gen_end = time.perf_counter()
    frames_per_block = pipeline.num_frame_per_block

    # Free DiT memory before decoding
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    # Decode latents to frames
    logger.info("Decoding latents to video frames...")
    all_frames = []
    for latent_block in all_latents:
        frames = encoder.decode_latent(latent_block)
        all_frames.append(frames)

    all_frames = torch.cat(all_frames, dim=0)
    total_frames = all_frames.shape[0]
    logger.info(f"Total frames generated: {total_frames}")

    # Calculate and report performance metrics
    latency = t_gen_start - t_model_ready
    generation_time = t_gen_end - t_gen_start
    throughput = total_frames / generation_time if generation_time > 0 else 0

    print(f"\n{'='*50}")
    print(f"PERFORMANCE METRICS")
    print(f"{'='*50}")
    print(f"Latency:          {latency:.2f}s (model ready -> generation start)")
    print(f"Blocks generated: {total_blocks}")
    print(f"Frames generated: {total_frames} ({total_blocks} blocks x {frames_per_block} frames)")
    print(f"Generation time:  {generation_time:.2f}s")
    print(f"Throughput:       {throughput:.2f} fps")
    print(f"{'='*50}\n")

    # Ensure output directory exists
    output_path = Path(resolve_output_path(args.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_video(all_frames, str(output_path), fps=args.fps)
    logger.info("Inference complete!")


def main():
    args = parse_args()

    # Resolve input paths (check workspace mount first for user files)
    args.audio = resolve_input_path(args.audio)
    args.image = resolve_input_path(args.image)

    # Check if running with torchrun (multi-GPU mode)
    is_multi_gpu = "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", 1)) > 1

    if is_multi_gpu:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        # Single GPU mode - initialize minimal distributed for RealVideo compatibility
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
        rank = 0
        world_size = 1
        local_rank = 0

    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    torch.manual_seed(args.seed + rank)
    torch.set_grad_enabled(False)

    print(f"{get_rank_prefix()} Initialized rank {rank}/{world_size} on device {device}")

    if is_multi_gpu and world_size != 2:
        if rank == 0:
            logger.error("Multi-GPU mode requires exactly 2 GPUs (rank 0: VAE, rank 1: DiT)")
        dist.destroy_process_group()
        sys.exit(1)

    # Download models (all ranks call this - HF cache is shared so only one downloads)
    # This avoids barrier timeout issues during long downloads (~50GB)
    checkpoint = download_models_if_needed(args.checkpoint, args.wan_model_path)

    if is_multi_gpu:
        dist.barrier()

    # Initialize sequence parallel groups for RealVideo (required for DiT model)
    from self_forcing.utils.parallel_state import initialize_parallel_states
    initialize_parallel_states()

    print("=" * 60)
    print("RealVideo - Batch Inference")
    print("=" * 60)
    print(f"Audio:    {args.audio}")
    print(f"Image:    {args.image}")
    print(f"Output:   {args.output}")
    print(f"Resolution: {args.width}x{args.height} @ {args.fps}fps")
    print(f"Denoising steps: {args.num_denoising_steps}")
    print(f"Mode:     {'Multi-GPU' if is_multi_gpu else 'Single GPU'}")
    print("=" * 60)

    if world_size == 1:
        # Single GPU mode - run everything on one device
        run_single_gpu(args, device, checkpoint)
    elif rank == 0:
        # VAE Encoder rank
        encoder = VAEEncoder(device=device, wan_model_path=args.wan_model_path)

        # Read and encode inputs
        logger.info("Encoding inputs...")
        image, sp_dim = read_image(args.image, args.height, args.width)
        audio = read_audio(args.audio)

        ref_latent = encoder.encode_image(image)
        prompt_embeds = encoder.encode_text(args.prompt)
        audio_embed, num_audio_blocks = encoder.encode_audio(audio, fps=args.fps)

        logger.info(f"ref_latent shape: {ref_latent.shape}")
        logger.info(f"prompt_embeds shape: {prompt_embeds.shape}")
        logger.info(f"audio_embed shape: {audio_embed.shape}")
        logger.info(f"num_audio_blocks: {num_audio_blocks}")

        # Prepare conditional dict
        conditional_dict = {
            "ref_latents": ref_latent,
            "prompt_embeds": prompt_embeds,
            "audio_input": audio_embed,
            "motion_latents": None,
            "motion_frames": [73, 19],
        }

        # Send tensor shapes to rank 1
        logger.info("Sending conditional dict to DiT rank...")
        shapes = torch.tensor([
            *ref_latent.shape,  # 5 dims
            *prompt_embeds.shape,  # 3 dims
            *audio_embed.shape,  # 4 dims (padded)
            num_audio_blocks,
            ord(sp_dim),
        ], dtype=torch.long, device=device)
        dist.broadcast(shapes, src=0)

        # Send tensors
        dist.broadcast(ref_latent.contiguous(), src=0)
        dist.broadcast(prompt_embeds.contiguous(), src=0)
        dist.broadcast(audio_embed.contiguous(), src=0)

        # Receive generated latents
        all_frames = []
        total_blocks = max(1, num_audio_blocks)

        for block_idx in range(total_blocks):
            # Receive shape
            shape_tensor = torch.zeros(5, dtype=torch.long, device=device)
            dist.broadcast(shape_tensor, src=1)
            shape = tuple(shape_tensor.tolist())

            # Receive latent block
            latent_block = torch.zeros(shape, dtype=torch.bfloat16, device=device)
            dist.broadcast(latent_block, src=1)

            logger.info(f"Received block {block_idx + 1}/{total_blocks}, shape: {shape}")

            # Decode to frames
            frames = encoder.decode_latent(latent_block)
            all_frames.append(frames)

        # Combine and save
        all_frames = torch.cat(all_frames, dim=0)
        total_frames = all_frames.shape[0]
        logger.info(f"Total frames generated: {total_frames}")

        # Receive timing info from rank 1
        timing_tensor = torch.zeros(5, dtype=torch.float64, device=device)
        dist.broadcast(timing_tensor, src=1)
        t_model_ready, t_gen_start, t_gen_end, num_blocks, frames_per_block = timing_tensor.tolist()

        # Calculate and report performance metrics
        latency = t_gen_start - t_model_ready
        generation_time = t_gen_end - t_gen_start
        throughput = total_frames / generation_time if generation_time > 0 else 0

        print(f"\n{'='*50}")
        print(f"PERFORMANCE METRICS")
        print(f"{'='*50}")
        print(f"Latency:          {latency:.2f}s (model ready -> generation start)")
        print(f"Blocks generated: {int(num_blocks)}")
        print(f"Frames generated: {total_frames} ({int(num_blocks)} blocks x {int(frames_per_block)} frames)")
        print(f"Generation time:  {generation_time:.2f}s")
        print(f"Throughput:       {throughput:.2f} fps")
        print(f"{'='*50}\n")

        # Ensure output directory exists
        output_path = Path(resolve_output_path(args.output))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_video(all_frames, str(output_path), fps=args.fps)

    else:
        # DiT Pipeline rank - receive shapes first
        shapes = torch.zeros(14, dtype=torch.long, device=device)
        dist.broadcast(shapes, src=0)

        ref_shape = tuple(shapes[:5].tolist())
        prompt_shape = tuple(shapes[5:8].tolist())
        audio_shape = tuple(shapes[8:12].tolist())
        num_audio_blocks = int(shapes[12].item())
        sp_dim = chr(int(shapes[13].item()))

        logger.info(f"Receiving tensors - ref: {ref_shape}, prompt: {prompt_shape}, audio: {audio_shape}")

        # Receive tensors
        ref_latent = torch.zeros(ref_shape, dtype=torch.bfloat16, device=device)
        dist.broadcast(ref_latent, src=0)

        prompt_embeds = torch.zeros(prompt_shape, dtype=torch.bfloat16, device=device)
        dist.broadcast(prompt_embeds, src=0)

        audio_embed = torch.zeros(audio_shape, dtype=torch.bfloat16, device=device)
        dist.broadcast(audio_embed, src=0)

        conditional_dict = {
            "ref_latents": ref_latent,
            "prompt_embeds": prompt_embeds,
            "audio_input": audio_embed,
            "motion_latents": None,
            "motion_frames": [73, 19],
        }

        # Initialize pipeline
        pipeline = DiTPipeline(
            device=device,
            wan_model_path=args.wan_model_path,
            checkpoint_path=checkpoint,
            num_denoising_steps=args.num_denoising_steps,
            compile_model=args.compile,
        )
        t_model_ready = time.perf_counter()

        logger.info(f"Received conditional dict, generating {num_audio_blocks} blocks...")

        # Initialize KV cache
        height, width = ref_latent.shape[-2:]
        pipeline._initialize_kv_cache(batch_size=1, height=height, width=width)
        pipeline.prefill_reference(conditional_dict, sp_dim)

        # Generate blocks with timing
        total_blocks = max(1, num_audio_blocks)
        t_gen_start = time.perf_counter()

        for block_idx in range(total_blocks):
            logger.info(f"Generating block {block_idx + 1}/{total_blocks}...")
            start_time = time.time()

            audio_ptr = min(block_idx * pipeline.num_frame_per_block,
                           audio_embed.shape[1] - pipeline.num_frame_per_block)

            latent_block = pipeline.generate_block(
                conditional_dict=conditional_dict,
                sp_dim=sp_dim,
                audio_ptr=audio_ptr,
            )

            elapsed = time.time() - start_time
            logger.info(f"Block {block_idx + 1} generated in {elapsed:.2f}s")

            # Send to rank 0
            shape_tensor = torch.tensor(latent_block.shape, dtype=torch.long, device=device)
            dist.broadcast(shape_tensor, src=1)
            dist.broadcast(latent_block.contiguous(), src=1)

        t_gen_end = time.perf_counter()

        # Send timing info to rank 0
        timing_tensor = torch.tensor([
            t_model_ready, t_gen_start, t_gen_end,
            float(total_blocks), float(pipeline.num_frame_per_block)
        ], dtype=torch.float64, device=device)
        dist.broadcast(timing_tensor, src=1)

    # Cleanup
    if is_multi_gpu:
        dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        logger.info("Inference complete!")


if __name__ == "__main__":
    main()
