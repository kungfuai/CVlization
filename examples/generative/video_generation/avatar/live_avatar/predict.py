#!/usr/bin/env python3
"""
LiveAvatar Batch Inference Script

Generates lip-synced talking head video from audio and reference image.
Based on LiveAvatar from Alibaba using Wan2.2-S2V-14B with LoRA fine-tuning.

Usage:
    python predict.py --audio input.wav --image reference.jpg --output output.mp4
"""
import os
import sys

# Set environment variables for single GPU mode before importing torch
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
# Force safetensors to load to CPU first, then move to GPU (workaround for device issues)
os.environ["SAFETENSORS_FAST_GPU"] = "0"

# Monkey-patch safetensors BEFORE any imports that use it
# This fixes "device cuda:0 is invalid" error - safetensors doesn't accept string
# device format like "cuda:0", it needs integer device IDs
# See: https://github.com/huggingface/safetensors/issues/481

def _convert_device(device):
    """Convert device to integer ID for safetensors compatibility.

    Safetensors doesn't accept 'cuda:N' strings or torch.device objects.
    It accepts integer device IDs (like 0, 1) or 'cpu'.
    See: https://github.com/huggingface/safetensors/issues/481
    """
    import torch
    # Handle torch.device objects
    if isinstance(device, torch.device):
        if device.type == "cuda":
            return device.index if device.index is not None else 0
        return str(device)  # 'cpu' or other device types
    # Handle 'cuda:N' strings
    if isinstance(device, str) and device.startswith("cuda:"):
        try:
            return int(device.split(":")[1])
        except (ValueError, IndexError):
            return 0
    # Handle 'cuda' without index
    if device == "cuda":
        return 0
    return device

# Patch safetensors.torch functions
import safetensors.torch as _st
_original_safe_open = _st.safe_open
_original_load_file = _st.load_file

class _PatchedSafeOpen:
    """Wrapper that converts device specs to integer IDs for safetensors."""
    def __init__(self, filename, framework="pt", device="cpu"):
        self._handle = _original_safe_open(filename, framework=framework, device=_convert_device(device))

    def __enter__(self):
        return self._handle.__enter__()

    def __exit__(self, *args):
        return self._handle.__exit__(*args)

    def keys(self):
        return self._handle.keys()

    def get_tensor(self, name):
        return self._handle.get_tensor(name)

def _patched_load_file(filename, device="cpu"):
    """Wrapper that converts string device specs to integer IDs."""
    return _original_load_file(filename, device=_convert_device(device))

# Patch the safe_open reference in load_file's globals BEFORE we replace it
# This ensures the original load_file uses our patched safe_open
if 'safe_open' in _original_load_file.__globals__:
    _original_load_file.__globals__['safe_open'] = _PatchedSafeOpen

_st.safe_open = _PatchedSafeOpen
_st.load_file = _patched_load_file

# Also patch accelerate which imports safe_load_file at module level
import accelerate.utils.modeling as _aum
_aum.safe_load_file = _patched_load_file

# Patch the safe_load_file reference in load_state_dict's globals too
if hasattr(_aum, 'load_state_dict') and 'safe_load_file' in _aum.load_state_dict.__globals__:
    _aum.load_state_dict.__globals__['safe_load_file'] = _patched_load_file

import argparse
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path

from cvlization.paths import resolve_input_path, resolve_output_path

warnings.filterwarnings('ignore')

import torch
import torch.distributed as dist

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="LiveAvatar Batch Inference")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file (WAV)")
    parser.add_argument("--image", type=str, required=True, help="Path to reference image")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video path")
    parser.add_argument("--prompt", type=str, default="A person talking naturally with expressive facial movements.",
                        help="Text prompt describing the scene")
    parser.add_argument("--size", type=str, default="704*384", help="Output video size (width*height)")
    parser.add_argument("--num_clips", type=int, default=100, help="Max number of clips to generate")
    parser.add_argument("--sample_steps", type=int, default=4, help="Number of denoising steps")
    parser.add_argument("--infer_frames", type=int, default=48, help="Frames per clip (default: 48)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--offload", action="store_true", default=True, help="Offload model to CPU when not in use")
    return parser.parse_args()


def download_models():
    """Download models from HuggingFace using built-in caching."""
    from huggingface_hub import snapshot_download, hf_hub_download

    # Download Wan2.2-S2V-14B base model
    logger.info("Loading Wan2.2-S2V-14B base model (downloading if not cached)...")
    base_model_path = snapshot_download(repo_id="Wan-AI/Wan2.2-S2V-14B")
    logger.info(f"Base model path: {base_model_path}")

    # Download LiveAvatar LoRA weights
    logger.info("Loading LiveAvatar LoRA weights (downloading if not cached)...")
    lora_path = snapshot_download(repo_id="Quark-Vision/Live-Avatar")
    logger.info(f"LoRA path: {lora_path}")

    return base_model_path, lora_path


def main():
    args = parse_args()

    # Resolve input paths (check workspace mount first for user files)
    args.audio = resolve_input_path(args.audio)
    args.image = resolve_input_path(args.image)

    # Check GPU
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This model requires a GPU.")
        sys.exit(1)

    # Initialize distributed process group (required by LiveAvatar even for single GPU)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=int(os.environ.get("RANK", 0)),
            world_size=int(os.environ.get("WORLD_SIZE", 1))
        )

    device = torch.device(f"cuda:{local_rank}")
    gpu_name = torch.cuda.get_device_name(local_rank)
    gpu_mem = torch.cuda.get_device_properties(local_rank).total_memory / 1024**3
    logger.info(f"Using GPU: {gpu_name} ({gpu_mem:.1f}GB)")

    if gpu_mem < 70:
        logger.warning(f"GPU has {gpu_mem:.1f}GB VRAM. LiveAvatar recommends 80GB+ for single-GPU mode.")

    # Download models
    base_model_path, lora_path = download_models()

    # Create symlink for checkpoint directory expected by LiveAvatar
    ckpt_dir = Path("/workspace/ckpt")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    wan_link = ckpt_dir / "Wan2.2-S2V-14B"
    if not wan_link.exists():
        os.symlink(base_model_path, wan_link)

    lora_link = ckpt_dir / "LiveAvatar"
    if not lora_link.exists():
        os.symlink(lora_path, lora_link)

    print("=" * 60)
    print("LiveAvatar - Batch Inference")
    print("=" * 60)
    print(f"Audio:    {args.audio}")
    print(f"Image:    {args.image}")
    print(f"Output:   {args.output}")
    print(f"Size:     {args.size}")
    print(f"Steps:    {args.sample_steps}")
    print(f"Offload:  {args.offload}")
    print("=" * 60)

    # Add LiveAvatar to path
    sys.path.insert(0, "/workspace/LiveAvatar")

    # Import LiveAvatar modules
    from liveavatar.models.wan.wan_2_2.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
    from liveavatar.models.wan.causal_s2v_pipeline import WanS2V
    from liveavatar.models.wan.wan_2_2.utils.utils import save_video, merge_video_audio
    from liveavatar.utils.args_config import parse_args_for_training_config
    from PIL import Image

    # Load training config for LoRA settings
    training_config_path = "/workspace/LiveAvatar/liveavatar/configs/s2v_causal_sft.yaml"
    training_settings = parse_args_for_training_config(training_config_path)

    # Get model config
    task = "s2v-14B"
    cfg = WAN_CONFIGS[task]

    logger.info("Creating WanS2V pipeline...")
    wan_s2v = WanS2V(
        config=cfg,
        checkpoint_dir=str(ckpt_dir / "Wan2.2-S2V-14B"),
        device_id=local_rank,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        sp_size=1,
        t5_cpu=False,
        convert_model_dtype=True,
        single_gpu=True,
        offload_kv_cache=False,
    )

    # Load LoRA weights
    logger.info("Loading LoRA weights...")
    wan_s2v.add_lora_to_model(
        wan_s2v.noise_model,
        lora_rank=training_settings['lora_rank'],
        lora_alpha=training_settings['lora_alpha'],
        lora_target_modules=training_settings['lora_target_modules'],
        init_lora_weights=training_settings['init_lora_weights'],
        pretrained_lora_path="Quark-Vision/Live-Avatar",
        load_lora_weight_only=False,
    )
    t_model_ready = time.perf_counter()

    # Generate video
    logger.info("Generating video...")
    t_gen_start = time.perf_counter()
    video, dataset_info = wan_s2v.generate(
        input_prompt=args.prompt,
        ref_image_path=args.image,
        audio_path=args.audio,
        enable_tts=False,
        num_repeat=args.num_clips,
        generate_size=args.size,
        max_area=MAX_AREA_CONFIGS.get(args.size, 270336),
        infer_frames=args.infer_frames,
        shift=cfg.sample_shift,
        sample_solver="euler",
        sampling_steps=args.sample_steps,
        guide_scale=0,
        seed=args.seed,
        offload_model=args.offload,
        init_first_frame=False,
        num_gpus_dit=1,
        enable_vae_parallel=False,
        enable_online_decode=False,
    )
    t_gen_end = time.perf_counter()

    # Calculate and report performance metrics
    generation_time = t_gen_end - t_gen_start
    latency = t_gen_start - t_model_ready
    # Total frames = num_clips * infer_frames
    total_frames = args.num_clips * args.infer_frames
    # Actual frames may be limited by audio length, use dataset_info if available
    actual_clips = dataset_info.get('num_clips', args.num_clips) if isinstance(dataset_info, dict) else args.num_clips
    actual_frames = actual_clips * args.infer_frames
    throughput = actual_frames / generation_time if generation_time > 0 else 0

    print(f"\n{'='*50}")
    print(f"PERFORMANCE METRICS")
    print(f"{'='*50}")
    print(f"Latency:          {latency:.2f}s (model ready -> generation start)")
    print(f"Clips generated:  {actual_clips}")
    print(f"Frames generated: {actual_frames} ({actual_clips} clips x {args.infer_frames} frames)")
    print(f"Generation time:  {generation_time:.2f}s")
    print(f"Throughput:       {throughput:.2f} fps")
    print(f"{'='*50}\n")

    # Save video
    logger.info(f"Saving video to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_video(
        tensor=video[None],
        save_file=str(output_path),
        fps=cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1)
    )

    # Merge audio
    logger.info("Merging audio with video...")
    merge_video_audio(video_path=str(output_path), audio_path=args.audio)

    logger.info(f"Video saved to: {output_path}")
    logger.info("Inference complete!")


if __name__ == "__main__":
    main()
