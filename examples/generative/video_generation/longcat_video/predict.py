#!/usr/bin/env python3
"""
LongCat-Video inference wrapper for CVlization.

Supports multiple generation modes:
- t2v: Text-to-Video generation
- i2v: Image-to-Video generation
- long_video: Long video generation (multi-segment)

Model: meituan-longcat/LongCat-Video (13.6B parameters)
"""

import os
import sys
import logging
import warnings

# Suppress verbose logging by default
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)]
)

import argparse
import datetime
import shutil
from pathlib import Path

import numpy as np
import PIL.Image
import torch
import torch.distributed as dist
from torchvision.io import write_video

# HuggingFace dataset for sample inputs
HF_DATASET_REPO = "zzsi/cvl"
HF_SAMPLE_SUBDIR = "longcat_video"

# Default paths
DEFAULT_OUTPUT = "outputs/output.mp4"
MODELS_DIR = Path("/models")
TEST_INPUTS_DIR = Path("/tmp/longcat_video_samples")


def download_sample_inputs():
    """Download sample inputs from HuggingFace dataset if not present."""
    from huggingface_hub import hf_hub_download

    TEST_INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    default_image = TEST_INPUTS_DIR / "sample.jpg"

    try:
        if not default_image.exists():
            logging.info("Downloading sample image from HuggingFace...")
            hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename=f"{HF_SAMPLE_SUBDIR}/sample.jpg",
                repo_type="dataset",
                local_dir=str(TEST_INPUTS_DIR),
            )
            src = TEST_INPUTS_DIR / HF_SAMPLE_SUBDIR / "sample.jpg"
            if src.exists():
                src.rename(default_image)

        # Clean up subdirectory
        subdir = TEST_INPUTS_DIR / HF_SAMPLE_SUBDIR
        if subdir.exists():
            shutil.rmtree(subdir, ignore_errors=True)

    except Exception as e:
        logging.warning(f"Could not download samples from HuggingFace: {e}")
        # Fall back to LongCat-Video repo assets
        assets = Path("/opt/LongCat-Video/assets")
        if assets.exists() and (assets / "girl.png").exists():
            if not default_image.exists():
                shutil.copy(assets / "girl.png", default_image)

    return default_image


def download_models_if_needed():
    """Download LongCat-Video model checkpoints if not present."""
    from huggingface_hub import snapshot_download

    model_dir = MODELS_DIR / "LongCat-Video"

    # Check if model is already downloaded
    if model_dir.exists() and (model_dir / "dit").exists():
        logging.info(f"Model already exists at {model_dir}")
        return model_dir

    logging.info("Downloading LongCat-Video from HuggingFace (this may take a while)...")
    snapshot_download(
        repo_id="meituan-longcat/LongCat-Video",
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
    )

    logging.info(f"Model downloaded to {model_dir}")
    return model_dir


def torch_gc():
    """Clean up GPU memory."""
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def init_distributed():
    """Initialize distributed environment for torchrun."""
    rank = int(os.environ.get('RANK', 0))
    num_gpus = torch.cuda.device_count()
    local_rank = rank % num_gpus
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600*24))
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()
    return local_rank, global_rank, num_processes


def load_pipeline(checkpoint_dir: str, local_rank: int, context_parallel_size: int = 1):
    """Load LongCat-Video pipeline components."""
    from transformers import AutoTokenizer, UMT5EncoderModel

    from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
    from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
    from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
    from longcat_video.context_parallel import context_parallel_util
    from longcat_video.context_parallel.context_parallel_util import init_context_parallel

    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()

    # Initialize context parallel
    init_context_parallel(
        context_parallel_size=context_parallel_size,
        global_rank=global_rank,
        world_size=num_processes
    )
    cp_size = context_parallel_util.get_cp_size()
    cp_split_hw = context_parallel_util.get_optimal_split(cp_size)

    logging.info(f"Loading model components from {checkpoint_dir}...")

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir, subfolder="tokenizer", torch_dtype=torch.bfloat16
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        checkpoint_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    vae = AutoencoderKLWan.from_pretrained(
        checkpoint_dir, subfolder="vae", torch_dtype=torch.bfloat16
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        checkpoint_dir, subfolder="scheduler", torch_dtype=torch.bfloat16
    )
    dit = LongCatVideoTransformer3DModel.from_pretrained(
        checkpoint_dir, subfolder="dit", cp_split_hw=cp_split_hw, torch_dtype=torch.bfloat16
    )

    pipe = LongCatVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
    )
    pipe.to(local_rank)

    return pipe


def generate_t2v(
    pipe,
    prompt: str,
    negative_prompt: str,
    local_rank: int,
    output_path: str,
    num_frames: int = 93,
    num_inference_steps: int = 50,
    guidance_scale: float = 4.0,
    seed: int = 42,
    resolution: str = "480p",
):
    """Generate video from text prompt."""
    generator = torch.Generator(device=local_rank)
    generator.manual_seed(seed + dist.get_rank())

    if resolution == "480p":
        height, width = 480, 832
    else:  # 720p
        height, width = 768, 1280

    logging.info(f"Generating T2V: {resolution} ({width}x{height}), {num_frames} frames, {num_inference_steps} steps")

    output = pipe.generate_t2v(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )[0]

    if local_rank == 0:
        output_tensor = torch.from_numpy(np.array(output))
        output_tensor = (output_tensor * 255).clamp(0, 255).to(torch.uint8)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        write_video(output_path, output_tensor, fps=15, video_codec="libx264", options={"crf": "18"})
        logging.info(f"Video saved to: {output_path}")

    return output


def generate_i2v(
    pipe,
    image_path: str,
    prompt: str,
    negative_prompt: str,
    local_rank: int,
    output_path: str,
    num_frames: int = 93,
    num_inference_steps: int = 50,
    guidance_scale: float = 4.0,
    seed: int = 42,
    resolution: str = "480p",
):
    """Generate video from image and text prompt."""
    from diffusers.utils import load_image

    generator = torch.Generator(device=local_rank)
    generator.manual_seed(seed + dist.get_rank())

    image = load_image(image_path)
    target_size = image.size  # (width, height)

    logging.info(f"Generating I2V from {image_path}: {resolution}, {num_frames} frames, {num_inference_steps} steps")

    output = pipe.generate_i2v(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        resolution=resolution,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )[0]

    if local_rank == 0:
        output_frames = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
        output_frames = [PIL.Image.fromarray(img) for img in output_frames]
        output_frames = [frame.resize(target_size, PIL.Image.BICUBIC) for frame in output_frames]

        output_tensor = torch.from_numpy(np.array(output_frames))
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        write_video(output_path, output_tensor, fps=15, video_codec="libx264", options={"crf": "18"})
        logging.info(f"Video saved to: {output_path}")

    return output


def generate_long_video(
    pipe,
    prompt: str,
    negative_prompt: str,
    local_rank: int,
    output_path: str,
    num_segments: int = 3,
    num_frames: int = 93,
    num_cond_frames: int = 13,
    num_inference_steps: int = 50,
    guidance_scale: float = 4.0,
    seed: int = 42,
    resolution: str = "480p",
):
    """Generate long video with multiple segments."""
    generator = torch.Generator(device=local_rank)
    generator.manual_seed(seed + dist.get_rank())

    if resolution == "480p":
        height, width = 480, 832
    else:
        height, width = 768, 1280

    logging.info(f"Generating long video: {num_segments} segments, {resolution}, {num_frames} frames/segment")

    # First segment: T2V
    output = pipe.generate_t2v(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )[0]

    video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
    video = [PIL.Image.fromarray(img) for img in video]
    del output
    torch_gc()

    target_size = video[0].size
    current_video = video
    all_generated_frames = list(video)

    # Continue with video continuation for remaining segments
    for segment_idx in range(num_segments - 1):
        if local_rank == 0:
            logging.info(f"Generating segment {segment_idx + 2}/{num_segments}...")

        output = pipe.generate_vc(
            video=current_video,
            prompt=prompt,
            negative_prompt=negative_prompt,
            resolution=resolution,
            num_frames=num_frames,
            num_cond_frames=num_cond_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            use_kv_cache=True,
            offload_kv_cache=False,
            enhance_hf=True,
        )[0]

        new_video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
        new_video = [PIL.Image.fromarray(img) for img in new_video]
        new_video = [frame.resize(target_size, PIL.Image.BICUBIC) for frame in new_video]
        del output

        all_generated_frames.extend(new_video[num_cond_frames:])
        current_video = new_video

    if local_rank == 0:
        output_tensor = torch.from_numpy(np.array(all_generated_frames))
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        write_video(output_path, output_tensor, fps=15, video_codec="libx264", options={"crf": "18"})
        logging.info(f"Long video saved to: {output_path} ({len(all_generated_frames)} frames)")

    return all_generated_frames


def resolve_path(path: str, check_exists: bool = True) -> str:
    """Resolve input path - check multiple locations."""
    if os.path.isabs(path) and os.path.exists(path):
        return path

    # Check relative to CVL workspace
    cvl_inputs = os.environ.get("CVL_INPUTS", "/mnt/cvl/workspace")
    cvl_path = os.path.join(cvl_inputs, path)
    if os.path.exists(cvl_path):
        return cvl_path

    # Check relative to /user_data
    user_path = os.path.join("/user_data", path)
    if os.path.exists(user_path):
        return user_path

    # Check relative to local examples
    local_path = os.path.join("/workspace/local", path)
    if os.path.exists(local_path):
        return local_path

    if check_exists:
        raise FileNotFoundError(f"Could not find file: {path}")
    return path


def resolve_output_path(path: str) -> str:
    """Resolve output path."""
    if os.path.isabs(path):
        return path

    cvl_outputs = os.environ.get("CVL_OUTPUTS", "/mnt/cvl/workspace")
    return os.path.join(cvl_outputs, path)


def main():
    parser = argparse.ArgumentParser(
        description="LongCat-Video inference - T2V, I2V, and Long Video generation"
    )
    parser.add_argument("--mode", type=str, default="i2v",
                        choices=["t2v", "i2v", "long_video"],
                        help="Generation mode (default: i2v)")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to input image (required for i2v mode)")
    parser.add_argument("--prompt", type=str,
                        default="A person sits peacefully, looking ahead with a calm expression.",
                        help="Text prompt describing the video")
    parser.add_argument("--negative-prompt", type=str,
                        default="Bright tones, overexposed, static, blurred details, subtitles, worst quality, low quality",
                        help="Negative prompt")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="Output video path")
    parser.add_argument("--resolution", type=str, default="480p",
                        choices=["480p", "720p"],
                        help="Output resolution (default: 480p)")
    parser.add_argument("--frames", type=int, default=93,
                        help="Frames per segment (default: 93)")
    parser.add_argument("--steps", type=int, default=50,
                        help="Inference steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=4.0,
                        help="Guidance scale (default: 4.0)")
    parser.add_argument("--segments", type=int, default=3,
                        help="Number of segments for long_video mode (default: 3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip model download check")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")

    args = parser.parse_args()

    # Re-enable verbose output if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        os.environ["TRANSFORMERS_VERBOSITY"] = "info"

    # Download models if needed
    if not args.skip_download:
        download_models_if_needed()

    # Handle input paths for I2V mode
    if args.mode == "i2v":
        if args.image is None:
            default_image = download_sample_inputs()
            args.image = str(default_image)
            logging.info(f"Using sample image: {args.image}")
        else:
            args.image = resolve_path(args.image)

    output_path = resolve_output_path(args.output)

    # Initialize distributed
    local_rank, global_rank, num_processes = init_distributed()

    if local_rank == 0:
        logging.info(f"Mode: {args.mode}")
        logging.info(f"Resolution: {args.resolution}")
        logging.info(f"Prompt: {args.prompt}")
        logging.info(f"Output: {output_path}")

    # Load pipeline
    model_dir = MODELS_DIR / "LongCat-Video"
    pipe = load_pipeline(str(model_dir), local_rank)

    # Generate video based on mode
    negative_prompt = args.negative_prompt

    if args.mode == "t2v":
        generate_t2v(
            pipe=pipe,
            prompt=args.prompt,
            negative_prompt=negative_prompt,
            local_rank=local_rank,
            output_path=output_path,
            num_frames=args.frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
            resolution=args.resolution,
        )
    elif args.mode == "i2v":
        generate_i2v(
            pipe=pipe,
            image_path=args.image,
            prompt=args.prompt,
            negative_prompt=negative_prompt,
            local_rank=local_rank,
            output_path=output_path,
            num_frames=args.frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
            resolution=args.resolution,
        )
    elif args.mode == "long_video":
        generate_long_video(
            pipe=pipe,
            prompt=args.prompt,
            negative_prompt=negative_prompt,
            local_rank=local_rank,
            output_path=output_path,
            num_segments=args.segments,
            num_frames=args.frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
            resolution=args.resolution,
        )

    logging.info("Done!")


if __name__ == "__main__":
    main()
