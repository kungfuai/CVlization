#!/usr/bin/env python3
"""
PersonaLive: Real-time Expressive Portrait Animation for Live Streaming

Generates portrait animations from a reference image and driving video.
Uses diffusion-based approach with custom motion encoding.

References:
- Paper: https://arxiv.org/abs/2512.11253
- GitHub: https://github.com/GVCLab/PersonaLive
- HuggingFace: https://huggingface.co/huaichang/PersonaLive
"""

import os
import sys
import logging
import warnings

# Suppress verbose logging by default
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging - only show errors by default
logging.basicConfig(level=logging.ERROR)
for logger_name in ["transformers", "diffusers", "torch", "huggingface_hub"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

import argparse
import gc
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from decord import VideoReader
from diffusers import AutoencoderKL
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection
from tqdm import tqdm

from src.models.motion_encoder.encoder import MotEncoder
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.liveportrait.motion_extractor import MotionExtractor
from src.pipelines.pipeline_pose2vid import Pose2VideoPipeline
from src.scheduler.scheduler_ddim import DDIMScheduler
from src.utils.util import save_videos_grid, crop_face

logger = logging.getLogger(__name__)

# HuggingFace model IDs for centralized caching
HF_BASE_MODEL = "lambdalabs/sd-image-variations-diffusers"
HF_VAE_MODEL = "stabilityai/sd-vae-ft-mse"
HF_PERSONALIVE = "huaichang/PersonaLive"

# PersonaLive checkpoint files (relative to pretrained_weights/personalive/)
PERSONALIVE_CHECKPOINTS = [
    "denoising_unet.pth",
    "reference_unet.pth",
    "motion_encoder.pth",
    "pose_guider.pth",
    "temporal_module.pth",
    "motion_extractor.pth",
]


def download_personalive_weights() -> dict:
    """Download PersonaLive weights from HuggingFace with centralized caching."""
    print(f"Downloading PersonaLive weights from {HF_PERSONALIVE}...")
    paths = {}
    for ckpt in PERSONALIVE_CHECKPOINTS:
        # Weights are in pretrained_weights/personalive/ subdirectory
        hf_path = f"pretrained_weights/personalive/{ckpt}"
        print(f"  Fetching {hf_path}...")
        paths[ckpt] = hf_hub_download(
            repo_id=HF_PERSONALIVE,
            filename=hf_path,
        )
    return paths


def get_hf_model_path(model_id: str) -> str:
    """Download HuggingFace model and return local cache path.

    Uses snapshot_download to get the full model directory for models
    that require local paths (like custom UNet loaders).
    """
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=model_id)


def detect_device():
    """Auto-detect device and dtype."""
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16  # PersonaLive uses fp16
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        dtype = torch.float32
        print("Using CPU (will be slow)")
    return device, dtype


def load_pipeline(config_path: str, device: str, dtype: torch.dtype):
    """Load PersonaLive pipeline with lazy HuggingFace downloads.

    Models are downloaded on first use and cached in ~/.cache/huggingface.
    Memory is managed carefully following the original repo's pattern.
    """
    config = OmegaConf.load(config_path)
    infer_config = OmegaConf.load(config.inference_config)

    weight_dtype = torch.float16 if config.weight_dtype == "fp16" else torch.float32

    # Download PersonaLive checkpoints first (while we have more free memory)
    ckpt_paths = download_personalive_weights()

    # Load VAE from HuggingFace (lazy download + centralized cache)
    print(f"Loading VAE from {HF_VAE_MODEL}...")
    vae = AutoencoderKL.from_pretrained(HF_VAE_MODEL).to(device, dtype=weight_dtype)

    # Load Reference UNet from HuggingFace base model
    print(f"Loading Reference UNet from {HF_BASE_MODEL}...")
    reference_unet = UNet2DConditionModel.from_pretrained(
        HF_BASE_MODEL,
        subfolder="unet",
    ).to(device=device, dtype=weight_dtype)

    # Load Denoising UNet (3D) from HuggingFace base model
    # Note: from_pretrained_2d requires local paths, so we use snapshot_download
    print(f"Loading Denoising UNet (3D) from {HF_BASE_MODEL}...")
    base_model_path = get_hf_model_path(HF_BASE_MODEL)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device=device)

    # Initialize custom modules
    print("Initializing Pose Guider...")
    pose_guider = PoseGuider().to(device=device, dtype=weight_dtype)

    print("Initializing Motion Encoder...")
    motion_encoder = MotEncoder().to(dtype=weight_dtype, device=device).eval()

    print("Initializing Motion Extractor...")
    pose_encoder = MotionExtractor(num_kp=21).to(device=device, dtype=weight_dtype).eval()

    # Load CLIP Image Encoder from HuggingFace
    print(f"Loading CLIP Image Encoder from {HF_BASE_MODEL}...")
    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        HF_BASE_MODEL,
        subfolder="image_encoder",
    ).to(dtype=weight_dtype, device=device)

    # Load PersonaLive pretrained weights with explicit memory management
    # Following the pattern from original repo's wrapper.py
    print("Loading PersonaLive checkpoints into models...")

    # Load pose_guider weights
    state_dict = torch.load(ckpt_paths["pose_guider.pth"], map_location="cpu")
    pose_guider.load_state_dict(state_dict, strict=True)
    del state_dict

    # Load motion_encoder weights
    state_dict = torch.load(ckpt_paths["motion_encoder.pth"], map_location="cpu")
    motion_encoder.load_state_dict(state_dict, strict=True)
    del state_dict

    # Load motion_extractor (pose_encoder) weights
    state_dict = torch.load(ckpt_paths["motion_extractor.pth"], map_location="cpu")
    pose_encoder.load_state_dict(state_dict, strict=False)
    del state_dict

    # Load reference_unet weights
    state_dict = torch.load(ckpt_paths["reference_unet.pth"], map_location="cpu")
    reference_unet.load_state_dict(state_dict, strict=True)
    del state_dict

    # Load denoising_unet weights
    state_dict = torch.load(ckpt_paths["denoising_unet.pth"], map_location="cpu")
    denoising_unet.load_state_dict(state_dict, strict=False)
    del state_dict

    # Load temporal_module weights (into denoising_unet)
    state_dict = torch.load(ckpt_paths["temporal_module.pth"], map_location="cpu")
    denoising_unet.load_state_dict(state_dict, strict=False)
    del state_dict

    # Force garbage collection to free CPU memory
    gc.collect()

    # Enable xformers for memory efficiency
    if is_xformers_available():
        reference_unet.enable_xformers_memory_efficient_attention()
        denoising_unet.enable_xformers_memory_efficient_attention()
        print("xformers memory efficient attention enabled")
    else:
        raise ValueError(
            "xformers is not available. Make sure it is installed correctly. "
            "Run: pip install xformers"
        )

    # Setup scheduler
    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    # Create pipeline
    pipe = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        motion_encoder=motion_encoder,
        pose_encoder=pose_encoder,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to(device)

    # Clear GPU cache
    torch.cuda.empty_cache()

    print("Pipeline loaded successfully!")
    return pipe


def run_inference(
    pipe,
    ref_image_path: str,
    driving_video_path: str,
    output_path: str,
    width: int = 512,
    height: int = 512,
    max_frames: int = 100,
    num_inference_steps: int = 4,
    seed: int = 42,
    device: str = "cuda",
):
    """Run portrait animation inference."""

    # Setup face mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    # Load reference image
    print(f"Loading reference image: {ref_image_path}")
    if ref_image_path.endswith('.mp4'):
        src_vid = VideoReader(ref_image_path)
        ref_img = src_vid[0].asnumpy()
        ref_img = Image.fromarray(ref_img).convert("RGB")
    else:
        ref_img = Image.open(ref_image_path).convert("RGB")

    # Load driving video
    print(f"Loading driving video: {driving_video_path}")
    control = VideoReader(driving_video_path)
    video_length = min(len(control) // 4 * 4, max_frames)
    sel_idx = range(len(control))[:video_length]
    control = control.get_batch(list(sel_idx)).asnumpy()  # N, H, W, C
    print(f"Processing {video_length} frames")

    # Crop faces
    ref_image_pil = ref_img.copy()
    ref_patch = crop_face(ref_image_pil, face_mesh)
    ref_face_pil = Image.fromarray(ref_patch).convert("RGB")

    pose_transform = transforms.Compose(
        [transforms.Resize((height, width)), transforms.ToTensor()]
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    # Process driving video frames
    dri_faces = []
    ori_pose_images = []
    for idx_control, pose_image_pil in tqdm(
        enumerate(control[:video_length]),
        total=video_length,
        desc='Cropping faces'
    ):
        pose_image_pil = Image.fromarray(pose_image_pil).convert("RGB")
        ori_pose_images.append(pose_image_pil)
        dri_face = crop_face(pose_image_pil, face_mesh)
        dri_face_pil = Image.fromarray(dri_face).convert("RGB")
        dri_faces.append(dri_face_pil)

    # Prepare tensors
    face_tensor_list = []
    ori_pose_tensor_list = []
    ref_tensor_list = []

    for idx, pose_image_pil in enumerate(ori_pose_images):
        face_tensor_list.append(pose_transform(dri_faces[idx]))
        ori_pose_tensor_list.append(pose_transform(pose_image_pil))
        ref_tensor_list.append(pose_transform(ref_image_pil))

    ref_tensor = torch.stack(ref_tensor_list, dim=0)
    ref_tensor = ref_tensor.transpose(0, 1).unsqueeze(0)

    face_tensor = torch.stack(face_tensor_list, dim=0)
    face_tensor = face_tensor.transpose(0, 1).unsqueeze(0)

    ori_pose_tensor = torch.stack(ori_pose_tensor_list, dim=0)
    ori_pose_tensor = ori_pose_tensor.transpose(0, 1).unsqueeze(0)

    # Run pipeline
    print(f"Running diffusion ({num_inference_steps} steps)...")
    gen_video = pipe(
        ori_pose_images,
        ref_image_pil,
        dri_faces,
        ref_face_pil,
        width,
        height,
        len(dri_faces),
        num_inference_steps=num_inference_steps,
        guidance_scale=1.0,
        generator=generator,
        temporal_window_size=4,
        temporal_adaptive_step=4,
    ).videos

    # Save output
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save generated video only
    save_videos_grid(gen_video, str(output_path), n_rows=1, fps=25, crf=18)
    print(f"Output saved to: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="PersonaLive: Real-time Portrait Animation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default demo files
  python predict.py

  # Custom inputs
  python predict.py --ref_image my_photo.jpg --driving_video motion.mp4

  # Adjust quality/speed
  python predict.py --steps 8 --max_frames 200
"""
    )
    parser.add_argument(
        "--ref_image",
        type=str,
        default="demo/ref_image.png",
        help="Path to reference image (portrait to animate)"
    )
    parser.add_argument(
        "--driving_video",
        type=str,
        default="demo/driving_video.mp4",
        help="Path to driving video (motion source)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output video path (default: outputs/output_TIMESTAMP.mp4)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/prompts/personalive_offline.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--width", "-W",
        type=int,
        default=512,
        help="Output width (default: 512)"
    )
    parser.add_argument(
        "--height", "-H",
        type=int,
        default=512,
        help="Output height (default: 512)"
    )
    parser.add_argument(
        "--max_frames", "-L",
        type=int,
        default=100,
        help="Maximum frames to process (default: 100)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Number of inference steps (default: 4)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Re-enable verbose output if requested
    if args.verbose:
        logging.basicConfig(level=logging.INFO, force=True)
        for logger_name in ["transformers", "diffusers", "torch"]:
            logging.getLogger(logger_name).setLevel(logging.INFO)

    # Detect device
    device, dtype = detect_device()

    # Set output path
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"outputs/output_{timestamp}.mp4"

    # Validate inputs
    if not Path(args.ref_image).exists():
        print(f"Error: Reference image not found: {args.ref_image}")
        return 1
    if not Path(args.driving_video).exists():
        print(f"Error: Driving video not found: {args.driving_video}")
        return 1
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        return 1

    # Load pipeline
    try:
        pipe = load_pipeline(args.config, device, dtype)
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Run inference
    try:
        output_path = run_inference(
            pipe=pipe,
            ref_image_path=args.ref_image,
            driving_video_path=args.driving_video,
            output_path=args.output,
            width=args.width,
            height=args.height,
            max_frames=args.max_frames,
            num_inference_steps=args.steps,
            seed=args.seed,
            device=device,
        )
        print(f"\nDone! Output: {output_path}")
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
