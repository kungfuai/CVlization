import os
import argparse
import logging
import math
from omegaconf import OmegaConf
from datetime import datetime
from pathlib import Path

import numpy as np
import torch.jit
from torchvision.datasets.folder import pil_loader
from torchvision.transforms.functional import pil_to_tensor, resize, center_crop
from torchvision.transforms.functional import to_pil_image


from mimicmotion.utils.geglu_patch import patch_geglu_inplace
patch_geglu_inplace()

from mimicmotion.pipelines.pipeline_mimicmotion import MimicMotionPipeline
from mimicmotion.utils.loader import create_pipeline
from mimicmotion.utils.utils import save_to_mp4
from mimicmotion.dwpose.preprocess import get_video_pose, get_image_pose

ASPECT_RATIO = 9 / 16
logging.basicConfig(level=logging.INFO, format="%(asctime)s: [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def apply_memory_optimizations(pipeline: MimicMotionPipeline, memory_config, device: torch.device):
    if memory_config is None:
        pipeline.to(device)
        return

    opts = OmegaConf.to_container(memory_config, resolve=True)
    if not isinstance(opts, dict):
        opts = {}
    cpu_offload = opts.get("cpu_offload", "none")
    offload_applied = False

    # Ensure Diffusers registers the device attribute before we try offloading.
    pipeline.to(device)

    try:
        if cpu_offload == "sequential":
            pipeline.enable_sequential_cpu_offload()
            offload_applied = True
        elif cpu_offload == "model":
            pipeline.enable_model_cpu_offload()
            offload_applied = True
    except (ImportError, ValueError) as exc:
        logger.warning("Unable to enable CPU offload (%s); keeping pipeline on %s", exc, device)

    if not offload_applied:
        # Already moved to requested device above
        pass

    if opts.get("attention_slicing") and hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()
    if opts.get("vae_slicing") and hasattr(pipeline, "enable_vae_slicing"):
        pipeline.enable_vae_slicing()
    if opts.get("vae_tiling") and hasattr(pipeline, "enable_vae_tiling"):
        pipeline.enable_vae_tiling()
    if opts.get("vae_cpu"):
        pipeline.vae.to("cpu")


def preprocess(video_path, image_path, resolution=576, sample_stride=2):
    """preprocess ref image pose and video pose

    Args:
        video_path (str): input video pose path
        image_path (str): reference image path
        resolution (int, optional):  Defaults to 576.
        sample_stride (int, optional): Defaults to 2.
    """
    image_pixels = pil_loader(image_path)
    image_pixels = pil_to_tensor(image_pixels) # (c, h, w)
    h, w = image_pixels.shape[-2:]
    ############################ compute target h/w according to original aspect ratio ###############################
    if h>w:
        w_target, h_target = resolution, int(resolution / ASPECT_RATIO // 64) * 64
    else:
        w_target, h_target = int(resolution / ASPECT_RATIO // 64) * 64, resolution
    h_w_ratio = float(h) / float(w)
    if h_w_ratio < h_target / w_target:
        h_resize, w_resize = h_target, math.ceil(h_target / h_w_ratio)
    else:
        h_resize, w_resize = math.ceil(w_target * h_w_ratio), w_target
    image_pixels = resize(image_pixels, [h_resize, w_resize], antialias=None)
    image_pixels = center_crop(image_pixels, [h_target, w_target])
    image_pixels = image_pixels.permute((1, 2, 0)).numpy()
    ##################################### get image&video pose value #################################################
    image_pose = get_image_pose(image_pixels)
    video_pose = get_video_pose(video_path, image_pixels, sample_stride=sample_stride)
    pose_pixels = np.concatenate([np.expand_dims(image_pose, 0), video_pose])
    image_pixels = np.transpose(np.expand_dims(image_pixels, 0), (0, 3, 1, 2))
    return torch.from_numpy(pose_pixels.copy()) / 127.5 - 1, torch.from_numpy(image_pixels) / 127.5 - 1


def run_pipeline(pipeline: MimicMotionPipeline, image_pixels, pose_pixels, device, task_config):
    image_pixels = [to_pil_image(img.to(torch.uint8)) for img in (image_pixels + 1.0) * 127.5]
    generator = torch.Generator(device=device)
    generator.manual_seed(task_config.seed)
    total_steps = getattr(task_config, "num_inference_steps", None)

    def progress_callback(pipeline, step: int, timestep, kwargs):
        if total_steps is None:
            return {}
        interval = max(total_steps // 10, 1)
        if step == 0 or (step + 1) % interval == 0 or (step + 1) == total_steps:
            logger.info(
                "Diffusion progress: %d/%d steps (t=%s)",
                step + 1,
                total_steps,
                int(timestep) if hasattr(timestep, "__int__") else timestep,
            )
        return {}

    frames = pipeline(
        image_pixels, image_pose=pose_pixels, num_frames=pose_pixels.size(0),
        tile_size=task_config.num_frames, tile_overlap=task_config.frames_overlap,
        height=pose_pixels.shape[-2], width=pose_pixels.shape[-1], fps=7,
        noise_aug_strength=task_config.noise_aug_strength, num_inference_steps=task_config.num_inference_steps,
        generator=generator, min_guidance_scale=task_config.guidance_scale, 
        max_guidance_scale=task_config.guidance_scale, decode_chunk_size=8, output_type="pt", device=device,
        callback_on_step_end=progress_callback,
    ).frames.cpu()
    video_frames = (frames * 255.0).to(torch.uint8)

    for vid_idx in range(video_frames.shape[0]):
        # deprecated first frame because of ref image
        _video_frames = video_frames[vid_idx, 1:]

    return _video_frames


@torch.no_grad()
def main(args):
    if not args.no_use_float16 :
        torch.set_default_dtype(torch.float16)

    infer_config = OmegaConf.load(args.inference_config)

    if getattr(infer_config, "memory_optimization", None) is None:
        infer_config.memory_optimization = OmegaConf.create({})

    if args.attention_slicing is not None:
        infer_config.memory_optimization.attention_slicing = args.attention_slicing == "on"
    if args.vae_slicing is not None:
        infer_config.memory_optimization.vae_slicing = args.vae_slicing == "on"

    if hasattr(infer_config, "dwpose"):
        dwpose_cfg = infer_config.dwpose
        if hasattr(dwpose_cfg, "det_path"):
            os.environ.setdefault("DWPOSE_MODEL_DET", str(dwpose_cfg.det_path))
        if hasattr(dwpose_cfg, "pose_path"):
            os.environ.setdefault("DWPOSE_MODEL_POSE", str(dwpose_cfg.pose_path))

    pipeline = create_pipeline(infer_config, device)
    apply_memory_optimizations(
        pipeline, getattr(infer_config, "memory_optimization", None), device
    )

    for task in infer_config.test_case:
        if args.num_inference_steps is not None:
            task.num_inference_steps = args.num_inference_steps
        if args.num_frames is not None:
            task.num_frames = args.num_frames
        if args.resolution is not None:
            task.resolution = args.resolution
        if args.sample_stride is not None:
            task.sample_stride = args.sample_stride
        ############################################## Pre-process data ##############################################
        pose_pixels, image_pixels = preprocess(
            task.ref_video_path, task.ref_image_path, 
            resolution=task.resolution, sample_stride=task.sample_stride
        )
        print(f"shapes: pose_pixels: {pose_pixels.shape}, image_pixels: {image_pixels.shape}")
        print(f"dtypes: pose_pixels: {pose_pixels.dtype}, image_pixels: {image_pixels.dtype}")
        print(f"unet dtype: {pipeline.unet.dtype}, num_frames: {task.num_frames}, tile_size: {task.num_frames}, tile_overlap: {task.frames_overlap}")
        ########################################### Run MimicMotion pipeline ###########################################
        _video_frames = run_pipeline(
            pipeline, 
            image_pixels, pose_pixels, 
            device, task
        )
        ################################### save results to output folder. ###########################################
        logger.info("Saving video (%d frames) to %s", _video_frames.shape[0], args.output_dir)
        save_to_mp4(
            _video_frames, 
            f"{args.output_dir}/{os.path.basename(task.ref_video_path).split('.')[0]}" \
            f"_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4",
            fps=task.fps,
        )

def set_logger(log_file=None, log_level=logging.INFO):
    log_handler = logging.FileHandler(log_file, "w")
    log_handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s]: %(message)s")
    )
    log_handler.setLevel(log_level)
    logger.addHandler(log_handler)


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--inference_config", type=str, default="configs/test.yaml") #ToDo
    parser.add_argument("--output_dir", type=str, default="outputs/", help="path to output")
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        help="Override inference steps for every task defined in the config",
    )
    parser.add_argument(
        "--attention_slicing",
        choices=["on", "off"],
        default=None,
        help="Force attention slicing on or off (overrides config).",
    )
    parser.add_argument(
        "--vae_slicing",
        choices=["on", "off"],
        default=None,
        help="Force VAE slicing on or off (overrides config).",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="Override number of frames processed per test case (must be <= reference video length).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Force square resolution (long side) for preprocessing (e.g. 384).",
    )
    parser.add_argument(
        "--sample_stride",
        type=int,
        default=None,
        help="Increase sampling stride on the reference video to skip frames (>=1).",
    )
    parser.add_argument("--no_use_float16",
                        action="store_true",
                        help="Whether use float16 to speed up inference",
    )
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    set_logger(args.log_file \
               if args.log_file is not None else f"{args.output_dir}/{datetime.now().strftime('%Y%m%d%H%M%S')}.log")
    main(args)
    logger.info(f"--- Finished ---")
