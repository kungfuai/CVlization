#!/usr/bin/env python3
"""
EGSTalker inference wrapper for CVLization.
Real-time audio-driven talking head generation using 3D Gaussian Splatting.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add egstalker to path
egstalker_path = '/workspace/egstalker'
if os.path.exists(egstalker_path):
    sys.path.insert(0, egstalker_path)
else:
    # Fallback: try current directory structure
    script_dir = Path(__file__).parent
    egstalker_path = script_dir / 'egstalker'
    if egstalker_path.exists():
        sys.path.insert(0, str(egstalker_path))

import torch
from render import render_sets
from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args
from utils.general_utils import safe_state
from argparse import ArgumentParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s: [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def setup_dataset_directory(audio_path, reference_path, output_dir):
    """
    Set up dataset directory structure expected by EGSTalker.
    
    Args:
        audio_path: Path to input audio file (.wav)
        reference_path: Path to reference dataset directory or image
        output_dir: Output directory for results
    """
    dataset_dir = Path(output_dir) / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy or link audio file
    audio_file = Path(audio_path)
    if audio_file.exists():
        import shutil
        target_audio = dataset_dir / audio_file.name
        if not target_audio.exists():
            shutil.copy(audio_file, target_audio)
        logger.info(f"Audio file prepared: {target_audio}")
    else:
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Handle reference path
    ref_path = Path(reference_path)
    if ref_path.is_dir():
        # If it's a directory, assume it's already a dataset
        logger.info(f"Using dataset directory: {ref_path}")
        return str(ref_path)
    elif ref_path.is_file() and ref_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        # If it's an image, create a minimal dataset structure
        images_dir = dataset_dir / "images"
        images_dir.mkdir(exist_ok=True)
        import shutil
        target_image = images_dir / ref_path.name
        shutil.copy(ref_path, target_image)
        logger.info(f"Reference image prepared: {target_image}")
        return str(dataset_dir)
    else:
        raise ValueError(f"Reference path must be a directory or image file: {reference_path}")


def find_model_checkpoint(model_path, iteration=None):
    """
    Find the model checkpoint directory.
    
    Args:
        model_path: Base model directory
        iteration: Specific iteration number, or None to find latest
    """
    model_dir = Path(model_path)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    
    if iteration is not None and iteration > 0:
        checkpoint_dir = model_dir / f"iteration_{iteration}"
        if checkpoint_dir.exists():
            return str(model_dir), iteration
        else:
            logger.warning(f"Checkpoint {iteration} not found, searching for latest...")
    
    # Find latest iteration (check both model_dir and model_dir/point_cloud)
    iterations = []
    search_dirs = [model_dir, model_dir / "point_cloud"]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for item in search_dir.iterdir():
            if item.is_dir() and item.name.startswith("iteration_"):
                try:
                    iter_num = int(item.name.split("_")[1])
                    iterations.append(iter_num)
                except ValueError:
                    continue

    if not iterations:
        raise FileNotFoundError(f"No checkpoints found in {model_path} or {model_path}/point_cloud")
    
    latest_iter = max(iterations)
    logger.info(f"Using checkpoint iteration: {latest_iter}")
    return str(model_dir), latest_iter


def main(args):
    """Main inference function."""
    logger.info("=" * 60)
    logger.info("EGSTalker Inference")
    logger.info("=" * 60)
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find model checkpoint
    model_path, iteration = find_model_checkpoint(args.model_path, args.iteration)
    logger.info(f"Model path: {model_path}")
    logger.info(f"Checkpoint iteration: {iteration}")
    
    # Set up dataset directory
    dataset_path = setup_dataset_directory(
        args.audio_path,
        args.reference_path,
        output_dir
    )
    logger.info(f"Dataset path: {dataset_path}")
    
    # Prepare audio features if needed
    # Note: EGSTalker expects .npy files for audio features
    # For now, we'll let EGSTalker handle this, or we need to preprocess
    audio_file = Path(args.audio_path)
    audio_npy = Path(dataset_path) / (audio_file.stem + ".npy")
    
    # Build command-line arguments for EGSTalker render.py
    egstalker_args = [
        "-s", dataset_path,
        "--model_path", model_path,
        "--iteration", str(iteration),
        "--batch", str(args.batch_size),
        "--skip_train",
        "--skip_test",
    ]
    
    # Always use custom audio for our use case
    audio_target = Path(dataset_path) / audio_file.name
    if not audio_target.exists():
        import shutil
        shutil.copy(audio_file, audio_target)
    
    egstalker_args.extend([
        "--custom_aud", audio_npy.name if audio_npy.exists() else "",
        "--custom_wav", audio_target.name
    ])
    
    if args.configs:
        egstalker_args.extend(["--configs", args.configs])
    
    logger.info(f"EGSTalker arguments: {' '.join(egstalker_args)}")
    
    # Create argument parser for EGSTalker and parse arguments
    # We need to temporarily modify sys.argv to pass arguments to EGSTalker's parser
    original_argv = sys.argv.copy()
    try:
        sys.argv = ['predict.py'] + egstalker_args
        
        parser = ArgumentParser(description="EGSTalker inference")
        model = ModelParams(parser, sentinel=True)
        pipeline = PipelineParams(parser)
        hyperparam = ModelHiddenParams(parser)
        parser.add_argument("--iteration", default=-1, type=int)
        parser.add_argument("--skip_train", action="store_true")
        parser.add_argument("--skip_test", action="store_true")
        parser.add_argument("--quiet", action="store_true")
        parser.add_argument("--skip_video", action="store_true")
        parser.add_argument("--configs", type=str)
        parser.add_argument("--batch", type=int, required=True)
        parser.add_argument("--custom_aud", type=str, default='')
        parser.add_argument("--custom_wav", type=str, default='')
        
        # Parse arguments
        combined_args = get_combined_args(parser)
    finally:
        sys.argv = original_argv
    
    # Merge config if provided
    if args.configs and os.path.exists(args.configs):
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        combined_args = merge_hparams(combined_args, config)
    
    # Initialize system state
    safe_state(args.quiet if hasattr(args, 'quiet') else False)
    combined_args.only_infer = True
    
    logger.info("Starting rendering...")
    
    # Run rendering
    with torch.no_grad():
        render_sets(
            model.extract(combined_args),
            hyperparam.extract(combined_args),
            iteration,
            pipeline.extract(combined_args),
            combined_args
        )
    
    logger.info(f"Rendering complete. Output saved to: {output_dir}")
    
    # Find output video
    output_video = None
    for pattern in ["*renders.mov", "*renders.mp4"]:
        videos = list(output_dir.rglob(pattern))
        if videos:
            output_video = videos[0]
            break
    
    if output_video:
        logger.info(f"Output video: {output_video}")
    else:
        logger.warning("Output video not found, check model_path for results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EGSTalker: Real-time audio-driven talking head generation"
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        required=True,
        help="Path to input audio file (.wav)"
    )
    parser.add_argument(
        "--reference_path",
        type=str,
        required=True,
        help="Path to reference dataset directory or reference image"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/outputs",
        help="Output directory for results"
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Model checkpoint iteration (default: use latest)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for rendering"
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help="Path to config file (optional)"
    )
    parser.add_argument(
        "--custom_audio",
        action="store_true",
        help="Use custom audio for inference"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_file = Path(args.output_dir) / f"egstalker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s: [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)
    
    try:
        main(args)
        logger.info("=" * 60)
        logger.info("Inference completed successfully")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        sys.exit(1)
