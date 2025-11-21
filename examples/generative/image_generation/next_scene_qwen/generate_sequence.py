#!/usr/bin/env python3
"""
Generate a multi-scene narrative sequence using Next-Scene LoRA.
"""

import argparse
import os
from pathlib import Path
import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image
import json


def load_pipeline(base_model, lora_model, lora_scale):
    """Load the diffusion pipeline with LoRA."""
    print(f"Loading base model: {base_model}")

    try:
        pipe = AutoPipelineForImage2Image.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
    except Exception as e:
        print(f"Could not load as Image2Image pipeline: {e}")
        print("Trying alternative loading method...")
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
        )

    pipe = pipe.to("cuda")

    # Load LoRA weights
    print(f"Loading LoRA: {lora_model}")
    try:
        pipe.load_lora_weights(lora_model)
        print(f"LoRA loaded with scale: {lora_scale}")
    except Exception as e:
        print(f"Warning: Could not load LoRA weights: {e}")

    # Enable optimizations
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("Enabled xformers memory efficient attention")
    except Exception:
        pass

    return pipe


def generate_scene(pipe, prompt, prev_image, args, scene_num):
    """Generate a single scene."""
    # Prefix prompt with "Next Scene:" if not already present
    if not prompt.lower().startswith("next scene"):
        prompt = f"Next Scene: {prompt}"

    print(f"\n--- Scene {scene_num} ---")
    print(f"Prompt: {prompt}")

    # Set seed if provided
    generator = None
    if args.seed is not None:
        # Use different seed for each scene
        seed = args.seed + scene_num
        generator = torch.Generator(device="cuda").manual_seed(seed)

    # Prepare generation kwargs
    gen_kwargs = {
        "prompt": prompt,
        "negative_prompt": args.negative_prompt,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "generator": generator,
    }

    # Add LoRA scale
    if hasattr(pipe, 'set_adapters'):
        gen_kwargs["cross_attention_kwargs"] = {"scale": args.lora_scale}

    # Use previous image if available
    if prev_image is not None:
        gen_kwargs["image"] = prev_image
        gen_kwargs["strength"] = args.strength
    else:
        # First scene - text-to-image
        gen_kwargs["width"] = args.width
        gen_kwargs["height"] = args.height

    # Generate
    output = pipe(**gen_kwargs)
    return output.images[0]


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-scene narrative sequence"
    )
    parser.add_argument(
        "--scenes",
        type=str,
        required=True,
        help="Path to JSON file with scene descriptions, or comma-separated prompts"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/sequence",
        help="Output directory for scene images"
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="Optional: combine scenes into video"
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="blurry, low quality, distorted, black bars",
        help="Negative prompt for all scenes"
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=30,
        help="Number of inference steps per scene"
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Guidance scale"
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=0.75,
        help="LoRA strength (0.7-0.8 recommended)"
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.6,
        help="Scene-to-scene transformation strength"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Output width"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=768,
        help="Output height"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="Frames per second if creating video"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="Base model ID"
    )
    parser.add_argument(
        "--lora-model",
        type=str,
        default="lovis93/next-scene-qwen-image-lora-2509",
        help="LoRA model ID"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse scenes
    if args.scenes.endswith('.json'):
        with open(args.scenes, 'r') as f:
            scenes_data = json.load(f)
            if isinstance(scenes_data, list):
                scenes = scenes_data
            elif isinstance(scenes_data, dict) and 'scenes' in scenes_data:
                scenes = scenes_data['scenes']
            else:
                raise ValueError("JSON must be a list or dict with 'scenes' key")
    else:
        # Comma-separated prompts
        scenes = [s.strip() for s in args.scenes.split(',')]

    print(f"Generating {len(scenes)} scenes")

    # Load pipeline once
    pipe = load_pipeline(args.base_model, args.lora_model, args.lora_scale)

    # Generate scenes sequentially
    prev_image = None
    scene_paths = []

    for i, scene_prompt in enumerate(scenes, 1):
        # Generate scene
        image = generate_scene(pipe, scene_prompt, prev_image, args, i)

        # Save scene
        scene_path = output_dir / f"scene_{i:03d}.png"
        image.save(scene_path)
        scene_paths.append(scene_path)
        print(f"Saved: {scene_path}")

        # Use this scene as input for next scene
        prev_image = image.resize((args.width, args.height))

    print(f"\n✓ Generated {len(scenes)} scenes in: {output_dir}")

    # Optionally create video
    if args.output_video:
        print(f"\nCreating video: {args.output_video}")
        try:
            import cv2
            import numpy as np

            # Load first image to get dimensions
            first_img = cv2.imread(str(scene_paths[0]))
            height, width = first_img.shape[:2]

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                args.output_video,
                fourcc,
                args.fps,
                (width, height)
            )

            # Write frames
            for scene_path in scene_paths:
                img = cv2.imread(str(scene_path))
                out.write(img)

            out.release()
            print(f"Video saved: {args.output_video}")
            print(f"Duration: {len(scenes) / args.fps:.1f} seconds @ {args.fps} fps")

        except ImportError:
            print("opencv-python not available, skipping video creation")
        except Exception as e:
            print(f"Error creating video: {e}")


if __name__ == "__main__":
    main()
