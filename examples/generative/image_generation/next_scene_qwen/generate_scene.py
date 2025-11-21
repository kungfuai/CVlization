#!/usr/bin/env python3
"""
Generate next scene using Qwen-Image-Edit with Next-Scene LoRA.
"""

import argparse
import os
from pathlib import Path
import torch
from PIL import Image


def main():
    parser = argparse.ArgumentParser(
        description="Generate next scene in cinematic sequence"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Scene description (will be prefixed with 'Next Scene:')"
    )
    parser.add_argument(
        "--input-image",
        type=str,
        default=None,
        help="Previous scene image (optional, for image-to-image)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/scene.png",
        help="Output image path"
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="blurry, low quality, distorted, black bars",
        help="Negative prompt"
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=30,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Guidance scale for generation"
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
        help="Transformation strength for image-to-image (0.0-1.0)"
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
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Base model ID (Stable Diffusion compatible model)"
    )
    parser.add_argument(
        "--lora-model",
        type=str,
        default="lovis93/next-scene-qwen-image-lora-2509",
        help="LoRA model ID"
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefix prompt with "Next Scene:" if not already present
    prompt = args.prompt
    if not prompt.lower().startswith("next scene"):
        prompt = f"Next Scene: {prompt}"

    print(f"Loading base model: {args.base_model}")

    # Load the appropriate pipeline based on whether we have input image
    if args.input_image:
        from diffusers import AutoPipelineForImage2Image as PipelineClass
        print("Using Image2Image pipeline")
    else:
        from diffusers import AutoPipelineForText2Image as PipelineClass
        print("Using Text2Image pipeline")

    pipe = PipelineClass.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )

    pipe = pipe.to("cuda")

    # Load LoRA weights
    print(f"Loading LoRA: {args.lora_model}")
    try:
        pipe.load_lora_weights(args.lora_model)
        print(f"LoRA loaded with scale: {args.lora_scale}")
    except Exception as e:
        print(f"Warning: Could not load LoRA weights: {e}")
        print("Continuing without LoRA...")

    # Enable optimizations
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("Enabled xformers memory efficient attention")
    except Exception as e:
        print(f"Could not enable xformers: {e}")

    # Set seed if provided
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        print(f"Using seed: {args.seed}")

    print(f"\nGenerating scene with prompt:")
    print(f"  {prompt}")
    print(f"Negative prompt: {args.negative_prompt}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Guidance scale: {args.guidance_scale}")

    # Prepare generation kwargs
    gen_kwargs = {
        "prompt": prompt,
        "negative_prompt": args.negative_prompt,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "generator": generator,
    }

    # Add LoRA scale if LoRA was loaded
    if hasattr(pipe, 'set_adapters'):
        gen_kwargs["cross_attention_kwargs"] = {"scale": args.lora_scale}

    # Load input image if provided
    if args.input_image:
        if not os.path.exists(args.input_image):
            raise FileNotFoundError(f"Input image not found: {args.input_image}")

        input_img = Image.open(args.input_image).convert("RGB")
        # Resize to match output dimensions
        input_img = input_img.resize((args.width, args.height))
        print(f"Using input image: {args.input_image}")

        gen_kwargs["image"] = input_img
        gen_kwargs["strength"] = args.strength
    else:
        # Text-to-image mode
        gen_kwargs["width"] = args.width
        gen_kwargs["height"] = args.height

    # Generate image
    output = pipe(**gen_kwargs)

    # Save image
    image = output.images[0]
    image.save(args.output)

    print(f"\nScene saved successfully to: {args.output}")
    print(f"Resolution: {image.size[0]}x{image.size[1]}")


if __name__ == "__main__":
    main()
