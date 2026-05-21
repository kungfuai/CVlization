"""Stable Diffusion v1.5 text-to-image inference.

Default behavior (no args) is preserved: an astronaut riding a horse on Mars.
CLI flags below let you supply a custom prompt, output path, seed, etc.
"""
import argparse
import os
import sys

DEFAULT_PROMPT = "a photo of an astronaut riding a horse on mars"


def parse_args():
    parser = argparse.ArgumentParser(description="Stable Diffusion v1.5 text-to-image")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt (use a string, or omit for the bundled demo)")
    parser.add_argument("--prompt-file", type=str, default=None,
                        help="Path to a text file containing the prompt (overrides --prompt)")
    parser.add_argument("--output", type=str, default="outputs/astronaut_rides_horse.png",
                        help="Output PNG path")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=50,
                        help="Denoising steps (SD 1.5 defaults to 50)")
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.prompt_file:
        prompt = open(args.prompt_file).read().strip()
    elif args.prompt:
        prompt = args.prompt
    else:
        prompt = DEFAULT_PROMPT

    import torch
    from diffusers import StableDiffusionPipeline

    try:
        from cvlization.paths import resolve_output_path
        out_path = resolve_output_path(args.output)
    except ImportError:
        out_path = args.output

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    ).to("cuda")

    image = pipe(
        prompt,
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator("cuda").manual_seed(args.seed),
    ).images[0]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    image.save(out_path)
    print(f"Image saved to {out_path}")


if __name__ == "__main__":
    main()
