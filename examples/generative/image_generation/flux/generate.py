"""FLUX.1-schnell text-to-image inference.

Default behavior (no args) is preserved: writes a KungFu.AI demo portrait to
outputs/flux-schnell.png. CLI flags below let you supply a custom prompt,
output path, seed, and image dimensions.
"""
import argparse
import os
import sys

DEFAULT_PROMPT = """
A charismatic speaker is captured mid-speech. She has long smooth hair that's slightly messy on top. She has an angular face, clean shaven, adorned with circular glasses with red rims, is animated as she gestures with she left hand. She is holding a black microphone in her right hand, speaking passionately.

The lady is wearing a light grey sweater over a white t-shirt. She's also wearing a simple black lanyard hanging around his neck. The lanyard badge has the text "KUNGFU.AI", very visible. The photo includes the whole upperbody.

Behind her, there is a blurred background with a white banner containing logos and text (including kungfu.ai written in pink), a professional conference setting.
"""


def parse_args():
    parser = argparse.ArgumentParser(description="FLUX.1-schnell text-to-image")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt (use a string, or omit for the bundled demo)")
    parser.add_argument("--prompt-file", type=str, default=None,
                        help="Path to a text file containing the prompt (overrides --prompt)")
    parser.add_argument("--output", type=str, default="outputs/flux-schnell.png",
                        help="Output PNG path")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4,
                        help="Denoising steps (schnell is calibrated at 4)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.prompt_file:
        prompt = open(args.prompt_file).read().strip()
    elif args.prompt:
        prompt = args.prompt
    else:
        prompt = DEFAULT_PROMPT

    # Heavy imports after argparse so --help is fast.
    import torch
    from diffusers import FluxPipeline

    # CVL workspace resolution (if available)
    try:
        from cvlization.paths import resolve_output_path
        out_path = resolve_output_path(args.output)
    except ImportError:
        out_path = args.output

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
    )
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.enable_sequential_cpu_offload()

    image = pipe(
        prompt,
        guidance_scale=0.0,
        num_inference_steps=args.steps,
        max_sequence_length=256,
        height=args.height,
        width=args.width,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    ).images[0]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    image.save(out_path)
    print(f"Image saved to {out_path}")


if __name__ == "__main__":
    main()
