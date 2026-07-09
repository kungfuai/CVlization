"""Generate one reference portrait per manifest item with FLUX.1-schnell.

Runs INSIDE the `flux` docker image (diffusers + FluxPipeline available).
Model is loaded once, then all items are generated in a loop. Resume-safe:
skips items whose output PNG already exists.

Usage (inside container):
    python flux_portraits.py <manifest.jsonl> <output_dir>
"""
import json
import os
import sys

import torch
from diffusers import FluxPipeline

# Light suffix: nudge toward upper-body framing without overriding the scene's
# own style description (important for the anime-bucket items).
FRAMING_SUFFIX = ", upper body in frame, single subject centered, sharp focus"


def main():
    manifest_path, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    items = [json.loads(l) for l in open(manifest_path) if l.strip()]

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
    )
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.enable_sequential_cpu_offload()

    done = 0
    for it in items:
        out = os.path.join(out_dir, f"{it['id']}.png")
        if os.path.exists(out):
            done += 1
            continue
        prompt = it["scene"] + FRAMING_SUFFIX
        img = pipe(
            prompt,
            guidance_scale=0.0,
            num_inference_steps=4,
            max_sequence_length=256,
            height=768,
            width=768,
            generator=torch.Generator("cpu").manual_seed(int(it["id"])),
        ).images[0]
        img.save(out)
        done += 1
        print(f"[{done}/{len(items)}] {it['id']} ({it['bucket']}) -> {out}", flush=True)

    print(f"flux_portraits done: {done}/{len(items)}")


if __name__ == "__main__":
    main()
