#!/usr/bin/env python3
"""
LLaVA-NeXT-Video-7B inference (video+text -> text) with transformers.
"""
import argparse
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import List

import decord
import requests
import torch
from transformers import AutoProcessor, LlavaNextVideoForConditionalGeneration

try:
    from cvlization.paths import resolve_input_path, resolve_output_path
except Exception:
    resolve_input_path = None  # type: ignore
    resolve_output_path = None  # type: ignore


decord.bridge.set_bridge("torch")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VIDEO_URL = "https://huggingface.co/datasets/Narsil/video_dummy/resolve/main/small.mp4"
DEFAULT_MODEL = os.environ.get("LLAVA_NEXT_VIDEO_MODEL_ID", "llava-hf/LLaVA-NeXT-Video-7B-hf")
CACHE_DIR = Path(os.environ.get("LLAVA_NEXT_VIDEO_CACHE", "/root/.cache/llava_next_video"))


def download_to_cache(url: str, suffix: str = ".mp4") -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(url.encode("utf-8")).hexdigest() + (suffix or ".mp4")
    dest = CACHE_DIR / name
    if dest.exists():
        return dest
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def load_video_frames(path: str, max_frames: int = 8) -> List[torch.Tensor]:
    if path.startswith(("http://", "https://")):
        path = str(download_to_cache(path, Path(path).suffix or ".mp4"))
    elif resolve_input_path:
        try:
            path = resolve_input_path(path)
        except Exception:
            pass

    vr = decord.VideoReader(path)
    total = len(vr)
    if total == 0:
        raise ValueError("Video contains no frames")
    idx = torch.linspace(0, total - 1, steps=min(max_frames, total)).long().tolist()
    frames = vr.get_batch(idx)  # (F, H, W, C) torch
    frames = frames.permute(0, 3, 1, 2)  # to (F, C, H, W)
    return [frame for frame in frames]


def save_output(text: str, path: Path, fmt: str, model_id: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        payload = {
            "text": text,
            "model": model_id,
            "timestamp": datetime.now().isoformat(),
        }
        import json

        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        path.write_text(text, encoding="utf-8")
    print(f"Saved to {path}")


def resolve_out(path: str) -> Path:
    if resolve_output_path:
        try:
            return Path(resolve_output_path(path))
        except Exception:
            return Path(path)
    return Path(path)


def run_caption(
    model_id: str,
    frames: List[torch.Tensor],
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device} ...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    conv = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[conv],
        videos=[frames],
        return_tensors="pt",
    ).to(device)

    if device == "cuda" and "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            use_cache=True,
        )[0]
    trimmed = output[inputs["input_ids"].shape[1]:]
    caption = processor.decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    return caption


def parse_args():
    parser = argparse.ArgumentParser(description="LLaVA-NeXT-Video-7B captioning")
    parser.add_argument("--video", help="Path or URL to video (mp4). Defaults to a small sample URL.")
    parser.add_argument("--prompt", default="Describe the video in detail.", help="Prompt text.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL, help="Model to load.")
    parser.add_argument("--max-frames", type=int, default=8, help="Frames to sample from video.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Generation cap.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling.")
    parser.add_argument("--output", default="outputs/llava_next_video.txt", help="Where to save.")
    parser.add_argument("--format", choices=["txt", "json"], default="txt")
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = args.video or DEFAULT_VIDEO_URL
    frames = load_video_frames(video_path, max_frames=args.max_frames)

    caption = run_caption(
        args.model_id,
        frames,
        args.prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
    )

    print("\n=== Caption ===")
    print(caption)
    print("===============")

    save_output(caption, resolve_out(args.output), args.format, args.model_id)


if __name__ == "__main__":
    main()
