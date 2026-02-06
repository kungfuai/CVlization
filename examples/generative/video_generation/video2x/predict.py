#!/usr/bin/env python3
"""
Real-ESRGAN: Video/Image Super-Resolution

Practical upscaling for images and videos using Real-ESRGAN models.
Supports RRDBNet and SRVGGNetCompact architectures with optional
GFPGAN face enhancement.

Models: https://github.com/xinntao/Real-ESRGAN
"""

import argparse
import os
import subprocess
import sys

import cv2
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from tqdm import tqdm

# CVL dual-mode execution support
try:
    from cvlization.paths import (
        get_input_dir,
        get_output_dir,
        resolve_input_path,
        resolve_output_path,
    )
    CVL_AVAILABLE = True
except ImportError:
    CVL_AVAILABLE = False

    def get_input_dir():
        return os.getenv("CVL_INPUTS", os.getcwd())

    def get_output_dir():
        output_dir = os.getenv("CVL_OUTPUTS", "./outputs")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def resolve_input_path(path):
        if path.startswith(("http://", "https://")) or path.startswith("/"):
            return path
        base = get_input_dir()
        return os.path.join(base, path) if os.getenv("CVL_INPUTS") else path

    def resolve_output_path(path):
        if path is None:
            path = "result.txt"
        if path.startswith("/"):
            return path
        base = get_output_dir()
        return os.path.join(base, path)


# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "RealESRGAN_x4plus": {
        "arch": "RRDBNet",
        "params": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
        "netscale": 4,
        "url": ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"],
    },
    "RealESRNet_x4plus": {
        "arch": "RRDBNet",
        "params": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
        "netscale": 4,
        "url": ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"],
    },
    "RealESRGAN_x2plus": {
        "arch": "RRDBNet",
        "params": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
        "netscale": 2,
        "url": ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"],
    },
    "RealESRGAN_x4plus_anime_6B": {
        "arch": "RRDBNet",
        "params": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4),
        "netscale": 4,
        "url": ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"],
    },
    "realesr-animevideov3": {
        "arch": "SRVGGNetCompact",
        "params": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type="prelu"),
        "netscale": 4,
        "url": ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"],
    },
    "realesr-general-x4v3": {
        "arch": "SRVGGNetCompact",
        "params": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu"),
        "netscale": 4,
        "url": [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
        ],
    },
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv"}

ASSETS_REPO = "zzsi/cvl"
SAMPLE_FILES = {
    "image": "video2x/0030.jpg",
    "video": "video2x/onepiece_demo.mp4",
}


def get_sample_asset(kind: str = "image") -> str:
    """Download a sample asset from HuggingFace if not available locally."""
    remote_path = SAMPLE_FILES[kind]
    filename = os.path.basename(remote_path)
    local_path = os.path.join(os.path.dirname(__file__), "test_inputs", filename)
    if os.path.exists(local_path):
        return local_path

    print(f"Downloading sample {kind}: {filename}")
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=ASSETS_REPO,
        filename=remote_path,
        repo_type="dataset",
    )
    return downloaded


def parse_args():
    parser = argparse.ArgumentParser(description="Real-ESRGAN image/video super-resolution")
    parser.add_argument("--input", "-i", default=None, help="Input image or video file (downloads sample if omitted)")
    parser.add_argument("--output", "-o", default=None, help="Output file path (auto-generated if omitted)")
    parser.add_argument(
        "--model", "-n",
        default="RealESRGAN_x4plus",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model name (default: RealESRGAN_x4plus)",
    )
    parser.add_argument("--outscale", "-s", type=float, default=None, help="Output upscaling factor (default: model netscale)")
    parser.add_argument("--denoise-strength", type=float, default=0.5, help="Denoise strength for realesr-general-x4v3 (0=weak, 1=strong)")
    parser.add_argument("--tile", "-t", type=int, default=0, help="Tile size for VRAM control (0=no tiling)")
    parser.add_argument("--tile-pad", type=int, default=10, help="Tile padding")
    parser.add_argument("--pre-pad", type=int, default=0, help="Pre-padding at each border")
    parser.add_argument("--face-enhance", action="store_true", help="Use GFPGAN for face enhancement")
    parser.add_argument("--fp32", action="store_true", help="Use fp32 instead of fp16")
    return parser.parse_args()


def get_model_and_upsampler(args):
    """Build the network and RealESRGANer upsampler."""
    cfg = MODEL_CONFIGS[args.model]

    if cfg["arch"] == "RRDBNet":
        model = RRDBNet(**cfg["params"])
    else:
        model = SRVGGNetCompact(**cfg["params"])

    netscale = cfg["netscale"]

    # Download model weights
    from basicsr.utils.download_util import load_file_from_url

    model_path = None
    for url in cfg["url"]:
        model_path = load_file_from_url(
            url=url, model_dir=os.path.join(os.path.dirname(__file__), "weights"),
            progress=True, file_name=None,
        )

    # DNI (dynamic network interpolation) for realesr-general-x4v3 denoise control
    dni_weight = None
    if args.model == "realesr-general-x4v3" and args.denoise_strength != 1:
        wdn_model_path = model_path.replace("realesr-general-x4v3", "realesr-general-wdn-x4v3")
        model_path = [model_path, wdn_model_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    import torch
    half = not args.fp32 and torch.cuda.is_available()
    gpu_id = 0 if torch.cuda.is_available() else None

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=half,
        gpu_id=gpu_id,
    )

    face_enhancer = None
    if args.face_enhance:
        from gfpgan import GFPGANer
        outscale = args.outscale or netscale
        face_enhancer = GFPGANer(
            model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
            upscale=outscale,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=upsampler,
        )

    return upsampler, face_enhancer, netscale


def process_image(img, upsampler, face_enhancer, outscale):
    """Upscale a single BGR image array."""
    if face_enhancer is not None:
        _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
    else:
        output, _ = upsampler.enhance(img, outscale=outscale)
    return output


def _video_has_audio(video_path):
    """Check if a video file has an audio stream."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", video_path],
            capture_output=True, text=True,
        )
        return bool(result.stdout.strip())
    except FileNotFoundError:
        return False


def process_video(input_path, output_path, upsampler, face_enhancer, outscale):
    """Upscale a video frame-by-frame, preserving audio with ffmpeg."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: cannot open video {input_path}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_w = int(width * outscale)
    out_h = int(height * outscale)

    has_audio = _video_has_audio(input_path)

    # Build ffmpeg command to encode upscaled frames
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}", "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "pipe:0",
    ]
    if has_audio:
        ffmpeg_cmd += ["-i", input_path, "-map", "0:v", "-map", "1:a", "-c:a", "copy"]
    ffmpeg_cmd += [
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p",
        output_path,
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        for _ in tqdm(range(total_frames), desc="Upscaling video", unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break
            output = process_image(frame, upsampler, face_enhancer, outscale)
            proc.stdin.write(output.tobytes())
    finally:
        cap.release()
        proc.stdin.close()
        proc.wait()

    if proc.returncode != 0:
        print("Error: ffmpeg encoding failed", file=sys.stderr)
        sys.exit(1)


def auto_output_path(input_path, outscale):
    """Generate an output path like 'photo_4x.png' from 'photo.png'."""
    base, ext = os.path.splitext(input_path)
    scale_str = f"{int(outscale)}x" if outscale == int(outscale) else f"{outscale}x"
    return f"{base}_{scale_str}{ext}"


def main():
    args = parse_args()

    if args.input is None:
        input_path = get_sample_asset("image")
    else:
        input_path = resolve_input_path(args.input)

    upsampler, face_enhancer, netscale = get_model_and_upsampler(args)
    outscale = args.outscale or netscale

    ext = os.path.splitext(input_path)[1].lower()
    is_video = ext in VIDEO_EXTENSIONS

    # Determine output path
    if args.output:
        output_path = resolve_output_path(args.output)
    else:
        output_path = resolve_output_path(auto_output_path(os.path.basename(input_path), outscale))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if is_video:
        print(f"Upscaling video: {input_path} -> {output_path} ({outscale}x, model={args.model})")
        process_video(input_path, output_path, upsampler, face_enhancer, outscale)
    else:
        print(f"Upscaling image: {input_path} -> {output_path} ({outscale}x, model={args.model})")
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Error: cannot read image {input_path}", file=sys.stderr)
            sys.exit(1)
        output = process_image(img, upsampler, face_enhancer, outscale)
        cv2.imwrite(output_path, output)

    print(f"Done: {output_path}")


if __name__ == "__main__":
    main()
