#!/usr/bin/env python3
"""
FlashVSR v1.1 video super-resolution (4x) inference.

- Default pipeline: tiny
- Optional: full
- Requires Block-Sparse-Attention (LCSA)
"""
import os
import sys
import re
import logging
import warnings
import tarfile
from pathlib import Path
from typing import Tuple, Optional

# Suppress verbose logging by default
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "0")
os.environ.setdefault("MLIR_ENABLE_DUMP", "0")

warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

logging.basicConfig(level=logging.ERROR)
for logger_name in ["transformers", "diffusers", "torch", "triton"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

import argparse
import numpy as np
from PIL import Image
import imageio
from tqdm import tqdm
import torch
from einops import rearrange

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
        return os.getcwd()

    def get_output_dir():
        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def resolve_input_path(path, base_dir):
        if os.path.isabs(path):
            return path
        return os.path.join(base_dir, path)

    def resolve_output_path(path, base_dir):
        if os.path.isabs(path):
            return path
        return os.path.join(base_dir, path)


def enable_verbose_logging():
    logging.basicConfig(level=logging.INFO, force=True)
    for logger_name in ["transformers", "diffusers", "torch", "triton"]:
        logging.getLogger(logger_name).setLevel(logging.INFO)
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
    os.environ["MLIR_ENABLE_DUMP"] = "1"


def detect_device() -> Tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        major_cc = torch.cuda.get_device_capability()[0]
        dtype = torch.bfloat16 if major_cc >= 8 else torch.float16
        return "cuda", dtype
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def check_block_sparse_attn():
    try:
        import block_sparse_attn  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Block-Sparse-Attention is required but not available. "
            "Please rebuild the Docker image to compile it."
        ) from exc


def _resolve_vendor_root(script_dir: Path) -> Path:
    """Return a vendor root that contains diffsynth, preferring local."""
    local = script_dir / "vendor"
    if (local / "diffsynth" / "__init__.py").exists():
        return local
    # Writable fallback inside HF cache (always mounted read-write)
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return cache / "cvl_vendor" / "flashvsr"


def add_vendor_paths(vendor_root: Path):
    vendor_root = vendor_root.resolve()
    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))
    wanvsr_dir = vendor_root / "WanVSR"
    if str(wanvsr_dir) not in sys.path:
        sys.path.insert(0, str(wanvsr_dir))


def tensor2video(frames: torch.Tensor):
    frames = rearrange(frames, "C T H W -> T H W C")
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    frames = [Image.fromarray(frame) for frame in frames]
    return frames


def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", os.path.basename(name))]


def list_images_natural(folder: str):
    exts = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]
    fs.sort(key=natural_key)
    return fs


def largest_8n1_leq(n):
    return 0 if n < 1 else ((n - 1) // 8) * 8 + 1


def is_video(path: str):
    return os.path.isfile(path) and path.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))


def pil_to_tensor_neg1_1(img: Image.Image, dtype=torch.bfloat16, device="cuda"):
    t = torch.from_numpy(np.asarray(img, np.uint8)).to(device=device, dtype=torch.float32)
    t = t.permute(2, 0, 1) / 255.0 * 2.0 - 1.0
    return t.to(dtype)


def save_video(frames, save_path, fps=30, quality=5):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    writer = imageio.get_writer(save_path, fps=fps, quality=quality)
    for f in tqdm(frames, desc=f"Saving {os.path.basename(save_path)}"):
        writer.append_data(np.array(f))
    writer.close()


def compute_scaled_and_target_dims(w0: int, h0: int, scale: float = 4.0, multiple: int = 128):
    if w0 <= 0 or h0 <= 0:
        raise ValueError("Invalid original size")
    if scale <= 0:
        raise ValueError("scale must be > 0")

    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))

    tW = (sW // multiple) * multiple
    tH = (sH // multiple) * multiple

    if tW == 0 or tH == 0:
        raise ValueError(
            f"Scaled size too small ({sW}x{sH}) for multiple={multiple}. "
            f"Increase scale (got {scale})."
        )

    return sW, sH, tW, tH


def upscale_then_center_crop(img: Image.Image, scale: float, tW: int, tH: int) -> Image.Image:
    w0, h0 = img.size
    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))

    if tW > sW or tH > sH:
        raise ValueError(
            f"Target crop ({tW}x{tH}) exceeds scaled size ({sW}x{sH}). "
            "Increase scale."
        )

    up = img.resize((sW, sH), Image.BICUBIC)
    l = (sW - tW) // 2
    t = (sH - tH) // 2
    return up.crop((l, t, l + tW, t + tH))


def prepare_input_tensor(path: str, scale: float = 4.0, dtype=torch.bfloat16, device="cuda"):
    if os.path.isdir(path):
        paths0 = list_images_natural(path)
        if not paths0:
            raise FileNotFoundError(f"No images in {path}")

        with Image.open(paths0[0]) as img0:
            w0, h0 = img0.size
        n0 = len(paths0)
        print(f"[{os.path.basename(path)}] Original Resolution: {w0}x{h0} | Original Frames: {n0}")

        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
        print(f"[{os.path.basename(path)}] Scaled (x{scale:.2f}): {sW}x{sH} -> Target (128-multiple): {tW}x{tH}")

        paths = paths0 + [paths0[-1]] * 4
        frames_n = largest_8n1_leq(len(paths))
        if frames_n == 0:
            raise RuntimeError(f"Not enough frames after padding in {path}. Got {len(paths)}.")
        paths = paths[:frames_n]
        print(f"[{os.path.basename(path)}] Target Frames (8n-3): {frames_n - 4}")

        frames = []
        for p in paths:
            with Image.open(p).convert("RGB") as img:
                img_out = upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH)
            frames.append(pil_to_tensor_neg1_1(img_out, dtype, device))
        vid = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)
        fps = 30
        return vid, tH, tW, frames_n, fps

    if is_video(path):
        reader = imageio.get_reader(path)
        first = Image.fromarray(reader.get_data(0)).convert("RGB")
        w0, h0 = first.size

        meta = {}
        try:
            meta = reader.get_meta_data()
        except Exception:
            pass
        fps_val = meta.get("fps", 30)
        fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 30

        def count_frames(r):
            try:
                nf = meta.get("nframes", None)
                if isinstance(nf, int) and nf > 0:
                    return nf
            except Exception:
                pass
            try:
                return r.count_frames()
            except Exception:
                n = 0
                try:
                    while True:
                        r.get_data(n)
                        n += 1
                except Exception:
                    return n

        total = count_frames(reader)
        if total <= 0:
            reader.close()
            raise RuntimeError(f"Cannot read frames from {path}")

        print(f"[{os.path.basename(path)}] Original Resolution: {w0}x{h0} | Original Frames: {total} | FPS: {fps}")

        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
        print(f"[{os.path.basename(path)}] Scaled (x{scale:.2f}): {sW}x{sH} -> Target (128-multiple): {tW}x{tH}")

        idx = list(range(total)) + [total - 1] * 4
        frames_n = largest_8n1_leq(len(idx))
        if frames_n == 0:
            reader.close()
            raise RuntimeError(f"Not enough frames after padding in {path}. Got {len(idx)}.")
        idx = idx[:frames_n]
        print(f"[{os.path.basename(path)}] Target Frames (8n-3): {frames_n - 4}")

        frames = []
        try:
            for i in idx:
                img = Image.fromarray(reader.get_data(i)).convert("RGB")
                img_out = upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH)
                frames.append(pil_to_tensor_neg1_1(img_out, dtype, device))
        finally:
            try:
                reader.close()
            except Exception:
                pass

        vid = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)
        return vid, tH, tW, frames_n, fps

    raise ValueError(f"Unsupported input: {path}")


def download_weights(model_dir: Optional[str], repo_id: str, token: Optional[str]) -> Path:
    if model_dir:
        model_path = Path(model_dir).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Weights directory not found: {model_path}")
        return model_path

    from huggingface_hub import snapshot_download

    cache_dir = os.environ.get("HF_HOME", None)
    model_path = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        token=token,
    )
    return Path(model_path).resolve()


def maybe_download_sample_input(input_arg: str, token: Optional[str]) -> Optional[Path]:
    if input_arg not in {"sample", "example0"}:
        return None
    from huggingface_hub import hf_hub_download

    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    local_dir = cache_root / "cvl_samples" / "flashvsr"
    local_dir.mkdir(parents=True, exist_ok=True)
    sample_path = hf_hub_download(
        repo_id="zzsi/cvl",
        repo_type="dataset",
        filename="flashvsr/example0.mp4",
        cache_dir=str(cache_root),
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        token=token,
    )
    return Path(sample_path).resolve()


def _safe_extract_tar(tar: tarfile.TarFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in tar.getmembers():
        member_path = (dest / member.name).resolve()
        if not str(member_path).startswith(str(dest) + os.sep):
            raise RuntimeError(f"Unsafe path in archive: {member.name}")
    tar.extractall(dest)


def _hf_download(filename: str, token: Optional[str]) -> str:
    from huggingface_hub import hf_hub_download

    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    local_dir = cache_root / "cvl_assets" / "flashvsr"
    local_dir.mkdir(parents=True, exist_ok=True)
    return hf_hub_download(
        repo_id="zzsi/cvl",
        repo_type="dataset",
        filename=filename,
        local_dir=str(local_dir),
        token=token,
    )


def ensure_vendor_code(vendor_root: Path, token: Optional[str]) -> None:
    """Download vendor Python source (diffsynth + WanVSR) if not present."""
    marker = vendor_root / "diffsynth" / "__init__.py"
    if marker.exists():
        return

    print("Downloading vendor source code from HuggingFace...")
    archive_path = _hf_download("flashvsr/vendor_code.tar.gz", token)
    vendor_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        _safe_extract_tar(tar, vendor_root)


def ensure_vendor_assets(vendor_root: Path, token: Optional[str]) -> None:
    required = [
        vendor_root / "diffsynth/tokenizer_configs/hunyuan_video/tokenizer_2/tokenizer.json",
        vendor_root / "WanVSR/prompt_tensor/posi_prompt.pth",
    ]
    if all(p.exists() for p in required):
        return

    archive_path = _hf_download("flashvsr/vendor_assets.tar.gz", token)
    vendor_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        _safe_extract_tar(tar, vendor_root)


def validate_weights(model_path: Path, mode: str):
    required = [
        "diffusion_pytorch_model_streaming_dmd.safetensors",
        "LQ_proj_in.ckpt",
        "TCDecoder.ckpt",
    ]
    if mode == "full":
        required.append("Wan2.1_VAE.pth")

    missing = [f for f in required if not (model_path / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing weight files in {model_path}: {', '.join(missing)}"
        )


def init_pipeline(model_path: Path, mode: str, dtype: torch.dtype):
    from diffsynth import ModelManager, FlashVSRTinyPipeline, FlashVSRFullPipeline
    from utils.utils import Causal_LQ4x_Proj
    from utils.TCDecoder import build_tcdecoder

    mm = ModelManager(torch_dtype=dtype, device="cpu")

    if mode == "full":
        mm.load_models([
            str(model_path / "diffusion_pytorch_model_streaming_dmd.safetensors"),
            str(model_path / "Wan2.1_VAE.pth"),
        ])
        pipe = FlashVSRFullPipeline.from_model_manager(mm, device="cuda")
    else:
        mm.load_models([
            str(model_path / "diffusion_pytorch_model_streaming_dmd.safetensors"),
        ])
        pipe = FlashVSRTinyPipeline.from_model_manager(mm, device="cuda")

    pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(
        in_dim=3, out_dim=1536, layer_num=1
    ).to("cuda", dtype=dtype)

    lq_proj_in_path = model_path / "LQ_proj_in.ckpt"
    if lq_proj_in_path.exists():
        pipe.denoising_model().LQ_proj_in.load_state_dict(
            torch.load(lq_proj_in_path, map_location="cpu"), strict=True
        )

    if mode == "tiny":
        multi_scale_channels = [512, 256, 128, 128]
        pipe.TCDecoder = build_tcdecoder(
            new_channels=multi_scale_channels, new_latent_channels=16 + 768
        )
        pipe.TCDecoder.load_state_dict(
            torch.load(model_path / "TCDecoder.ckpt", map_location="cpu"),
            strict=False,
        )
    else:
        pipe.vae.model.encoder = None
        pipe.vae.model.conv1 = None

    pipe.to("cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv()
    pipe.load_models_to_device(["dit", "vae"])
    return pipe


def parse_args():
    parser = argparse.ArgumentParser(description="FlashVSR v1.1 video super-resolution")
    parser.add_argument(
        "--input",
        default="sample",
        help="Input video file or image folder (default: sample)",
    )
    parser.add_argument("--output", default="outputs/flashvsr_out.mp4", help="Output video path")
    parser.add_argument("--mode", choices=["tiny", "full"], default="tiny", help="Pipeline mode")
    parser.add_argument("--scale", type=float, default=4.0, help="Upscale factor (recommended: 4.0)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--sparse_ratio", type=float, default=2.0, help="Sparse ratio (1.5 or 2.0 recommended)")
    parser.add_argument("--local_range", type=int, default=11, help="Local attention range (9 or 11 recommended)")
    parser.add_argument("--weights_dir", default=None, help="Path to FlashVSR-v1.1 weights (optional)")
    parser.add_argument("--hf_token", default=None, help="Hugging Face token (optional)")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup and exit")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (shows compilation, warnings, etc.)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        enable_verbose_logging()

    device, dtype = detect_device()
    if device != "cuda":
        raise RuntimeError("FlashVSR requires CUDA GPU support.")

    check_block_sparse_attn()

    script_dir = Path(__file__).resolve().parent
    vendor_root = _resolve_vendor_root(script_dir)

    input_base = get_input_dir()
    output_base = get_output_dir()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    ensure_vendor_code(vendor_root, hf_token)
    add_vendor_paths(vendor_root)
    ensure_vendor_assets(vendor_root, hf_token)

    sample_path = maybe_download_sample_input(args.input, hf_token)
    if sample_path is not None:
        input_path = str(sample_path)
    else:
        input_path = resolve_input_path(args.input, input_base)

    output_path = resolve_output_path(args.output, output_base)

    model_path = download_weights(args.weights_dir, "JunhaoZhuang/FlashVSR-v1.1", hf_token)
    validate_weights(model_path, args.mode)

    if args.dry_run:
        print("Dry run successful. Environment and weights are ready.")
        return

    torch.manual_seed(args.seed)

    pipe = init_pipeline(model_path, args.mode, dtype)

    lq, th, tw, frames_n, fps = prepare_input_tensor(
        input_path, scale=args.scale, dtype=dtype, device="cuda"
    )

    with torch.no_grad():
        video = pipe(
            prompt="",
            negative_prompt="",
            cfg_scale=1.0,
            num_inference_steps=1,
            seed=args.seed,
            LQ_video=lq,
            num_frames=frames_n,
            height=th,
            width=tw,
            is_full_block=False,
            if_buffer=True,
            topk_ratio=args.sparse_ratio * 768 * 1280 / (th * tw),
            kv_ratio=3.0,
            local_range=args.local_range,
            color_fix=True,
        )

    video = tensor2video(video)
    save_video(video, output_path, fps=fps, quality=6)
    print(f"Saved output to: {output_path}")


if __name__ == "__main__":
    main()
