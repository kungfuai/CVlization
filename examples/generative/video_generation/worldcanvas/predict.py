#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import torchvision
from PIL import Image

VENDOR_ROOT = Path(__file__).resolve().parent / "vendor" / "WorldCanvas"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

from diffsynth import save_video
from diffsynth.pipelines.WorldCanvas import ModelConfig, WorldCanvasPipeline

COLORS = np.array(
    [
        (230, 25, 75),
        (67, 99, 216),
        (56, 195, 56),
        (255, 225, 25),
        (145, 30, 180),
        (70, 240, 240),
        (245, 130, 49),
    ],
    dtype=np.uint8,
)
COLOR_NAMES = ["red", "blue", "green", "yellow", "purple", "cyan", "orange"]
SPECIAL_NAMES = ["+++", "@@@", "~~~", "$$$", "^^^", "&&&", "---"]


def create_circle_image(radius, color, size=None):
    if size is None:
        size = int(2 * radius) + 1
    image = np.zeros((size, size, 3), dtype=np.uint8)
    center = size // 2
    y, x = np.ogrid[:size, :size]
    dist_squared = (x - center) ** 2 + (y - center) ** 2
    mask = dist_squared <= radius**2
    for i in range(3):
        image[:, :, i][mask] = color[i]
    return image


def crop_and_resize(image, target_height, target_width):
    width, height = image.size
    scale = max(target_width / width, target_height / height)
    image = torchvision.transforms.functional.resize(
        image,
        (round(height * scale), round(width * scale)),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
    )
    image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
    return image


def load_first_frame(file_path, height, width):
    suffix = Path(file_path).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        frame = Image.open(file_path).convert("RGB")
    else:
        reader = imageio.get_reader(file_path)
        frame = Image.fromarray(reader.get_data(0)).convert("RGB")
        reader.close()
    frame = crop_and_resize(frame, height, width)
    return frame


def load_json(file_path, height, width, if_color, if_special_corr):
    with open(file_path, "r", encoding="utf-8") as handle:
        content = json.load(handle)

    data = {}
    if "_crop.json" in file_path:
        if "id_caption_map" not in content[-2].keys():
            return None
        if len(content[-2]["id_caption_map"]) == 0:
            return None
    else:
        if "id_caption_map" not in content[-1].keys():
            return None
        if len(content[-1]["id_caption_map"]) == 0:
            return None

    if "_crop.json" in file_path:
        data["crop"] = content[-1]
        content = content[:-1]

    id_caption_map = content[-1]["id_caption_map"]
    text_prompt = ""
    id_caption_order = {}
    color_idx = 0

    appears_ids = []

    data["tracking_points"] = []
    data["vis"] = []
    data["rs"] = []
    data["ids"] = []
    data["point_masks"] = []

    for tp in content:
        if "tracking" in list(tp.keys()):
            data["tracking_points"].append(tp["tracking"])
            if (
                (tp["tracking"][0][0] < 0 or tp["tracking"][0][0] >= width)
                or (tp["tracking"][0][1] < 0 or tp["tracking"][0][1] >= height)
            ):
                appears_ids.append(str(tp["id"]))
            data["vis"].append(tp["tracking_vis_value"])
            data["rs"].append(tp["r"])
            data["ids"].append(tp["id"])
            data["point_masks"].append(tp["mask_cluster"])

    if if_color == 1:
        id_color_map = {}
        color_n = list(range(7))
        np.random.shuffle(color_n)

        for tid, ca in id_caption_map.items():
            id_color_map[tid] = COLORS[color_n[color_idx]]
            text_prompt += COLOR_NAMES[color_n[color_idx]]
            if tid in appears_ids:
                text_prompt += " mask appears: "
            else:
                text_prompt += " mask: "
            text_prompt += re.sub(r"\s+", " ", ca).strip()
            text_prompt += os.linesep
            id_caption_order[tid] = color_idx
            color_idx += 1

        data["id_color_map"] = id_color_map
    elif if_special_corr == 1:
        id_special_map = {}
        for tid, ca in id_caption_map.items():
            id_special_map[tid] = color_idx + 1
            if tid not in appears_ids:
                text_prompt += "Object "
                text_prompt += SPECIAL_NAMES[color_idx]
                text_prompt += " : "
            else:
                text_prompt += "Object "
                text_prompt += SPECIAL_NAMES[color_idx]
                text_prompt += " appears: "
            text_prompt += re.sub(r"\s+", " ", ca).strip()
            text_prompt += os.linesep
            id_caption_order[tid] = color_idx
            color_idx += 1
        data["id_special_map"] = id_special_map
    else:
        for tid, ca in id_caption_map.items():
            if tid not in appears_ids:
                text_prompt += f"Object {color_idx + 1}: "
            else:
                text_prompt += f"Object {color_idx + 1} appears: "
            text_prompt += re.sub(r"\s+", " ", ca).strip()
            text_prompt += os.linesep
            id_caption_order[tid] = color_idx
            color_idx += 1

    data["text_prompt"] = text_prompt[:-1]
    data["id_caption_order"] = id_caption_order

    return data


def ensure_models():
    from huggingface_hub import hf_hub_download, snapshot_download

    hf_token = os.environ.get("HF_TOKEN")
    t5_repo = os.environ.get("WORLDCANVAS_T5_VAE_REPO", "Wan-AI/Wan2.2-I2V-A14B")
    dit_repo = os.environ.get("WORLDCANVAS_DIT_REPO", "hlwang06/WorldCanvas")
    tokenizer_dataset_repo = os.environ.get("WORLDCANVAS_TOKENIZER_DATASET_REPO", "zzsi/cvl")
    tokenizer_dataset_path = os.environ.get(
        "WORLDCANVAS_TOKENIZER_DATASET_PATH",
        "worldcanvas/tokenizer_configs/hunyuan_video/tokenizer_2",
    )

    t5_path = hf_hub_download(
        repo_id=t5_repo,
        filename="models_t5_umt5-xxl-enc-bf16.pth",
        token=hf_token,
    )
    vae_path = hf_hub_download(
        repo_id=t5_repo,
        filename="Wan2.1_VAE.pth",
        token=hf_token,
    )

    tokenizer_files = [
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    tokenizer_root = None
    for filename in tokenizer_files:
        downloaded = hf_hub_download(
            repo_id=tokenizer_dataset_repo,
            repo_type="dataset",
            filename=f"{tokenizer_dataset_path}/{filename}",
            token=hf_token,
        )
        if tokenizer_root is None:
            tokenizer_root = Path(downloaded).parent

    if tokenizer_root is None or not tokenizer_root.exists():
        raise FileNotFoundError("Tokenizer files were not downloaded correctly.")

    high_path = hf_hub_download(
        repo_id=dit_repo,
        filename="WorldCanvas_dit/WorldCanvas/high_model.safetensors",
        token=hf_token,
    )
    low_path = hf_hub_download(
        repo_id=dit_repo,
        filename="WorldCanvas_dit/WorldCanvas/low_model.safetensors",
        token=hf_token,
    )

    return {
        "t5": t5_path,
        "vae": vae_path,
        "tokenizer": str(tokenizer_root),
        "dit_high": high_path,
        "dit_low": low_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="WorldCanvas inference")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--image", type=str, default=None, help="Path to input image/video")
    parser.add_argument("--control", type=str, default=None, help="Path to trajectory JSON")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--height", type=int, default=480, help="Output height")
    parser.add_argument("--width", type=int, default=832, help="Output width")
    parser.add_argument("--num-frames", type=int, default=81, help="Number of frames")
    parser.add_argument("--steps", type=int, default=50, help="Denoising steps")
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=(
            "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
            "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
            "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        ),
        help="Negative prompt",
    )
    return parser.parse_args()


def resolve_sample_paths(args):
    sample_image_name = "door.png"
    sample_control_name = "door.json"

    image_path = Path(args.image) if args.image else None
    control_path = Path(args.control) if args.control else None

    if image_path is None or control_path is None:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError

        token = os.environ.get("HF_TOKEN")
        try:
            if image_path is None:
                image_path = Path(
                    hf_hub_download(
                        repo_id="zzsi/cvl",
                        repo_type="dataset",
                        filename=f"worldcanvas/{sample_image_name}",
                        token=token,
                    )
                )
            if control_path is None:
                control_path = Path(
                    hf_hub_download(
                        repo_id="zzsi/cvl",
                        repo_type="dataset",
                        filename=f"worldcanvas/{sample_control_name}",
                        token=token,
                    )
                )
        except EntryNotFoundError:
            local_dir = VENDOR_ROOT / "examples"
            image_path = image_path or local_dir / sample_image_name
            control_path = control_path or local_dir / sample_control_name
            print(
                "HF dataset assets not found; falling back to vendored samples "
                f"under {local_dir}"
            )

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not control_path.exists():
        raise FileNotFoundError(f"Control JSON not found: {control_path}")

    return image_path, control_path


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path, control_path = resolve_sample_paths(args)

    models = ensure_models()

    pipe = WorldCanvasPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        # Use an indexed CUDA device to satisfy mem_get_info() in diffsynth.
        device="cuda:0",
        model_configs=[
            ModelConfig(path=models["t5"], offload_device="cpu"),
            ModelConfig(path=models["dit_high"], offload_device="cpu"),
            ModelConfig(path=models["dit_low"], offload_device="cpu"),
            ModelConfig(path=models["vae"], offload_device="cpu"),
        ],
        tokenizer_config=ModelConfig(path=models["tokenizer"]),
        redirect_common_files=False,
    )
    pipe.enable_vram_management()
    pipe.to("cuda")

    if_color = 0
    if_mask = 1
    if_vr = 0
    if_special_corr = 0
    controls = ["gaussian_channel", "vae_channel"]
    vae_channel = "point"

    sample_json = load_json(
        str(control_path),
        height=args.height,
        width=args.width,
        if_color=if_color,
        if_special_corr=if_special_corr,
    )
    if sample_json is None:
        raise RuntimeError("Control JSON has no captions or tracking data.")
    current_prompt = sample_json["text_prompt"]

    input_image = load_first_frame(str(image_path), args.height, args.width)

    video = pipe(
        prompt=current_prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        json_data=sample_json,
        tiled=True,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        input_image=input_image,
        switch_DiT_boundary=0.9,
        num_inference_steps=args.steps,
        if_color=if_color,
        if_mask=if_mask,
        if_special_corr=if_special_corr,
        if_vr=if_vr,
        controls=controls,
        vae_channel=vae_channel,
    )

    video_array = np.stack(video, axis=0)
    video_array_no_point = video_array.copy()

    for i, tp in enumerate(sample_json["tracking_points"]):
        radius = 10
        if if_color == 1:
            if str(sample_json["ids"][i]) in sample_json["id_color_map"].keys():
                color_traj_area = create_circle_image(
                    radius, sample_json["id_color_map"][str(sample_json["ids"][i])]
                )
            else:
                color_traj_area = create_circle_image(radius, np.array((255, 255, 255), dtype=np.uint8))
        else:
            color_traj_area = create_circle_image(radius, np.array((255, 255, 255), dtype=np.uint8))

        frame_count = min(args.num_frames, len(tp))
        for j in range(frame_count):
            w, h = tp[j]
            if sample_json["vis"][i][j] > 0.5:
                x1 = int(max(w - radius, 0))
                x2 = int(min(w + radius, args.width - 1))
                y1 = int(max(h - radius, 0))
                y2 = int(min(h + radius, args.height - 1))
                if (x2 - x1) < 1 or (y2 - y1) < 1:
                    continue
                need_map = cv2.resize(color_traj_area, (x2 - x1 + 1, y2 - y1 + 1))
                video_array[j, y1 : y2 + 1, x1 : x2 + 1, :] = torch.tensor(
                    need_map, dtype=torch.float32
                )

    pil_frames = [Image.fromarray(frame) for frame in video_array]
    pil_frames_no_point = [Image.fromarray(frame) for frame in video_array_no_point]

    stem = image_path.stem
    overlay_path = output_dir / f"{stem}_seed{args.seed}_overlay.mp4"
    clean_path = output_dir / f"{stem}_seed{args.seed}.mp4"

    save_video(pil_frames, str(overlay_path), fps=16, quality=5)
    save_video(pil_frames_no_point, str(clean_path), fps=16, quality=5)

    print(f"Saved overlay video: {overlay_path}")
    print(f"Saved clean video: {clean_path}")


if __name__ == "__main__":
    main()
