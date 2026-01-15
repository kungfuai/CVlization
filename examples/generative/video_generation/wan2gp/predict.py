#!/usr/bin/env python3
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image
import soundfile as sf
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm
from inspect import signature

from cvlization.paths import resolve_input_path, resolve_output_path

VENDOR_ROOT = Path(__file__).parent / "vendor" / "wan2gp"
sys.path.insert(0, str(VENDOR_ROOT))

from mmgp import offload

from shared.utils import files_locator as fl
from shared.utils.audio_video import save_video, combine_video_with_audio_tracks
from shared.utils.utils import convert_image_to_tensor
from shared.utils.loras_mutipliers import parse_loras_multipliers


MODEL_DEFAULTS = {
    "ltx2_19B": "ltx2_19B.json",
    "ltx2_distilled": "ltx2_distilled.json",
    "wan_t2v": "t2v.json",
    "wan_t2v_1.3B": "t2v_1.3B.json",
    "wan_i2v": "i2v.json",
    "longcat_video": "longcat_video.json",
    "longcat_avatar": "longcat_avatar.json",
}


def load_defaults(model_key: str):
    defaults_path = VENDOR_ROOT / "defaults" / MODEL_DEFAULTS[model_key]
    with open(defaults_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    model_def = data["model"]
    ui_defaults = {k: v for k, v in data.items() if k != "model"}
    return model_def, ui_defaults


def select_model_url(model_def: dict, quantization: str) -> str:
    urls = model_def.get("URLs") or []
    if isinstance(urls, str):
        urls = [urls]
    if not urls:
        raise ValueError("No model URLs found in defaults.")
    if quantization == "fp8":
        for url in urls:
            if "fp8" in url.lower():
                return url
    if quantization == "int8":
        for url in urls:
            if "int8" in url.lower() or "quanto" in url.lower():
                return url
    return urls[0]


def compute_list(entry):
    if not entry:
        return []
    if isinstance(entry, (list, tuple)):
        return [os.path.basename(str(item)) for item in entry if item]
    return [os.path.basename(str(entry))]


def download_model_files(download_def, checkpoint_dir: Path):
    hf_supports_tqdm = "tqdm_class" in signature(hf_hub_download).parameters
    for entry in download_def:
        repo_id = entry["repoId"]
        source_folders = entry.get("sourceFolderList", [""])
        file_lists = entry.get("fileList", [])
        for source_folder, files in zip(source_folders, file_lists):
            files_iter = files
            if not hf_supports_tqdm:
                files_iter = tqdm(files, desc=f"Downloading {repo_id}", unit="file")
            for filename in files_iter:
                if not filename:
                    continue
                local_path = checkpoint_dir / source_folder / filename if source_folder else checkpoint_dir / filename
                if local_path.exists():
                    continue
                local_path.parent.mkdir(parents=True, exist_ok=True)
                hf_kwargs = {}
                if hf_supports_tqdm:
                    hf_kwargs["tqdm_class"] = tqdm
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(checkpoint_dir),
                    subfolder=source_folder or None,
                    **hf_kwargs,
                )


def resolve_handler(base_model_type: str):
    if base_model_type in {"ltx2_19B"}:
        from models.ltx2 import ltx2_handler

        return ltx2_handler.family_handler
    if base_model_type in {"longcat_video", "longcat_avatar"}:
        from models.longcat import longcat_handler

        return longcat_handler.family_handler
    if base_model_type in {"t2v", "i2v"}:
        from models.wan import wan_handler

        return wan_handler.family_handler
    raise ValueError(f"Unsupported base model type: {base_model_type}")


def download_lora_files(model_def_raw: dict, checkpoint_dir: Path):
    """Download LoRA files specified in model defaults."""
    lora_urls = model_def_raw.get("loras", [])
    if not lora_urls:
        return []

    lora_paths = []
    for url in lora_urls:
        filename = os.path.basename(url)
        local_path = checkpoint_dir / filename
        if local_path.exists():
            lora_paths.append(str(local_path))
            continue

        # Parse HuggingFace URL
        if "huggingface.co" in url:
            parts = url.replace("https://huggingface.co/", "").split("/resolve/")
            if len(parts) == 2:
                repo_id = parts[0]
                filename = parts[1].split("/", 1)[-1] if "/" in parts[1] else parts[1]
                from huggingface_hub import hf_hub_download

                path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(checkpoint_dir))
                lora_paths.append(path)
    return lora_paths


def prepare_model(model_key: str, args):
    model_def_raw, ui_defaults = load_defaults(model_key)
    base_model_type = model_def_raw["architecture"]
    handler = resolve_handler(base_model_type)
    default_model_def = handler.query_model_def(base_model_type, model_def_raw)
    model_def = default_model_def or {}
    model_def.update(model_def_raw)

    model_url = select_model_url(model_def_raw, args.quantization)
    model_filename = os.path.basename(model_url)

    text_encoder_quantization = "int8" if args.quantization == "int8" else None
    download_def = handler.query_model_files(
        compute_list,
        base_model_type,
        model_filename,
        text_encoder_quantization,
    )
    download_model_files(download_def, Path(args.checkpoint_dir))

    # Download LoRA files (e.g., distilled LoRA for LTX2 two-stage)
    lora_paths = download_lora_files(model_def_raw, Path(args.checkpoint_dir))

    model_path = fl.locate_file(model_filename)
    model, pipe = handler.load_model(
        model_filename=model_path,
        model_type=model_key,
        base_model_type=base_model_type,
        model_def=model_def,
        quantizeTransformer=args.quantization == "int8",
        text_encoder_quantization=text_encoder_quantization,
        dtype=torch.bfloat16,
        VAE_dtype=torch.float32,
    )

    # Configure mmgp offloading
    offload.shared_state["_attention"] = args.attention

    # Handle wrapped pipe structure for non-distilled models (matching wgp.py init_pipe logic)
    # The wrapped pipe has structure: {"pipe": inner_pipe, "loras": [...]}
    if isinstance(pipe, dict) and "pipe" in pipe:
        pipe_for_offload = pipe["pipe"]
        # Extract loras list from wrapped pipe (e.g., ["text_embedding_projection", "text_embeddings_connector"])
        loras_models = list(pipe.get("loras", []))
    else:
        pipe_for_offload = pipe
        loras_models = []

    # Add transformer(s) to loras list (matching wgp.py logic)
    if "transformer" in pipe_for_offload:
        loras_models.append("transformer")
    if "transformer2" in pipe_for_offload:
        loras_models.append("transformer2")

    offload.profile(
        pipe_for_offload,
        profile_no=args.mmgp_profile,
        quantizeTransformer=(args.quantization == "int8"),
        loras=loras_models if loras_models else None,
    )

    # Load LoRAs via mmgp if available (e.g., distilled LoRA for LTX2)
    loras_slists = None
    if lora_paths and hasattr(model, "get_trans_lora"):
        trans_lora, _ = model.get_trans_lora()
        if trans_lora is not None:
            # Parse LoRA multipliers using original wan2gp logic
            lora_multipliers = model_def_raw.get("loras_multipliers", [])
            num_steps = args.steps or ui_defaults.get("num_inference_steps", 40)
            guidance_phases = model_def.get("guidance_max_phases", 2)

            # Use the actual parse_loras_multipliers from wan2gp
            initial_strengths, loras_slists, errors = parse_loras_multipliers(
                lora_multipliers,
                len(lora_paths),
                num_steps,
                nb_phases=guidance_phases,
            )
            if errors:
                print(f"Warning: LoRA multiplier parsing error: {errors}")

            # Create preprocessor wrapper (matching wgp.py get_loras_preprocessor)
            raw_preprocessor = getattr(trans_lora, "preprocess_loras", None)
            if raw_preprocessor is not None:
                # Wrap to include model_type argument
                def preprocess_sd(sd, preprocessor=raw_preprocessor, mtype=base_model_type):
                    return preprocessor(mtype, sd)
            else:
                preprocess_sd = None

            split_map = getattr(trans_lora, "split_linear_modules_map", None)

            # Load LoRAs into transformer (matching original wgp.py logic)
            offload.load_loras_into_model(
                trans_lora,
                lora_paths,
                initial_strengths,
                activate_all_loras=True,
                preprocess_sd=preprocess_sd,
                split_linear_modules_map=split_map,
            )
            print(f"Loaded {len(lora_paths)} LoRA(s) via mmgp")

    return model, model_def, ui_defaults, base_model_type, loras_slists


def save_with_audio(video_tensor, audio_np, audio_sr, output_path: Path):
    if audio_np is None:
        save_video(video_tensor, str(output_path))
        return
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_video = Path(tmp_dir) / "video.mp4"
        temp_audio = Path(tmp_dir) / "audio.wav"
        save_video(video_tensor, str(temp_video))
        sf.write(temp_audio, audio_np, audio_sr)
        combine_video_with_audio_tracks(
            str(temp_video),
            [str(temp_audio)],
            str(output_path),
        )


def run_ltx2(model, model_def, ui_defaults, args, loras_slists=None):
    prompt = args.prompt or ui_defaults.get("prompt")
    if not prompt:
        raise ValueError("Prompt is required for LTX-2.")
    steps = args.steps or ui_defaults.get("num_inference_steps", 40)
    frames = args.frames or ui_defaults.get("video_length", 121)
    height = args.height or 1024
    width = args.width or 1536
    fps = args.fps or model_def.get("fps", 24)
    output_path = resolve_output_path(args.output)

    audio_guide = resolve_input_path(args.audio) if args.audio else None

    result = model.generate(
        input_prompt=prompt,
        n_prompt=args.negative_prompt or "",
        sampling_steps=steps,
        guide_scale=args.guidance_scale or ui_defaults.get("guidance_scale", 4.0),
        frame_num=frames,
        height=height,
        width=width,
        fps=fps,
        seed=args.seed or 0,
        audio_guide=str(audio_guide) if audio_guide else None,
        audio_scale=args.audio_strength if audio_guide else None,
        loras_slists=loras_slists,
    )
    save_with_audio(result["x"], result.get("audio"), result.get("audio_sampling_rate", 24000), output_path)
    print(f"Saved LTX-2 output to {output_path}")


def run_longcat(model, model_def, ui_defaults, args, is_avatar: bool):
    prompt = args.prompt or ui_defaults.get("prompt")
    if not prompt:
        raise ValueError("Prompt is required for LongCat.")
    steps = args.steps or ui_defaults.get("num_inference_steps", 50)
    frames = args.frames or ui_defaults.get("video_length", 93)
    height = args.height or (480 if not is_avatar else 512)
    width = args.width or (832 if not is_avatar else 512)
    output_path = resolve_output_path(args.output)

    audio_guide = resolve_input_path(args.audio) if args.audio else None

    video = model.generate(
        input_prompt=prompt,
        n_prompt=args.negative_prompt or "",
        sampling_steps=steps,
        frame_num=frames,
        height=height,
        width=width,
        guide_scale=args.guidance_scale or 4.0,
        seed=args.seed or 0,
        audio_guide=str(audio_guide) if audio_guide else None,
    )
    save_video(video, str(output_path), fps=args.fps or (16 if is_avatar else 15))
    print(f"Saved LongCat output to {output_path}")


def run_wan(model, model_def, ui_defaults, args, is_i2v: bool):
    prompt = args.prompt or ui_defaults.get("prompt")
    if not prompt:
        raise ValueError("Prompt is required for Wan.")
    steps = args.steps or ui_defaults.get("num_inference_steps", 50)
    frames = args.frames or ui_defaults.get("video_length", 81)
    height = args.height or 720
    width = args.width or 1280
    output_path = resolve_output_path(args.output)

    input_frames = None
    if is_i2v:
        if not args.image:
            raise ValueError("Wan I2V requires --image.")
        image_path = resolve_input_path(args.image)
        image = Image.open(image_path).convert("RGB")
        image_tensor = convert_image_to_tensor(image)
        input_frames = image_tensor.unsqueeze(0).unsqueeze(0)

    video = model.generate(
        input_prompt=prompt,
        input_frames=input_frames,
        width=width,
        height=height,
        frame_num=frames,
        sampling_steps=steps,
        guide_scale=args.guidance_scale or 5.0,
        n_prompt=args.negative_prompt or "",
        seed=args.seed or -1,
        shift=args.shift,
        sample_solver=args.solver,
        fit_into_canvas=True,
    )
    save_video(video, str(output_path), fps=args.fps or 24)
    print(f"Saved Wan output to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Wan2GP CLI (LTX-2, Wan, LongCat)")
    parser.add_argument("--model", choices=MODEL_DEFAULTS.keys(), default="ltx2_19B")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--negative-prompt", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--audio", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--quantization", choices=["bf16", "int8", "fp8"], default="bf16")
    parser.add_argument("--checkpoint-dir", type=str, default=os.path.expanduser("~"))
    parser.add_argument("--audio-strength", type=float, default=1.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--solver", type=str, default="unipc")
    parser.add_argument(
        "--mmgp-profile",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=4,
        help="MMGP memory profile: 1=HighRAM_HighVRAM, 2=HighRAM_LowVRAM, "
        "3=LowRAM_HighVRAM, 4=LowRAM_LowVRAM (default), 5=VeryLowRAM_LowVRAM",
    )
    parser.add_argument(
        "--attention",
        type=str,
        default="flash",
        choices=["flash", "sdpa", "sage", "sage2", "sage3", "xformers", "auto"],
        help="Attention implementation to use (default: flash)",
    )

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fl.set_checkpoints_paths([str(checkpoint_dir)])

    model, model_def, ui_defaults, base_model_type, loras_slists = prepare_model(args.model, args)

    if base_model_type == "ltx2_19B":
        run_ltx2(model, model_def, ui_defaults, args, loras_slists=loras_slists)
    elif base_model_type == "longcat_video":
        run_longcat(model, model_def, ui_defaults, args, is_avatar=False)
    elif base_model_type == "longcat_avatar":
        run_longcat(model, model_def, ui_defaults, args, is_avatar=True)
    elif base_model_type == "t2v":
        run_wan(model, model_def, ui_defaults, args, is_i2v=False)
    elif base_model_type == "i2v":
        run_wan(model, model_def, ui_defaults, args, is_i2v=True)
    else:
        raise ValueError(f"Unsupported base model type: {base_model_type}")


if __name__ == "__main__":
    main()
