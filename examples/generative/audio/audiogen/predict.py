#!/usr/bin/env python3
"""
AudioGen text-to-sound generation with AudioCraft.

Code: MIT license via AudioCraft
Model weights: CC-BY-NC 4.0
"""
import os
import sys
import logging
import warnings

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.ERROR)
for logger_name in ["audiocraft", "transformers", "torch", "xformers"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

import argparse
import shutil
from pathlib import Path
from typing import Optional

import torch
import soundfile as sf

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


DEFAULT_MODEL = "facebook/audiogen-medium"
DEFAULT_PROMPT = "footsteps on gravel followed by birds chirping in a quiet park"
HF_DATA_REPO = "zzsi/cvl"
HF_DATA_PREFIX = "audiogen"
SAMPLE_FILES = {
    "sirens_and_a_humming_engine_approach_and_pass.mp3": "sample_data/sirens_and_a_humming_engine_approach_and_pass.mp3",
}


def configure_verbose_logging() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    for logger_name in ["audiocraft", "transformers", "torch", "xformers"]:
        logging.getLogger(logger_name).setLevel(logging.INFO)


def detect_device(requested_device: Optional[str] = None) -> str:
    if requested_device and requested_device != "auto":
        return requested_device
    if torch.cuda.is_available():
        print("Using CUDA device")
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Using MPS device")
        return "mps"
    print("Using CPU. AudioGen inference will be slow.")
    return "cpu"


def load_text(text: Optional[str], input_path: Optional[str]) -> str:
    if text:
        return text.strip()
    if input_path:
        with open(input_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return DEFAULT_PROMPT


def ensure_sample_data(cache_root: Optional[Path] = None) -> Path:
    """Download small sample assets from zzsi/cvl if they are not cached."""
    if cache_root is None:
        cache_root = (
            Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
            / "cvl_data"
            / HF_DATA_PREFIX
        )

    from huggingface_hub import hf_hub_download

    cache_root.mkdir(parents=True, exist_ok=True)
    for local_name, repo_path in SAMPLE_FILES.items():
        target = cache_root / local_name
        if target.exists():
            continue
        downloaded = hf_hub_download(
            repo_id=HF_DATA_REPO,
            filename=f"{HF_DATA_PREFIX}/{repo_path}",
            repo_type="dataset",
        )
        shutil.copy2(downloaded, target)
    return cache_root


def load_model(model_id: str, device: str):
    from audiocraft.models import AudioGen

    print(f"Loading AudioGen model: {model_id}")
    model = AudioGen.get_pretrained(model_id, device=device)
    print(f"Model loaded on {model.device}")
    return model


def set_generation_params(
    model,
    duration: float,
    temperature: float,
    top_k: int,
    top_p: float,
    cfg_coef: float,
) -> None:
    use_sampling = temperature > 0
    model.set_generation_params(
        duration=duration,
        use_sampling=use_sampling,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        cfg_coef=cfg_coef,
    )


def load_audio(path: str, duration: Optional[float] = None):
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio.T)
    if duration is not None and duration > 0:
        waveform = waveform[..., : int(duration * sample_rate)]
    return waveform, sample_rate


def generate_sound(
    model,
    prompt: str,
    prompt_audio_path: Optional[str],
    prompt_duration: Optional[float],
    progress: bool,
):
    if prompt_audio_path:
        print(f"Loading audio prompt: {prompt_audio_path}")
        prompt_waveform, prompt_sample_rate = load_audio(prompt_audio_path, prompt_duration)
        print("Generating sound continuation from text and audio prompt...")
        return model.generate_continuation(
            prompt_waveform,
            prompt_sample_rate=prompt_sample_rate,
            descriptions=[prompt],
            progress=progress,
        )

    print("Generating sound from text...")
    return model.generate([prompt], progress=progress)


def save_audio(wav: torch.Tensor, sample_rate: int, output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    audio = wav.detach().cpu()
    if audio.dim() == 3:
        audio = audio[0]
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    audio_np = audio.numpy().T
    sf.write(str(output_file), audio_np, sample_rate)
    print(f"Saved audio to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate sound effects from text with AudioCraft AudioGen.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--text", type=str, default=None, help="Prompt text")
    parser.add_argument("--input", type=str, default=None, help="Path to a text prompt file")
    parser.add_argument("--output", type=str, default="audiogen.wav", help="Output WAV path")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="AudioGen model id")
    parser.add_argument("--duration", type=float, default=10.0, help="Generated audio duration in seconds")
    parser.add_argument(
        "--prompt-audio",
        type=str,
        default=None,
        help="Optional audio prompt for continuation. Use 'sample' for the bundled sirens sample.",
    )
    parser.add_argument("--prompt-duration", type=float, default=2.0, help="Seconds to use from the audio prompt")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature. Use 0 for greedy decoding")
    parser.add_argument("--top-k", type=int, default=250, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.0, help="Top-p sampling. 0 disables top-p")
    parser.add_argument("--cfg-coef", type=float, default=3.0, help="Classifier-free guidance coefficient")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"], help="Inference device")
    parser.add_argument("--no-progress", action="store_true", help="Disable generation progress output")
    parser.add_argument("--prepare-sample-data", action="store_true", help="Download bundled sample inputs and exit")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verbose:
        configure_verbose_logging()

    input_dir = get_input_dir()
    output_dir = get_output_dir()

    input_path = resolve_input_path(args.input, input_dir) if args.input else None
    if args.prepare_sample_data:
        sample_dir = ensure_sample_data()
        print(f"Sample data ready in {sample_dir}")
        return 0

    if args.prompt_audio == "sample":
        prompt_audio_path = str(ensure_sample_data() / "sirens_and_a_humming_engine_approach_and_pass.mp3")
    else:
        prompt_audio_path = resolve_input_path(args.prompt_audio, input_dir) if args.prompt_audio else None
    output_path = resolve_output_path(args.output, output_dir)

    prompt = load_text(args.text, input_path)
    if not prompt:
        print("Prompt is empty.", file=sys.stderr)
        return 2

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = detect_device(args.device)
    model = load_model(args.model, device)
    set_generation_params(
        model=model,
        duration=args.duration,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        cfg_coef=args.cfg_coef,
    )

    wav = generate_sound(
        model,
        prompt,
        prompt_audio_path=prompt_audio_path,
        prompt_duration=args.prompt_duration,
        progress=not args.no_progress,
    )
    save_audio(wav, model.sample_rate, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
