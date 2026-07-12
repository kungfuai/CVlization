#!/usr/bin/env python3
"""
OmniVoice: Massively multilingual zero-shot TTS (646 languages).

k2-fsa's non-autoregressive diffusion language model for text-to-speech
with zero-shot voice cloning and voice design capabilities.

License: Apache-2.0 (code); CC-BY-NC (pretrained weights)
Model: https://huggingface.co/k2-fsa/OmniVoice
"""
import os
import sys
import logging
import warnings

# Suppress verbose logging by default
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.ERROR)
for logger_name in ["transformers", "torch", "accelerate"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

import argparse
import shutil
from pathlib import Path

import soundfile as sf
import torch

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


MODEL_ID = "k2-fsa/OmniVoice"
SAMPLE_RATE = 24000

HF_DATA_REPO = "zzsi/cvl"
HF_DATA_PREFIX = "omnivoice"


def ensure_sample_data(cache_root=None):
    """Download canonical reference audio from HuggingFace if not cached."""
    from huggingface_hub import hf_hub_download

    if cache_root is None:
        hf_home = Path(
            os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
        )
        cache_root = hf_home / "cvl_data" / "omnivoice"

    cache_root = Path(cache_root)
    ref_wav = cache_root / "ref_speech.wav"

    if ref_wav.exists():
        print(f"Sample data already cached at {cache_root}")
        return str(cache_root)

    print(f"Downloading sample data from {HF_DATA_REPO}...")
    cache_root.mkdir(parents=True, exist_ok=True)

    files_to_download = ["ref_speech.wav"]
    for rel_path in files_to_download:
        downloaded = hf_hub_download(
            repo_id=HF_DATA_REPO,
            filename=f"{HF_DATA_PREFIX}/{rel_path}",
            repo_type="dataset",
        )
        local_target = cache_root / rel_path
        local_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, local_target)
        print(f"  Cached: {rel_path}")

    print(f"Sample data cached at {cache_root}")
    return str(cache_root)


def detect_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Using CUDA device: {gpu_name} ({vram:.1f} GB)")
    else:
        device = "cpu"
        print("Using CPU (inference will be slow)")
    return device


def load_model(model_id: str, device: str, dtype=None):
    """Load OmniVoice model."""
    from omnivoice import OmniVoice

    if dtype is None:
        dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading {model_id} on {device} ({dtype})...")
    model = OmniVoice.from_pretrained(
        model_id,
        device_map=f"{device}:0" if device == "cuda" else device,
        dtype=dtype,
    )
    print("Model loaded successfully")
    return model


def load_text(text_path: str) -> str:
    """Load text from file."""
    with open(text_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def run_voice_cloning(model, text, ref_audio, ref_text=None, num_step=32, speed=1.0):
    """Generate speech cloning a reference voice."""
    print(f"Voice cloning: {len(text)} chars, ref={ref_audio}")
    audio = model.generate(
        text=text,
        ref_audio=ref_audio,
        ref_text=ref_text,
        num_step=num_step,
        speed=speed,
    )
    return audio[0]


def run_voice_design(model, text, instruct, num_step=32, speed=1.0):
    """Generate speech with designed voice attributes."""
    print(f"Voice design: {len(text)} chars, instruct='{instruct}'")
    audio = model.generate(
        text=text,
        instruct=instruct,
        num_step=num_step,
        speed=speed,
    )
    return audio[0]


def save_audio(wav, sample_rate: int, output_path: str):
    """Save generated audio to file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, wav, sample_rate)
    duration = len(wav) / sample_rate
    size_kb = Path(output_path).stat().st_size / 1024
    print(f"Audio saved to {output_path} ({duration:.1f}s, {size_kb:.0f} KB)")


def main():
    parser = argparse.ArgumentParser(
        description="OmniVoice: Multilingual zero-shot TTS with voice cloning and voice design",
        epilog=(
            "Examples:\n"
            "  # Voice design (no reference audio):\n"
            "  python predict.py --text 'Hello world' --instruct 'female, British accent'\n"
            "\n"
            "  # Voice cloning (with reference audio):\n"
            "  python predict.py --text 'Hello world' --ref-audio ref.wav --ref-text 'transcript'\n"
            "\n"
            "  # From text file:\n"
            "  python predict.py --input sample.txt --instruct 'male, low pitch'\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--text", type=str, default=None, help="Text to synthesize (direct input)"
    )
    parser.add_argument(
        "--input", type=str, default=None, help="Path to text file to synthesize"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output WAV file path (default: speech.wav)",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Reference audio for voice cloning (3-10s WAV recommended)",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Transcript of reference audio (optional, improves quality)",
    )
    parser.add_argument(
        "--instruct",
        type=str,
        default=None,
        help="Voice design attributes (e.g. 'female, low pitch, British accent')",
    )
    parser.add_argument(
        "--num-step",
        type=int,
        default=32,
        help="Diffusion steps: higher=better quality (default: 32, fast: 16)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed factor (default: 1.0)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, force=True)
        for logger_name in ["transformers", "torch", "accelerate", "omnivoice"]:
            logging.getLogger(logger_name).setLevel(logging.INFO)

    # Resolve paths
    INP = get_input_dir()
    OUT = get_output_dir()

    if args.output is None:
        args.output = "speech.wav"
    output_path = Path(resolve_output_path(args.output, OUT))

    # Determine text
    if args.text is not None:
        text = args.text
    elif args.input is not None:
        input_path = resolve_input_path(args.input, INP)
        if not Path(input_path).exists():
            print(f"Error: Input file '{input_path}' not found")
            return 1
        text = load_text(input_path)
    else:
        text = "Hello! This is OmniVoice, a multilingual text to speech model supporting over six hundred languages."

    # Resolve reference audio path
    ref_audio_path = None
    if args.ref_audio:
        ref_audio_path = resolve_input_path(args.ref_audio, INP)
        if not Path(ref_audio_path).exists():
            print(f"Error: Reference audio '{ref_audio_path}' not found")
            return 1

    # Determine mode
    if ref_audio_path:
        mode = "voice_cloning"
    elif args.instruct:
        mode = "voice_design"
    else:
        # Default: use canonical reference audio for voice cloning demo
        print("No --ref-audio or --instruct provided; using canonical reference audio.")
        sample_dir = ensure_sample_data()
        ref_audio_path = os.path.join(sample_dir, "ref_speech.wav")
        mode = "voice_cloning"

    # Detect device and load model
    device = detect_device()
    try:
        model = load_model(MODEL_ID, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Run inference
    try:
        if mode == "voice_cloning":
            wav = run_voice_cloning(
                model,
                text,
                ref_audio=ref_audio_path,
                ref_text=args.ref_text,
                num_step=args.num_step,
                speed=args.speed,
            )
        else:
            wav = run_voice_design(
                model,
                text,
                instruct=args.instruct,
                num_step=args.num_step,
                speed=args.speed,
            )
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Save output
    try:
        save_audio(wav, SAMPLE_RATE, str(output_path))
    except Exception as e:
        print(f"Error saving audio: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Summary
    print(f"\nSummary:")
    print(f"  Mode: {mode}")
    print(f"  Text: {text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print(f"  Output: {output_path}")
    if ref_audio_path:
        print(f"  Reference audio: {ref_audio_path}")
    if args.instruct:
        print(f"  Voice design: {args.instruct}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
