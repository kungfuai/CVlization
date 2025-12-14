#!/usr/bin/env python3
"""
VoxCPM1.5: Tokenizer-free Text-to-Speech with voice cloning.

OpenBMB's end-to-end TTS model with zero-shot voice cloning,
streaming synthesis, and bilingual (Chinese/English) support.

License: Apache-2.0
Model: https://huggingface.co/openbmb/VoxCPM1.5
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


MODEL_ID = "openbmb/VoxCPM1.5"


def detect_device():
    """Auto-detect device."""
    if torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA device")
    else:
        device = "cpu"
        print("Using CPU (inference will be slow)")
    return device


def load_model(model_id: str):
    """Load VoxCPM model."""
    from voxcpm import VoxCPM

    print(f"Loading {model_id}...")
    model = VoxCPM.from_pretrained(model_id)
    print("Model loaded successfully")
    return model


def load_text(text_path: str) -> str:
    """Load text from file."""
    with open(text_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def run_inference(
    model,
    text: str,
    prompt_wav_path: str = None,
    prompt_text: str = None,
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
    normalize: bool = True,
    denoise: bool = False,
    streaming: bool = False,
):
    """Generate speech from text."""
    print(f"Generating speech for {len(text)} characters...")

    if streaming:
        import numpy as np
        chunks = []
        for chunk in model.generate_streaming(
            text=text,
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=normalize,
            denoise=denoise,
        ):
            chunks.append(chunk)
        wav = np.concatenate(chunks)
    else:
        wav = model.generate(
            text=text,
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=normalize,
            denoise=denoise,
            retry_badcase=True,
            retry_badcase_max_times=3,
        )

    return wav, model.tts_model.sample_rate


def save_audio(wav, sample_rate: int, output_path: str):
    """Save generated audio to file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, wav, sample_rate)
    print(f"Audio saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="VoxCPM1.5: Text-to-speech with voice cloning",
        epilog="Examples:\n"
               "  python predict.py --text 'Hello world'\n"
               "  python predict.py --input sample.txt\n"
               "  python predict.py --text 'Hello' --prompt-audio voice.wav --prompt-text 'Reference'",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to synthesize (direct input)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to text file to synthesize"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output WAV file path (default: outputs/speech.wav)"
    )
    parser.add_argument(
        "--prompt-audio",
        type=str,
        default=None,
        help="Reference audio for voice cloning"
    )
    parser.add_argument(
        "--prompt-text",
        type=str,
        default=None,
        help="Transcript of reference audio"
    )
    parser.add_argument(
        "--cfg-value",
        type=float,
        default=2.0,
        help="Guidance strength (default: 2.0)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10,
        help="Inference timesteps - higher=better quality (default: 10)"
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable text normalization"
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Enable denoising (limits output to 16kHz)"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming synthesis"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Re-enable verbose output if requested
    if args.verbose:
        logging.basicConfig(level=logging.INFO, force=True)
        for logger_name in ["transformers", "torch", "accelerate", "voxcpm"]:
            logging.getLogger(logger_name).setLevel(logging.INFO)

    # Resolve paths
    INP = get_input_dir()
    OUT = get_output_dir()

    # Determine output path
    if args.output is None:
        args.output = "speech.wav"
    output_path = Path(resolve_output_path(args.output, OUT))

    # Get text to synthesize
    if args.text is not None:
        text = args.text
    elif args.input is not None:
        input_path = resolve_input_path(args.input, INP)
        if not Path(input_path).exists():
            print(f"Error: Input file '{input_path}' not found")
            return 1
        text = load_text(input_path)
    else:
        # Default sample text
        text = "Hello! This is VoxCPM, an end-to-end text to speech model from OpenBMB."

    # Resolve prompt audio path if provided
    prompt_wav_path = None
    if args.prompt_audio:
        prompt_wav_path = resolve_input_path(args.prompt_audio, INP)
        if not Path(prompt_wav_path).exists():
            print(f"Error: Prompt audio '{prompt_wav_path}' not found")
            return 1

    # Detect device
    detect_device()

    # Load model
    try:
        model = load_model(MODEL_ID)
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Run inference
    try:
        wav, sample_rate = run_inference(
            model,
            text,
            prompt_wav_path=prompt_wav_path,
            prompt_text=args.prompt_text,
            cfg_value=args.cfg_value,
            inference_timesteps=args.timesteps,
            normalize=not args.no_normalize,
            denoise=args.denoise,
            streaming=args.streaming,
        )
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Save output
    try:
        save_audio(wav, sample_rate, str(output_path))
    except Exception as e:
        print(f"Error saving audio: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Print summary
    print(f"\nSummary:")
    print(f"  Text: {text[:50]}{'...' if len(text) > 50 else ''}")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Output: {output_path}")
    if prompt_wav_path:
        print(f"  Voice cloning: {prompt_wav_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
