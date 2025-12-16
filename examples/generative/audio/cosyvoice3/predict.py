#!/usr/bin/env python3
"""
Fun-CosyVoice3-0.5B: Zero-shot multilingual text-to-speech inference.

Model is downloaded on first run to centralized cache (~10GB).
Supports zero-shot voice cloning, cross-lingual synthesis, and instruction-following.

License: Apache 2.0
Model: https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
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

logging.basicConfig(level=logging.WARNING)
for logger_name in ["transformers", "torch", "modelscope"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Add CosyVoice to path if not already there
COSYVOICE_ROOT = "/opt/CosyVoice"
MATCHA_ROOT = os.path.join(COSYVOICE_ROOT, "third_party/Matcha-TTS")
for path in [COSYVOICE_ROOT, MATCHA_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

import argparse
from pathlib import Path

import torch
import torchaudio

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


# Model identifier - downloaded lazily on first run
MODEL_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
DEFAULT_PROMPT_WAV = "/opt/CosyVoice/asset/zero_shot_prompt.wav"


def detect_device():
    """Auto-detect device."""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("Using CPU (will be slow)")
    return device


def load_model(model_id: str = MODEL_ID, use_fp16: bool = False):
    """
    Load CosyVoice3 model with lazy downloading.

    Model is downloaded to centralized cache on first run:
    - HuggingFace: ~/.cache/huggingface (or HF_HOME)
    - ModelScope: ~/.cache/modelscope (or MODELSCOPE_CACHE)
    """
    from cosyvoice.cli.cosyvoice import AutoModel

    print(f"Loading model {model_id}...")
    print(f"  Cache: {os.environ.get('MODELSCOPE_CACHE', '~/.cache/modelscope')}")

    # AutoModel handles downloading if model not present locally
    # Downloads to MODELSCOPE_CACHE or HF_HOME
    model = AutoModel(model_dir=model_id, fp16=use_fp16)

    print(f"Model loaded (sample_rate={model.sample_rate})")
    return model


def load_text(text_path: str) -> str:
    """Load text from file."""
    with open(text_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def run_inference(
    model,
    text: str,
    prompt_text: str,
    prompt_wav: str,
    mode: str = "zero_shot",
    instruct_text: str = None,
    stream: bool = False,
    speed: float = 1.0,
):
    """Generate speech from text."""
    print(f"Generating ({mode}): {text[:50]}{'...' if len(text) > 50 else ''}")

    outputs = []

    if mode == "zero_shot":
        for _, result in enumerate(model.inference_zero_shot(
            text, prompt_text, prompt_wav, stream=stream, speed=speed
        )):
            outputs.append(result['tts_speech'])

    elif mode == "cross_lingual":
        for _, result in enumerate(model.inference_cross_lingual(
            text, prompt_wav, stream=stream, speed=speed
        )):
            outputs.append(result['tts_speech'])

    elif mode == "instruct":
        if instruct_text is None:
            instruct_text = "You are a helpful assistant.<|endofprompt|>"
        for _, result in enumerate(model.inference_instruct2(
            text, instruct_text, prompt_wav, stream=stream, speed=speed
        )):
            outputs.append(result['tts_speech'])
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return outputs


def save_audio(audio_tensors: list, output_path: str, sample_rate: int = 24000):
    """Save generated audio to file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if len(audio_tensors) == 1:
        torchaudio.save(output_path, audio_tensors[0], sample_rate)
    else:
        combined = torch.cat(audio_tensors, dim=1)
        torchaudio.save(output_path, combined, sample_rate)

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fun-CosyVoice3-0.5B TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize")
    parser.add_argument("--input", type=str, default=None, help="Text file path")
    parser.add_argument("--output", type=str, default=None, help="Output WAV path")
    parser.add_argument("--mode", type=str, choices=["zero_shot", "cross_lingual", "instruct"],
                        default="zero_shot", help="Inference mode")
    parser.add_argument("--prompt-wav", type=str, default=None, help="Reference audio")
    parser.add_argument("--prompt-text", type=str,
                        default="希望你以后能够做的比我还好呦。",
                        help="Reference text (must match prompt audio)")
    parser.add_argument("--instruct", type=str, default=None, help="Instruction for instruct mode")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed")
    parser.add_argument("--fp16", action="store_true", help="Use FP16")
    parser.add_argument("--stream", action="store_true", help="Streaming mode")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, force=True)

    # Resolve paths
    INP = get_input_dir()
    OUT = get_output_dir()

    if args.output is None:
        output_path = Path(OUT) / "speech.wav"
    else:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path

    # Get text
    if args.text is not None:
        text = args.text
    elif args.input is not None:
        input_path = resolve_input_path(args.input, INP)
        if not Path(input_path).exists():
            print(f"Error: {input_path} not found")
            return 1
        text = load_text(input_path)
    else:
        text = "八百标兵奔北坡，北坡炮兵并排跑。"

    # Get prompt audio
    if args.prompt_wav is not None:
        prompt_wav = resolve_input_path(args.prompt_wav, INP)
        if not Path(prompt_wav).exists():
            print(f"Error: {prompt_wav} not found")
            return 1
    else:
        prompt_wav = DEFAULT_PROMPT_WAV

    # Detect device
    device = detect_device()

    # Load model (lazy download on first run)
    try:
        model = load_model(MODEL_ID, use_fp16=args.fp16 and device == "cuda")
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    # Prepare instruct text
    instruct_text = None
    if args.mode == "instruct":
        if args.instruct:
            instruct_text = f"You are a helpful assistant. {args.instruct}<|endofprompt|>"
        else:
            instruct_text = "You are a helpful assistant.<|endofprompt|>"

    # Run inference
    try:
        outputs = run_inference(
            model, text, args.prompt_text, prompt_wav,
            mode=args.mode, instruct_text=instruct_text,
            stream=args.stream, speed=args.speed,
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    # Save output
    save_audio(outputs, str(output_path), sample_rate=model.sample_rate)

    return 0


if __name__ == "__main__":
    sys.exit(main())
