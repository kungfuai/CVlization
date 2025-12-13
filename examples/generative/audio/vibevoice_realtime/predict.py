#!/usr/bin/env python3
"""
VibeVoice-Realtime-0.5B: Real-time streaming text-to-speech inference.

Microsoft's lightweight TTS model with ~300ms first-audio latency.
Based on Qwen2.5-0.5B, generates up to ~10 minutes of speech.

License: MIT
Model: https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B
"""
import os
import sys
import logging
import warnings

# Suppress verbose logging by default
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.ERROR)
for logger_name in ["transformers", "torch", "accelerate"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

import argparse
import json
from pathlib import Path

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


MODEL_ID = "microsoft/VibeVoice-Realtime-0.5B"


def detect_device():
    """Auto-detect device and dtype."""
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        print(f"Using CUDA device with bfloat16")
    else:
        device = "cpu"
        dtype = torch.float32
        print("Using CPU with float32 (inference will be slow)")
    return device, dtype


def load_model(model_id: str, device: str, dtype, use_flash_attn: bool = True):
    """Load VibeVoice model and processor."""
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference
    )
    from vibevoice.processor.vibevoice_streaming_processor import (
        VibeVoiceStreamingProcessor
    )

    print(f"Loading {model_id}...")

    processor = VibeVoiceStreamingProcessor.from_pretrained(model_id)

    # Determine attention implementation
    attn_impl = "flash_attention_2" if use_flash_attn and device == "cuda" else "eager"
    if attn_impl == "eager" and use_flash_attn:
        print("flash_attention_2 not available, falling back to eager attention")

    try:
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device if device == "cuda" else None,
            attn_implementation=attn_impl
        )
    except Exception as e:
        if "flash" in str(e).lower():
            print("Flash attention failed, retrying with eager attention...")
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map=device if device == "cuda" else None,
                attn_implementation="eager"
            )
        else:
            raise

    if device != "cuda":
        model = model.to(device)

    model.eval()
    model.set_ddpm_inference_steps(num_steps=5)

    print(f"Model loaded successfully on {device}")
    return model, processor


def load_text(text_path: str) -> str:
    """Load text from file."""
    with open(text_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def get_voice_prompt(speaker_name: str, device: str = "cuda"):
    """Load voice prompt from the VibeVoice voices directory."""
    # Voice files are stored in the VibeVoice installation
    voices_dir = "/opt/VibeVoice/demo/voices/streaming_model"

    if not os.path.exists(voices_dir):
        raise FileNotFoundError(
            f"Voices directory not found at {voices_dir}. "
            "Ensure VibeVoice is properly installed."
        )

    # Build speaker name to file mapping
    voice_files = {}
    for f in os.listdir(voices_dir):
        if f.endswith('.pt'):
            # Extract speaker name from filename (e.g., "en-Carter_man.pt" -> "Carter")
            name = os.path.splitext(f)[0]
            # Remove language prefix and gender suffix
            parts = name.split('-')
            if len(parts) > 1:
                name = parts[1]
            if '_' in name:
                name = name.split('_')[0]
            voice_files[name] = os.path.join(voices_dir, f)
            # Also map the full filename without extension
            voice_files[os.path.splitext(f)[0]] = os.path.join(voices_dir, f)

    # Find matching voice
    voice_path = None
    speaker_lower = speaker_name.lower()

    # Try exact match first
    if speaker_name in voice_files:
        voice_path = voice_files[speaker_name]
    else:
        # Try case-insensitive partial match
        for name, path in voice_files.items():
            if name.lower() == speaker_lower or speaker_lower in name.lower():
                voice_path = path
                break

    if voice_path is None:
        available = sorted(set(voice_files.keys()))
        raise ValueError(
            f"Speaker '{speaker_name}' not found. Available voices: {available}"
        )

    print(f"Loading voice prompt from: {voice_path}")
    voice_prompt = torch.load(voice_path, map_location=device, weights_only=False)
    return voice_prompt


def run_inference(
    model,
    processor,
    text: str,
    speaker_name: str = "Carter",
    cfg_scale: float = 1.5,
    device: str = "cuda"
):
    """Generate speech from text."""
    # Get voice prompt for speaker
    voice_prompt = get_voice_prompt(speaker_name, device)

    # Process input
    inputs = processor.process_input_with_cached_prompt(
        text=text,
        cached_prompt=voice_prompt,
        padding=True,
        return_tensors="pt"
    )

    # Move inputs to device
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    # Generate speech
    print(f"Generating speech for {len(text)} characters...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            cfg_scale=cfg_scale,
            tokenizer=processor.tokenizer,
            generation_config={"do_sample": False},
            all_prefilled_outputs=voice_prompt
        )

    return outputs


def save_audio(processor, outputs, output_path: str):
    """Save generated audio to file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    processor.save_audio(outputs.speech_outputs[0], output_path)
    print(f"Audio saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="VibeVoice-Realtime-0.5B: Text-to-speech inference",
        epilog="Examples:\n"
               "  python predict.py --text 'Hello world'\n"
               "  python predict.py --input sample.txt --speaker Carter\n"
               "  python predict.py --text 'Hello' --output speech.wav",
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
        "--speaker",
        type=str,
        default="Carter",
        help="Speaker voice name (default: Carter)"
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.5,
        help="Classifier-free guidance scale (default: 1.5)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--no-flash-attn",
        action="store_true",
        help="Disable flash attention (use eager attention)"
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
        for logger_name in ["transformers", "torch", "accelerate"]:
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
        text = "Hello! This is VibeVoice, a real-time text to speech model from Microsoft."

    if len(text) < 4:
        print("Warning: Very short text (<4 chars) may produce degraded output")

    # Detect device
    if args.device:
        device = args.device
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    else:
        device, dtype = detect_device()

    # Load model
    try:
        model, processor = load_model(
            MODEL_ID,
            device,
            dtype,
            use_flash_attn=not args.no_flash_attn
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Run inference
    try:
        outputs = run_inference(
            model,
            processor,
            text,
            speaker_name=args.speaker,
            cfg_scale=args.cfg_scale,
            device=device
        )
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Save output
    try:
        save_audio(processor, outputs, str(output_path))
    except Exception as e:
        print(f"Error saving audio: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Print summary
    print(f"\nSummary:")
    print(f"  Text: {text[:50]}{'...' if len(text) > 50 else ''}")
    print(f"  Speaker: {args.speaker}")
    print(f"  Output: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
