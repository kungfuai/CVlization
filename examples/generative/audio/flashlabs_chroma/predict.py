#!/usr/bin/env python3
"""FlashLabs Chroma-4B voice-cloning dialogue (file-in / file-out).

Chroma is a multimodal causal LM (Qwen2.5-Omni-3B reasoner + Llama3 backbone
+ Llama3 decoder + Mimi codec, 24 kHz). One user-turn audio in -> one
response audio out, with optional voice cloning from a reference audio
prompt. This wrapper is the headless / batch counterpart of upstream's
local_voice_chat.py (which uses pyaudio for mic/speaker).

The HF model `FlashLabs/Chroma-4B` is auto-gated: first run requires
HF_TOKEN with the gate accepted at huggingface.co/FlashLabs/Chroma-4B.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from cvlization.paths import resolve_input_path, resolve_output_path
except ImportError:

    def resolve_input_path(path: str, input_dir: Optional[Path] = None) -> str:
        if path.startswith(("http://", "https://")) or path.startswith("/"):
            return path
        return str(Path(path).expanduser())

    def resolve_output_path(
        path: Optional[str] = None,
        output_dir: Optional[Path] = None,
        default_filename: str = "result.wav",
    ) -> str:
        output_root = output_dir or Path("outputs")
        output_root.mkdir(parents=True, exist_ok=True)
        path = path or default_filename
        return path if path.startswith("/") else str((output_root / path).resolve())


HF_DATA_REPO = "zzsi/cvl"
HF_SAMPLE_FILE = "livetalk/example.wav"
EXAMPLE_NAME = "flashlabs_chroma"
DEFAULT_MODEL_ID = "FlashLabs/Chroma-4B"


def _env(name: str, default: str) -> str:
    v = os.getenv(name, "").strip()
    return v if v else default


def ensure_sample_audio(cache_root: Optional[Path] = None) -> Path:
    if cache_root is None:
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        cache_root = hf_home / "cvl_data" / EXAMPLE_NAME
    sample_path = cache_root / "example.wav"
    if sample_path.exists():
        return sample_path
    cache_root.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    print(f"Downloading sample audio from {HF_DATA_REPO}/{HF_SAMPLE_FILE}...")
    downloaded = hf_hub_download(
        repo_id=HF_DATA_REPO, filename=HF_SAMPLE_FILE, repo_type="dataset",
    )
    shutil.copy2(downloaded, sample_path)
    return sample_path


def resolve_audio_arg(audio: Optional[str]) -> str:
    if not audio or audio == "sample":
        return str(ensure_sample_audio())
    return resolve_input_path(audio)


def maybe_resample_to_target(audio_path: str, target_sr: int = 16000) -> str:
    """Chroma's audio encoder expects 16 kHz mono PCM for the user-query input."""
    import soundfile as sf
    info = sf.info(audio_path)
    if info.samplerate == target_sr and info.channels == 1:
        return audio_path
    import librosa
    print(f"Resampling {info.samplerate} Hz / {info.channels}ch -> {target_sr} Hz / mono")
    y, _ = librosa.load(audio_path, sr=target_sr, mono=True)
    out_path = str(Path(audio_path).with_suffix(f".{target_sr}.wav"))
    sf.write(out_path, y, target_sr)
    return out_path


def load_model(model_id: str, quant: str, device_map: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    kwargs = {"trust_remote_code": True, "device_map": device_map, "low_cpu_mem_usage": True}
    if quant == "4bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif quant == "bf16":
        kwargs["dtype"] = torch.bfloat16
    else:  # fp16
        kwargs["dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    # Keep the Mimi codec in float32 -- matches upstream's local_voice_chat.py
    # workaround for "Input type (Half) and bias type (float) should be the same".
    if hasattr(model, "codec_model"):
        model.codec_model.to(dtype=torch.float32)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


def run_inference(args) -> str:
    import torch
    import soundfile as sf

    user_audio = maybe_resample_to_target(resolve_audio_arg(args.audio), target_sr=16000)
    if args.prompt_audio:
        prompt_audio_path = maybe_resample_to_target(
            resolve_input_path(args.prompt_audio), target_sr=16000
        )
    else:
        # Default voice-clone reference = the user's own audio. Self-cloning is
        # the simplest neutral default and avoids shipping celebrity samples.
        prompt_audio_path = user_audio
    prompt_text = args.prompt_text

    print(f"User audio:      {user_audio}")
    print(f"Prompt audio:    {prompt_audio_path}{'  (self-clone)' if prompt_audio_path == user_audio else ''}")
    print(f"Prompt text:     {prompt_text!r}")
    print(f"Model:           {args.model} ({args.quant})")

    print("Loading model... (first run downloads weights)")
    model, processor = load_model(args.model, args.quant, args.device_map)
    device = next(model.parameters()).device

    conversation = [[
        {"role": "system", "content": [{"type": "text", "text": args.system}]},
        {"role": "user", "content": [{"type": "audio", "audio": user_audio}]},
    ]]
    inputs = processor(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
        prompt_audio=[prompt_audio_path],
        prompt_text=[prompt_text],
    )

    model_inputs = {k: v.to(device) for k, v in inputs.items() if k != "prompt_text"}
    # Match upstream's dtype handling: keep audio inputs fp32, downcast others.
    for k, v in model_inputs.items():
        if hasattr(v, "dtype") and v.dtype.is_floating_point:
            if k == "input_values":
                model_inputs[k] = v.to(dtype=torch.float32)
            else:
                model_inputs[k] = v.to(dtype=torch.float16 if args.quant != "bf16" else torch.bfloat16)

    print(f"Generating (max_new_tokens={args.max_new_tokens}, temperature={args.temperature})...")
    output = model.generate(
        **model_inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=max(args.temperature, 1e-5),
    )

    print("Decoding via Mimi codec...")
    audio_values = model.codec_model.decode(output.permute(0, 2, 1)).audio_values
    audio_numpy = audio_values[0].cpu().detach().numpy()
    # Defensive shape handling (C, T) -> (T,) or (T, C) -> mono.
    if audio_numpy.ndim > 1 and audio_numpy.shape[0] < audio_numpy.shape[1]:
        audio_numpy = audio_numpy.T
    if audio_numpy.ndim > 1:
        audio_numpy = audio_numpy.mean(axis=1)

    output_path = resolve_output_path(args.output)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio_numpy, 24000)  # Chroma's native rate.
    print(f"Saved Chroma response to: {output_path} ({Path(output_path).stat().st_size:,} bytes)")
    return output_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FlashLabs Chroma-4B spoken dialogue with optional voice cloning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--audio", default="sample",
                   help="User-turn audio. Path / URL / 'sample' for the bundled CVL clip.")
    p.add_argument("--prompt-audio", default=_env("CHROMA_PROMPT_AUDIO", ""),
                   help="Reference voice to clone. Empty (default) self-clones from --audio.")
    p.add_argument("--prompt-text", default=_env("CHROMA_PROMPT_TEXT", "Tell me a story."),
                   help="Transcript of --prompt-audio (helps the voice-cloning conditioner).")
    p.add_argument("--system", default=_env("CHROMA_SYSTEM",
                   "You are Chroma, a helpful voice assistant. Respond conversationally."))
    p.add_argument("--output", default="chroma_response.wav")
    p.add_argument("--model", default=_env("CHROMA_MODEL_ID", DEFAULT_MODEL_ID))
    p.add_argument("--quant", default=_env("CHROMA_QUANT", "bf16"),
                   choices=["bf16", "fp16", "4bit"],
                   help="bf16 (default, ~8GB VRAM), fp16, or 4bit (bnb, ~4GB VRAM).")
    p.add_argument("--device-map", default=_env("CHROMA_DEVICE_MAP", "auto"))
    p.add_argument("--max-new-tokens", type=int, default=int(_env("CHROMA_MAX_NEW_TOKENS", "256")))
    p.add_argument("--temperature", type=float, default=float(_env("CHROMA_TEMPERATURE", "0.7")))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_inference(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
