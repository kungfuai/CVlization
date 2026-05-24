#!/usr/bin/env python3
"""Qwen3-Omni-30B-A3B-Instruct spoken dialogue (file-in / text + audio out).

Multimodal causal LM that accepts text / image / video / audio and emits
text + audio (24 kHz) using one of three baked-in voices (Ethan, Chelsie,
Aiden). This wrapper is the headless / batch counterpart of upstream's
web_demo.py (which uses gradio for mic / image / video / chat I/O).

Default checkpoint is the cpatonn AWQ-4bit (compressed-tensors
pack-quantized int4) so the model fits on ~24 GB cards. Set
QWEN3_OMNI_QUANT=bf16 (or --quant bf16) to use the full
Qwen/Qwen3-Omni-30B-A3B-Instruct weights, which need ~80 GB VRAM for a
15 s context. Both checkpoints are Apache 2.0 and not gated.
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
# Reuses the flashlabs_chroma sibling preset's dialogue prompt -- a 1.5 s
# "Hi, who are you?" wav -- as the default user-turn audio.
HF_SAMPLE_FILE = "flashlabs_chroma/hi_who_are_you.wav"
EXAMPLE_NAME = "qwen3_omni"

# Default checkpoint per quantization mode.
MODEL_ID_BF16 = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_ID_AWQ4 = "cpatonn/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit"

SPEAKERS = ("Ethan", "Chelsie", "Aiden")
SAMPLE_RATE = 24000


def _env(name: str, default: str) -> str:
    v = os.getenv(name, "").strip()
    return v if v else default


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


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


def default_model_for_quant(quant: str) -> str:
    return MODEL_ID_AWQ4 if quant == "4bit" else MODEL_ID_BF16


def load_model(model_id: str, quant: str, attn_impl: str):
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    kwargs = {"device_map": "auto"}
    # FA2 requires bf16/fp16; "auto" lets transformers pick from config.json
    # (bf16 for the full model; the AWQ-4bit ckpt also reports bf16 for
    # non-quantized layers). Override to bf16 explicitly to avoid surprises.
    kwargs["dtype"] = "auto" if quant == "4bit" else torch.bfloat16
    if attn_impl != "auto":
        kwargs["attn_implementation"] = attn_impl

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(model_id, **kwargs)
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_id)
    return model, processor


def build_conversation(user_audio: str, system_prompt: str):
    messages = []
    if system_prompt:
        messages.append({"role": "system",
                         "content": [{"type": "text", "text": system_prompt}]})
    messages.append({"role": "user",
                     "content": [{"type": "audio", "audio": user_audio}]})
    return messages


def run_inference(args) -> str:
    import torch
    import soundfile as sf
    from qwen_omni_utils import process_mm_info

    user_audio = resolve_audio_arg(args.audio)
    text_only = args.text_only

    print(f"User audio:      {user_audio}")
    print(f"Model:           {args.model}  (quant={args.quant}, attn={args.attn_impl})")
    print(f"Speaker:         {args.speaker if not text_only else '(text-only, talker disabled)'}")
    if args.system:
        print(f"System prompt:   {args.system!r}")

    print("Loading model... (first run downloads weights)")
    model, processor = load_model(args.model, args.quant, args.attn_impl)
    if text_only:
        # Frees ~10 GB by dropping the Talker stage entirely.
        if hasattr(model, "disable_talker"):
            model.disable_talker()

    conversation = build_conversation(user_audio, args.system)

    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False,
    )
    audios, images, videos = process_mm_info(
        conversation, use_audio_in_video=False,
    )
    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=False,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    gen_kwargs = dict(
        thinker_return_dict_in_generate=True,
        thinker_max_new_tokens=args.max_new_tokens,
        use_audio_in_video=False,
    )
    if args.temperature > 0:
        gen_kwargs.update(
            thinker_do_sample=True,
            thinker_temperature=args.temperature,
        )
    else:
        gen_kwargs.update(thinker_do_sample=False)

    if text_only:
        gen_kwargs["return_audio"] = False
    else:
        gen_kwargs["speaker"] = args.speaker

    print(f"Generating (max_new_tokens={args.max_new_tokens}, temperature={args.temperature})...")
    with torch.no_grad():
        result = model.generate(**inputs, **gen_kwargs)

    # generate() returns either (text_ids, audio) or just text_ids when audio
    # is disabled. Normalise.
    if isinstance(result, tuple):
        text_ids, audio = result
    else:
        text_ids, audio = result, None

    # Decode the text reply (the visible "Thinker" output).
    input_len = inputs["input_ids"].shape[1]
    sequences = text_ids.sequences if hasattr(text_ids, "sequences") else text_ids
    reply_text = processor.batch_decode(
        sequences[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    # ---- Distinct text-side reply (analog of the Moshi/Chroma fix) ----
    border = "=" * 60
    print()
    print(border)
    print("Qwen3-Omni text reply:")
    print(border)
    print(reply_text if reply_text else "(empty text reply)")
    print(border)
    print()

    if audio is None:
        print("Audio generation skipped (text-only mode).")
        return ""

    audio_numpy = audio.reshape(-1).float().detach().cpu().numpy()
    output_path = resolve_output_path(args.output)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio_numpy, SAMPLE_RATE)
    duration_s = len(audio_numpy) / SAMPLE_RATE
    print(f"Saved Qwen3-Omni audio reply to: {output_path} "
          f"({Path(output_path).stat().st_size:,} bytes, {duration_s:.2f} s)")
    return output_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qwen3-Omni-30B-A3B-Instruct: audio in -> text + audio out.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--audio", default="sample",
                   help="User-turn audio. Path / URL / 'sample' for the bundled CVL clip.")
    p.add_argument("--system", default=_env("QWEN3_OMNI_SYSTEM", ""),
                   help="Optional system prompt. Upstream's snippet uses none by default.")
    p.add_argument("--output", default="qwen3_omni_response.wav",
                   help="Output wav path (24 kHz mono).")

    default_quant = _env("QWEN3_OMNI_QUANT", "bf16")
    p.add_argument("--quant", default=default_quant, choices=["bf16", "4bit"],
                   help="'bf16' (default) = full model (~80 GB at 15 s ctx); "
                        "'4bit' = compressed-tensors AWQ-4bit (cpatonn) — "
                        "currently produces garbled output under transformers "
                        "5.9 + compressed-tensors 0.11/0.12 (tracked as a "
                        "known issue; see README).")
    p.add_argument("--model", default=_env("QWEN3_OMNI_MODEL_ID", ""),
                   help="Override model id. Default depends on --quant: "
                        f"{MODEL_ID_AWQ4} (4bit) or {MODEL_ID_BF16} (bf16).")
    p.add_argument("--speaker", default=_env("QWEN3_OMNI_SPEAKER", "Ethan"),
                   choices=list(SPEAKERS),
                   help="Talker voice. Three baked-in choices.")
    p.add_argument("--text-only", action="store_true",
                   default=_bool_env("QWEN3_OMNI_TEXT_ONLY"),
                   help="Disable the Talker stage; only print/return text "
                        "(saves ~10 GB VRAM).")
    p.add_argument("--max-new-tokens", type=int,
                   default=int(_env("QWEN3_OMNI_MAX_NEW_TOKENS", "256")),
                   help="Thinker max new tokens (text response length cap). "
                        "Audio length scales with this — 256 tokens is ~5–10 s "
                        "of speech.")
    p.add_argument("--temperature", type=float,
                   default=float(_env("QWEN3_OMNI_TEMPERATURE", "0.7")),
                   help="Thinker sampling temperature. 0 -> greedy.")
    p.add_argument("--attn-impl", default=_env("QWEN3_OMNI_ATTN", "auto"),
                   choices=["auto", "sdpa", "eager", "flash_attention_2"],
                   help="Attention backend. 'auto' lets transformers pick "
                        "(usually sdpa). 'flash_attention_2' needs the "
                        "flash-attn wheel installed.")
    args = p.parse_args()
    if not args.model:
        args.model = default_model_for_quant(args.quant)
    return args


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
        import traceback
        print(f"ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
