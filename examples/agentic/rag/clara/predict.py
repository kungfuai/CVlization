#!/usr/bin/env python3
"""
Simple CLaRa inference demo (stage2/stage3) using locally vendored modeling code.
Downloads a HF checkpoint, copies the bundled modeling_clara.py into the cache,
and runs a single QA over a tiny document set.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from modeling_clara import CLaRa, CLaRaConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CLaRa QA inference.")
    parser.add_argument("--model", default=os.getenv("MODEL_ID", "apple/CLaRa-7B-Instruct"),
                        help="HF model id or local path.")
    parser.add_argument("--revision", default=os.getenv("REVISION"),
                        help="Optional HF revision/commit.")
    parser.add_argument("--compression-dir", default=os.getenv("COMPRESSION_DIR", "compression-16"),
                        help="Which compression variant directory to use inside the repo (e.g., compression-16).")
    parser.add_argument("--mode", choices=["text", "paraphrase", "questions"], default=os.getenv("MODE", "text"),
                        help="Which generation path to use: text (stage1_2), paraphrase (stage1), or questions (stage3).")
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="Max new tokens to generate.")
    parser.add_argument("--prompt", default="What does CLaRa do?",
                        help="User question.")
    parser.add_argument("--docs", nargs="*", default=[],
                        help="Optional documents (space-separated). If empty, uses the upstream plant QA toy set.")
    parser.add_argument("--output", default="outputs/result.txt",
                        help="Where to write the generated answer.")
    return parser.parse_args()


def ensure_local_model(model_id: str, revision: str | None, token: str | None,
                       compression_dir: str) -> Path:
    if token:
        print(f"Using HF token (last 4): ...{token[-4:]}")
    else:
        print("Warning: HF_TOKEN not set; trying anonymous download.")
    # Quick sanity check to surface auth errors early.
    from huggingface_hub import HfApi
    HfApi(token=token).model_info(model_id, revision=revision)

    # Pull weights/tokenizer; keep source code local by copying the vendored modeling file.
    allow_patterns = [
        "*.json", "*.safetensors", "*.bin", "*.model", "*.txt",
        "tokenizer.*", "generation_config.json", "*.py",
        f"{compression_dir}/*", f"{compression_dir}/**"
    ]
    model_dir = Path(snapshot_download(
        repo_id=model_id,
        revision=revision,
        token=token,
        allow_patterns=allow_patterns,
        local_files_only=False,
        resume_download=True,
    ))
    variant_dir = model_dir / compression_dir
    if not variant_dir.exists():
        raise FileNotFoundError(
            f"Compression directory '{compression_dir}' not found in {model_dir}. "
            f"Available: {[p.name for p in model_dir.iterdir() if p.is_dir()]}"
        )

    local_modeling = Path(__file__).parent / "modeling_clara.py"
    if local_modeling.exists():
        shutil.copy(local_modeling, model_dir / "modeling_clara.py")
        shutil.copy(local_modeling, variant_dir / "modeling_clara.py")
    return variant_dir


def load_model(model_dir: Path, device: torch.device) -> tuple[CLaRa, AutoTokenizer]:
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    config = CLaRaConfig.from_pretrained(model_dir, trust_remote_code=False)
    # Some released configs point decoder/compressor to local paths; map them to HF ids if so.
    decoder_override = None
    compr_override = None
    if str(config.decoder_model_name).startswith("/mnt/ceph_rbd/model/"):
        decoder_override = "mistralai/Mistral-7B-Instruct-v0.2"
    if str(config.compr_base_model_name).startswith("/mnt/ceph_rbd/model/"):
        compr_override = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    kwargs = {
        "config": config,
        "torch_dtype": dtype,
        "trust_remote_code": False,
    }
    if decoder_override:
        kwargs["decoder_model_name"] = decoder_override
    if compr_override:
        kwargs["compr_base_model_name"] = compr_override
    model = CLaRa.from_pretrained(model_dir, **kwargs)
    model.to(device)
    model.eval()
    return model, tokenizer


def default_docs() -> List[str]:
    # Matches the toy example in the upstream inference notebook.
    return [
        "Weldenia is a monotypic genus of flowering plant in the family Commelinaceae, first described in 1829. "
        "It has one single species: Weldenia candida, which grows originally in Mexico and Guatemala.",
        "Hagsatera is a genus of flowering plants from the orchid family, Orchidaceae. "
        "There are two known species, native to Mexico and Guatemala.",
        "Alsobia is a genus of flowering plants in the family Gesneriaceae, native to Mexico, Guatemala and Costa Rica. "
        "The two species are succulent, stoloniferous herbs and were previously included in the genus 'Episcia'. "
        "Recent molecular studies have supported the separation of 'Alsobia' from 'Episcia'.",
    ]


def main():
    args = parse_args()
    if args.revision == "":
        args.revision = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    model_dir = ensure_local_model(args.model, args.revision, token, args.compression_dir)
    model, tokenizer = load_model(model_dir, device)

    docs = args.docs if args.docs else default_docs()
    questions = [args.prompt]

    if args.mode == "text":
        generations = model.generate_from_text(
            questions=questions,
            documents=[docs],
            max_new_tokens=args.max_tokens,
        )
        answer = generations[0] if isinstance(generations, list) else str(generations)
    elif args.mode == "paraphrase":
        # Paraphrase mode expects documents shaped [batch, num_docs]; reuse the same questions list length.
        generations = model.generate_from_paraphrase(
            questions=questions,
            documents=[docs],
            max_new_tokens=args.max_tokens,
        )
        answer = generations[0] if isinstance(generations, list) else str(generations)
    else:  # questions (stage3 end-to-end). Only supported when the checkpoint includes query_reasoner_adapter.
        if "query_reasoner_adapter" not in getattr(model, "adapter_keys", []):
            raise RuntimeError(
                "questions mode requires a stage3 checkpoint with query_reasoner_adapter "
                "(e.g., MODEL_ID=apple/CLaRa-7B-E2E and COMPRESSION_DIR=compression-16). "
                "Current checkpoint lacks query_reasoner_adapter."
            )
        generations, topk = model.generate_from_questions(
            questions=questions,
            documents=[docs],
            max_new_tokens=args.max_tokens,
        )
        # topk indices come back as tensor; keep only text output
        answer = generations[0] if isinstance(generations, list) else str(generations)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(answer.strip(), encoding="utf-8")
    print(f"\nAnswer:\n{answer.strip()}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    sys.exit(main())
