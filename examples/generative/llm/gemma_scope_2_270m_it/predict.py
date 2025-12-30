import argparse
import html
import json
import os
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

# CVL dual-mode execution support
try:
    from cvlization.paths import get_output_dir, resolve_output_path

    CVL_AVAILABLE = True
except ImportError:
    CVL_AVAILABLE = False

    def get_output_dir() -> Path:
        out = Path("./outputs")
        out.mkdir(parents=True, exist_ok=True)
        return out

    def resolve_output_path(
        path: str | None = None,
        output_dir: Path | None = None,
        default_filename: str = "result.txt",
    ) -> str:
        output_dir = output_dir or get_output_dir()
        path = path or default_filename
        return path if path.startswith("/") else str((output_dir / path).resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gemma Scope 2 analysis for Gemma 3 270M IT with optional exports."
    )
    parser.add_argument(
        "--prompt",
        default="Explain what a sparse autoencoder is in plain language.",
        help="User prompt to analyze.",
    )
    parser.add_argument(
        "--system",
        default="",
        help="Optional system instruction to prepend when using chat template.",
    )
    parser.add_argument(
        "--model_id",
        default=os.environ.get("MODEL_ID", "google/gemma-3-270m-it"),
        help="Base Gemma model ID for activation extraction.",
    )
    parser.add_argument(
        "--sae_release",
        default="google/gemma-scope-2-270m-it",
        help="Hugging Face repo ID containing SAE weights.",
    )
    parser.add_argument(
        "--sae_category",
        default="resid_post",
        help="Folder name within the repo (e.g. resid_post, attn_out, mlp_out).",
    )
    parser.add_argument(
        "--sae_id",
        default="layer_12_width_16k_l0_small",
        help="SAE identifier within the release.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=12,
        help="Layer index for hidden_states (0-based, excludes embeddings).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top SAE features to keep per token.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Number of tokens to generate before analysis (0 = prompt only).",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument(
        "--chat",
        action="store_true",
        default=True,
        help="Use the tokenizer chat template when available.",
    )
    parser.add_argument(
        "--no_chat",
        action="store_false",
        dest="chat",
        help="Disable chat template formatting.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device placement for model and SAE.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Computation dtype (auto chooses bfloat16 on CUDA, float32 on CPU).",
    )
    parser.add_argument(
        "--export",
        action="append",
        choices=["html", "json", "jsonl"],
        help="Export artifacts (repeatable).",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Directory for exported artifacts (defaults to CVL outputs when available).",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
        os.environ["HF_TOKEN"] = token
    return token or ""


def ensure_hf_cache() -> None:
    hf_home = os.environ.get("HF_HOME", "~/.cache/huggingface")
    cache_dir = Path(hf_home).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))


def can_use_cuda() -> bool:
    if not torch.cuda.is_available():
        return False
    if torch.cuda.device_count() == 0:
        return False
    try:
        torch.tensor(0, device="cuda")
    except (RuntimeError, AssertionError):
        return False
    return True


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not can_use_cuda():
            print("CUDA requested but unavailable; falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cuda" if can_use_cuda() else "cpu")


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if dtype_arg == "float32":
        return torch.float32
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def sanitize_cuda_device(device: torch.device) -> torch.device:
    if device.type != "cuda":
        return device
    if torch.cuda.device_count() == 0:
        return torch.device("cpu")
    index = device.index
    if index is None:
        return torch.device("cuda:0")
    if index >= torch.cuda.device_count():
        return torch.device("cpu")
    return torch.device(f"cuda:{index}")


def build_prompt(tokenizer, prompt: str, system: str, use_chat: bool) -> str:
    if use_chat and hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    if system:
        return f"{system}\n\n{prompt}"
    return prompt


class JumpReLUSAE(nn.Module):
    def __init__(self, d_in: int, d_sae: int, affine_skip_connection: bool = False) -> None:
        super().__init__()
        self.w_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.w_dec = nn.Parameter(torch.zeros(d_sae, d_in))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        if affine_skip_connection:
            self.affine_skip_connection = nn.Parameter(torch.zeros(d_in, d_in))
        else:
            self.affine_skip_connection = None

    def encode(self, input_acts: torch.Tensor) -> torch.Tensor:
        pre_acts = input_acts @ self.w_enc + self.b_enc
        mask = pre_acts > self.threshold
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.w_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acts = self.encode(x)
        recon = self.decode(acts)
        if self.affine_skip_connection is not None:
            return recon + x @ self.affine_skip_connection
        return recon


def load_sae(
    repo_id: str,
    category: str,
    sae_id: str,
    device: torch.device,
    dtype: torch.dtype,
    token: str,
) -> JumpReLUSAE:
    filename = f"{category}/{sae_id}/params.safetensors"
    params_path = hf_hub_download(
        repo_id=repo_id, filename=filename, token=token or None
    )
    params = load_file(params_path)
    d_in, d_sae = params["w_enc"].shape
    sae = JumpReLUSAE(d_in, d_sae)
    sae.load_state_dict(params)
    sae.eval()
    if device.type == "cuda":
        sae = sae.to(device)
        if dtype in (torch.float16, torch.bfloat16):
            sae = sae.to(dtype=dtype)
    return sae


def normalize_token_text(token: str) -> str:
    if token.startswith("▁"):
        return " " + token[1:]
    if token in {"<bos>", "<eos>", "<pad>"}:
        return ""
    return token


def gather_feature_data(
    token_ids: Sequence[int],
    tokens: Sequence[str],
    top_values: torch.Tensor,
    top_indices: torch.Tensor,
    prompt_len: int,
) -> List[dict]:
    rows = []
    for idx, token_id in enumerate(token_ids):
        features = []
        for feat_idx, feat_val in zip(top_indices[idx], top_values[idx]):
            features.append(
                {"feature_id": int(feat_idx), "activation": float(feat_val)}
            )
        rows.append(
            {
                "position": idx,
                "token_id": int(token_id),
                "token": tokens[idx],
                "text": normalize_token_text(tokens[idx]),
                "is_generated": idx >= prompt_len,
                "features": features,
            }
        )
    return rows


def write_json(output: dict, path: Path) -> None:
    path.write_text(json.dumps(output, indent=2))


def write_jsonl(output: dict, path: Path) -> None:
    lines = []
    meta = {
        "model_id": output["model_id"],
        "sae_release": output["sae_release"],
        "sae_id": output["sae_id"],
        "layer": output["layer"],
    }
    for row in output["tokens"]:
        payload = dict(meta)
        payload.update(row)
        lines.append(json.dumps(payload))
    path.write_text("\n".join(lines) + "\n")


def build_html(output: dict, max_activations: Iterable[float]) -> str:
    max_vals = list(max_activations)
    max_val = max(max_vals) if max_vals else 1.0
    min_val = min(max_vals) if max_vals else 0.0
    span_parts = []
    for row, activation in zip(output["tokens"], max_vals):
        if max_val > min_val:
            norm = (activation - min_val) / (max_val - min_val)
        else:
            norm = 0.0
        alpha = 0.15 + 0.65 * norm
        tooltip = ", ".join(
            f"f{feat['feature_id']}:{feat['activation']:.3f}"
            for feat in row["features"]
        )
        token_text = html.escape(row["text"])
        span_parts.append(
            "<span class=\"tok\" style=\"background: rgba(255, 195, 0, "
            f"{alpha:.3f})\" title=\"{html.escape(tooltip)}\">{token_text}</span>"
        )
    body = "".join(span_parts)
    meta = (
        f"Model: {html.escape(output['model_id'])} | "
        f"SAE: {html.escape(output['sae_release'])} / "
        f"{html.escape(output['sae_category'])} / "
        f"{html.escape(output['sae_id'])} | "
        f"Layer: {output['layer']}"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Gemma Scope 2 Report</title>
  <style>
    :root {{
      color-scheme: light;
    }}
    body {{
      font-family: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
      margin: 2rem;
      background: #f6f2ea;
      color: #1f1a14;
    }}
    .meta {{
      font-size: 0.95rem;
      margin-bottom: 1rem;
    }}
    .tokens {{
      font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
      background: #fffaf0;
      border-radius: 12px;
      padding: 1rem;
      box-shadow: 0 4px 24px rgba(0,0,0,0.06);
      line-height: 1.8;
      white-space: pre-wrap;
    }}
    .tok {{
      padding: 0.1rem 0.2rem;
      border-radius: 4px;
    }}
    .legend {{
      margin-top: 1rem;
      font-size: 0.9rem;
      opacity: 0.75;
    }}
  </style>
</head>
<body>
  <h1>Gemma Scope 2 Token Report</h1>
  <div class="meta">{meta}</div>
  <div class="tokens">{body}</div>
  <div class="legend">Hover tokens for top SAE features.</div>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    ensure_hf_cache()
    token = resolve_token()

    device = resolve_device(args.device)
    device = sanitize_cuda_device(device)
    dtype = resolve_dtype(args.dtype, device)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=token or None)
    prompt_text = build_prompt(tokenizer, args.prompt, args.system, args.chat)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map=None,
        token=token or None,
        low_cpu_mem_usage=True,
    )
    model.eval()
    if device.type == "cuda":
        model.to(device)
    model_device = device

    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    if args.max_new_tokens > 0:
        gen_kwargs = {"max_new_tokens": args.max_new_tokens}
        if args.greedy:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = args.temperature
            gen_kwargs["top_p"] = args.top_p
        with torch.no_grad():
            generated = model.generate(**inputs, **gen_kwargs)
        full_ids = generated
    else:
        full_ids = inputs["input_ids"]

    full_ids = full_ids.to(model_device)
    attention_mask = torch.ones_like(full_ids)

    with torch.no_grad():
        outputs = model(
            full_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False
        )

    hidden_states = outputs.hidden_states
    layer_index = args.layer + 1
    if layer_index >= len(hidden_states):
        raise ValueError(
            f"Layer {args.layer} out of range. hidden_states has {len(hidden_states) - 1} layers."
        )
    resid = hidden_states[layer_index]

    sae_device = sanitize_cuda_device(model_device)
    sae = load_sae(
        args.sae_release,
        args.sae_category,
        args.sae_id,
        sae_device,
        dtype,
        token,
    )
    with torch.no_grad():
        features = sae.encode(resid)

    top_k = max(1, min(args.top_k, features.shape[-1]))
    top_values, top_indices = torch.topk(features[0], k=top_k, dim=-1)

    token_ids = full_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    prompt_len = inputs["input_ids"].shape[1]
    token_rows = gather_feature_data(
        token_ids, tokens, top_values.cpu(), top_indices.cpu(), prompt_len
    )

    output = {
        "prompt": args.prompt,
        "system": args.system,
        "prompt_text": prompt_text,
        "model_id": args.model_id,
        "sae_release": args.sae_release,
        "sae_category": args.sae_category,
        "sae_id": args.sae_id,
        "layer": args.layer,
        "top_k": top_k,
        "max_new_tokens": args.max_new_tokens,
        "tokens": token_rows,
    }

    out_dir = Path(args.out_dir) if args.out_dir else get_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    exports = set(args.export or [])
    if "json" in exports:
        out_path = resolve_output_path(
            "gemma_scope_2_report.json", output_dir=out_dir
        )
        write_json(output, Path(out_path))
    if "jsonl" in exports:
        out_path = resolve_output_path(
            "gemma_scope_2_report.jsonl", output_dir=out_dir
        )
        write_jsonl(output, Path(out_path))
    if "html" in exports:
        max_per_token = [
            max((feat["activation"] for feat in row["features"]), default=0.0)
            for row in token_rows
        ]
        html_report = build_html(output, max_per_token)
        out_path = resolve_output_path(
            "gemma_scope_2_report.html", output_dir=out_dir
        )
        Path(out_path).write_text(html_report)

    print(f"Analyzed {len(token_rows)} tokens at layer {args.layer}.")
    if exports:
        print(f"Exports written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
