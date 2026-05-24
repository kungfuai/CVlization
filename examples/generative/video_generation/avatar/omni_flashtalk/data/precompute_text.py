"""Precompute T5 text embeddings for each manifest item.

OmniAvatar's student DiT is trained with text + audio + ref conditioning.
Our distillation was passing zero text, which puts the student in an
out-of-distribution regime (cross-attn over all zeros) and produces garbage
predictions regardless of the loss recipe.

This script loads OmniAvatar's WanTextEncoder + umt5-xxl tokenizer, encodes
each item's `full_prompt` from manifest_assets.jsonl, and appends the result
to that item's `latents/<id>.pt` under key `text_context` (shape [text_len=512,
text_dim=4096], dtype bf16).

Run inside OmniAvatar's venv:
    python precompute_text.py --data-dir ~/zz/omni_flashtalk_data \\
        --t5-path ~/zz/omni_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth \\
        --tokenizer-dir ~/zz/omni_models/Wan2.1-T2V-1.3B/google/umt5-xxl
"""
import argparse
import json
import os
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--t5-path", required=True,
                    help="path to models_t5_umt5-xxl-enc-bf16.pth")
    ap.add_argument("--tokenizer-dir", required=True,
                    help="dir containing tokenizer.json + tokenizer_config.json")
    ap.add_argument("--omniavatar-root", default="/home/whadmin/zz/OmniAvatar")
    args = ap.parse_args()

    sys.path.insert(0, args.omniavatar_root)
    from OmniAvatar.models.wan_video_text_encoder import WanTextEncoder
    from OmniAvatar.prompters.wan_prompter import WanPrompter

    device = "cuda"
    print("loading T5 (~11GB, ~30s)...")
    text_encoder = WanTextEncoder()
    sd = torch.load(args.t5_path, map_location="cpu", weights_only=False)
    # The .pth holds raw T5 weights; try direct then via converter
    try:
        text_encoder.load_state_dict(sd, strict=False)
    except Exception:
        sd = text_encoder.state_dict_converter().from_civitai(sd)
        text_encoder.load_state_dict(sd, strict=False)
    text_encoder = text_encoder.to(device=device, dtype=torch.bfloat16).eval()

    prompter = WanPrompter(tokenizer_path=args.tokenizer_dir, text_len=512)
    prompter.fetch_models(text_encoder=text_encoder)

    items = [json.loads(l) for l in
             open(os.path.join(args.data_dir, "manifest_assets.jsonl")) if l.strip()]
    done = skipped = 0
    for it in items:
        pt_path = os.path.join(args.data_dir, "latents", f"{it['id']}.pt")
        if not os.path.exists(pt_path):
            print(f"  WARN {it['id']}: no latents .pt"); continue
        blob = torch.load(pt_path, map_location="cpu", weights_only=False)
        if "text_context" in blob:
            skipped += 1
            continue
        with torch.no_grad():
            emb = prompter.encode_prompt(it["full_prompt"], positive=True, device=device)
        # emb shape: [1, text_len, text_dim]. Squeeze batch, cast to bf16 for storage.
        blob["text_context"] = emb.squeeze(0).to(torch.bfloat16).cpu()
        torch.save(blob, pt_path)
        done += 1
        if done <= 3 or done % 25 == 0:
            print(f"[{done + skipped}/{len(items)}] {it['id']}  "
                  f"text_context shape={tuple(blob['text_context'].shape)}  "
                  f"norm={float(blob['text_context'].float().norm()):.1f}",
                  flush=True)

    print(f"done: {done} new + {skipped} skipped")


if __name__ == "__main__":
    main()
