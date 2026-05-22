"""S1: encode SoulX target videos + ref portraits into VAE latents.

Uses OmniAvatar's WanVideoVAE (Wan2.1 VAE) so the latents land in the
student's native latent space. For each manifest item, writes a
`latents/<id>.pt` with:
    video_latent : [16, T_lat, H/8, W/8]   teacher target (the KD signal)
    ref_latent   : [16, 1,     H/8, W/8]   ref-portrait latent (conditioning)

Run inside the OmniAvatar venv on a GPU host:
    python encode_targets.py --data-dir ~/zz/omni_flashtalk_data \
        --vae ~/zz/omni_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth [--limit N]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/zz/OmniAvatar"))
from OmniAvatar.models.wan_video_vae import WanVideoVAE


def load_vae(vae_path, device):
    vae = WanVideoVAE()
    sd = torch.load(vae_path, map_location="cpu")
    sd = vae.state_dict_converter().from_civitai(sd)
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    print(f"  VAE load: missing={len(missing)} unexpected={len(unexpected)}")
    return vae.to(device=device, dtype=torch.bfloat16).eval()


def read_video(path):
    """Return float tensor [3, T, H, W] in [-1, 1]."""
    import decord
    vr = decord.VideoReader(path)
    frames = vr.get_batch(range(len(vr))).asnumpy()  # [T, H, W, 3] uint8
    t = torch.from_numpy(frames).float() / 127.5 - 1.0
    return t.permute(3, 0, 1, 2).contiguous()  # [3, T, H, W]


def read_image(path, size_hw=None):
    """Return float tensor [3, 1, H, W] in [-1, 1] (single frame).

    If size_hw=(H, W) is given, the image is resized to match — OmniAvatar's
    pipeline resizes the ref image to the video frame size before encoding so
    the ref latent's spatial dims line up with the video latent.
    """
    from PIL import Image
    img = Image.open(path).convert("RGB")
    if size_hw is not None:
        h, w = size_hw
        img = img.resize((w, h))  # PIL resize takes (W, H)
    t = torch.from_numpy(np.array(img)).float() / 127.5 - 1.0
    return t.permute(2, 0, 1).unsqueeze(1).contiguous()  # [3, 1, H, W]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all items")
    args = ap.parse_args()

    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    device = "cuda"
    out_dir = os.path.join(data_dir, "latents")
    os.makedirs(out_dir, exist_ok=True)

    vae = load_vae(os.path.expanduser(args.vae), device)

    items = [json.loads(l) for l in open(os.path.join(data_dir, "manifest_assets.jsonl")) if l.strip()]
    if args.limit:
        items = items[: args.limit]

    done = 0
    for it in items:
        out = os.path.join(out_dir, f"{it['id']}.pt")
        if os.path.exists(out):
            done += 1
            continue
        vid_mp4 = os.path.join(data_dir, "soulx_targets", f"{it['id']}.mp4")
        ref_png = os.path.join(data_dir, it["image_path"])
        with torch.no_grad():
            video = read_video(vid_mp4).to(device=device, dtype=torch.bfloat16)
            # resize the ref portrait to the video frame size so latents align
            vid_hw = (video.shape[2], video.shape[3])
            ref = read_image(ref_png, size_hw=vid_hw).to(device=device, dtype=torch.bfloat16)
            video_latent = vae.encode([video], device=device)[0]   # [16, T_lat, h, w]
            ref_latent = vae.encode([ref], device=device)[0]       # [16, 1, h, w]
        torch.save(
            {"video_latent": video_latent.cpu(), "ref_latent": ref_latent.cpu()}, out
        )
        done += 1
        print(f"[{done}/{len(items)}] {it['id']}: "
              f"video {tuple(video.shape)} -> latent {tuple(video_latent.shape)}  "
              f"ref -> {tuple(ref_latent.shape)}  "
              f"lat mean={video_latent.float().mean():.3f} std={video_latent.float().std():.3f}",
              flush=True)

    print(f"encode_targets done: {done}/{len(items)}")


if __name__ == "__main__":
    main()
