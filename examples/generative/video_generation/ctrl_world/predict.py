#!/usr/bin/env python3
"""Ctrl-World: Controllable world model for robot manipulation.

Replays recorded robot trajectories through a learned action-conditioned
world model, generating predicted video alongside ground truth for visual
comparison. Based on Stable Video Diffusion with action + text conditioning.

Paper: https://arxiv.org/abs/2510.10125
"""

import os
import sys
import logging
import warnings

# Suppress verbose logging before heavy imports
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.ERROR)
for _logger_name in ["transformers", "diffusers", "torch", "accelerate"]:
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

import argparse
import json
import datetime
import shutil
from pathlib import Path

import numpy as np
import torch
import einops
import mediapy
from decord import VideoReader, cpu as decord_cpu
from huggingface_hub import snapshot_download, hf_hub_download

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
        d = os.path.join(os.getcwd(), "outputs")
        os.makedirs(d, exist_ok=True)
        return d

    def resolve_input_path(path, base_dir=None):
        if os.path.isabs(path):
            return path
        return os.path.join(base_dir or ".", path)

    def resolve_output_path(path, base_dir=None):
        if os.path.isabs(path):
            return path
        return os.path.join(base_dir or ".", path)


# Vendored Ctrl-World modules (models/ directory alongside this file)
from config import wm_args
from models.ctrl_world import CrtlWorld
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

def ensure_models(svd_path, clip_path, ckpt_path):
    """Download model weights from HuggingFace if paths are not provided."""
    if svd_path is None:
        print("Downloading SVD model (stabilityai/stable-video-diffusion-img2vid)...")
        svd_path = snapshot_download("stabilityai/stable-video-diffusion-img2vid")
        print(f"  -> {svd_path}")

    if clip_path is None:
        print("Downloading CLIP model (openai/clip-vit-base-patch32)...")
        clip_path = snapshot_download("openai/clip-vit-base-patch32")
        print(f"  -> {clip_path}")

    if ckpt_path is None:
        print("Downloading Ctrl-World checkpoint (yjguo/Ctrl-World)...")
        ckpt_path = hf_hub_download("yjguo/Ctrl-World", filename="checkpoint-10000.pt")
        print(f"  -> {ckpt_path}")

    return svd_path, clip_path, ckpt_path


HF_DATA_REPO = "zzsi/cvl"
HF_DATA_PREFIX = "ctrl_world"


def ensure_sample_data(cache_root=None):
    """Download sample data + stats from HuggingFace zzsi/cvl if not cached.

    Returns (dataset_dir, stat_path) pointing to cached local paths.
    """
    if cache_root is None:
        cache_root = Path(os.environ.get(
            "HF_HOME", Path.home() / ".cache" / "huggingface"
        )) / "cvl_data" / "ctrl_world"

    cache_root = Path(cache_root)
    dataset_dir = cache_root / "sample_data" / "droid_subset"
    stat_path = cache_root / "dataset_meta_info" / "droid_subset" / "stat.json"

    # Check if already downloaded (presence of stat.json + at least one annotation)
    if stat_path.exists() and (dataset_dir / "annotation" / "val" / "899.json").exists():
        return str(dataset_dir), str(stat_path)

    print("Downloading Ctrl-World sample data from HuggingFace (zzsi/cvl)...")

    # Files to download (relative to HF_DATA_PREFIX in the dataset repo)
    hf_files = [
        "sample_data/droid_subset/annotation/val/199.json",
        "sample_data/droid_subset/annotation/val/899.json",
        "sample_data/droid_subset/annotation/val/1799.json",
        "sample_data/droid_subset/annotation/val/18599.json",
        "dataset_meta_info/droid_subset/stat.json",
    ]
    # Video files for each trajectory
    for traj_id in ["199", "899", "1799", "18599"]:
        for cam in ["0", "1", "2"]:
            hf_files.append(f"sample_data/droid_subset/videos/val/{traj_id}/{cam}.mp4")

    for rel_path in hf_files:
        hf_path = f"{HF_DATA_PREFIX}/{rel_path}"
        local_target = cache_root / rel_path
        if local_target.exists():
            continue
        downloaded = hf_hub_download(
            repo_id=HF_DATA_REPO,
            filename=hf_path,
            repo_type="dataset",
        )
        # Copy from HF cache to our organized layout
        local_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, local_target)

    print(f"  Sample data cached at: {cache_root}")
    return str(dataset_dir), str(stat_path)


# ---------------------------------------------------------------------------
# Replay agent
# ---------------------------------------------------------------------------

class ReplayAgent:
    """Loads Ctrl-World model and runs trajectory replay inference."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = cfg.dtype

        # Load world model
        print("Loading Ctrl-World model...")
        self.model = CrtlWorld(cfg)
        state_dict = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).to(self.dtype)
        self.model.eval()
        print(f"Model loaded on {self.device} ({self.dtype})")

        # Normalization statistics
        with open(cfg.data_stat_path, "r") as f:
            data_stat = json.load(f)
            self.state_p01 = np.array(data_stat["state_01"])[None, :]
            self.state_p99 = np.array(data_stat["state_99"])[None, :]

    def normalize_bound(self, data, data_min, data_max, eps=1e-8):
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, -1, 1)

    def load_trajectory(self, traj_id, start_idx=0, steps=8):
        """Load annotation + video frames and encode to VAE latent space."""
        cfg = self.cfg
        skip = cfg.skip_step

        annotation_path = f"{cfg.val_dataset_dir}/annotation/val/{traj_id}.json"
        with open(annotation_path) as f:
            anno = json.load(f)
            try:
                length = len(anno["action"])
            except KeyError:
                length = anno["video_length"]

        frames_ids = np.arange(start_idx, start_idx + steps * skip, skip)
        max_ids = np.ones_like(frames_ids) * (length - 1)
        frames_ids = np.minimum(frames_ids, max_ids).astype(int)

        instruction = anno["texts"][0]
        car_action = np.array(anno["states"])[frames_ids]
        joint_pos = np.array(anno["joints"])[frames_ids]

        video_dict = []
        video_latents = []
        for vid_info in anno["videos"]:
            video_path = f"{cfg.val_dataset_dir}/{vid_info['video_path']}"
            vr = VideoReader(video_path, ctx=decord_cpu(0), num_threads=2)
            try:
                raw = vr.get_batch(range(length)).asnumpy()
            except AttributeError:
                raw = vr.get_batch(range(length)).numpy()
            raw = raw[frames_ids]
            video_dict.append(raw)

            # Encode frames to latent space
            x = torch.from_numpy(raw).to(self.dtype).to(self.device)
            x = x.permute(0, 3, 1, 2) / 255.0 * 2 - 1
            vae = self.model.pipeline.vae
            with torch.no_grad():
                latents = []
                for i in range(0, len(x), 32):
                    batch = x[i : i + 32]
                    latent = (
                        vae.encode(batch)
                        .latent_dist.sample()
                        .mul_(vae.config.scaling_factor)
                    )
                    latents.append(latent)
                x = torch.cat(latents, dim=0)
            video_latents.append(x)

        return car_action, joint_pos, video_dict, video_latents, instruction

    def forward_wm(self, action_cond, video_latent_true, current_latent, his_cond=None, text=None):
        """Run one world-model interaction step."""
        cfg = self.cfg
        pipeline = self.model.pipeline

        # Normalize action to [-1, 1]
        action_norm = self.normalize_bound(action_cond, self.state_p01, self.state_p99)
        action_tensor = torch.tensor(action_norm).unsqueeze(0).to(self.device).to(self.dtype)

        with torch.no_grad():
            if text is not None:
                text_token = self.model.action_encoder(
                    action_tensor, text, self.model.tokenizer, self.model.text_encoder
                )
            else:
                text_token = self.model.action_encoder(action_tensor)

            _, latents = CtrlWorldDiffusionPipeline.__call__(
                pipeline,
                image=current_latent,
                text=text_token,
                width=cfg.width,
                height=int(cfg.height * 3),
                num_frames=cfg.num_frames,
                history=his_cond,
                num_inference_steps=cfg.num_inference_steps,
                decode_chunk_size=cfg.decode_chunk_size,
                max_guidance_scale=cfg.guidance_scale,
                fps=cfg.fps,
                motion_bucket_id=cfg.motion_bucket_id,
                mask=None,
                output_type="latent",
                return_dict=False,
                frame_level_cond=True,
            )

        # Rearrange multi-view latents: (B, F, C, 3*H, W) -> (3*B, F, C, H, W)
        latents = einops.rearrange(latents, "b f c (m h) (n w) -> (b m n) f c h w", m=3, n=1)

        # Decode ground truth
        true_video = self._decode_latents(torch.stack(video_latent_true, dim=0), pipeline)
        # Decode predicted
        pred_video = self._decode_latents(latents, pipeline)

        # Concatenate: ground truth on top, predicted on bottom; views side by side
        videos_cat = np.concatenate([true_video, pred_video], axis=-3)
        videos_cat = np.concatenate([v for v in videos_cat], axis=-2).astype(np.uint8)

        return videos_cat, true_video, pred_video, latents

    def _decode_latents(self, latents_4d, pipeline):
        """Decode a (V, F, C, H, W) latent tensor to uint8 numpy video."""
        cfg = self.cfg
        bsz, frame_num = latents_4d.shape[:2]
        flat = latents_4d.flatten(0, 1)
        decoded = []
        for i in range(0, flat.shape[0], cfg.decode_chunk_size):
            chunk = flat[i : i + cfg.decode_chunk_size] / pipeline.vae.config.scaling_factor
            decoded.append(pipeline.vae.decode(chunk, num_frames=chunk.shape[0]).sample)
        video = torch.cat(decoded, dim=0).reshape(bsz, frame_num, *decoded[0].shape[1:])
        video = ((video / 2.0 + 0.5).clamp(0, 1) * 255)
        return video.detach().float().cpu().numpy().transpose(0, 1, 3, 4, 2).astype(np.uint8)


# ---------------------------------------------------------------------------
# Replay loop
# ---------------------------------------------------------------------------

def run_replay(agent, val_id, start_idx, output_dir):
    """Run trajectory replay and save comparison video."""
    cfg = agent.cfg
    interact_num = cfg.interact_num
    pred_step = cfg.pred_step
    num_history = cfg.num_history

    total_steps = int(pred_step * interact_num + 8)
    print(f"\nReplaying trajectory {val_id} (start={start_idx}, {interact_num} steps)...")

    eef_gt, joint_pos_gt, video_dict, video_latents, instruction = agent.load_trajectory(
        val_id, start_idx=start_idx, steps=total_steps
    )
    print(f"  Instruction: {instruction}")

    # Initialize history buffers
    video_to_save = []
    his_cond = []
    his_eef = []
    first_latent = torch.cat([v[0] for v in video_latents], dim=1).unsqueeze(0)

    for _ in range(num_history * 4):
        his_cond.append(first_latent)
        his_eef.append(eef_gt[0:1])

    # Autoregressive rollout
    for i in range(interact_num):
        start_id = int(i * (pred_step - 1))
        end_id = start_id + pred_step
        video_latent_true = [v[start_id:end_id] for v in video_latents]
        cartesian_pose = eef_gt[start_id:end_id]

        print(f"  Step {i + 1}/{interact_num} ...", end=" ", flush=True)

        # Build action conditioning from history + current chunk
        history_idx = [0, 0, -8, -6, -4, -2]
        his_pose = np.concatenate([his_eef[idx] for idx in history_idx], axis=0)
        action_cond = np.concatenate([his_pose, cartesian_pose], axis=0)
        his_cond_input = torch.cat([his_cond[idx] for idx in history_idx], dim=0).unsqueeze(0)
        current_latent = his_cond[-1]

        videos_cat, _, _, predicted_latents = agent.forward_wm(
            action_cond,
            video_latent_true,
            current_latent,
            his_cond=his_cond_input,
            text=instruction if cfg.text_cond else None,
        )
        print("done")

        # Update history
        his_eef.append(cartesian_pose[pred_step - 1 : pred_step])
        his_cond.append(
            torch.cat([v[pred_step - 1] for v in predicted_latents], dim=1).unsqueeze(0)
        )

        if i == interact_num - 1:
            video_to_save.append(videos_cat)
        else:
            video_to_save.append(videos_cat[: pred_step - 1])

    # Save output video
    video = np.concatenate(video_to_save, axis=0)
    text_slug = instruction.replace(" ", "_").replace(",", "").replace(".", "")[:30]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"replay_{val_id}_{text_slug}_{ts}.mp4"
    output_path = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    mediapy.write_video(output_path, video, fps=4)
    print(f"  Saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ctrl-World: Replay robot trajectories through a learned world model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Default: replay trajectory 899, 12 interaction steps (~6 min on A100)
  python predict.py

  # Replay all 3 sample trajectories
  python predict.py --val_ids 899 18599 199

  # Quick run with fewer interaction steps
  python predict.py --interact_num 3

  # Point to locally cached models
  python predict.py --svd_model_path /data/svd --clip_model_path /data/clip --ckpt_path /data/ctrl_world.pt
""",
    )
    parser.add_argument(
        "--svd_model_path", type=str, default=None,
        help="Path to SVD model dir (auto-downloads from HuggingFace if not set)",
    )
    parser.add_argument(
        "--clip_model_path", type=str, default=None,
        help="Path to CLIP model dir (auto-downloads from HuggingFace if not set)",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=None,
        help="Path to Ctrl-World .pt checkpoint (auto-downloads from HuggingFace if not set)",
    )
    parser.add_argument(
        "--val_ids", type=str, nargs="+", default=["899"],
        help="Trajectory IDs to replay (default: 899). Available: 899 18599 199",
    )
    parser.add_argument(
        "--interact_num", type=int, default=None,
        help="Number of interaction steps per trajectory (default: 12)",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=50,
        help="Diffusion denoising steps (default: 50)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, force=True)
        for name in ["transformers", "diffusers", "torch", "accelerate"]:
            logging.getLogger(name).setLevel(logging.INFO)

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Resolve output directory (outputs/ subdirectory, not cwd root)
    output_dir = str(resolve_output_path(args.output or "ctrl_world_outputs/"))

    # Ensure model weights are available
    svd_path, clip_path, ckpt_path = ensure_models(
        args.svd_model_path, args.clip_model_path, args.ckpt_path
    )

    # Ensure sample data is available (lazy download from HuggingFace)
    dataset_dir, stat_path = ensure_sample_data()

    # Build config from vendored wm_args dataclass
    cfg = wm_args(task_type="replay")
    cfg.svd_model_path = svd_path
    cfg.clip_model_path = clip_path
    cfg.ckpt_path = ckpt_path
    cfg.val_model_path = ckpt_path
    cfg.val_dataset_dir = dataset_dir
    cfg.data_stat_path = stat_path
    cfg.num_inference_steps = args.num_inference_steps
    if args.interact_num is not None:
        cfg.interact_num = args.interact_num
    cfg.val_id = args.val_ids
    cfg.start_idx = [8] * len(args.val_ids)
    cfg.instruction = [""] * len(args.val_ids)

    # Load model and run replay
    agent = ReplayAgent(cfg)

    output_paths = []
    for val_id, start_idx in zip(cfg.val_id, cfg.start_idx):
        path = run_replay(agent, val_id, start_idx, output_dir)
        output_paths.append(path)

    print(f"\nDone! Generated {len(output_paths)} video(s):")
    for p in output_paths:
        print(f"  {p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
