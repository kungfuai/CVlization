#!/usr/bin/env python3
"""Stable-Worldmodel DMControl: World Model Loading + Expert Policy Rollout

This script demonstrates two things:
  1. Loading the LeJEPA world model from lejepa_weights.ckpt (the PyTorch
     Lightning training checkpoint) without using the broken object checkpoint.
  2. Running an expert policy rollout in DeepMind Control Suite (cheetah task)
     to produce a video.

Background — why the object checkpoint cannot be used
------------------------------------------------------
The official inference path uses lejepa_epoch_50_object.ckpt, which was
pickled with a private training codebase containing a top-level 'module'
package (classes: ARPredictor, ConditionalBlock, JEPA, MLP, Attention, …).
This package is absent from every released stable-worldmodel tag, so
torch.load(weights_only=False) raises "No module named 'module'".

Our fix (Option C): use lejepa_weights.ckpt instead
----------------------------------------------------
lejepa_weights.ckpt is a plain PyTorch Lightning checkpoint dict with a
'state_dict' key. We reconstruct the model architecture here from scratch
by matching the exact weight key names and tensor shapes observed in the
checkpoint, then call model.load_state_dict() — no pickle class-path
dependencies.

Architecture (from config.yaml + checkpoint shape inspection):
  encoder:        ViT-tiny (hidden=192, 12 layers, patch_size=14, img=224)
  action_encoder: Conv1d(30→10) → Linear(10→768) → SiLU → Linear(768→192)
  predictor:      6-layer DiT-style transformer (16 heads, dim_head=64)
  pred_proj:      Linear(192→2048) → BatchNorm → ReLU → Linear(2048→192)
  projector:      same as pred_proj  (training-only contrastive heads)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# CVL dual-mode path support
try:
    from cvlization.paths import resolve_output_path
except ImportError:
    def resolve_output_path(path):
        return Path(path)


# ---------------------------------------------------------------------------
# World model architecture — reconstructed from lejepa_weights.ckpt shapes
# ---------------------------------------------------------------------------

class ActionEmbedder(nn.Module):
    """Encodes a 5-step action window (30-dim) to a 192-dim embedding.

    Weight keys:
      action_encoder.patch_embed.*   Conv1d(30, 10, kernel_size=1)
      action_encoder.embed.{0,2}.*   Linear(10→768) → SiLU → Linear(768→192)
    """
    def __init__(self, in_dim: int = 30, out_dim: int = 192, hidden_dim: int = 768):
        super().__init__()
        self.patch_embed = nn.Conv1d(in_dim, 10, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(10, hidden_dim),       # embed.0
            nn.SiLU(),                        # embed.1
            nn.Linear(hidden_dim, out_dim),   # embed.2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 30)
        B, T, _ = x.shape
        x = x.permute(0, 2, 1)      # (B, 30, T)
        x = self.patch_embed(x)      # (B, 10, T)
        x = x.permute(0, 2, 1)      # (B, T, 10)
        return self.embed(x)         # (B, T, 192)


class SelfAttention(nn.Module):
    """Pre-norm self-attention accepting optional adaLN shift/scale.

    Weight keys:
      attn.norm.*        LayerNorm(192)
      attn.to_qkv.*      Linear(192, 3072, bias=False)
      attn.to_out.0.*    Linear(1024, 192)
    """
    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float = 0.0):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),  # to_out.0
            nn.Dropout(dropout),         # to_out.1
        )

    def forward(self, x: torch.Tensor, shift=None, scale=None) -> torch.Tensor:
        B, N, _ = x.shape
        h = self.norm(x)
        if shift is not None:
            h = h * (1.0 + scale) + shift
        q, k, v = self.to_qkv(h).chunk(3, dim=-1)
        q, k, v = (t.view(B, N, self.heads, self.dim_head).transpose(1, 2)
                   for t in (q, k, v))
        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.to_out(out)


class FeedForward(nn.Module):
    """MLP whose first Sequential element (net.0) is a LayerNorm.

    Indices 2 and 3 (SiLU, Dropout) carry no parameters, so only
    net.{0,1,4} appear in the state dict.

    Weight keys:
      mlp.net.0.*   LayerNorm(192)
      mlp.net.1.*   Linear(192, 2048)
      mlp.net.4.*   Linear(2048, 192)
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),           # net.0
            nn.Linear(dim, hidden_dim),  # net.1
            nn.SiLU(),                   # net.2
            nn.Dropout(dropout),         # net.3
            nn.Linear(hidden_dim, dim),  # net.4
        )

    def forward(self, x: torch.Tensor, shift=None, scale=None) -> torch.Tensor:
        h = self.net[0](x)   # LayerNorm
        if shift is not None:
            h = h * (1.0 + scale) + shift
        h = self.net[1](h)   # Linear
        h = self.net[2](h)   # SiLU
        h = self.net[3](h)   # Dropout
        h = self.net[4](h)   # Linear
        return h


class ConditionalBlock(nn.Module):
    """DiT-style transformer block with adaLN-zero conditioning.

    Weight keys:
      adaLN_modulation.{0,1}.*  Sequential(SiLU, Linear(192→1152))
      attn.*                     SelfAttention
      mlp.*                      FeedForward
    """
    def __init__(self, dim: int, heads: int, dim_head: int,
                 mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),                   # .0
            nn.Linear(dim, 6 * dim),     # .1
        )
        self.attn = SelfAttention(dim, heads, dim_head, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # c: (B, T, dim) — per-token action conditioning
        factors = self.adaLN_modulation(c)   # (B, T, 6*dim)
        s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp = factors.chunk(6, dim=-1)
        x = x + g_msa * self.attn(x, shift=s_msa, scale=sc_msa)
        x = x + g_mlp * self.mlp(x, shift=s_mlp, scale=sc_mlp)
        return x


class ConditionalTransformer(nn.Module):
    """Stack of ConditionalBlocks with final LayerNorm.

    Weight keys:
      transformer.norm.*      LayerNorm(192)
      transformer.layers.*    ModuleList of ConditionalBlocks
    """
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int,
                 mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            ConditionalBlock(dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, c)
        return self.norm(x)


class ARPredictor(nn.Module):
    """Autoregressive frame predictor with learned temporal positional embedding.

    Operates on (B, T, 192) frame embeddings conditioned on (B, T, 192) action
    embeddings via per-token adaLN in each ConditionalBlock.

    Weight keys:
      predictor.pos_embedding           shape (1, 3, 192)
      predictor.transformer.*           ConditionalTransformer
    """
    def __init__(self, num_frames: int, dim: int, depth: int, heads: int,
                 dim_head: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_frames, dim))
        self.transformer = ConditionalTransformer(dim, depth, heads, dim_head,
                                                  mlp_dim, dropout)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x, c: (B, T, dim)
        x = x + self.pos_embedding[:, :x.shape[1]]
        return self.transformer(x, c)


class Projector(nn.Module):
    """2-layer projection head used for contrastive loss during training.

    Index 2 (activation) has no parameters so only net.{0,1,3} appear
    in the state dict.

    Weight keys:
      *.net.0.*   Linear(192, 2048)
      *.net.1.*   BatchNorm1d(2048)
      *.net.3.*   Linear(2048, 192)
    """
    def __init__(self, in_dim: int = 192, hidden_dim: int = 2048, out_dim: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),   # net.0
            nn.BatchNorm1d(hidden_dim),       # net.1
            nn.ReLU(),                        # net.2
            nn.Linear(hidden_dim, out_dim),   # net.3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class JEPA(nn.Module):
    """Joint Embedding Predictive Architecture for world modelling.

    Reconstructed from lejepa_weights.ckpt by matching exact weight key names
    and shapes — no pickle class-path dependencies.

    The Lightning checkpoint stores weights under 'model.*' keys; we strip
    that prefix when calling load_state_dict().
    """
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder                             # ViT-tiny
        self.action_encoder = ActionEmbedder(30, 192, 768)
        self.predictor = ARPredictor(
            num_frames=3, dim=192, depth=6,
            heads=16, dim_head=64, mlp_dim=2048, dropout=0.0,
        )
        self.pred_proj = Projector(192, 2048, 192)         # training-only
        self.projector = Projector(192, 2048, 192)          # training-only

    @torch.no_grad()
    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """(B, T, 3, H, W) → (B, T, 192) mean-pooled patch embeddings."""
        B, T, C, H, W = frames.shape
        flat = frames.view(B * T, C, H, W)
        out = self.encoder(pixel_values=flat, interpolate_pos_encoding=True)
        patches = out.last_hidden_state[:, 1:, :]   # drop CLS  (B*T, P, 192)
        emb = patches.mean(dim=1)                    # (B*T, 192)
        return emb.view(B, T, 192)

    @torch.no_grad()
    def predict_next(self, history_frames: torch.Tensor,
                     action_windows: torch.Tensor) -> torch.Tensor:
        """Predict next-frame embedding from T=3 history frames.

        Args:
            history_frames:  (B, T, 3, H, W)  T = history_size = 3
            action_windows:  (B, T, 30)        5-step action window per frame
        Returns:
            (B, 192) predicted next-frame embedding
        """
        frame_emb = self.encode_frames(history_frames)     # (B, T, 192)
        action_emb = self.action_encoder(action_windows)   # (B, T, 192)
        predicted = self.predictor(frame_emb, c=action_emb)  # (B, T, 192)
        return predicted[:, -1, :]                          # last = next-frame


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_jepa(device: str = "cpu") -> JEPA:
    """Instantiate JEPA with a ViT-tiny encoder (randomly initialized)."""
    from transformers import ViTModel, ViTConfig
    vit_cfg = ViTConfig(
        hidden_size=192,
        num_hidden_layers=12,
        num_attention_heads=3,      # 192 / 64 = 3 heads
        intermediate_size=768,
        image_size=224,
        patch_size=14,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    encoder = ViTModel(vit_cfg)
    return JEPA(encoder).to(device)


def load_jepa_weights(model: JEPA, ckpt_path: str) -> dict:
    """Load 'model.*' keys from a Lightning checkpoint into JEPA."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    # Strip the 'model.' prefix added by the Lightning wrapper
    model_sd = {k[len("model."):]: v
                for k, v in sd.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    stats = {
        "epoch": ckpt.get("epoch", "?"),
        "total_keys": len(sd),
        "model_keys_loaded": len(model_sd),
        "missing": missing,
        "unexpected": unexpected,
    }
    return stats


# ---------------------------------------------------------------------------
# Demo: verify architecture with a synthetic forward pass
# ---------------------------------------------------------------------------

def demo_forward_pass(model: JEPA, device: str) -> dict:
    """Run one predict_next() call with random data; return output stats."""
    B, T, H, W = 1, 3, 224, 224
    frames = torch.randn(B, T, 3, H, W, device=device)
    actions = torch.zeros(B, T, 30, device=device)  # zero-conditioned
    pred = model.predict_next(frames, actions)        # (1, 192)
    return {
        "input_shape": f"({B}, {T}, 3, {H}, {W})",
        "action_shape": f"({B}, {T}, 30)",
        "output_shape": str(tuple(pred.shape)),
        "output_norm": float(pred.norm(dim=-1).mean()),
    }


# ---------------------------------------------------------------------------
# Dataset helpers for quality evaluation
# ---------------------------------------------------------------------------

def ensure_h5_extracted(asset_dir: Path) -> Path:
    """Extract cheetah/run.h5 from the expert zstd tarball (cached next to it)."""
    out = asset_dir / "datasets" / "expert" / "cheetah_run.h5"
    if out.exists():
        return out
    tarball = asset_dir / "datasets" / "expert" / "dmc_expert.tar.zst"
    if not tarball.exists():
        raise FileNotFoundError(
            f"Dataset tarball not found: {tarball}\n"
            "Run: python download_assets.py --splits expert"
        )
    import subprocess
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["tar", "--use-compress-program=zstd", "-xf", str(tarball),
         "-C", str(out.parent), "--strip-components=3", "dmc/expert/cheetah/run.h5"],
        check=True,
    )
    (out.parent / "run.h5").rename(out)  # same filesystem, always safe
    return out


def load_episode_data(h5_path: Path, episode_idx: int = 0,
                      max_frames: int | None = None):
    """Load pixel frames and actions for one episode. Returns (frames, actions)."""
    import hdf5plugin  # registers Blosc filter — must import before h5py reads data
    import h5py
    with h5py.File(str(h5_path), "r") as f:
        start = int(f["ep_offset"][episode_idx])
        length = int(f["ep_len"][episode_idx])
        if max_frames:
            length = min(length, max_frames)
        end = start + length
        frames  = np.array(f["pixels"][start:end])   # (T, 224, 224, 3) uint8
        actions = np.array(f["action"][start:end])   # (T, 6) float32
    return frames, actions


def encode_frames_batched(encoder: nn.Module, frames_np: np.ndarray,
                          device: str, batch_size: int = 16) -> torch.Tensor:
    """(N, 224, 224, 3) uint8  →  (N, 192) mean-pooled ViT embeddings."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    encoder.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(frames_np), batch_size):
            x = torch.from_numpy(frames_np[i:i + batch_size]).float().to(device) / 255.0
            x = (x.permute(0, 3, 1, 2) - mean) / std
            out = encoder(pixel_values=x, interpolate_pos_encoding=True)
            embs.append(out.last_hidden_state[:, 1:, :].mean(dim=1).cpu())
    return torch.cat(embs, dim=0)  # (N, 192)


def build_action_windows(actions_np: np.ndarray, indices) -> np.ndarray:
    """For each frame index t return a 30-dim window: actions[t-4:t+1] (zero-padded)."""
    windows = []
    for t in indices:
        w = actions_np[max(0, t - 4): t + 1]
        if len(w) < 5:
            w = np.concatenate([np.zeros((5 - len(w), 6), dtype=np.float32), w])
        windows.append(w.flatten())
    return np.array(windows, dtype=np.float32)  # (N, 30)


def predict_sequence(model: "JEPA", all_embs: torch.Tensor,
                     actions_np: np.ndarray, device: str) -> torch.Tensor:
    """Run the predictor over all rolling windows; return (N-3, 192) pred embeddings."""
    N = len(all_embs)
    pred_embs = []
    model.eval()
    with torch.no_grad():
        for t in range(2, N - 1):
            hist = all_embs[t - 2: t + 1].unsqueeze(0).to(device)       # (1,3,192)
            aw   = build_action_windows(actions_np, [t - 2, t - 1, t])  # (3,30)
            aw_t = torch.from_numpy(aw).float().unsqueeze(0).to(device)  # (1,3,30)
            ae   = model.action_encoder(aw_t)                            # (1,3,192)
            ps   = model.predictor(hist, ae)                             # (1,3,192)
            pred_embs.append(ps[0, -1].cpu())
    return torch.stack(pred_embs)  # (N-3, 192)


# ---------------------------------------------------------------------------
# t-SNE embedding visualization
# ---------------------------------------------------------------------------

def run_tsne_viz(model: "JEPA", asset_dir: Path, output_dir: Path,
                 device: str, max_frames: int = 300) -> Path:
    """Encode an episode, run predictor, visualize actual vs predicted in 2D (t-SNE)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    print(f"\nLoading episode data for t-SNE (up to {max_frames} frames)...")
    h5_path = ensure_h5_extracted(asset_dir)
    frames_np, actions_np = load_episode_data(h5_path, max_frames=max_frames)
    N = len(frames_np)
    print(f"  {N} frames")

    print("Encoding frames...")
    actual_embs = encode_frames_batched(model.encoder, frames_np, device)  # (N,192)

    print("Running predictor...")
    pred_embs = predict_sequence(model, actual_embs, actions_np, device)   # (N-3,192)

    print("Running t-SNE (may take ~30 s)...")
    all_np = torch.cat([actual_embs, pred_embs], dim=0).numpy()
    coords = TSNE(n_components=2, perplexity=30, random_state=42,
                  max_iter=1000).fit_transform(all_np)

    actual_2d = coords[:N]
    pred_2d   = coords[N:]

    fig, ax = plt.subplots(figsize=(10, 7))
    t_vals = np.arange(N)
    sc = ax.scatter(actual_2d[:, 0], actual_2d[:, 1], c=t_vals,
                    cmap="Blues", s=30, zorder=3, edgecolors="steelblue",
                    linewidths=0.3)
    ax.plot(actual_2d[:, 0], actual_2d[:, 1], "-", color="steelblue",
            alpha=0.5, linewidth=1.2, zorder=2)
    # Lines connecting each predicted point to the actual frame it should match
    for p, a in zip(pred_2d, actual_2d[3:]):
        ax.plot([p[0], a[0]], [p[1], a[1]], "-", color="#e07050",
                alpha=0.3, linewidth=0.6)
    ax.scatter(pred_2d[:, 0], pred_2d[:, 1], c=np.arange(len(pred_embs)),
               cmap="Reds", s=55, marker="*", zorder=4, edgecolors="#aa2200",
               linewidths=0.3)
    plt.colorbar(sc, ax=ax, label="Time step")
    # Explicit legend with large, fully-opaque markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",
               markersize=10, label="Actual frames (trajectory)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="crimson",
               markersize=14, label="Predicted next-frame embedding"),
        Line2D([0], [0], color="#e07050", linewidth=1.5, alpha=0.8,
               label="Prediction → ground-truth target"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, framealpha=0.9)
    ax.set_title(
        "JEPA embeddings — actual trajectory vs predicted next-frame positions (t-SNE)\n"
        "Lines connect each prediction to its ground-truth target",
        fontsize=11,
    )
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")

    out_path = output_dir / "tsne_embeddings.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] t-SNE plot → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# NN retrieval demo video
# ---------------------------------------------------------------------------

def _label_bar(texts: list[str], cell_w: int = 224, h: int = 28) -> np.ndarray:
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (cell_w * len(texts), h), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    for i, text in enumerate(texts):
        draw.text((cell_w * i + cell_w // 2, h // 2), text,
                  fill=(220, 220, 220), anchor="mm")
    return np.array(img)


def run_nn_demo_video(model: "JEPA", asset_dir: Path, output_dir: Path,
                      device: str, fps: int = 5,
                      max_frames: int = 400) -> Path:
    """Nearest-neighbour retrieval demo: predicted embedding → closest dataset frame."""
    import imageio

    print(f"\nLoading episode data for NN demo (up to {max_frames} frames)...")
    h5_path = ensure_h5_extracted(asset_dir)
    frames_np, actions_np = load_episode_data(h5_path, max_frames=max_frames)
    N = len(frames_np)
    print(f"  {N} frames (database size)")

    print("Encoding all frames...")
    all_embs      = encode_frames_batched(model.encoder, frames_np, device)  # (N,192)
    all_embs_norm = F.normalize(all_embs, dim=-1)                            # unit vecs

    labels    = ["  t-2  ", "  t-1  ", "   t   ", "actual t+1", "NN predicted"]
    label_bar = _label_bar(labels)

    video_frames = []
    cos_sims     = []
    model.eval()
    with torch.no_grad():
        for t in range(2, min(N - 1, 80)):
            hist = all_embs[t - 2: t + 1].unsqueeze(0).to(device)
            aw   = build_action_windows(actions_np, [t - 2, t - 1, t])
            aw_t = torch.from_numpy(aw).float().unsqueeze(0).to(device)
            ae   = model.action_encoder(aw_t)
            ps   = model.predictor(hist, ae)
            pred_norm = F.normalize(ps[0, -1:].cpu(), dim=-1)  # (1,192)

            # Cosine similarity against the whole database; mask nearby frames
            sims = (all_embs_norm @ pred_norm.T).squeeze(-1)   # (N,)
            mask = torch.ones(N, dtype=torch.bool)
            mask[max(0, t - 4): min(N, t + 5)] = False
            sims[~mask] = -1.0
            nn_idx = int(sims.argmax())
            cos_sims.append(float(sims[nn_idx]))

            row = np.concatenate([
                frames_np[t - 2], frames_np[t - 1], frames_np[t],
                frames_np[t + 1],       # ground-truth next frame
                frames_np[nn_idx],      # NN-retrieved frame
            ], axis=1)                  # (224, 224*5, 3)
            video_frames.append(np.concatenate([label_bar, row], axis=0))

    print(f"  Mean cos_sim(predicted, NN match): {float(np.mean(cos_sims)):.3f}")

    out_path = output_dir / "nn_demo.mp4"
    imageio.mimwrite(str(out_path), video_frames, fps=fps, codec="libx264",
                     output_params=["-crf", "20"])
    print(f"[ok] NN demo video → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Latent-space animated video (actual frames + t-SNE, in sync)
# ---------------------------------------------------------------------------

def run_latent_video(model: "JEPA", asset_dir: Path, output_dir: Path,
                     device: str, fps: int = 10,
                     max_frames: int = 200) -> Path:
    """Animated side-by-side: actual episode frame (left) + t-SNE latent space (right).

    Mirrors visualize_trajectories.py from the original repo.  At each step the
    world model predicts the next latent position (red star); the actual trajectory
    is shown as faded blue dots.  Both panels advance together so the viewer can
    watch the environment while simultaneously seeing where the world model thinks
    it is in representation space.
    """
    import imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from sklearn.manifold import TSNE
    from PIL import Image

    print(f"\nLoading episode data for latent video (up to {max_frames} frames)...")
    h5_path = ensure_h5_extracted(asset_dir)
    frames_np, actions_np = load_episode_data(h5_path, max_frames=max_frames)
    N = len(frames_np)
    print(f"  {N} frames")

    print("Encoding frames...")
    actual_embs = encode_frames_batched(model.encoder, frames_np, device)  # (N, 192)

    print("Running predictor...")
    pred_embs = predict_sequence(model, actual_embs, actions_np, device)   # (N-3, 192)
    # pred_embs[i] is the world model's prediction for frame i+3

    print("Running t-SNE (may take ~30 s)...")
    all_np = torch.cat([actual_embs, pred_embs], dim=0).numpy()
    coords = TSNE(n_components=2, perplexity=30, random_state=42,
                  max_iter=1000).fit_transform(all_np)
    actual_2d = coords[:N]   # (N, 2)
    pred_2d   = coords[N:]   # (N-3, 2)

    pad  = 3.0
    xlim = (coords[:, 0].min() - pad, coords[:, 0].max() + pad)
    ylim = (coords[:, 1].min() - pad, coords[:, 1].max() + pad)

    CELL = 224
    DPI  = 96
    fig, ax = plt.subplots(figsize=(CELL / DPI, CELL / DPI), dpi=DPI)

    # Static background: full actual trajectory (faded) — drawn once
    ax.plot(actual_2d[:, 0], actual_2d[:, 1], "-",
            color="steelblue", alpha=0.15, lw=0.8, zorder=1)
    ax.scatter(actual_2d[:, 0], actual_2d[:, 1],
               c=np.arange(N), cmap="Blues", s=6, alpha=0.18,
               zorder=2, linewidths=0)

    # Dynamic artists — updated each frame without re-creating the figure
    cur_dot,   = ax.plot([], [], "o", color="steelblue", ms=6, zorder=4,
                         mec="navy", mew=0.8)
    pred_star, = ax.plot([], [], "*", color="crimson",   ms=10, zorder=5,
                         mec="#aa0000", mew=0.5)
    conn_line, = ax.plot([], [], "-", color="#e07050", alpha=0.8, lw=1.2, zorder=3)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    ax.set_title("latent space", fontsize=7, pad=2)
    fig.tight_layout(pad=0.3)

    canvas = FigureCanvasAgg(fig)
    label_bar = _label_bar(["actual frame", "latent space"], cell_w=CELL)

    print("Rendering video frames...")
    video_frames = []
    for t in range(3, N - 1):
        pred_idx = t - 3

        cur_dot.set_data([actual_2d[t, 0]], [actual_2d[t, 1]])
        pred_star.set_data([pred_2d[pred_idx, 0]], [pred_2d[pred_idx, 1]])
        conn_line.set_data(
            [pred_2d[pred_idx, 0], actual_2d[t, 0]],
            [pred_2d[pred_idx, 1], actual_2d[t, 1]],
        )

        canvas.draw()
        w, h = canvas.get_width_height()
        right_arr = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).copy()
        right_arr = right_arr.reshape(h, w, 4)[:, :, :3]  # RGBA → RGB
        right = np.array(Image.fromarray(right_arr).resize((CELL, CELL), Image.LANCZOS))

        row = np.concatenate([frames_np[t], right], axis=1)   # (224, 448, 3)
        video_frames.append(np.concatenate([label_bar, row], axis=0))

    plt.close(fig)

    out_path = output_dir / "latent_video.mp4"
    imageio.mimwrite(str(out_path), video_frames, fps=fps, codec="libx264",
                     output_params=["-crf", "20"])
    print(f"[ok] latent video → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# WM alongside live policy rollout
# ---------------------------------------------------------------------------

def run_wm_rollout_video(model: "JEPA", asset_dir: Path, output_dir: Path,
                         device: str, fps: int = 10, steps: int = 200) -> Path:
    """Run expert policy live; at each step WM predicts next frame via NN retrieval.

    Three phases:
      1. Step the real MuJoCo environment under the expert SAC policy, recording
         each frame and the action taken.
      2. Encode all rollout frames with the JEPA encoder.
      3. For every rolling 3-frame window, run the WM predictor → find the nearest
         dataset frame → compose side-by-side video.

    Columns: actual next frame (left) | WM-predicted NN frame (right).
    This is a live rollout, not a dataset replay — the world model is forecasting
    from real policy trajectories it has never seen.
    """
    import imageio

    # ---- NN database from dataset ----------------------------------------
    print("\nBuilding NN database from dataset...")
    h5_path = ensure_h5_extracted(asset_dir)
    db_frames, _ = load_episode_data(h5_path, max_frames=400)
    db_embs      = encode_frames_batched(model.encoder, db_frames, device)  # (M, 192)
    db_embs_norm = F.normalize(db_embs, dim=-1)
    print(f"  {len(db_frames)} frames in NN database")

    # ---- Live policy rollout ---------------------------------------------
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    from run_eval import _make_cheetah_env
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    ckpt     = (asset_dir / "models" / "swm-dmc-expert-policies" /
                "cheetah" / "expert_policy" / "expert_policy.zip")
    vec_norm = (asset_dir / "models" / "swm-dmc-expert-policies" /
                "cheetah" / "expert_policy" / "vec_normalize.pkl")
    os.environ.setdefault("MUJOCO_GL", "egl")

    sb3_env = DummyVecEnv([_make_cheetah_env])
    sb3_env = VecNormalize.load(str(vec_norm), sb3_env)
    sb3_env.training    = False
    sb3_env.norm_reward = False
    sac_model = SAC.load(str(ckpt), env=sb3_env, device=device)

    gym_env = _make_cheetah_env()
    obs, _ = gym_env.reset()

    print(f"Running live rollout ({steps} steps)...")
    all_frames  = []
    all_actions = []
    for _ in range(steps):
        all_frames.append(gym_env.render().copy())                   # (H, W, C) uint8
        obs_norm = sb3_env.normalize_obs(obs.reshape(1, -1))
        action, _ = sac_model.predict(obs_norm, deterministic=True)  # (1, 6)
        all_actions.append(action[0])
        obs, _, done, _, _ = gym_env.step(action[0])
        if done:
            obs, _ = gym_env.reset()
    all_frames.append(gym_env.render().copy())                       # final frame
    gym_env.close()
    sb3_env.close()

    frames_np  = np.stack(all_frames)    # (steps+1, H, W, C)
    actions_np = np.stack(all_actions)   # (steps, 6)
    print(f"  Collected {len(frames_np)} frames")

    # ---- Encode rollout frames -------------------------------------------
    print("Encoding rollout frames...")
    roll_embs = encode_frames_batched(model.encoder, frames_np, device)  # (steps+1, 192)

    # ---- Predict + NN retrieve -------------------------------------------
    print("Running WM predictor + NN retrieval...")
    labels    = ["actual next frame", "WM predicted (NN)"]
    label_bar = _label_bar(labels, cell_w=224)

    video_frames = []
    cos_sims     = []
    model.eval()
    with torch.no_grad():
        for t in range(2, steps):
            hist = roll_embs[t - 2: t + 1].unsqueeze(0).to(device)    # (1, 3, 192)
            aw   = build_action_windows(actions_np, [t - 2, t - 1, t])
            aw_t = torch.from_numpy(aw).float().unsqueeze(0).to(device)
            ae   = model.action_encoder(aw_t)
            ps   = model.predictor(hist, ae)
            pred_norm = F.normalize(ps[0, -1:].cpu(), dim=-1)          # (1, 192)

            sims   = (db_embs_norm @ pred_norm.T).squeeze(-1)          # (M,)
            nn_idx = int(sims.argmax())
            cos_sims.append(float(sims[nn_idx]))

            row = np.concatenate([frames_np[t + 1], db_frames[nn_idx]], axis=1)
            video_frames.append(np.concatenate([label_bar, row], axis=0))

    print(f"  Mean cos_sim(predicted, NN match): {float(np.mean(cos_sims)):.3f}")

    out_path = output_dir / "wm_rollout.mp4"
    imageio.mimwrite(str(out_path), video_frames, fps=fps, codec="libx264",
                     output_params=["-crf", "20"])
    print(f"[ok] WM rollout video → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Expert policy rollout (delegates to run_eval.py)
# ---------------------------------------------------------------------------

def run_expert_rollout(asset_dir: Path, video_out_dir: Path,
                       steps: int, device: str) -> Path | None:
    import subprocess
    cmd = [
        sys.executable, "run_eval.py",
        "--asset-dir", str(asset_dir),
        "--steps", str(steps),
        "--num-envs", "1",
        "--device", device,
        "--video-out-dir", str(video_out_dir),
    ]
    print(f"Running expert policy rollout ({steps} steps)...")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("[warn] rollout returned non-zero; video may be incomplete")
        return None
    videos = sorted(video_out_dir.glob("*.mp4"))
    return videos[0] if videos else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="stable-worldmodel DMControl: world model loading + expert rollout",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run (download → load model → rollout → video)
  python predict.py

  # Skip rollout, just verify world model loads
  python predict.py --skip-rollout

  # t-SNE embedding visualization (actual trajectory vs predicted positions)
  python predict.py --skip-rollout --tsne

  # NN retrieval demo video (predicted embedding → nearest dataset frame)
  python predict.py --skip-rollout --demo-video

  # Both quality checks at once
  python predict.py --skip-rollout --tsne --demo-video

  # Animated latent-space video (actual frames + t-SNE in sync)
  python predict.py --skip-rollout --latent-video

  # WM prediction alongside live policy rollout
  python predict.py --skip-rollout --wm-rollout
""",
    )
    parser.add_argument("--asset-dir", type=Path,
                        default=Path(os.environ.get(
                            "STABLEWM_HOME",
                            Path.home() / ".cache" / "cvlization" / "stable-worldmodel"
                        )) / "assets",
                        help="Directory with downloaded HuggingFace assets")
    parser.add_argument("--steps", type=int, default=200,
                        help="Expert policy rollout steps (default: 200)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device for world model (default: cpu)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: stable_worldmodel_outputs/)")
    parser.add_argument("--skip-rollout", action="store_true",
                        help="Skip expert policy rollout; only verify world model")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip asset download (assets already present)")
    parser.add_argument("--tsne", action="store_true",
                        help="Encode episode frames, run predictor, visualize with t-SNE")
    parser.add_argument("--demo-video", action="store_true",
                        help="NN retrieval demo: predicted embedding → nearest dataset frame")
    parser.add_argument("--latent-video", action="store_true",
                        help="Animated video: actual frames (left) + t-SNE latent space (right)")
    parser.add_argument("--wm-rollout", action="store_true",
                        help="Live policy rollout with WM prediction: actual next frame vs NN-retrieved")
    args = parser.parse_args()

    output_dir = Path(resolve_output_path(args.output or "stable_worldmodel_outputs/"))
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir

    # ---- Step 1: Download assets ----------------------------------------
    if not args.skip_download:
        print("Downloading assets from HuggingFace...")
        import subprocess
        r = subprocess.run(
            [sys.executable, "download_assets.py",
             "--target-dir", str(args.asset_dir), "--splits", "expert"],
            capture_output=False,
        )
        if r.returncode != 0:
            print("[error] asset download failed")
            return 1

    # ---- Step 2: Load world model from lejepa_weights.ckpt ---------------
    ckpt_path = args.asset_dir / "models" / "swm-dmc-cheetah" / "lejepa_weights.ckpt"
    if not ckpt_path.exists():
        print(f"[error] checkpoint not found: {ckpt_path}")
        return 1

    print(f"\nBuilding JEPA architecture...")
    model = build_jepa(device=args.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    print(f"Loading weights from {ckpt_path.name}...")
    stats = load_jepa_weights(model, str(ckpt_path))
    print(f"  Trained for {stats['epoch']} epochs")
    print(f"  Keys loaded: {stats['model_keys_loaded']} / {stats['total_keys']}")
    if stats["missing"]:
        print(f"  [info] {len(stats['missing'])} missing keys "
              f"(decoder/sigreg not loaded — training-only)")

    # ---- Step 3: Synthetic forward pass to verify all shapes ------------
    print("\nRunning synthetic forward pass to verify architecture...")
    fp = demo_forward_pass(model, args.device)
    print(f"  Input frames:   {fp['input_shape']}")
    print(f"  Action windows: {fp['action_shape']}")
    print(f"  Predicted emb:  {fp['output_shape']}  norm={fp['output_norm']:.3f}")
    print("[ok] World model loaded and verified")

    # ---- Step 4: Expert policy rollout -----------------------------------
    video_path = None
    if not args.skip_rollout:
        video_path = run_expert_rollout(
            asset_dir=args.asset_dir,
            video_out_dir=video_dir,
            steps=args.steps,
            device=args.device,
        )

    # ---- Step 5: t-SNE embedding visualization ---------------------------
    tsne_path = None
    if args.tsne:
        tsne_path = run_tsne_viz(model, args.asset_dir, output_dir, args.device)

    # ---- Step 6: NN retrieval demo video ---------------------------------
    nn_video_path = None
    if args.demo_video:
        nn_video_path = run_nn_demo_video(model, args.asset_dir, output_dir, args.device)

    # ---- Step 7: Latent-space animated video -----------------------------
    latent_video_path = None
    if args.latent_video:
        latent_video_path = run_latent_video(model, args.asset_dir, output_dir, args.device)

    # ---- Step 8: WM alongside live rollout -------------------------------
    wm_rollout_path = None
    if args.wm_rollout:
        wm_rollout_path = run_wm_rollout_video(
            model, args.asset_dir, output_dir, args.device, steps=args.steps,
        )

    # ---- Summary ---------------------------------------------------------
    print(f"\n{'='*55}")
    print("Done.")
    print(f"  World model: loaded from lejepa_weights.ckpt (epoch {stats['epoch']})")
    print(f"  Output dir:  {output_dir}")
    if video_path:
        print(f"  Rollout video:      {video_path}")
    if tsne_path:
        print(f"  t-SNE plot:         {tsne_path}")
    if nn_video_path:
        print(f"  NN demo video:      {nn_video_path}")
    if latent_video_path:
        print(f"  Latent video:       {latent_video_path}")
    if wm_rollout_path:
        print(f"  WM rollout video:   {wm_rollout_path}")
    print(f"{'='*55}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
