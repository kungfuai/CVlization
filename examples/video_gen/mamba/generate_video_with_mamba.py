import os

import numpy as np
import torch
import wandb
from tqdm import tqdm
from einops import rearrange

from examples.video_gen.mamba.mamba_classifier import MambaClassifier
from cvlization.torch.net.vae.video_vqvae import VQVAE

def load_vae() -> torch.nn.Module:
    api = wandb.Api()

    model_full_name = "zzsi_kungfu/videogpt/model-kbu39ped:v11"

    artifact_dir = f"artifacts/{model_full_name.split('/')[-1]}"
    if os.path.exists(artifact_dir):
        print(f"Model already exists at {artifact_dir}")
    else:
        artifact_dir = api.artifact(model_full_name).download()

    vae = VQVAE.load_from_checkpoint(
        artifact_dir + "/model.ckpt",
    )

    vae.eval()

    return vae

def load_single_val_sample(device: str) -> torch.Tensor:
    data = np.load("flying_mnist_tokens_32frames_train.npy")
    single_sample = torch.tensor(data, dtype=torch.long)[-1:]
    assert single_sample.shape == (1, data.shape[1]), \
        f"Expected (1, {data.shape[1]}), got {single_sample.shape}"
    return single_sample.clone().to(device)

def load_mamba(device: str) -> torch.nn.Module:
    mamba = MambaClassifier(
        n_tokens=5120,
        seq_len=32768,
        mamba_n_embed=128,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        n_mamba_layers=1,
        device=device,
    )
    checkpoint_path = "mamba_model.pth"
    assert os.path.exists(checkpoint_path), f"Expected {checkpoint_path} to exist"
    mamba.load_state_dict(torch.load(checkpoint_path))
    mamba.eval()
    return mamba

def generate_sequence(mamba: torch.nn.Module, device: str) -> torch.Tensor:
    """
    Returns a tensor of shape (1, seq_len).
    """
    input_tensor = torch.zeros(1, 32768).long().to(device)
    pbar = tqdm(range(input_tensor.shape[1]), desc="Generating sequence")
    with torch.no_grad():
        for ix in pbar:
            logits = mamba(input_tensor)
            input_tensor[0, ix] = torch.argmax(logits[0, ix])
    return input_tensor

def generate_video(
        vae: torch.nn.Module,
        sequence: torch.Tensor,
        log_name: str,
) -> torch.Tensor:
    t, h, w = 8, 64, 64
    sequence = rearrange(sequence, "(b t h w) -> b t h w", b=1, t=t, h=h, w=w)
    assert sequence.shape == (1, t, h, w), f"expected (1, {t}, {h}, {w}), got {sequence.shape}"
    with torch.no_grad():
        z = vae.vq.codes_to_vec(sequence)
        assert len(z.shape) == 5
        assert z.shape == (1, 4, t, h, w)
        video = vae.decoder(z)
        video = (video - video.min()) / (
            video.max() - video.min() + 1e-6
        )
        video = (video * 255).to(torch.uint8)
        video = rearrange(video, "b c t h w -> t c h (b w)")
        assert video.shape[1] == 3, f"shape of video is {video.shape}"
        display = wandb.Video(video.detach().cpu(), fps=5, format="mp4")
        wandb.log({log_name: display})

if __name__ == "__main__":
    device = "cuda:0"
    vae = load_vae().to(device)
    mamba = load_mamba(device).to(device)
    val_sample = load_single_val_sample(device)
    generated_sequence = generate_sequence(mamba, device)
    generate_video(vae, val_sample, "val_sample")
    generate_video(vae, generated_sequence, "generated_sequence")
    print("Done")
