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

def load_data() -> torch.Tensor:
    data = np.load("flying_mnist_tokens_32frames_train.npy")
    assert data.shape == (1000, 8, 64, 64), f"Expected (1000, 8, 64, 64), got {data.shape}"
    # Move `8` to last dim, then flatten after batch dimension.
    # data = np.moveaxis(data, 1, -1)
    # assert data.shape == (1000, 64, 64, 8), f"Expected (1000, 64, 64, 8), got {data.shape}"
    # data = data.reshape(-1, 8*64*64)
    # assert data.shape == (1000, 8*64*64), f"Expected (1000, 8*64*64), got {data.shape}"
    return torch.tensor(data, dtype=torch.long)

def load_single_train_sample(data: torch.Tensor) -> torch.Tensor:
    single_sample = data[0].unsqueeze(0)
    return single_sample.clone()

def load_single_val_sample(data: torch.Tensor) -> torch.Tensor:
    single_sample = data[-1].unsqueeze(0)
    return single_sample.clone()

def load_mamba(device: str) -> torch.nn.Module:
    mamba = MambaClassifier(
        n_tokens=5120,
        # seq_len=32768,
        seq_len=int(2*64*64),
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

def generate_sequence(mamba: torch.nn.Module, prompt_sequence: torch.Tensor, device: str) -> torch.Tensor:
    """
    Returns a tensor of shape (1, seq_len).
    """
    assert prompt_sequence.shape == (1, 8, 64, 64), f"Expected (1, 8, 64, 64), got {prompt_sequence.shape}"
    output_video = prompt_sequence[:, :2].to(device) # Prompt with first two clips.
    with torch.no_grad():
        for ix in range(2, 8):
            assert ix == output_video.shape[1], f"Expected {ix}, got {output_video.shape[1]}"
            model_input = output_video[:, ix-2:ix].view(1, -1)
            assert model_input.shape == (1, 2*64*64), f"Expected (1, 2*64*64), got {model_input.shape}"
            logits = mamba(model_input)[:,-int(64*64):] # (1, 2*64*64, 5120) -> (1, 64*64, 5120)
            # take the argmax of the last dimension.
            next_clip = logits.argmax(dim=-1).view(1, 64, 64)
            assert next_clip.shape == (1, 64, 64), f"Expected (1, 64, 64), got {next_clip.shape}"
            output_video = torch.cat([output_video, next_clip.unsqueeze(0)], dim=1)
    assert output_video.shape == (1, 8, 64, 64), f"Expected (1, 8, 64, 64), got {output_video.shape}"
    return output_video.detach().cpu()

def decode_video(
        vae: torch.nn.Module,
        sequence: torch.Tensor,
) -> torch.Tensor:
    t, h, w = 8, 64, 64
    # sequence = rearrange(sequence, "(b h w t) -> b t h w", b=1, t=t, h=h, w=w)
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
        return video.detach().cpu()

if __name__ == "__main__":
    wandb.init(project="mamba_videos")

    device = "cuda:0"

    data = load_data()
    train_vid = load_single_train_sample(data)
    val_vid = load_single_val_sample(data)

    vae = load_vae().to(device)
    mamba = load_mamba(device).to(device)

    gen_seq_from_train = generate_sequence(mamba, train_vid, device)
    print("Generated sequence from train")
    gen_seq_from_val = generate_sequence(mamba, val_vid, device)
    print("Generated sequence from val")

    train_sanity_vid = decode_video(vae, train_vid.to(device))
    print("Decoded train vid")
    val_sanity_vid = decode_video(vae, val_vid.to(device))
    print("Decoded val vid")
    train_generated_vid = decode_video(vae, gen_seq_from_train.to(device))
    print("Decoded train generated vid")
    val_generated_vid = decode_video(vae, gen_seq_from_val.to(device))
    print("Decoded val generated vid")

    wandb.log({
        "train_sanity_vid": wandb.Video(train_sanity_vid, fps=5, format="mp4"),
        "val_sanity_vid": wandb.Video(val_sanity_vid, fps=5, format="mp4"),
        "train_generated_vid": wandb.Video(train_generated_vid, fps=5, format="mp4"),
        "val_generated_vid": wandb.Video(val_generated_vid, fps=5, format="mp4"),
    })

    print("Done")

