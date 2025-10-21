"""
Given a latent nanogpt checkpoint and its associated VAE, generate a video using the latent space of the VAE.

This is work in progress..
"""

import argparse
import os
import torch
import torch.nn.functional as F
from einops import rearrange
from nanogpt import GPT
from train_dit import create_vae

def main():
    # load nanogpt model
    nanogpt_state = torch.load("out-latent-nanogpt/ckpt.pt")
    print(nanogpt_state.keys())
    model_args = nanogpt_state["model_args"]
    config = nanogpt_state["config"]
    config = argparse.Namespace(**{**config, **model_args})
    state_dict = nanogpt_state["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model = GPT(config)
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    start_ids = [2433]
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    
    # generate
    temperature = 0.8
    top_k = 200
    max_new_tokens = 2048
    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    print(y)
    y = rearrange(y[:, :2048], "b (t h w) -> b t h w", t=8, h=64, w=64)

    # load vae
    vae = create_vae(wandb_model_name="zzsi_kungfu/videogpt/model-kbu39ped:v11")
    z = vae.vq.code_to_vecs(y)
    video = vae.decode(z)
    print(video.shape)
    # save video to mp4
    frames = video.permute(0, 2, 3, 1)
    import torchvision

    torchvision.io.write_video("sampled.mp4", frames, fps=6)


if __name__ == "__main__":
    main()
    

