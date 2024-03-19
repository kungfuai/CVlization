import os

import torch
import wandb

# from train_dit import create_vae

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

# def load_vae(wandb_model_name: str = None, hf_model_name: str = "stabilityai/sd-vae-ft-mse") -> AutoencoderKL:
#     if wandb_model_name:
#         vae = load_model_from_wandb(wandb_model_name)
#         return vae
#     vae = AutoencoderKL.from_pretrained(hf_model_name)
#     return vae


if __name__ == "__main__":
    print(
        load_vae(),
    )
