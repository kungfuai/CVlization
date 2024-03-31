import os
from pathlib import Path
from typing import Iterable
import torch
from torch.utils.data import DataLoader
from einops import rearrange
import wandb
from diffusers.models import AutoencoderKL
from cvlization.torch.net.vae.video_vqvae import VQVAE


def load_model_from_wandb(
    model_full_name: str = "zzsi_kungfu/videogpt/model-tjzu02pg:v17",
) -> dict:
    api = wandb.Api()
    # skip if the file already exists
    artifact_dir = f"artifacts/{model_full_name.split('/')[-1]}"
    if os.path.exists(artifact_dir):
        print(f"Model already exists at {artifact_dir}")
    else:
        artifact_dir = api.artifact(model_full_name).download()
    # The file is model.ckpt.
    state_dict = torch.load(artifact_dir + "/model.ckpt")
    # print(list(state_dict.keys()))
    hyper_parameters = state_dict["hyper_parameters"]
    args = hyper_parameters["args"]

    # args = Namespace(**hyper_parameters)
    # print(args)
    model = VQVAE.load_from_checkpoint(artifact_dir + "/model.ckpt")
    # model = VQVAE(args=args)
    # model.load_state_dict(state_dict["state_dict"])
    return model


def create_vae(
    wandb_model_name: str = None, hf_model_name: str = "stabilityai/sd-vae-ft-mse"
) -> AutoencoderKL:
    if wandb_model_name:
        vae = load_model_from_wandb(wandb_model_name)
        return vae
    vae = AutoencoderKL.from_pretrained(hf_model_name)
    return vae

def extract_latents(
    vae, dataset, batch_size: int = 32, output_device: str = "cpu"
) -> Iterable[torch.Tensor]:
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch in dl:
        x = batch["video"]
        assert x.ndim == 5, "videos must have 4 dimensions besides the batch dim"
        assert x.shape[1] == 3, "videos must have 3 channels at dim 1"
        x = x.to(vae.device)
        with torch.no_grad():
            z = vae.encoder(x)
            vq_output = vae.vq(z)
            if isinstance(vq_output, dict):
                z_q = vq_output["z_recon"]
            else:
                z_q = vq_output
            # unbatch
            for z_q_i in z_q.to(output_device):
                yield z_q_i


def extract_token_ids(
    vae,
    dataset,
    batch_size: int = 32,
    latent_sequence_length=None,
    latent_height=None,
    latent_width=None,
    output_device: str = "cpu",
) -> Iterable[torch.IntTensor]:
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch in dl:
        x = batch["video"]
        # print("x:", x.mean())
        assert x.ndim == 5, "videos must have 4 dimensions besides the batch dim"
        assert x.shape[1] == 3, "videos must have 3 channels at dim 1"
        x = x.to(vae.device)
        with torch.no_grad():
            z = vae.encoder(x)
            # z shape is
            # print("z:", z.mean())
            token_ids = vae.vq.vec_to_codes(z)
            # print("token_ids:", token_ids.float().mean())
            token_ids = token_ids.to(output_device)  # shape is (b, t * h * w)
            if (
                latent_sequence_length is not None
                and latent_height is not None
                and latent_width is not None
            ):
                token_ids = rearrange(
                    token_ids,
                    "b (t h w) -> b t h w",
                    t=latent_sequence_length,
                    h=latent_height,
                    w=latent_width,
                )
            # unbatch
            for token_ids_i in token_ids:
                # print("token_ids_i:", token_ids_i.float().mean())
                yield token_ids_i


def main():
    """
    Extract video latents.
    """
    from argparse import ArgumentParser
    from cvlization.dataset.flying_mnist import FlyingMNISTDatasetBuilder
    from latents import extract_token_ids
    from tqdm import tqdm
    import numpy as np
    
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="flying_mnist", help="Dataset name. E.g. flying_mnist_11k")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for token extraction")
    parser.add_argument("--vae", type=str, default="zzsi_kungfu/videogpt/model-kbu39ped:v11", help="VAE model name")
    args = parser.parse_args()
    dataset_name = args.dataset

    max_frames_per_video = 32
    resolution = 256
    db = FlyingMNISTDatasetBuilder(
        dataset_name=dataset_name,
        max_frames_per_video=max_frames_per_video, resolution=resolution
    )
    train_ds = db.training_dataset()
    if ":" in args.vae:
        vae = create_vae(wandb_model_name=args.vae)
    else:
        vae = create_vae(hf_model_name=args.vae)

    vae = vae.to("cuda")
    all_latents = []
    for j, latents in tqdm(
        enumerate(
            extract_latents(
                vae,
                train_ds,
                batch_size=args.batch_size,
                output_device="cpu",
            )
        )
    ):
        assert latents.ndim == 4, f"Latents must have 4 dimensions besides the batch dim, got {latents.shape}"
        all_latents.append(latents.unsqueeze(0).numpy())  # .reshape(1, -1))
        # print("all_token_ids:", all_token_ids[-1].astype(float).mean())
        # if j > 1:
        #     break
    all_latents = np.concatenate(all_latents, 0)
    print(all_latents[0])
    print(all_latents.shape, all_latents.dtype)
    model_id = args.vae.split("/")[-1].split(":")[0]
    # save
    Path("data/latents").mkdir(exist_ok=True, parents=True)
    np.save(
        f"data/latents/{dataset_name}__{model_id}_latents_{max_frames_per_video}frames_train.npy", all_latents
    )


if __name__ == "__main__":
    main()