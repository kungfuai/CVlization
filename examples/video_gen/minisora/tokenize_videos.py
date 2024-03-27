import os
import torch
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


def tokenize():
    from argparse import ArgumentParser
    from cvlization.dataset.flying_mnist import FlyingMNISTDatasetBuilder
    from latents import extract_token_ids
    from tqdm import tqdm
    import numpy as np
    
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="flying_mnist", help="Dataset name. E.g. flying_mnist_11k")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for token extraction")
    args = parser.parse_args()
    dataset_name = args.dataset

    max_frames_per_video = 32
    resolution = 256
    db = FlyingMNISTDatasetBuilder(
        dataset_name=dataset_name,
        max_frames_per_video=max_frames_per_video, resolution=resolution
    )
    train_ds = db.training_dataset()
    vae = create_vae(wandb_model_name="zzsi_kungfu/videogpt/model-kbu39ped:v11")
    spatial_compression = 4
    temporal_compression = 4
    vae = vae.to("cuda")
    all_token_ids = []
    for j, token_ids in tqdm(
        enumerate(
            extract_token_ids(
                vae,
                train_ds,
                batch_size=args.batch_size,
                output_device="cpu",
                latent_sequence_length=max_frames_per_video // temporal_compression,
                latent_height=resolution // spatial_compression,
                latent_width=resolution // spatial_compression,
            )
        )
    ):
        # print(token_ids.shape)
        all_token_ids.append(token_ids.unsqueeze(0).numpy())  # .reshape(1, -1))
        # print("all_token_ids:", all_token_ids[-1].astype(float).mean())
        # if j > 1:
        #     break
    all_token_ids = np.concatenate(all_token_ids, 0)
    print(all_token_ids[0])
    print(all_token_ids.shape, all_token_ids.dtype)
    # save
    np.save(
        f"{dataset_name}_tokens_{max_frames_per_video}frames_train.npy", all_token_ids
    )


if __name__ == "__main__":
    tokenize()
