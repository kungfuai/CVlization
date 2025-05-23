def tokenize():
    from argparse import ArgumentParser
    from cvlization.dataset.flying_mnist import FlyingMNISTDatasetBuilder
    from latents import extract_token_ids, create_vae
    from tqdm import tqdm
    import numpy as np
    from pathlib import Path
    
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="flying_mnist", help="Dataset name. E.g. flying_mnist_11k")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for token extraction")
    parser.add_argument("--vae", type=str, default="zzsi_kungfu/videogpt/model-kbu39ped:v11", help="VAE model name")
    args = parser.parse_args()
    dataset_name = args.dataset
    model_id = args.vae.split("/")[-1].split(":")[0]

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
    Path("data/latents").mkdir(exist_ok=True, parents=True)
    np.save(
        f"data/latents/{dataset_name}__{model_id}_tokens_{max_frames_per_video}frames_train.npy", all_token_ids
    )


if __name__ == "__main__":
    tokenize()
