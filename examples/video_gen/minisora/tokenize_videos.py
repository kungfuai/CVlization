def tokenize():
    from argparse import ArgumentParser
    from cvlization.dataset.flying_mnist import FlyingMNISTDatasetBuilder
    from latents import extract_token_ids, create_vae
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
