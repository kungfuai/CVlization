import argparse
import torch
from lightning import pytorch as pl
from cvlization.torch.training_pipeline.vae.video_vqvae import VQVAE, VQVAETrainingPipeline


def main():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser = VQVAE.add_model_specific_args(parser)
    parser.add_argument("--dataset", type=str, default="flying_mnist")
    parser.add_argument(
        "--track", action="store_true", help="Whether to track the experiment"
    )
    # parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--sequence_length", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--accumulate_grad_batches", type=int, default=8)
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)
    parser.add_argument("--save_every_n_epochs", type=int, default=1)
    parser.add_argument(
        "--reinit_every_n_epochs",
        type=int,
        default=None,
        help="reinitialize VQ codebook every n epochs",
    )
    parser.add_argument("--low_utilization_cost", type=float, default=0)
    parser.add_argument("--watch_gradients", action="store_true")
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument(
        "--network_variant", type=str, default="encode111111_decode111111"
    )
    parser.add_argument("--kl_loss_weight", type=float, default=1.0)
    parser.add_argument("--commitment_cost", type=float, default=0.25)

    args = parser.parse_args()

    # Do this for 3090
    torch.set_float32_matmul_precision('medium')

    pipeline = VQVAETrainingPipeline(args)
    if args.dataset == "flying_mnist":
        from cvlization.dataset.flying_mnist import FlyingMNISTDatasetBuilder

        dataset_builder = FlyingMNISTDatasetBuilder(
            resolution=args.resolution, max_frames_per_video=args.sequence_length
        )
    else:
        print("Loading from huggingface...")
        from cvlization.dataset.huggingface import HuggingfaceDatasetBuilder

        dataset_builder = HuggingfaceDatasetBuilder(
            dataset_name=args.dataset, train_split="train", val_split="validation"
        )
    pipeline.fit(dataset_builder)


if __name__ == "__main__":
    main()
