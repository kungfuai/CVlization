import argparse
from torch.utils.data import DataLoader
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from vqvae import VQVAE


class VQVAETrainingPipeline:
    def __init__(self, args):
        self.args = args

    def fit(self, dataset_builder):
        self.model = self._create_model()
        train_loader, val_loader = self._create_dataloaders(dataset_builder)
        trainer = self._create_trainer()
        trainer.fit(self.model, train_loader, val_loader)

    def _create_model(self):
        model = VQVAE(self.args)
        return model

    def _create_dataloaders(self, dataset_builder):
        train_ds = dataset_builder.training_dataset()
        val_ds = dataset_builder.validation_dataset()
        train_loader = DataLoader(
            train_ds,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
        )
        return train_loader, val_loader

    def _create_trainer(self):
        args = self.args
        callbacks = []
        callbacks.append(
            ModelCheckpoint(
                monitor="val/recon_loss",
                mode="min",
                every_n_epochs=args.save_every_n_epochs,
            )
        )

        kwargs = dict()
        if args.gpus > 1:
            kwargs = dict(distributed_backend="ddp", gpus=args.gpus)

        logger = (
            WandbLogger(project="videogpt", log_model="all") if args.track else None
            # WandbLogger(project="videogpt", log_model=False)
            if args.track
            else None
        )
        if logger is not None:
            logger.log_hyperparams(args)
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            # max_steps=args.max_steps,
            callbacks=callbacks,
            accumulate_grad_batches=args.accumulate_grad_batches,
            limit_train_batches=args.limit_train_batches,
            limit_val_batches=args.limit_val_batches,
            log_every_n_steps=50,
            logger=logger,
            **kwargs,
        )
        if args.track and args.watch_gradients:
            trainer.logger.experiment.watch(self.model)
        # trainer = pl.Trainer.from_argparse_args(
        #     args, callbacks=callbacks, max_steps=20000, **kwargs
        # )
        return trainer


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

    import torch
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
