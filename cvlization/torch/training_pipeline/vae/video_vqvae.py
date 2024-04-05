import argparse
from torch.utils.data import DataLoader
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from cvlization.torch.net.vae.video_vqvae import VQVAE


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