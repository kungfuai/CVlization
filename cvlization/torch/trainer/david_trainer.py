# TODO: move this to training_pipeline/image_classification
import torch
from torch import nn
from ...base_trainer import BaseTrainer
from ..net.image_classification.davidnet.core import (
    PiecewiseLinear,
    Const,
    union,
    Timer,
    preprocess,
    pad,
    normalise,
    transpose,
    Transform,
    Crop,
    Cutout,
    FlipLR,
)
from ..net.image_classification.davidnet.torch_backend import (
    SGD,
    MODEL,
    LOSS,
    OPTS,
    Table,
    train_epoch,
    x_ent_loss,
    # trainable_params,
    cifar10,
    cifar10_mean,
    cifar10_std,
    DataLoader as DataLoaderDavid,
)


class DavidTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        epochs: int = 10,
        train_batch_size: int = 512,
        use_cached_cifar10: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.epochs = epochs
        self.batch_size = train_batch_size
        self.use_cached_cifar10 = use_cached_cifar10

    def _training_loop(self):
        from functools import partial
        import numpy as np

        batch_size = self.batch_size
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        class WrappedDataLoader:
            def __init__(self, dataloader):
                self.dataloader = dataloader

            def __iter__(self):
                for x, y in self.dataloader:
                    yield {"input": x.to(device).half(), "target": y.to(device).long()}

            def __len__(self):
                return len(self.dataloader)

        if self.use_cached_cifar10:
            dataset = cifar10("./data")
            transforms = [
                partial(
                    normalise,
                    mean=np.array(cifar10_mean, dtype=np.float32),
                    std=np.array(cifar10_std, dtype=np.float32),
                ),
                partial(transpose, source="NHWC", target="NCHW"),
            ]
            train_transforms = [Crop(32, 32), FlipLR(), Cutout(8, 8)]
            train_set = list(
                zip(
                    *preprocess(
                        dataset["train"], [partial(pad, border=4)] + transforms
                    ).values()
                )
            )
            test_set = list(zip(*preprocess(dataset["valid"], transforms).values()))
            train_batches = DataLoaderDavid(
                Transform(train_set, train_transforms),
                batch_size,
                shuffle=True,
                set_random_choices=True,
                drop_last=True,
            )
            val_batches = DataLoaderDavid(
                test_set, batch_size, shuffle=False, drop_last=False
            )
        else:
            train_batches = WrappedDataLoader(self.train_dataset)
            val_batches = WrappedDataLoader(self.val_dataset)

        model = self.model

        loss = x_ent_loss
        epochs = self.epochs
        lr_schedule = PiecewiseLinear([0, 5, epochs], [0, 0.4, 0])
        n_train_batches = int(50000 / batch_size)
        timer = Timer(synch=torch.cuda.synchronize)
        opts = [
            SGD(
                # trainable_params(model).values(),
                model.parameters(),
                {
                    "lr": (
                        lambda step: lr_schedule(step / n_train_batches) / batch_size
                    ),
                    "weight_decay": Const(5e-4 * batch_size),
                    "momentum": Const(0.9),
                },
            )
        ]
        logs, state = Table(), {MODEL: model, LOSS: loss, OPTS: opts}
        for epoch in range(epochs):
            logs.append(
                union(
                    {"epoch": epoch + 1},
                    train_epoch(state, timer, train_batches, val_batches),
                )
            )
