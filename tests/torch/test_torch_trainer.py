import logging
from cvlization.torch.torch_model_factory import TorchModelFactory
from cvlization.torch.torch_trainer import TorchTrainer
from cvlization.torch.encoder.torch_image_backbone import create_image_backbone
from cvlization.torch.encoder.torch_image_encoder import TorchImageEncoder
from cvlization.torch.torch_dataset import MapDataset
from ..ml_dataset_utils import prepare_ml_datasets

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def create_model(model_inputs, model_targets):
    factory = TorchModelFactory(
        model_inputs=model_inputs,
        model_targets=model_targets,
        image_encoder=TorchImageEncoder(
            # TODO: use a smaller network to increase speed.
            backbone=create_image_backbone(
                name="simple_conv", pretrained=False, in_chans=1
            ),
            pool_name="flatten",
        ),
        optimizer_name="Adam",
        optimizer_kwargs={"lr": 0.01},
    )
    model_checkpoint = factory()
    model = model_checkpoint.model
    return model


def test_mnist_multiclass(tmpdir):
    train_data, val_data = prepare_ml_datasets()
    train_ds = MapDataset(train_data)
    val_ds = MapDataset(val_data)
    first_batch = train_ds[0]
    x, y, _ = first_batch
    assert x is not None
    model = create_model(train_data.model_inputs, train_data.model_targets)
    trainer = TorchTrainer(
        model=model,
        model_inputs=train_data.model_inputs,
        model_targets=train_data.model_targets,
        train_dataset=train_data,
        val_dataset=val_data,
        train_steps_per_epoch=10,
        train_batch_size=16,
        epochs=3,
        log_dir=str(tmpdir.join("lightning_logs")),
        experiment_tracker=None,
    )
    trainer.train()
    metrics = trainer.get_metrics()
    assert "train_AUROC" in metrics
    assert metrics["train_AUROC"] > 0.53
    # assert metrics["train_Accuracy"] > 0.15


if __name__ == "__main__":
    test_mnist_multiclass()
