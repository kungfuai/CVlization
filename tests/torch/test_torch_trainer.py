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
        optimizer_kwargs={"lr": 0.005},
    )
    model_checkpoint = factory()
    model = model_checkpoint.model
    return model


def test_mnist_multiclass(tmpdir):
    from pytorch_lightning import seed_everything

    seed_everything(0, workers=True)
    train_data, val_data = prepare_ml_datasets()
    train_ds = MapDataset(train_data)
    val_ds = MapDataset(val_data)
    first_batch = train_ds[0]
    x, y, _ = first_batch
    assert x is not None
    print("model targets:", train_data.model_targets)
    model = create_model(train_data.model_inputs, train_data.model_targets)
    trainer = TorchTrainer(
        model=model,
        model_inputs=train_data.model_inputs,
        model_targets=train_data.model_targets,
        train_dataset=train_data,
        val_dataset=val_data,
        train_steps_per_epoch=5,
        train_batch_size=32,
        epochs=2,
        log_dir=str(tmpdir.join("lightning_logs")),
        experiment_tracker=None,
    )
    trainer.train()
    metrics = trainer.get_metrics()
    # Metrics now include task-specific prefixes; ensure binary AUROC is tracked for the even/odd head.
    assert "train_digit_is_even_BinaryAUROC" in metrics
    assert metrics["train_digit_is_even_BinaryAUROC"] > 0.5


if __name__ == "__main__":
    test_mnist_multiclass()
