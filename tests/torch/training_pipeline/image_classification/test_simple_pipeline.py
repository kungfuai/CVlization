from cvlization.data.mock_dataset import RandomImageClassificationDatasetBuilder
from cvlization.torch.training_pipeline.image_classification.simple_pipeline import SimpleImageClassificationPipeline

# Use pytest -s tests/torch/training_pipeline/image_classification/test_simple_pipeline.py
#   to see the console output of training.

def test_simple_image_classification_pipeline(tmpdir):
    mlruns_dir = tmpdir.join("mlruns")
    Config = SimpleImageClassificationPipeline.Config
    config = Config(
        num_classes=9,
        batch_size=8,
        tracking_uri=str(mlruns_dir),
        train_steps_per_epoch=1,
        val_steps_per_epoch=1,
        epochs=2,
        pretrained=False,
    )
    p = SimpleImageClassificationPipeline(config)
    dataset_builder = RandomImageClassificationDatasetBuilder(num_classes=9, height=32, width=28, sample_size=8)
    p.fit(dataset_builder)