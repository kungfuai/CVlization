from tensorflow import keras
from cvlization.legacy_training_pipeline import LegacyTrainingPipeline as TrainingPipeline
from cvlization.lab.model_specs import ImageClassification
from cvlization.lab.datasets import get_dataset_builder_registry
from cvlization.specs import MLFramework, ModelSpec


def test_training_pipeline_can_use_customized_keras_model():
    def my_model(x):
        x = keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
        return x

    prediction_task = ImageClassification(
        n_classes=10,
        num_channels=1,
        image_height=28,
        image_width=28,
        channels_first=False,
    )
    # Dataset and transforms.
    dataset_builders = get_dataset_builder_registry()
    dsb = dataset_builders["mnist_torchvision"]
    # Model, optimizer and hyperparams.
    p = TrainingPipeline(
        ml_framework=MLFramework.TENSORFLOW,
        model=ModelSpec(
            image_backbone=my_model,
            permute_image=False,
            model_inputs=prediction_task.get_model_inputs(),
            model_targets=prediction_task.get_model_targets(),
        ),
        train_batch_size=128,
        val_batch_size=16,
        train_steps_per_epoch=10,
        val_steps_per_epoch=2,
        lr=0.001,
    )
    p.create_model()

    # For debugging. --------------------------------
    debug = False
    if debug:
        train_data = dsb.training_dataset(batch_size=2)
        train_data = dsb.transform_training_dataset_tf(train_data)
        batch = next(iter(train_data))
        assert batch[0][0].shape == (2, 28, 28, 1)
        import tensorflow as tf

        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(p.model(batch[0]))

        gradients = tape.gradient(loss, p.model.trainable_variables)
        print(gradients)
    # End debugging. --------------------------------

    p.create_dataloaders(dsb).create_trainer().run()
