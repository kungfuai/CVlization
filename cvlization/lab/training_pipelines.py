from typing import Dict

from cvlization import data
from ..training_pipeline import TrainingPipeline, TrainingPipelineConfig
from ..specs import MLFramework


def training_pipelines() -> Dict[str, TrainingPipeline]:
    return {
        "noop": TrainingPipeline(
            framework=MLFramework.PYTORCH,
            config=TrainingPipelineConfig(
                data_only=True,
            ),
        ),
        "noop_torch": TrainingPipeline(
            framework=MLFramework.PYTORCH,
            config=TrainingPipelineConfig(
                data_only=True,
            ),
        ),
        "noop_tf": TrainingPipeline(
            framework=MLFramework.TENSORFLOW,
            config=TrainingPipelineConfig(
                data_only=True,
            ),
        ),
        "resnet50_tf": TrainingPipeline(
            framework=MLFramework.TENSORFLOW,
            config=TrainingPipelineConfig(
                image_backbone="ResNet50",
                image_pool="flatten",
                pretrained=True,
                epochs=50,
                train_batch_size=256,
                val_batch_size=256,
                train_steps_per_epoch=200,
                optimizer_name="Adam",
                lr=0.0001,
                n_gradients=1,
                dropout=0,
            ),
            # TODO: compare with https://keras.io/zh/examples/cifar10_resnet/
        ),
        "mobilenet_tf": TrainingPipeline(
            framework=MLFramework.TENSORFLOW,
            config=TrainingPipelineConfig(
                image_backbone="MobileNetV2",
                image_pool="flatten",
                pretrained=True,
                epochs=50,
                train_batch_size=256,
                val_batch_size=256,
                train_steps_per_epoch=200,
                optimizer_name="Adam",
                lr=0.0003,
                n_gradients=1,
            ),
        ),
        "resnet18_tf": TrainingPipeline(
            framework=MLFramework.TENSORFLOW,
            config=TrainingPipelineConfig(
                image_backbone="resnet18v2",
                input_shape=[32, 32, 3],
                image_pool="flatten",
                epochs=100,
                train_batch_size=256,
                val_batch_size=256,
                train_steps_per_epoch=200,
                optimizer_name="Adam",
                lr=0.01,
                n_gradients=1,
                dropout=0,
                experiment_tracker=None,
            ),
        ),
        "resnet18_smallimage_tf": TrainingPipeline(
            framework=MLFramework.TENSORFLOW,
            config=TrainingPipelineConfig(
                image_backbone="resnet18v2_smallimage",
                input_shape=[32, 32, 3],
                image_pool="flatten",
                epochs=100,
                train_batch_size=256,
                val_batch_size=256,
                train_steps_per_epoch=200,
                optimizer_name="Adam",
                lr=0.01,
                n_gradients=1,
                dropout=0,
                experiment_tracker=None,
            ),
        ),
        "simple_tf": TrainingPipeline(
            framework=MLFramework.TENSORFLOW,
            config=TrainingPipelineConfig(
                image_backbone="simple",
                image_pool="flatten",
                epochs=100,
                train_batch_size=256,
                val_batch_size=256,
                train_steps_per_epoch=200,
                optimizer_name="Adam",
                # optimizer_name="AdamW",
                lr=0.01,
                n_gradients=1,
                dropout=0,
                experiment_tracker=None,
            ),
            # TODO: https://www.tensorflow.org/addons/tutorials/optimizers_cyclicallearningrate
        ),
        "davidnet_torch": TrainingPipeline(
            framework=MLFramework.PYTORCH,
            config=TrainingPipelineConfig(
                # https://myrtle.ai/learn/how-to-train-your-resnet/
                image_backbone="davidnet",
                image_pool="flatten",
                dense_layer_sizes=[10],
                pretrained=False,
                epochs=25,
                train_batch_size=512,
                val_batch_size=512,
                train_steps_per_epoch=100,
                optimizer_name="SGD_david",
                # optimizer_name="SGD",
                optimizer_kwargs={
                    "lr": 0.01,
                    "momentum": 0.9,
                    "weight_decay": 5e-4,
                    "nesterov": True,
                },
                lr_scheduler_name="OneCycleLR",
                # TODO: lr_scheduler epochs should typically be epochs + 1
                #   This closely-coupled behavior should be handled automatically.
                #   Only max_lr needs to be set.
                lr_scheduler_kwargs={
                    "max_lr": 0.1,
                    "epochs": 26,
                    "steps_per_epoch": 100,
                },
                n_gradients=1,
                permute_image=False,
                # customize_conv1=True,
                experiment_tracker=None,
                precision="fp16",
            ),
        ),
        "resnet18_torch": TrainingPipeline(
            framework=MLFramework.PYTORCH,
            config=TrainingPipelineConfig(
                image_backbone="resnet18",
                image_pool="flatten",
                pretrained=True,
                epochs=20,
                train_batch_size=512,
                val_batch_size=256,
                train_steps_per_epoch=100,
                optimizer_name="SGD",
                optimizer_kwargs={
                    "lr": 0.05,
                    "momentum": 0.9,
                    "weight_decay": 5e-4,
                },
                lr_scheduler_name="OneCycleLR",
                # TODO: lr_scheduler epochs should typically be epochs + 1
                #   This closely-coupled behavior should be handled automatically.
                #   Only max_lr needs to be set.
                lr_scheduler_kwargs={
                    "max_lr": 0.1,
                    "epochs": 21,
                    "steps_per_epoch": 100,
                },
                n_gradients=1,
                permute_image=False,
                customize_conv1=True,
                # num_workers=0,
                experiment_tracker=None,
                precision="fp16",
            ),
        ),
        "resnet18_torch_gc": TrainingPipeline(
            # with gradient accumulation (gc)
            framework=MLFramework.PYTORCH,
            config=TrainingPipelineConfig(
                image_backbone="resnet18",
                image_pool="flatten",
                pretrained=False,
                epochs=20,
                train_batch_size=int(256 / 8),
                val_batch_size=256,
                train_steps_per_epoch=200 * 8,
                optimizer_name="SGD",
                optimizer_kwargs={"lr": 0.05, "momentum": 0.9, "weight_decay": 5e-4},
                lr_scheduler_name="OneCycleLR",
                # TODO: lr_scheduler epochs should typically be epochs + 1
                #   This closely-coupled behavior should be handled automatically.
                #   Only max_lr needs to be set.
                lr_scheduler_kwargs={
                    "max_lr": 0.1,
                    "epochs": 21,
                    "steps_per_epoch": 200 * 8,
                },
                n_gradients=8,
                permute_image=False,
                customize_conv1=True,
                # num_workers=0,
                experiment_tracker=None,
            ),
        ),
    }
