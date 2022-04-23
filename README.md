# A Computer vision library easy to customize

## Requirements

- Docker

## Quickstart

[Colab notebook: running experiments on cifar10](https://colab.research.google.com/drive/1FkZcZnJC_z-PuFSYM91kU1-d63-LecMJ?usp=sharing)

## Quickstart for development
```
bin/build.sh
bin/test.sh
```

Then,

```
bin/train.sh
```

or

```
# If you prefer to run it outside of docker:
bin/train_no_docker.sh
```

to run an experiment on CIFAR10.

## Using the library in a project

Copy `cvlization` directory as a python module to your project.

## When to use this library

### Exploring new model architectures, loss functions, optimizers, image augmentations

In a project with many moving parts and lots of data, running experiments can be very time consuming and prone to errors.

When testing out a new model architecture or optimizer, do a "lab train" first. And this library is aiming to make it easy. Typically a custom model architecture is based on a commonly used base model (e.g ResNet, ViT). Do the customization on top of the base model, pick a public dataset (available in this library, e.g. CIFAR10) that is similar to your target domain, run a lab train with:

```
bin/train.sh
# or python -m cvlization.lab.experiment
```

This experiment can take minutes or hours to finish (depending on the hardware, model architecture and optimizer), and its metrics will give you a good idea of whether the direction is worth pursuing.

Here is an example training pipeline:

```
TrainingPipeline(
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
        experiment_tracker="wandb",
    ),
)
```

The list of available image backbones can be found by `cvlization.keras.image_backbone_names()` and `cvlization.torch.image_backbone_names()` for keras and torch models respectively.

If you like to use your own image backbone, in tensorflow / keras, you can implement a python model function (e.g. `lambda x: keras.layers.Dense(10)`), a `keras.Layer` or a `keras.Model`. In torch, you can implement a `nn.Module`. Then pass that as the `image_backbone` argument.


### Develop multi-task models or using multi-modal inputs

The declarative pattern of the library is to make it easy to build multi-task models. The model's flow can be described as the following pseudo-code:


```
encoded = [encoder(input_tensor) for input_tensor in inputs]
aggregated = aggregate(encoded)
outputs = [head(aggregated) for head in heads]
```

Encoders, aggregators and heads are all components of the model (they can also be models themselves, e.g. `nn.Module`, `keras.Model`). When you need to change what additional inputs the model receives, or what additional targets the model should predict, you can delare this in a configuration, and the library handles model creation and dataset preparation based on your configuration. This can save lots of manual code changes that are error prone.

Encoders are concerned with extracting features from specific inputs. For example, an image encoder knows how to extract features (e.g. pooled vector, feature pyramids) from an image.

Aggregators are concerned with fusing different types of inputs. For example, if the inputs include multiple images or both images and categorical variables, the aggregator knows how to combine them. The logic to combine them can be use case specific. It can be a simple concatenation (with optional broadcasting), attention layers. It can also be customized to combine certain inputs first, and combine other inputs later.

Heads are concered with model targets and loss functions.

To customize the model, you can replace any encoder, aggregator or head with a python function, or `nn.Module` or `keras.Layer/Model` that takes in input tensors and returns output tensors.

### Train models with confidence

The training pipeline in the library does a series of automated quality checks and logs intermediate results for visual inspection:

- It checks the model can be serialized and deserialized, saved and loaded.
- It logs input images after image augmentation, right before the batch enters the training loop.

### Structure of this project

The following python sub-modules are included:

- `specs`: This module contains data classes and enums that specifies of 
the model, metrics, loss functions, optimizers. It is declarative and does
not concern itself with the actual implementation. It is agnostic to the deep learning frameworks.
The right level of abstraction can be hard to find. It strives to be small enough to not micro-manage how modeling should be done, 
but not too small such that customization options are too limited.
- `keras`: This module contains the implementation of layers, models, optimizers, losses, metrics, training loops in keras.
- `torch`: This module contains the implementation of layers, models, optimizers, losses, metrics, training loops in torch.
- `lab`: This module contains utilities to make it easy to run and track experiments with the help of popular tools like `ray`, `mlflow`.


## Why this repo?

This repo serves 3 purposes:

1. Make it easier to customize and operate computer vision model training for client projects, than if we start from standard tooling like tensorflow, torch, torchvision.
2. Provide common quality check and diagnostic tools for the model training workflow.
3. Make it easier to vet new computer vision research, by providing benchmark dataset connectors, and by having an abstraction that can accomodate training scripts in many research repos.
