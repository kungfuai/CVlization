This is for training an unconditional image generator with DDPM.

## Quickstart

```
bash examples/image_gen/diffuser_unconditional/build.sh
bash examples/image_gen/diffuser_unconditional/train.sh  # --help for options
```

By default it trains on the `huggan/flowers-102-categories` dataset with 64x64 resolution.

If you like to run on a different dataset, such as cifar10, you can do:

```
bash examples/image_gen/diffuser_unconditional/train.sh --dataset_name uoft-cs/cifar10 --resolution 32
```

And if you want to log the training to wandb, you can do:

```
bash examples/image_gen/diffuser_unconditional/train.sh --logger wandb
```

## TODO

- calculate FID

## Reference

Adapted from https://github.com/huggingface/diffusers/blob/main/examples/unconditional_image_generation/train_unconditional.py