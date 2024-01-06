## Quickstart

In project root directory, run:

```
bash examples/image_gen/uva_energy/build.sh
bash examples/image_gen/uva_energy/train.sh
```

The training outputs will be saved to a tensorboard directory in `logs/uva_energy`.

To use weights and biases to track the experiment, do

```
bash examples/image_gen/uva_energy/train.sh --logger wandb
```

To use a huggingface dataset (e.g. `huggan/flowers-102-categories`) do:

```
bash examples/image_gen/uva_energy/train.sh --logger wandb --dataset huggan/flowers-102-categories --image_shape 3,64,64 --epochs 100 --batch_size 32
```