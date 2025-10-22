https://github.com/sail-sg/MDT

To train on the flying MNIST frames, download the image latents:

```
mkdir -p data/latents
cd data/latents

wget https://storage.googleapis.com/research-datasets-public/minisora/data/latents/flying_mnist_11k__sd-vae-ft-mse_latents_32frames_train.npy

cd ../../
```

Then,

```
bash examples/image_gen/mdt/build.sh
# To train on flying mnist frames:
# The --track option will track the experiment in wandb.
bash examples/image_gen/mdt/train.sh --track
```
