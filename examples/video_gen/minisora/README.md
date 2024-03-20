## Quickstart

Dataload the Flying MNIST dataset: https://storage.googleapis.com/research-datasets-public/flying_mnist.tar.gz and extract to `<CVlization project root>/data/`. The directory structure should be:

```
data/
    flying_mnist/
        train/00000.mp4
              ...
        val/00000.mp4
            ...
```

Then

```bash
bash examples/video_gen/minisora/build.sh

# train vae model
bash examples/video_gen/minisora/train.sh python train_vqvae.py --batch_size 2 --resolution 256 --sequence_length 32 --embedding_dim 4 --n_codes 5120 --limit_train_batches 1.0 --limit_val_batches 0.25 --epochs 100 --save_every_n_epochs 5 --low_utilization_cost 0.1 --network_variant s4t4_b_vq --lr 0.001 --kl_loss_weight 0.01 --commitment_cost 0.25 --track
# --reinit_every_n_epochs 1
# --accumulate_grad_batches 1

# tokenze the video using vae
bash examples/video_gen/minisora/train.sh python tokenize_videos.py
```

Now VAE is trained and videos are tokenized. From this point, you have several options:

1. Train a diffusion model

Using DiT (adapted from PKU's OpenSora):
```
# train diffusion model
# With a VAE trained on flying MNIST:
bash examples/video_gen/minisora/train.sh python train_dit.py --model "Latte-S/2" --vae_model "zzsi_kungfu/videogpt/model-kbu39ped:v11" --batch_size 2 --num_clips_per_video 10 --lr 0.00002 --resolution 256 --sequence_length 4 --latent_input_size 64 --ae_temporal_stride 4 --ae_spatial_stride 4 --learn_sigma --ckpt_every 1000000 --sample_every 2000 --log_every 20 --epochs 100 --track

# or with a StablilityAI pretrained VAE:
bash examples/video_gen/minisora/train.sh python train_dit.py --model "Latte-T/2" --batch_size 2 --lr 0.00002 --resolution 256 --sequence_length 4 --latent_input_size 32 --ae_temporal_stride 1 --ae_spatial_stride 8 --learn_sigma --ckpt_every 1000000 --sample_every 100 --log_every 20 --epochs 100 --track
```

Using spatial temporal DiT (adatped from ColossalAI's OpenSora):

```
# This will use a VAE trained on Flying MNIST
bash examples/video_gen/minisora/train.sh python iddpm.py
```

2. Train an autoregressive transformer-based language model (next token predictor)

```
# Instead of training a diffusion model, one can also train a next token predictor.
bash examples/video_gen/minisora/train.sh python train_latent_nanogpt.py
```

3. Train a autoregressive MAMBA-based language model

4. Train a flow-matching model

## Reference

- [Nanogpt](https://github.com/karpathy/nanoGPT)
- https://github.com/PKU-YuanGroup/Open-Sora-Plan
- https://github.com/hpcaitech/Open-Sora
- https://github.com/wilson1yan/VideoGPT