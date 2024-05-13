## Overview

Train a Sora-like model, but with small scope, high quality and low cost.

## Quickstart

### Download data

Dataload the Flying MNIST dataset with [11k videos](https://storage.googleapis.com/research-datasets-public/minisora/flying_mnist_11k.tar.gz) and extract to `<CVlization project root>/data/`.


```bash
mkdir -p data
cd data
wget https://storage.googleapis.com/research-datasets-public/minisora/flying_mnist_11k.tar.gz
tar -xzf flying_mnist_11k.tar.gz
rm flying_mnist_11k.tar.gz
cd ..
```

There is also larger dataset with [110k videos](https://storage.googleapis.com/research-datasets-public/minisora/flying_mnist_110k.tar.gz).

The directory structure should be:

```
data/
    flying_mnist/
        train/00000.mp4
              ...
        val/00000.mp4
            ...
```

### Build a docker image

```bash
bash examples/video_gen/minisora/build.sh
```

### Train a spatial-temporal VQ-VAE

You can skip this step if you want to use a pretrained VAE.

```bash
# train vae model
bash examples/video_gen/minisora/train.sh python train_vqvae.py --batch_size 2 --resolution 256 --sequence_length 32 --embedding_dim 4 --n_codes 5120 --limit_train_batches 1.0 --limit_val_batches 0.25 --epochs 150 --save_every_n_epochs 5 --low_utilization_cost 0.1 --network_variant s8_b --lr 0.001 --kl_loss_weight 0.01 --commitment_cost 0.25 --track
# --reinit_every_n_epochs 1
# --accumulate_grad_batches 1
```

To train a VAE without quantization (more suitable for diffusion):
```bash
bash examples/video_gen/minisora/train.sh python train_vqvae.py --batch_size 2 --resolution 256 --sequence_length 32 --embedding_dim 4 --n_codes 5120 --limit_train_batches 1.0 --limit_val_batches 0.25 --epochs 150 --save_every_n_epochs 5 --low_utilization_cost 0.1 --network_variant vae_s4t4_b --lr 0.001 --kl_loss_weight 0.1 --commitment_cost 0.25 --track
```

### Use VAE to extract latents or tokenize videos

The following script uses a pretrained VAE to extract latents or tokenize videos. An WANDB API key is needed. If you want to use your own VAE, please adapt the script.

```bash
# extract latents from the video using vae
bash examples/video_gen/minisora/train.sh python latents.py --dataset flying_mnist --batch_size 1 --vae stabilityai/sd-vae-ft-mse # zzsi_kungfu/videogpt/model-nilqq143:v14
```

```bash
# tokenize the video using vae
bash examples/video_gen/minisora/train.sh python tokenize_videos.py --dataset flying_mnist --batch_size 8
```

Some precomputed latents for your convenience:

```
https://storage.googleapis.com/research-datasets-public/minisora/data/latents/flying_mnist_11k__sd-vae-ft-mse_latents_32frames_train.npy (4.9GB)
https://storage.googleapis.com/research-datasets-public/minisora/data/latents/flying_mnist_11k__model-kbu39ped_tokens_32frames_train.npy (2.4GB)
```

### Train a latent generative model

Now VAE is trained and videos are tokenized. From this point, you have several options:

1. Train a diffusion model

Using DiT (adapted from PKU's OpenSora):
```bash
# train diffusion model
# With a VAE trained on flying MNIST:
bash examples/video_gen/minisora/train.sh python train_dit.py --model "Latte-S/2" --vae_model "zzsi_kungfu/videogpt/model-kbu39ped:v11" --batch_size 2 --num_clips_per_video 10 --lr 0.00002 --resolution 256 --sequence_length 4 --latent_input_size 64 --ae_temporal_stride 4 --ae_spatial_stride 4 --learn_sigma --ckpt_every 1000000 --sample_every 2000 --log_every 20 --epochs 100 --track

# or with a StablilityAI pretrained VAE:
bash examples/video_gen/minisora/train.sh python train_dit.py --model "Latte-T/2" --batch_size 2 --lr 0.00002 --resolution 256 --sequence_length 4 --latent_input_size 32 --ae_temporal_stride 1 --ae_spatial_stride 8 --learn_sigma --ckpt_every 1000000 --sample_every 5000 --log_every 100 --epochs 10000 --track
```

Using spatial temporal DiT (adatped from ColossalAI's OpenSora):

```bash
# This will use a VAE trained on Flying MNIST (VAE and latents files must match)
bash examples/video_gen/minisora/train.sh python iddpm.py --batch_size 4 --accumulate_grad_batches 8 --latent_frames_to_generate 8 --diffusion_steps 1000 --max_steps 1000000 --log_every 50 --sample_every 2000 --clip_grad 1.0 --vae zzsi_kungfu/videogpt/model-nilqq143:v14 --latents_input_file data/latents/flying_mnist__model-nilqq143_latents_32frames_train.npy --track

# or train a larger net
bash examples/video_gen/minisora/train.sh python iddpm.py --batch_size 1 --accumulate_grad_batches 32 --depth 16 --num_heads 12 --hidden_size 768 --max_steps 1000000 --log_every 50 --sample_every 2000 --diffusion_steps 1000 --clip_grad 1.0 --latent_frames_to_generate 8 --tokens_input_file data/latents/flying_mnist_110k__model-kbu39ped_tokens_32frames_train.npy --vae zzsi_kungfu/videogpt/model-kbu39ped:v11 --track

bash examples/video_gen/minisora/train.sh python iddpm.py --batch_size 1 --accumulate_grad_batches 32 --depth 16 --num_heads 12 --hidden_size 768 --max_steps 1000000 --log_every 500 --sample_every 5000 --checkpoint_every 500000 --diffusion_steps 1000 --clip_grad 1.0 --latent_frames_to_generate 32 --latents_input_file data/latents/flying_mnist_110k__sd-vae-ft-mse_latents_32frames_train.npy --vae stabilityai/sd-vae-ft-mse --resume_from zzsi_kungfu/flying_mnist/denoiser_model:v29 --track


# or train with stablediffusion VAE
bash examples/video_gen/minisora/train.sh python iddpm.py --batch_size 1 --accumulate_grad_batches 32 --depth 16 --num_heads 12 --hidden_size 768 --max_steps 1000000 --log_every 50 --sample_every 2000 --diffusion_steps 1000 --clip_grad 1.0 --latent_frames_to_generate 32 --latents_input_file data/latents/flying_mnist__sd-vae-ft-mse_latents_32frames_train.npy --vae stabilityai/sd-vae-ft-mse --track
```

2. Train an autoregressive transformer-based language model (next token predictor)

```bash
# Instead of training a diffusion model, one can also train a next token predictor.
bash examples/video_gen/minisora/train.sh python train_latent_nanogpt.py --block_size 512 --tokens_input_file data/latents/flying_mnist__model-kbu39ped_tokens_32frames_train.npy --sample_interval 2000 --batch_size 8 --gradient_accumulation_steps 4 --max_iters 10000000 --wandb_log
```

A variant of GPT (under development):

```bash
# Instead of training a diffusion model, one can also train a next token predictor.
bash examples/video_gen/minisora/train.sh python train_latent_mdgpt.py --block_size 512 --sparse_block_size 512 --sample_interval 1000 --num_latent_frames 8 --batch_size 8 --gradient_accumulation_steps 4 --use_1d_pos_emb --max_iters 100000000 --only_predict_last --wandb_log
```

3. Train a autoregressive MAMBA-based language model

(for now, quality is low)

```
bash examples/video_gen/minisora/train.sh python train_latent_mamba.py --n_layer 32 --n_embed 1280 --batch_size 8 --gradient_accumulation_steps 1 --block_size 512 --track
```

## Reference

- [Nanogpt](https://github.com/karpathy/nanoGPT)
- https://github.com/PKU-YuanGroup/Open-Sora-Plan
- https://github.com/hpcaitech/Open-Sora
- https://github.com/wilson1yan/VideoGPT
- https://github.com/lucidrains/magvit2-pytorch