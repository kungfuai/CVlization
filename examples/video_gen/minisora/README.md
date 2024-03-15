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
bash examples/video_gen/videogpt/build.sh

# train vae model
bash examples/video_gen/videogpt/train.sh python train_vqvae.py --batch_size 2 --resolution 256 --sequence_length 4 --embedding_dim 4 --n_codes 5120 --limit_train_batches 1.0 --limit_val_batches 0.25 --epochs 100 --save_every_n_epochs 5 --low_utilization_cost 0.1 --network_variant vae_s4t4_b_vq --lr 0.001 --kl_loss_weight 0.001 --commitment_cost 0.25 --track
# --reinit_every_n_epochs 1
# --accumulate_grad_batches 1

# train diffusion model
bash examples/video_gen/videogpt/train.sh python train_dit.py --model "Latte-T/2" --vae_model "zzsi_kungfu/videogpt/model-tjzu02pg:v17" --batch_size 32 --resolution 256 --sequence_length 4 --latent_input_size 64 --ae_temporal_stride 4 --ae_spatial_stride 4 --learn_sigma --ckpt_every 1000000 --sample_every 50 --log_every 20 --epochs 100 --track


```


## Reference

Adapted from https://github.com/wilson1yan/VideoGPT