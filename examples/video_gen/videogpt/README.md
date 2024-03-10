## Quickstart

```bash
bash examples/video_gen/videogpt/build.sh
bash examples/video_gen/videogpt/train.sh python train_vqvae.py --batch_size 1 --resolution 128 --sequence_length 16 --embedding_dim 128 --n_codes 1200 --limit_train_batches 1.0 --limit_val_batches 0.1 --epochs 30 --save_every_n_epochs 1  --low_utilization_cost 0 --reinit_every_n_epochs 1 --accumulate_grad_batches 16 --network_variant s4t4_b --track
# --reinit_every_n_epochs 1
# train diffusion model
```


## Reference

Adapted from https://github.com/wilson1yan/VideoGPT