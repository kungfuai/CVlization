## Quickstart

```bash
bash examples/video_gen/videogpt/build.sh
bash examples/video_gen/videogpt/train.sh python train_vqvae.py --batch_size 1 --resolution 128 --limit_train_batches 0.2 --limit_val_batches 1 --epochs 10 --save_every_n_epochs 1 --track
# train diffusion model
```


## Reference

Adapted from https://github.com/wilson1yan/VideoGPT