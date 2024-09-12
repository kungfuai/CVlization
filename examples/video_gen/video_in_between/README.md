## Given 2 keyframes, fill in the frames in between



Adapted from this [paper in Aug 2024](https://arxiv.org/abs/2408.15239). The python dependencies are modified from the original work (see `Dockerfile`).

```bash
bash examples/video_gen/video_in_between/build.sh
```

Download the model files:
```bash
cd examples/video_gen/video_in_between
pip install gdown
gdown 1H7vgiNVbxSeeleyJOqhoyRbJ97kGWGOK --folder
```

Now change directory back to the root of the repo and run the following command:

```bash
# Feel free to edit the `predict.sh` script to change the input keyframes and parameters
bash examples/video_gen/video_in_between/predict.sh
```

On a 3090 GPU, it takes about 40 minutes using 22.4GB of VRAM, with 50 sampling (denoising) steps and a `decode_chunk_size` of 4. A result `.gif` file will be generated in `results/`.