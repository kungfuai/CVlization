## Quickstart

Download the model weights:

```bash
bash download_models.sh
```

Build the docker image:

```bash
bash examples/video_gen/animate_x/build.sh
```

Extract the pose from example video(s):

```bash
bash examples/video_gen/animate_x/extract_pose.sh
```

Animate an image using example videos:

```bash
bash examples/video_gen/animate_x/predict.sh
```

Edit the `Animate_X_infer.yaml`  file to customize the generation. You can specify one or more (image, video) prompts, change the video resolution (aspect ratio) and other parameters.



