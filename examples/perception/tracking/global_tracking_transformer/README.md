# Global Tracking Transformer

Adapted from [https://github.com/xingyizhou/GTR](https://github.com/xingyizhou/GTR).

Download an [example video](https://drive.google.com/file/d/18E9eRgZBaYlH_O6gB2Evrv96swa8MabU/view?usp=sharing):

```bash
# at the root of the repository
gdown -O data/soccer1_clip1.mp4 https://drive.google.com/uc?id=18E9eRgZBaYlH_O6gB2Evrv96swa8MabU
```

Build the docker image:

```bash
bash examples/object_tracking/global_tracking_transformer/build.sh
```

Download the pretrained model:

```bash
cd examples/object_tracking/global_tracking_transformer
mkdir -p models
gdown -O models/ https://drive.google.com/uc?id=1TqkLpFZvOMY5HTTaAWz25RxtLHdzQ-CD
```

Run the docker container:

```bash
# at the root of the repository
bash examples/object_tracking/global_tracking_transformer/predict.sh
```
