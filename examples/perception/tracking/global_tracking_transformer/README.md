# Global Tracking Transformer

Adapted from [https://github.com/xingyizhou/GTR](https://github.com/xingyizhou/GTR).

## Quick Start

Build the Docker image (downloads model and example video automatically):

```bash
cvl run global_tracking_transformer build
```

Run prediction on the example video:

```bash
cvl run global_tracking_transformer predict
```

## Manual Setup (Alternative)

If not using CVL, download the model and example video manually:

```bash
# Download pretrained model
mkdir -p models
gdown -O models/gtr_model.pth https://drive.google.com/uc?id=1TqkLpFZvOMY5HTTaAWz25RxtLHdzQ-CD

# Download example video
mkdir -p data
gdown -O data/soccer1_clip1.mp4 https://drive.google.com/uc?id=18E9eRgZBaYlH_O6gB2Evrv96swa8MabU

# Build and run
bash build.sh
bash predict.sh
```
