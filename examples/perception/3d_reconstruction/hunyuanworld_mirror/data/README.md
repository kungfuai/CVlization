# Data Directory

Place your input images in the `images/` subdirectory.

## Included Example

The directory includes the "Desk" scene from the HunyuanWorld-Mirror repository:
- 2 images of a desk scene (multi-view)
- From: https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror/tree/main/examples/realistic/Desk

## Adding Your Own Images

```
data/
└── images/
    ├── *.jpg        # Your images here
    └── ...
```

## Supported Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- WebP (.webp)

## Recommendations

- **Image count**: 2-30 images recommended (model supports more)
- **Resolution**: Any resolution (resized to 518px by default)
- **Overlap**: Adjacent images should have 30-70% overlap for best results
- **Multi-view**: Better results with multiple viewpoints of the same scene
- **Video**: Also supports video input (.mp4, .avi, .mov, .webm, .gif)

## Example Datasets

Public datasets for testing:
- DTU MVS Dataset: https://roboimagedata.compute.dtu.dk/
- Tanks and Temples: https://www.tanksandtemples.org/
- MipNeRF360: https://jonbarron.info/mipnerf360/
