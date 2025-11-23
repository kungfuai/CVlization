# Data Directory

Place your input images in the `images/` subdirectory.

## Directory Structure

```
data/
└── images/
    ├── 02.png       # Example image (Berlin scene from official HF Space)
    └── ...          # Add your own images here
```

## Supported Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- Other common image formats

## Recommendations

- **Image count**: 2-50 images recommended (model supports up to 2000)
- **Resolution**: Any resolution (model adapts automatically)
- **Overlap**: Adjacent images should have 30-70% overlap for best results
- **Lighting**: Consistent lighting across views works best
- **Object coverage**: Multiple views from different angles

## Example Datasets

You can test with these public datasets:
- [DTU MVS Dataset](https://roboimagedata.compute.dtu.dk/)
- [Tanks and Temples](https://www.tanksandtemples.org/)
- [MipNeRF360](https://jonbarron.info/mipnerf360/)

Or simply take photos of an object/scene from multiple angles with your phone!
