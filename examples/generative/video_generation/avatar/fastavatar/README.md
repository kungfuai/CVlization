# FastAvatar: Instant 3D Face Reconstruction from Single Image

Feed-forward 3D Gaussian Splatting for instant, pose-invariant face reconstruction from a single unconstrained image.

## Overview

FastAvatar achieves real-time 3D face reconstruction from a single image using a feed-forward approach (no test-time optimization required). The system combines:

- **Encoder**: Face embedding → latent W vector (using InsightFace)
- **DINO Model**: Image → 3D point cloud geometric guidance
- **Decoder**: W vector + 3D points → 3D Gaussian splats

**Key Features:**
- Instant reconstruction (feed-forward, no optimization)
- Pose-invariant (works from any viewing angle)
- Real-time performance (~1 second per image)
- Good 3D geometry reconstruction

**Important Note:**
This implementation uses **feedforward-only mode** for speed. Novel view quality (especially side views) is limited. For high-quality novel view synthesis, the original FastAvatar paper uses test-time optimization with multi-view data. See **Limitations** section for details.

## Quick Start

### 1. Build Docker Image

```bash
docker build -t fastavatar:latest .
```

Build time: ~10-15 minutes

### 2. Run Inference

```bash
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  -v $HOME/.cache/fastavatar:$HOME/.cache/fastavatar \
  fastavatar:latest \
  python predict.py --image data/example.png --output_dir results
```

**First run**: Downloads pretrained weights (~1.5GB) from Google Drive to `~/.cache/fastavatar/`

**Output**:
- `results/splats.ply` - 3D Gaussian point cloud
- `results/w_vector.npy` - Predicted latent vector
- `results/dino_points.npy` - DINO 3D point predictions
- `results/summary.json` - Reconstruction statistics

### 3. Visualize Results

Upload `results/splats.ply` to [Supersplat Editor](https://superspl.at/editor) for interactive 3D visualization.

## Usage

### Basic Inference

```bash
python predict.py --image path/to/face.jpg
```

### Custom Output Directory

```bash
python predict.py --image path/to/face.jpg --output_dir my_results
```

### CPU Inference

```bash
python predict.py --image path/to/face.jpg --device cpu
```

### Using Pre-downloaded Weights

```bash
python predict.py \
  --image path/to/face.jpg \
  --encoder_checkpoint path/to/encoder.pth \
  --decoder_checkpoint path/to/decoder.pth \
  --dino_checkpoint path/to/dino_encoder.pth
```

## Model Architecture

### Encoder (View-Invariant)
- **Input**: 512-dim InsightFace embedding
- **Output**: W vector (latent code)
- **Purpose**: Map face identity to generative latent space

### DINO Encoder
- **Input**: Face image (224x224)
- **Output**: 3D point cloud (~10K points)
- **Purpose**: Provide geometric initialization for Gaussian means

### Decoder (Conditional GS Generator)
- **Input**: W vector + 3D points
- **Output**: 3D Gaussian parameters (means, scales, rotations, colors, opacities)
- **Purpose**: Generate high-quality 3D representation

## Pretrained Weights

Weights are automatically downloaded from Google Drive on first run:
- **encoder_neutral_flame.pth** (~100MB) - Face embedding encoder
- **decoder_neutral_flame.pth** (~500MB) - Gaussian splat decoder
- **dino_encoder.pth** (~800MB) - DINOv2-based 3D point predictor
- **averaged_model.ply** (~50MB) - Mean face shape for initialization

Total size: ~1.5GB

Cache location: `~/.cache/fastavatar/pretrained_weights/`

## Input Requirements

- **Image format**: JPG, PNG
- **Face constraints**:
  - Single visible face
  - Reasonably well-lit
  - Works with various poses (not limited to frontal)
- **Resolution**: Automatically resized internally

## Performance

- **Inference time**: ~0.5-1 second per image (GPU)
- **GPU memory**: ~4-6GB
- **Output Gaussians**: Typically 10,000-15,000 points

## Limitations

### Novel View Quality (Important)

This implementation uses **feedforward-only mode** (`no_guidance`), which has inherent quality limitations for novel view synthesis:

**What works well:**
- 3D geometry reconstruction (point positions, scales, rotations)
- Frontal/near-frontal view rendering
- Fast preview generation
- Basic 3D structure

**Known limitations:**
- Novel views (especially profiles/side views) may have poor quality:
  - Inconsistent colors across viewing angles
  - Dark or washed-out appearance from non-frontal views
  - May not accurately match input person's appearance
- This is **by design**, not a bug

**Root cause:**
- Feedforward mode outputs unconstrained spherical harmonic (SH) coefficients
- No photometric regularization during single-image inference
- SH values can be extreme (range: ±100+), causing view-dependent artifacts

**For high-quality novel views:**
The original FastAvatar paper achieves high quality using **test-time optimization** (`full_guidance` mode):
- Requires: Multi-view images (8-16 views) + FLAME parameters + camera poses
- Process: 400-800 optimization iterations with photometric loss
- Time: 5-10 minutes per subject
- Result: SH coefficients regularized, consistent appearance across all views

This implementation prioritizes **speed over quality** - instant reconstruction without test-time optimization. For production use requiring high-quality novel view synthesis, you would need to:

1. Capture multi-view images with known camera poses
2. Extract FLAME parameters for each view
3. Implement test-time optimization loop (see `FINDINGS.md` for details)

See `FINDINGS.md` for detailed technical analysis and comparison with ground truth multi-view data.

## Docker Configuration

The Dockerfile uses:
- Base: PyTorch 2.1.0 + CUDA 12.1 + cuDNN 8
- Key dependencies: gsplat, insightface, mediapipe, LPIPS
- Automatic model weight caching

## Troubleshooting

### Out of Memory

Reduce batch size or use CPU:
```bash
python predict.py --image face.jpg --device cpu
```

### Download Fails

Manually download weights from [Google Drive](https://drive.google.com/file/d/1_XPTo_1rgzxvGQcRI7Toa3iGagytPTjK/view) and extract to `~/.cache/fastavatar/pretrained_weights/`

### No Face Detected

Ensure input image contains a clearly visible face. The model uses InsightFace for face detection.

## Dataset Bias Disclaimer

This model was trained on datasets that may not fully represent the diversity of real-world populations. Performance may vary across different demographic groups (age, gender, ethnicity). Users should interpret results with care and consider retraining on more balanced datasets for production use.

## Citation

```bibtex
@article{liang2025fastavatar,
  title={FastAvatar: Instant 3D Gaussian Splatting for Faces from Single Unconstrained Poses},
  author={Liang, Hao and Ge, Zhixuan and Tiwari, Ashish and Majee, Soumendu and Godaliyadda, GM and Veeraraghavan, Ashok and Balakrishnan, Guha},
  journal={arXiv preprint arXiv:2508.18389},
  year={2025}
}
```

## References

- **Paper**: [FastAvatar: Instant 3D Gaussian Splatting](https://arxiv.org/pdf/2508.18389)
- **Official Repo**: [hliang2/FastAvatar](https://github.com/hliang2/FastAvatar)
- **Website**: [hliang2.github.io/FastAvatar](https://hliang2.github.io/FastAvatar/)
- **3D Gaussian Splatting**: [gsplat](https://github.com/nerfstudio-project/gsplat)
- **Face Recognition**: [InsightFace](https://github.com/deepinsight/insightface)
- **Vision Transformer**: [DINOv2](https://github.com/facebookresearch/dinov2)

## License

This example follows the FastAvatar repository license (MIT). Please ensure appropriate rights and consent when using face data.
