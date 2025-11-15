# BFM Model Download Utility

This module provides automatic downloading and caching of the Basel Face Model (BFM) 2009 from HuggingFace.

## Quick Start

### Download BFM Model

```bash
# From project root
python -m data_utils.bfm_model.download_bfm

# Check if model is cached
python -m data_utils.bfm_model.download_bfm --check
```

### Use in Python

```python
from data_utils.bfm_model import download_bfm_model, get_bfm_model_path

# Download model (if not cached) and get path
bfm_dir = download_bfm_model()

# Just get path (will auto-download if missing)
bfm_dir = get_bfm_model_path()
```

## Cache Location

Models are cached in: `~/.cache/egstalker/bfm/`

This can be overridden with the `EGSTALKER_CACHE` environment variable:

```bash
export EGSTALKER_CACHE=/path/to/cache
```

## Files Downloaded

### Core Model
- `01_MorphableModel.mat` (53MB) - Main BFM 2009 morphable model

### Supporting Files (from SadTalker)
- `BFM09_model_info.mat` - Model metadata
- `BFM_exp_idx.mat` - Expression indices
- `BFM_front_idx.mat` - Front-face indices
- `Exp_Pca.bin` - Expression PCA data
- `facemodel_info.mat` - Face model info
- `similarity_Lm3D_all.mat` - Landmark similarity data
- `std_exp.txt` - Standard expression data

### Preprocessing Files (from AD-NeRF)
- `exp_info.npy` (32MB) - Expression model parameters (mu_exp, base_exp, sig_exp)
- `keys_info.npy` (7KB) - Keypoint indices for 68-point facial landmarks
- `topology_info.npy` (4MB) - Face mesh topology and sub-indices

## Docker Integration

The cache directory can be mounted from the host for persistence:

```bash
docker run --rm --gpus all \
  -v $(pwd):/workspace/host \
  -v ~/.cache/egstalker:~/.cache/egstalker \
  -w /workspace/host \
  egstalker:bfm \
  python data_utils/process.py video.mp4 --tracker bfm
```

## Source

Models and preprocessing files are downloaded from:
- **SadTalker HuggingFace repository**: https://huggingface.co/wsj1995/sadTalker (BFM model and supporting files)
- **AD-NeRF GitHub repository**: https://github.com/YudongGuo/AD-NeRF (preprocessing .npy files)

This provides a convenient alternative to manual registration at the Basel Face website.

## License

The BFM 2009 model is subject to the Basel Face Model license. By downloading and using this model, you agree to the terms at: https://faces.dmi.unibas.ch/bfm/

The download utility code is part of the CVlization project and follows its license.
