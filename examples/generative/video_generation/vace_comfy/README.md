# Note

Under construction. This doesn't quite work yet.

# VACE ComfyUI Implementation

This folder contains the VACE (Video Animation Control Extension) implementation for ComfyUI, adapted from the WAN 2.1 workflow.

## Structure

### Core Files
- **`nodes.py`** - Essential ComfyUI core nodes (copied from wan_comfy) - **REQUIRED**
- **`nodes_vace.py`** - VACE-specific nodes including `WanVaceToVideo` and `CreateFadeMaskAdvanced`
- **`nodes_model_advanced.py`** - Advanced model sampling nodes for VACE
- **`nodes_images.py`** - Image processing and batch utility nodes  
- **`nodes_extra.py`** - Additional nodes for video processing and missing dependencies
- **`predict.py`** - Main execution script implementing the VACE workflow

### GGUF Support Files
- **`comfyui_gguf.py`** - Main GGUF integration (based on [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF))
- **`comfyui_gguf_*.py`** - Supporting GGUF modules for quantized model loading

### Workflow JSON
- **`workflows/Wan2.1_VACE_4images.json`** - Original ComfyUI workflow definition

## Installation

### Prerequisites
This implementation requires the ComfyUI-GGUF plugin functionality for loading quantized GGUF models. The required GGUF files are already integrated.

### Download Tokenizers
```bash
# From the video_generation directory
bash download_tokenizers.sh
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

**Important**: The `gguf>=0.13.0` package is **required** for loading the quantized WAN 2.1 VACE model (`Wan2.1-VACE-14B-Q6_K.gguf`).

## Key Nodes

### VACE-Specific Nodes
- **`WanVaceToVideo`** - Core VACE node for video conditioning and latent preparation
- **`CreateFadeMaskAdvanced`** - Advanced fade mask creation with interpolation

### Model Processing Nodes  
- **`ModelSamplingSD3`** - Model sampling configuration
- **`CFGZeroStar`** - CFG Zero Star implementation  
- **`UNetTemporalAttentionMultiply`** - Temporal attention scaling
- **`SkipLayerGuidanceDiT`** - Skip layer guidance for DiT models

### GGUF Nodes
- **`UnetLoaderGGUF`** - Loads quantized GGUF models (from ComfyUI-GGUF)

### Video Utility Nodes
- **`VHS_VideoCombine`** - Video output generation
- **`GetImageSize+`** - Enhanced image size detection

## Usage

Run the main prediction script:
```bash
python predict.py -p "your prompt" -i path/to/image1.jpg path/to/image2.jpg
```

### Model Paths
By default, the script expects models in `/root/.cache/models/wan/`. You can change this with:
```bash
python predict.py --models_root_dir /path/to/your/models
```

### Required Models
- `Wan2.1-VACE-14B-Q6_K.gguf` - Quantized VACE UNET model
- `umt5_xxl_fp8_e4m3fn_scaled.safetensors` - Text encoder
- `wan_2.1_vae.safetensors` - VAE decoder

## GGUF Integration

This implementation includes a full integration of [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) by city96. The GGUF support enables:

- **Quantized Model Loading**: Load GGUF quantized models for reduced VRAM usage
- **Efficient Inference**: Optimized operations for quantized weights
- **Memory Management**: Smart memory handling for large models

### GGUF Benefits
- **Reduced VRAM**: Quantized models use significantly less GPU memory
- **Faster Loading**: Quantized models load faster than full precision
- **Quality Preservation**: Q6_K quantization maintains high quality for transformer/DiT models

## Dependencies

The implementation requires all core ComfyUI nodes from `nodes.py` plus the additional VACE-specific nodes. The `nodes.py` file contains essential classes like:
- `CLIPTextEncode`, `CLIPLoader`, `VAELoader`, `VAEDecode`
- `KSampler`, `EmptyLatentImage`, `LoadImage`, `SaveImage`
- And many other fundamental ComfyUI nodes

Without `nodes.py`, the `predict.py` script will fail with import errors since it relies on `NODE_CLASS_MAPPINGS` from the core nodes module.

## Credits

- **ComfyUI-GGUF**: [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) (Apache-2.0 License)
- **VACE Extension**: [VACE Extension is the next level beyond FLF2V](https://www.reddit.com/r/StableDiffusion/comments/1kqw177/vace_extension_is_the_next_level_beyond_flf2v/)

## Reference
- [VACE Extension is the next level beyond FLF2V](https://www.reddit.com/r/StableDiffusion/comments/1kqw177/vace_extension_is_the_next_level_beyond_flf2v/)
