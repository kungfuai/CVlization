#!/usr/bin/env python3

"""
Example script demonstrating how to use the video extension functionality.

This script shows how to:
1. Load the required models
2. Load an existing video or frames
3. Extend the video with new generated content
"""

import os
import sys
import torch
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.memory import get_cuda_free_memory_gb

# Import our new video extension module
from video_extension import extend_video_frames, create_models_dict, load_video_frames

# Set HF cache directory
os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

def load_models():
    """Load all required models for video extension."""
    print("Loading models...")
    
    # Check VRAM
    free_mem_gb = get_cuda_free_memory_gb(0)  # Assuming GPU 0
    high_vram = free_mem_gb > 60
    print(f'Free VRAM {free_mem_gb} GB')
    print(f'High-VRAM Mode: {high_vram}')
    
    # Load models
    text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
    text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
    tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
    tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
    vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()
    
    feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
    image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()
    
    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()
    
    # Set models to eval mode
    vae.eval()
    text_encoder.eval()
    text_encoder_2.eval()
    image_encoder.eval()
    transformer.eval()
    
    # Configure models
    if not high_vram:
        vae.enable_slicing()
        vae.enable_tiling()
    
    transformer.high_quality_fp32_output_for_inference = True
    
    # Set dtypes
    transformer.to(dtype=torch.bfloat16)
    vae.to(dtype=torch.float16)
    image_encoder.to(dtype=torch.float16)
    text_encoder.to(dtype=torch.float16)
    text_encoder_2.to(dtype=torch.float16)
    
    # Disable gradients
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    
    if high_vram:
        # Move to GPU if high VRAM
        text_encoder.to('cuda')
        text_encoder_2.to('cuda')
        image_encoder.to('cuda')
        vae.to('cuda')
        transformer.to('cuda')
    
    models = create_models_dict(
        text_encoder, text_encoder_2, tokenizer, tokenizer_2,
        vae, transformer, feature_extractor, image_encoder
    )
    
    print("Models loaded successfully!")
    return models, high_vram


def example_extend_video():
    """Example of extending a video."""
    
    # Configuration
    video_path = "./data/example_video.mp4"  # Path to your existing video
    prompt = "The character continues dancing gracefully with flowing movements"
    extend_seconds = 3.0
    seed = 42
    
    # Load models
    models, high_vram = load_models()
    
    # Check if video file exists, otherwise provide instructions
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        print("Please provide a video file to extend, or use load_video_frames() with frame arrays.")
        print("For testing, you can:")
        print("1. Place a video file at the specified path")
        print("2. Or modify the video_path variable to point to your video")
        print("3. Or use the load_video_frames() function to load frames programmatically")
        return
    
    def progress_callback(percentage, message):
        print(f"[{percentage:3d}%] {message}")
    
    print(f"\nStarting video extension...")
    print(f"Input video: {video_path}")
    print(f"Prompt: {prompt}")
    print(f"Extension: {extend_seconds} seconds")
    print(f"Seed: {seed}")
    print("-" * 50)
    
    try:
        # Extend the video
        output_path = extend_video_frames(
            existing_frames=video_path,
            prompt=prompt,
            models=models,
            extend_seconds=extend_seconds,
            seed=seed,
            steps=25,
            use_teacache=True,
            output_dir="./outputs",
            high_vram=high_vram,
            progress_callback=progress_callback
        )
        
        print(f"\n✓ Video extension completed!")
        print(f"Output saved to: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Error during video extension: {e}")
        import traceback
        traceback.print_exc()


def example_extend_from_frames():
    """Example of extending video from frame arrays."""
    
    # This example shows how to use the function with frame arrays instead of a video file
    print("Example: Extending video from frame arrays")
    
    # Load models
    models, high_vram = load_models()
    
    # Example: Load frames from a video file first
    video_path = "./data/example_video.mp4"
    if os.path.exists(video_path):
        frames = load_video_frames(video_path, max_frames=30)  # Load first 30 frames
        
        print(f"Loaded {len(frames)} frames from {video_path}")
        print(f"Frame shape: {frames[0].shape}")
        
        # Now extend these frames
        output_path = extend_video_frames(
            existing_frames=frames,  # Pass frames directly
            prompt="The action continues with smooth transitions",
            models=models,
            extend_seconds=2.0,
            seed=123,
            high_vram=high_vram,
            output_dir="./outputs"
        )
        
        print(f"Extended video saved to: {output_path}")
    else:
        print(f"Video file not found: {video_path}")
        print("Please provide a video file for this example.")


if __name__ == "__main__":
    print("FramePack Video Extension Example")
    print("=" * 40)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "frames":
            example_extend_from_frames()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python example_video_extension.py [frames]")
    else:
        example_extend_video() 