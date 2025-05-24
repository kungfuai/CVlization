#!/usr/bin/env python3

import os
import cv2
import torch
import einops
import numpy as np
from PIL import Image
from typing import List, Union, Optional, Tuple

# Import necessary components from diffusers_helper
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode
from diffusers_helper.utils import (
    save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, 
    resize_and_center_crop, generate_timestamp
)
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import (
    cpu, gpu, unload_complete_models, load_model_as_complete,
    move_model_to_device_with_memory_preservation, 
    offload_model_from_device_for_memory_preservation
)
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket


def load_video_frames(video_path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """
    Load video frames from a video file.
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to load (None for all frames)
    
    Returns:
        List of frames as numpy arrays
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        frame_count += 1
        
        if max_frames and frame_count >= max_frames:
            break
    
    cap.release()
    return frames


def prepare_frames_for_encoding(frames: List[np.ndarray], target_height: int, target_width: int) -> torch.Tensor:
    """
    Prepare video frames for VAE encoding by resizing and converting to tensor format.
    
    Args:
        frames: List of frames as numpy arrays
        target_height: Target height for resizing
        target_width: Target width for resizing
    
    Returns:
        Tensor in format (1, C, T, H, W) ready for VAE encoding
    """
    processed_frames = []
    
    for frame in frames:
        # Resize and center crop
        processed_frame = resize_and_center_crop(frame, target_width, target_height)
        
        # Convert to tensor and normalize to [-1, 1]
        frame_tensor = torch.from_numpy(processed_frame).float() / 127.5 - 1
        frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC -> CHW
        processed_frames.append(frame_tensor)
    
    # Stack frames: (C, H, W) -> (C, T, H, W)
    video_tensor = torch.stack(processed_frames, dim=1)
    
    # Add batch dimension: (C, T, H, W) -> (1, C, T, H, W)
    video_tensor = video_tensor.unsqueeze(0)
    
    return video_tensor


@torch.no_grad()
def extend_video_frames(
    existing_frames: Union[List[np.ndarray], str],
    prompt: str,
    models: dict,
    extend_seconds: float = 2.0,
    negative_prompt: str = "",
    seed: int = 31337,
    latent_window_size: int = 9,
    steps: int = 25,
    cfg_scale: float = 1.0,
    distilled_cfg_scale: float = 10.0,
    cfg_rescale: float = 0.0,
    gpu_memory_preservation: float = 6.0,
    use_teacache: bool = True,
    mp4_crf: int = 16,
    output_dir: str = "./outputs",
    high_vram: bool = False,
    progress_callback: Optional[callable] = None
) -> str:
    """
    Extend existing video frames with new generated content.
    
    Args:
        existing_frames: List of frame arrays or path to video file
        prompt: Text prompt for generation
        models: Dictionary containing the required models:
            - text_encoder: LlamaModel
            - text_encoder_2: CLIPTextModel  
            - tokenizer: LlamaTokenizerFast
            - tokenizer_2: CLIPTokenizer
            - vae: AutoencoderKLHunyuanVideo
            - transformer: HunyuanVideoTransformer3DModelPacked
            - feature_extractor: SiglipImageProcessor
            - image_encoder: SiglipVisionModel
        extend_seconds: How many seconds to extend the video
        negative_prompt: Negative prompt (currently not used when cfg_scale=1.0)
        seed: Random seed
        latent_window_size: Size of latent generation window
        steps: Number of diffusion steps
        cfg_scale: CFG scale (1.0 = no CFG)
        distilled_cfg_scale: Distilled guidance scale
        cfg_rescale: CFG rescale factor
        gpu_memory_preservation: GPU memory to preserve (GB)
        use_teacache: Whether to use TeaCache acceleration
        mp4_crf: MP4 compression quality (lower = better quality)
        output_dir: Output directory for generated videos
        high_vram: Whether system has high VRAM (>60GB)
        progress_callback: Optional callback for progress updates
    
    Returns:
        Path to the generated extended video file
    """
    # Extract models
    text_encoder = models['text_encoder']
    text_encoder_2 = models['text_encoder_2'] 
    tokenizer = models['tokenizer']
    tokenizer_2 = models['tokenizer_2']
    vae = models['vae']
    transformer = models['transformer']
    feature_extractor = models['feature_extractor']
    image_encoder = models['image_encoder']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    job_id = generate_timestamp()
    
    def update_progress(message: str, percentage: int = 0):
        if progress_callback:
            progress_callback(percentage, message)
        else:
            print(f"[{percentage:3d}%] {message}")
    
    update_progress("Loading and processing frames...", 0)
    
    # Load frames if path is provided
    if isinstance(existing_frames, str):
        existing_frames = load_video_frames(existing_frames)
    
    if not existing_frames:
        raise ValueError("No frames provided or loaded")
    
    # Determine target dimensions from first frame
    H, W = existing_frames[0].shape[:2]
    height, width = find_nearest_bucket(H, W, resolution=640)
    
    # Prepare frames for encoding
    frames_tensor = prepare_frames_for_encoding(existing_frames, height, width)
    
    # Calculate extension parameters
    total_latent_sections = int(max(round((extend_seconds * 30) / (latent_window_size * 4)), 1))
    
    update_progress("Encoding text prompt...", 10)
    
    # Clean GPU if needed
    if not high_vram:
        unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
    
    # Text encoding
    if not high_vram:
        load_model_as_complete(text_encoder, target_device=gpu)
        load_model_as_complete(text_encoder_2, target_device=gpu)
    
    llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
    
    if cfg_scale == 1:
        llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
    else:
        llama_vec_n, clip_l_pooler_n = encode_prompt_conds(negative_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
    
    llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
    llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
    
    update_progress("Encoding existing video frames...", 20)
    
    # VAE encoding of existing frames
    if not high_vram:
        load_model_as_complete(vae, target_device=gpu)
    
    existing_latents = vae_encode(frames_tensor, vae)
    
    update_progress("Processing last frame for image conditioning...", 30)
    
    # CLIP Vision encoding of last frame for conditioning
    if not high_vram:
        load_model_as_complete(image_encoder, target_device=gpu)
    
    last_frame = existing_frames[-1]
    last_frame_processed = resize_and_center_crop(last_frame, width, height)
    image_encoder_output = hf_clip_vision_encode(last_frame_processed, feature_extractor, image_encoder)
    image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
    
    # Convert to appropriate dtypes
    llama_vec = llama_vec.to(transformer.dtype)
    llama_vec_n = llama_vec_n.to(transformer.dtype)
    clip_l_pooler = clip_l_pooler.to(transformer.dtype)
    clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
    image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)
    
    update_progress("Starting video extension generation...", 40)
    
    # Initialize generation state with existing content
    rnd = torch.Generator("cpu").manual_seed(seed)
    num_frames = latent_window_size * 4 - 3
    
    # Initialize history with existing latents instead of zeros
    existing_latent_frames = existing_latents.shape[2]
    history_latents = torch.zeros(size=(1, 16, existing_latent_frames + 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
    history_latents[:, :, :existing_latent_frames, :, :] = existing_latents.cpu()
    
    # Initialize pixel history with existing frames
    history_pixels = vae_decode(existing_latents, vae).cpu()
    total_generated_latent_frames = existing_latent_frames
    
    # Set up latent padding sequence for extension
    latent_paddings = reversed(range(total_latent_sections))
    if total_latent_sections > 4:
        latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
    
    # Extension generation loop
    for section_idx, latent_padding in enumerate(latent_paddings):
        is_last_section = latent_padding == 0
        latent_padding_size = latent_padding * latent_window_size
        
        progress_pct = 40 + int(50 * section_idx / len(list(latent_paddings)))
        update_progress(f"Generating extension section {section_idx + 1}/{len(list(latent_paddings))}...", progress_pct)
        
        print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')
        
        # Set up indices for this section
        indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
        clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
        clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
        
        # Use last frame of existing content as start latent
        start_latent = existing_latents[:, :, -1:, :, :]
        clean_latents_pre = start_latent.to(history_latents)
        clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, -1-2-16:, :, :].split([1, 2, 16], dim=2)
        clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
        
        # Load transformer for generation
        if not high_vram:
            unload_complete_models()
            move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)
        
        if use_teacache:
            transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
        else:
            transformer.initialize_teacache(enable_teacache=False)
        
        def callback(d):
            current_step = d['i'] + 1
            step_progress = int(100.0 * current_step / steps)
            section_progress = f"Section {section_idx + 1}/{len(list(latent_paddings))}, Step {current_step}/{steps}"
            # Don't update main progress here to avoid too frequent updates
            return
        
        # Generate new frames
        generated_latents = sample_hunyuan(
            transformer=transformer,
            sampler='unipc',
            width=width,
            height=height,
            frames=num_frames,
            real_guidance_scale=cfg_scale,
            distilled_guidance_scale=distilled_cfg_scale,
            guidance_rescale=cfg_rescale,
            num_inference_steps=steps,
            generator=rnd,
            prompt_embeds=llama_vec,
            prompt_embeds_mask=llama_attention_mask,
            prompt_poolers=clip_l_pooler,
            negative_prompt_embeds=llama_vec_n,
            negative_prompt_embeds_mask=llama_attention_mask_n,
            negative_prompt_poolers=clip_l_pooler_n,
            device=gpu,
            dtype=torch.bfloat16,
            image_embeddings=image_encoder_last_hidden_state,
            latent_indices=latent_indices,
            clean_latents=clean_latents,
            clean_latent_indices=clean_latent_indices,
            clean_latents_2x=clean_latents_2x,
            clean_latent_2x_indices=clean_latent_2x_indices,
            clean_latents_4x=clean_latents_4x,
            clean_latent_4x_indices=clean_latent_4x_indices,
            callback=callback,
        )
        
        if is_last_section:
            generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)
        
        total_generated_latent_frames += int(generated_latents.shape[2])
        history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)
        
        # Decode and blend new frames
        if not high_vram:
            offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
            load_model_as_complete(vae, target_device=gpu)
        
        real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]
        
        if section_idx == 0:
            # First extension section - decode all frames including existing ones
            history_pixels = vae_decode(real_history_latents, vae).cpu()
        else:
            # Subsequent sections - blend with overlap
            section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
            overlapped_frames = latent_window_size * 4 - 3
            
            current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
            history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)
        
        if not high_vram:
            unload_complete_models()
        
        if is_last_section:
            break
    
    update_progress("Saving extended video...", 90)
    
    # Save final video
    output_filename = os.path.join(output_dir, f'{job_id}_extended.mp4')
    save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)
    
    update_progress("Video extension complete!", 100)
    
    return output_filename


def create_models_dict(
    text_encoder, text_encoder_2, tokenizer, tokenizer_2,
    vae, transformer, feature_extractor, image_encoder
) -> dict:
    """
    Helper function to create the models dictionary required by extend_video_frames.
    
    Returns:
        Dictionary with all required models
    """
    return {
        'text_encoder': text_encoder,
        'text_encoder_2': text_encoder_2,
        'tokenizer': tokenizer, 
        'tokenizer_2': tokenizer_2,
        'vae': vae,
        'transformer': transformer,
        'feature_extractor': feature_extractor,
        'image_encoder': image_encoder
    } 