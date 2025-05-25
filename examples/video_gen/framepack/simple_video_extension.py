#!/usr/bin/env python3

import os
import cv2
import torch
import numpy as np
from PIL import Image
from typing import List, Union, Optional, Callable

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
    offload_model_from_device_for_memory_preservation,
    fake_diffusers_current_device
)
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket


@torch.no_grad()
def vae_encode_safe(frames_tensor, vae):
    """
    Safe VAE encode that handles Half precision issues by temporarily converting to float32.
    """
    original_dtype = vae.dtype
    
    # Convert VAE to float32 if it's in half precision to avoid replication_pad3d issues
    if vae.dtype == torch.float16:
        vae = vae.to(torch.float32)
    
    try:
        result = vae_encode(frames_tensor, vae)
    finally:
        # Convert back to original dtype
        if original_dtype == torch.float16:
            vae = vae.to(original_dtype)
    
    return result


@torch.no_grad()
def vae_decode_safe(latents, vae, image_mode=False):
    """
    Safe VAE decode that handles Half precision issues by temporarily converting to float32.
    """
    original_dtype = vae.dtype
    
    # Convert VAE to float32 if it's in half precision to avoid replication_pad3d issues
    if vae.dtype == torch.float16:
        vae = vae.to(torch.float32)
    
    try:
        result = vae_decode(latents, vae, image_mode)
    finally:
        # Convert back to original dtype
        if original_dtype == torch.float16:
            vae = vae.to(original_dtype)
    
    return result


def load_video_frames(video_path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """Load video frames from a video file."""
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
    """Prepare video frames for VAE encoding."""
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
def extend_video_conservative(
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
    Conservative video extension that only uses recent frames as context.
    
    This approach respects FramePack's limitations by:
    1. Using only the last ~9 frames as context (not the full video)
    2. Treating the last frame as the "start image" like original i2v
    3. Using normal FramePack sectional generation from there
    
    Limitations:
    - Very limited temporal context (~0.3 seconds)
    - May not smoothly continue complex long-term motions
    - Untested - model wasn't trained for this use case
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
    
    # CRITICAL: Only use recent frames as context (respects FramePack's limitations)
    max_context_frames = min(latent_window_size, len(existing_frames))
    recent_frames = existing_frames[-max_context_frames:]
    
    print(f"Using only last {len(recent_frames)} frames as context (out of {len(existing_frames)} total)")
    print(f"Context duration: ~{len(recent_frames)/30:.2f} seconds")
    
    # Determine target dimensions from first frame
    H, W = recent_frames[0].shape[:2]
    height, width = find_nearest_bucket(H, W, resolution=640)
    
    # Prepare recent frames for encoding
    frames_tensor = prepare_frames_for_encoding(recent_frames, height, width)
    
    # Calculate extension parameters
    total_latent_sections = int(max(round((extend_seconds * 30) / (latent_window_size * 4)), 1))
    
    update_progress("Encoding text prompt...", 10)
    
    # Clean GPU if needed
    if not high_vram:
        unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
    
    # Text encoding (same as original)
    if not high_vram:
        fake_diffusers_current_device(text_encoder, gpu)
        load_model_as_complete(text_encoder_2, target_device=gpu)
    
    llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
    
    if cfg_scale == 1:
        llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
    else:
        llama_vec_n, clip_l_pooler_n = encode_prompt_conds(negative_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
    
    llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
    llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
    
    update_progress("Encoding recent video frames...", 20)
    
    # VAE encoding of recent frames
    if not high_vram:
        load_model_as_complete(vae, target_device=gpu)
    
    recent_latents = vae_encode_safe(frames_tensor, vae)
    
    update_progress("Processing last frame for image conditioning...", 30)
    
    # CLIP Vision encoding of last frame (like original i2v)
    if not high_vram:
        load_model_as_complete(image_encoder, target_device=gpu)
    
    last_frame = recent_frames[-1]
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
    
    print("🔧 Initializing generation state...")
    
    # Initialize generation state (CONSERVATIVE APPROACH)
    rnd = torch.Generator("cpu").manual_seed(seed)
    num_frames = latent_window_size * 4 - 3
    
    # Use last frame as start_latent (like original i2v)
    start_latent = recent_latents[:, :, -1:, :, :]
    print(f"✓ Start latent shape: {start_latent.shape}")
    
    # Initialize history buffer with recent context + standard buffer
    # This is the key difference: we include recent frames in the history
    recent_latent_frames = recent_latents.shape[2]
    history_latents = torch.zeros(size=(1, 16, recent_latent_frames + 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
    
    # Place recent frames at the beginning of history
    history_latents[:, :, :recent_latent_frames, :, :] = recent_latents.cpu()
    print(f"✓ History buffer initialized: {history_latents.shape}")
    
    # Initialize pixel history - we'll decode everything together in the first section
    print("📋 Deferring pixel decode until first generation section...")
    history_pixels = None  # Will be set in first loop iteration
    total_generated_latent_frames = recent_latent_frames
    
    print(f"✓ Initialized with {recent_latent_frames} recent frames")
    print("✓ Will decode all frames together after first generation")
    
    # Set up latent padding sequence for extension (same as original)
    latent_paddings = list(reversed(range(total_latent_sections)))
    if total_latent_sections > 4:
        latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
    
    print(f"📋 Generation plan: {len(latent_paddings)} sections with padding sequence: {latent_paddings}")
    print(f"🎯 Target frames per section: {num_frames}")
    print()
    
    # Extension generation loop (adapted from original)
    for section_idx, latent_padding in enumerate(latent_paddings):
        is_last_section = latent_padding == 0
        latent_padding_size = latent_padding * latent_window_size
        
        progress_pct = 40 + int(50 * section_idx / len(latent_paddings))
        update_progress(f"Generating extension section {section_idx + 1}/{len(latent_paddings)}...", progress_pct)
        
        print(f'🎬 === SECTION {section_idx + 1}/{len(latent_paddings)} ===')
        print(f'   Padding size: {latent_padding_size}, Last section: {is_last_section}')
        
        # Set up indices for this section (same as original)
        print("🔧 Setting up indices and conditioning...")
        indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
        clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
        clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
        
        # CRITICAL: Use recent history for conditioning (not arbitrary old frames)
        clean_latents_pre = start_latent.to(history_latents)
        
        # Get conditioning from recent history (last 1+2+16 frames of current history)
        history_end = history_latents[:, :, -1-2-16:, :, :]
        clean_latents_post, clean_latents_2x, clean_latents_4x = history_end.split([1, 2, 16], dim=2)
        clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
        print(f"   ✓ Clean latents shape: {clean_latents.shape}")
        
        # Load transformer for generation
        print("🚀 Loading transformer model...")
        if not high_vram:
            unload_complete_models()
            move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)
        print("   ✓ Transformer loaded")
        
        print("⚡ Initializing TeaCache...")
        if use_teacache:
            transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            print("   ✓ TeaCache enabled")
        else:
            transformer.initialize_teacache(enable_teacache=False)
            print("   ✓ TeaCache disabled")
        
        def callback(d):
            current_step = d['i'] + 1
            step_progress = int(100.0 * current_step / steps)
            print(f"     Step {current_step}/{steps} ({step_progress}%)")
            return
        
        # Generate new frames (same as original)
        print(f"🎨 Starting diffusion sampling ({steps} steps)...")
        import time
        start_time = time.time()
        
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
        
        sampling_time = time.time() - start_time
        print(f"   ✓ Diffusion sampling complete! ({sampling_time:.1f}s)")
        print(f"   ✓ Generated latents shape: {generated_latents.shape}")
        
        if is_last_section:
            generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)
            print(f"   ✓ Added start latent for final section")
        
        total_generated_latent_frames += int(generated_latents.shape[2])
        history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)
        print(f"   ✓ Updated history: {total_generated_latent_frames} total latent frames")
        
        # Decode and blend new frames
        print("🎬 Decoding and blending frames...")
        if not high_vram:
            offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
            load_model_as_complete(vae, target_device=gpu)
        
        real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]
        
        decode_start = time.time()
        if section_idx == 0:
            # First extension section - decode all frames including recent ones
            print(f"   🎬 Decoding {total_generated_latent_frames} frames (existing + first section)...")
            history_pixels = vae_decode_safe(real_history_latents, vae).cpu()
        else:
            # Subsequent sections - blend with overlap (same as original)
            section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
            overlapped_frames = latent_window_size * 4 - 3
            
            print(f"   🎬 Decoding and blending {section_latent_frames} frames...")
            current_pixels = vae_decode_safe(real_history_latents[:, :, :section_latent_frames], vae).cpu()
            history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)
        
        decode_time = time.time() - decode_start
        print(f"   ✓ VAE decode complete! ({decode_time:.1f}s)")
        print(f"   ✓ Current video shape: {history_pixels.shape}")
        
        current_video_length = max(0, (total_generated_latent_frames * 4 - 3) / 30)
        print(f"   📊 Current video length: {current_video_length:.2f} seconds")
        
        if not high_vram:
            unload_complete_models()
        
        if is_last_section:
            print(f"🏁 Final section complete!")
            break
        
        print()
    
    update_progress("Saving extended video...", 90)
    
    # CRITICAL FIX: Reorder frames so original frames come first, then generated frames
    print("🔄 Reordering frames: [Generated] + [Original] → [Original] + [Generated]")
    total_frames = history_pixels.shape[2]
    generated_frames = total_frames - len(recent_frames)
    
    # Split the video: first part is generated frames, second part is original frames
    generated_part = history_pixels[:, :, :generated_frames, :, :]  # New frames (currently first)
    original_part = history_pixels[:, :, generated_frames:, :, :]   # Original frames (currently last)
    
    # Reorder: put original frames first, then generated frames
    history_pixels_reordered = torch.cat([original_part, generated_part], dim=2)
    
    print(f"   ✓ Original frames: {original_part.shape[2]} frames")
    print(f"   ✓ Generated frames: {generated_part.shape[2]} frames") 
    print(f"   ✓ Final order: Original → Generated")
    
    # Save final video with correct temporal order
    output_filename = os.path.join(output_dir, f'{job_id}_extended.mp4')
    save_bcthw_as_mp4(history_pixels_reordered, output_filename, fps=30, crf=mp4_crf)
    
    update_progress("Video extension complete!", 100)
    
    total_frames = history_pixels.shape[2]
    extension_frames = total_frames - len(recent_frames)
    
    print(f"\n=== Conservative Video Extension Complete ===")
    print(f"Original context: {len(recent_frames)} frames ({len(recent_frames)/30:.2f}s)")
    print(f"Generated extension: {extension_frames} frames ({extension_frames/30:.2f}s)")
    print(f"Total output: {total_frames} frames ({total_frames/30:.2f}s)")
    print(f"Extended video: {output_filename}")
    
    return output_filename


def create_models_dict(
    text_encoder, text_encoder_2, tokenizer, tokenizer_2,
    vae, transformer, feature_extractor, image_encoder
) -> dict:
    """Helper function to create the models dictionary."""
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

def extend_video(
    input_video_path: str,
    output_dir: str,
    job_id: str,
    prompt: str,
    negative_prompt: str = "",
    num_frames: int = 30,  # Number of frames to generate
    context_frames: int = 9,  # Number of recent frames to use as context (default: 3 seconds at 30fps)
    height: int = 576,
    width: int = 1024,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = 42,
    mp4_crf: int = 23,
    mp4_preset: str = "medium",
    device: str = "cuda",
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> str:
    """
    Extend a video using FramePack's video extension capabilities.
    
    Args:
        input_video_path: Path to input video file
        output_dir: Directory to save output files
        job_id: Unique identifier for this generation job
        prompt: Text prompt describing the video extension
        negative_prompt: Text prompt for what to avoid
        num_frames: Number of new frames to generate
        context_frames: Number of recent frames to use as context (default: 9 = 3 seconds at 30fps)
        height: Output video height
        width: Output video width
        num_inference_steps: Number of denoising steps
        guidance_scale: How closely to follow the prompt
        seed: Random seed for generation
        mp4_crf: Video quality (lower = better, 18-28 is good)
        mp4_preset: Video encoding preset
        device: Device to run generation on
        progress_callback: Optional callback for progress updates
    
    Returns:
        Path to the generated video file
    """

    # Load and prepare input video
    print("📽️ Loading input video...")
    video_frames = load_video_frames(input_video_path, height, width)
    print(f"✓ Loaded {len(video_frames)} frames")
    
    # Use last N frames as context (default: 3 seconds at 30fps)
    if len(video_frames) < context_frames:
        print(f"⚠️ Warning: Input video has only {len(video_frames)} frames, using all as context")
        context_frames = len(video_frames)
    
    recent_frames = video_frames[-context_frames:]  # Use specified number of context frames
    print(f"✓ Using last {context_frames} frames as context")
    
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
    
    # Prepare recent frames for encoding
    frames_tensor = prepare_frames_for_encoding(recent_frames, height, width)
    
    # Calculate extension parameters
    total_latent_sections = int(max(round((num_frames / 4)), 1))
    
    update_progress("Encoding text prompt...", 10)
    
    # Clean GPU if needed
    if device != "cpu":
        unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
    
    # Text encoding (same as original)
    if device != "cpu":
        fake_diffusers_current_device(text_encoder, device)
        load_model_as_complete(text_encoder_2, target_device=device)
    
    llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
    
    if guidance_scale == 1:
        llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
    else:
        llama_vec_n, clip_l_pooler_n = encode_prompt_conds(negative_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
    
    llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
    llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
    
    update_progress("Encoding recent video frames...", 20)
    
    # VAE encoding of recent frames
    if device != "cpu":
        load_model_as_complete(vae, target_device=device)
    
    recent_latents = vae_encode_safe(frames_tensor, vae)
    
    update_progress("Processing last frame for image conditioning...", 30)
    
    # CLIP Vision encoding of last frame (like original i2v)
    if device != "cpu":
        load_model_as_complete(image_encoder, target_device=device)
    
    last_frame = recent_frames[-1]
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
    
    print("🔧 Initializing generation state...")
    
    # Initialize generation state (CONSERVATIVE APPROACH)
    rnd = torch.Generator("cpu").manual_seed(seed)
    num_frames = 4 * context_frames - 3
    
    # Use last frame as start_latent (like original i2v)
    start_latent = recent_latents[:, :, -1:, :, :]
    print(f"✓ Start latent shape: {start_latent.shape}")
    
    # Initialize history buffer with recent context + standard buffer
    # This is the key difference: we include recent frames in the history
    recent_latent_frames = recent_latents.shape[2]
    history_latents = torch.zeros(size=(1, 16, recent_latent_frames + 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
    
    # Place recent frames at the beginning of history
    history_latents[:, :, :recent_latent_frames, :, :] = recent_latents.cpu()
    print(f"✓ History buffer initialized: {history_latents.shape}")
    
    # Initialize pixel history - we'll decode everything together in the first section
    print("📋 Deferring pixel decode until first generation section...")
    history_pixels = None  # Will be set in first loop iteration
    total_generated_latent_frames = recent_latent_frames
    
    print(f"✓ Initialized with {recent_latent_frames} recent frames")
    print("✓ Will decode all frames together after first generation")
    
    # Set up latent padding sequence for extension (same as original)
    latent_paddings = list(reversed(range(total_latent_sections)))
    if total_latent_sections > 4:
        latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
    
    print(f"📋 Generation plan: {len(latent_paddings)} sections with padding sequence: {latent_paddings}")
    print(f"🎯 Target frames per section: {num_frames}")
    print()
    
    # Extension generation loop (adapted from original)
    for section_idx, latent_padding in enumerate(latent_paddings):
        is_last_section = latent_padding == 0
        latent_padding_size = latent_padding * context_frames
        
        progress_pct = 40 + int(50 * section_idx / len(latent_paddings))
        update_progress(f"Generating extension section {section_idx + 1}/{len(latent_paddings)}...", progress_pct)
        
        print(f'🎬 === SECTION {section_idx + 1}/{len(latent_paddings)} ===')
        print(f'   Padding size: {latent_padding_size}, Last section: {is_last_section}')
        
        # Set up indices for this section (same as original)
        print("🔧 Setting up indices and conditioning...")
        indices = torch.arange(0, sum([1, latent_padding_size, context_frames, 1, 2, 16])).unsqueeze(0)
        clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, context_frames, 1, 2, 16], dim=1)
        clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
        
        # CRITICAL: Use recent history for conditioning (not arbitrary old frames)
        clean_latents_pre = start_latent.to(history_latents)
        
        # Get conditioning from recent history (last 1+2+16 frames of current history)
        history_end = history_latents[:, :, -1-2-16:, :, :]
        clean_latents_post, clean_latents_2x, clean_latents_4x = history_end.split([1, 2, 16], dim=2)
        clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
        print(f"   ✓ Clean latents shape: {clean_latents.shape}")
        
        # Load transformer for generation
        print("🚀 Loading transformer model...")
        if device != "cpu":
            unload_complete_models()
            move_model_to_device_with_memory_preservation(transformer, target_device=device, preserved_memory_gb=6)
        print("   ✓ Transformer loaded")
        
        print("⚡ Initializing TeaCache...")
        transformer.initialize_teacache(enable_teacache=True, num_steps=num_inference_steps)
        print("   ✓ TeaCache enabled")
        
        def callback(d):
            current_step = d['i'] + 1
            step_progress = int(100.0 * current_step / num_inference_steps)
            print(f"     Step {current_step}/{num_inference_steps} ({step_progress}%)")
            return
        
        # Generate new frames (same as original)
        print(f"🎨 Starting diffusion sampling ({num_inference_steps} steps)...")
        import time
        start_time = time.time()
        
        generated_latents = sample_hunyuan(
            transformer=transformer,
            sampler='unipc',
            width=width,
            height=height,
            frames=num_frames,
            real_guidance_scale=guidance_scale,
            distilled_guidance_scale=10.0,
            guidance_rescale=0.0,
            num_inference_steps=num_inference_steps,
            generator=rnd,
            prompt_embeds=llama_vec,
            prompt_embeds_mask=llama_attention_mask,
            prompt_poolers=clip_l_pooler,
            negative_prompt_embeds=llama_vec_n,
            negative_prompt_embeds_mask=llama_attention_mask_n,
            negative_prompt_poolers=clip_l_pooler_n,
            device=device,
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
        
        sampling_time = time.time() - start_time
        print(f"   ✓ Diffusion sampling complete! ({sampling_time:.1f}s)")
        print(f"   ✓ Generated latents shape: {generated_latents.shape}")
        
        if is_last_section:
            generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)
            print(f"   ✓ Added start latent for final section")
        
        total_generated_latent_frames += int(generated_latents.shape[2])
        history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)
        print(f"   ✓ Updated history: {total_generated_latent_frames} total latent frames")
        
        # Decode and blend new frames
        print("🎬 Decoding and blending frames...")
        if device != "cpu":
            offload_model_from_device_for_memory_preservation(transformer, target_device=device, preserved_memory_gb=8)
            load_model_as_complete(vae, target_device=device)
        
        real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]
        
        decode_start = time.time()
        if section_idx == 0:
            # First extension section - decode all frames including recent ones
            print(f"   🎬 Decoding {total_generated_latent_frames} frames (existing + first section)...")
            history_pixels = vae_decode_safe(real_history_latents, vae).cpu()
        else:
            # Subsequent sections - blend with overlap (same as original)
            section_latent_frames = (context_frames * 2 + 1) if is_last_section else (context_frames * 2)
            overlapped_frames = context_frames * 4 - 3
            
            print(f"   🎬 Decoding and blending {section_latent_frames} frames...")
            current_pixels = vae_decode_safe(real_history_latents[:, :, :section_latent_frames], vae).cpu()
            history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)
        
        decode_time = time.time() - decode_start
        print(f"   ✓ VAE decode complete! ({decode_time:.1f}s)")
        print(f"   ✓ Current video shape: {history_pixels.shape}")
        
        current_video_length = max(0, (total_generated_latent_frames * 4 - 3) / 30)
        print(f"   📊 Current video length: {current_video_length:.2f} seconds")
        
        if device != "cpu":
            unload_complete_models()
        
        if is_last_section:
            print(f"🏁 Final section complete!")
            break
        
        print()
    
    update_progress("Saving extended video...", 90)
    
    # CRITICAL FIX: Reorder frames so original frames come first, then generated frames
    print("🔄 Reordering frames: [Generated] + [Original] → [Original] + [Generated]")
    total_frames = history_pixels.shape[2]
    generated_frames = total_frames - len(recent_frames)
    
    # Split the video: first part is generated frames, second part is original frames
    generated_part = history_pixels[:, :, :generated_frames, :, :]  # New frames (currently first)
    original_part = history_pixels[:, :, generated_frames:, :, :]   # Original frames (currently last)
    
    # Reorder: put original frames first, then generated frames
    history_pixels_reordered = torch.cat([original_part, generated_part], dim=2)
    
    print(f"   ✓ Original frames: {original_part.shape[2]} frames")
    print(f"   ✓ Generated frames: {generated_part.shape[2]} frames") 
    print(f"   ✓ Final order: Original → Generated")
    
    # Save final video with correct temporal order
    output_filename = os.path.join(output_dir, f'{job_id}_extended.mp4')
    save_bcthw_as_mp4(history_pixels_reordered, output_filename, fps=30, crf=mp4_crf)
    
    update_progress("Video extension complete!", 100)
    
    total_frames = history_pixels.shape[2]
    extension_frames = total_frames - len(recent_frames)
    
    print(f"\n=== Conservative Video Extension Complete ===")
    print(f"Original context: {len(recent_frames)} frames ({len(recent_frames)/30:.2f}s)")
    print(f"Generated extension: {extension_frames} frames ({extension_frames/30:.2f}s)")
    print(f"Total output: {total_frames} frames ({total_frames/30:.2f}s)")
    print(f"Extended video: {output_filename}")
    
    return output_filename 