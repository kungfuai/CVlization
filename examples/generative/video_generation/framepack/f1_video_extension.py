#!/usr/bin/env python3

import os
import cv2
import torch
import numpy as np
from PIL import Image
from typing import List, Union, Optional, Callable
import traceback
import einops
import pathlib
from tqdm import tqdm

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


def extract_last_frame_as_image(video_frames: List[np.ndarray]) -> np.ndarray:
    """Extract the last frame from video frames to use as starting image."""
    if not video_frames:
        raise ValueError("No video frames provided")
    return video_frames[-1]


@torch.no_grad()
def video_encode(video_path: str, resolution: int = 640, no_resize: bool = False, 
                vae=None, vae_batch_size: int = 16, device: str = "cuda", 
                width: Optional[int] = None, height: Optional[int] = None):
    """
    Encode a video into latent representations using the VAE.
    
    Args:
        video_path: Path to the input video file.
        resolution: Target resolution for bucket finding.
        no_resize: Whether to skip resizing and use native resolution.
        vae: AutoencoderKLHunyuanVideo model.
        vae_batch_size: Number of frames to process per batch.
        device: Device for computation (e.g., "cuda").
        width, height: Target resolution for resizing frames.
    
    Returns:
        start_latent: Latent of the first frame (for compatibility with original code).
        input_image_np: First frame as numpy array (for CLIP vision encoding).
        history_latents: Latents of all frames (shape: [1, channels, frames, height//8, width//8]).
        fps: Frames per second of the input video.
        target_height: Final height used.
        target_width: Final width used.
        input_video_pixels: Original video frames as tensor.
    """
    # Normalize video path for Windows compatibility
    video_path = str(pathlib.Path(video_path).resolve())
    print(f"Processing video: {video_path}")

    # Check CUDA availability and fallback to CPU if needed
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU")
        device = "cpu"

    try:
        # Load video using OpenCV (fallback if decord not available)
        try:
            import decord
            print("Using decord for video loading...")
            vr = decord.VideoReader(video_path)
            fps = vr.get_avg_fps()
            num_real_frames = len(vr)
            print(f"Video loaded: {num_real_frames} frames, FPS: {fps}")
            
            # Truncate to nearest latent size (multiple of 4)
            latent_size_factor = 4
            num_frames = (num_real_frames // latent_size_factor) * latent_size_factor
            if num_frames != num_real_frames:
                print(f"Truncating video from {num_real_frames} to {num_frames} frames for latent size compatibility")
            num_real_frames = num_frames
            
            # Read frames
            print("Reading video frames...")
            frames = vr.get_batch(range(num_real_frames)).asnumpy()  # Shape: (num_real_frames, height, width, channels)
            print(f"Frames read: {frames.shape}")
            
        except ImportError:
            print("Decord not available, using OpenCV for video loading...")
            # Fallback to OpenCV
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Truncate to nearest latent size (multiple of 4)
            latent_size_factor = 4
            num_frames = (total_frames // latent_size_factor) * latent_size_factor
            if num_frames != total_frames:
                print(f"Truncating video from {total_frames} to {num_frames} frames for latent size compatibility")
            
            frames = []
            for i in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            cap.release()
            frames = np.array(frames)
            num_real_frames = len(frames)
            print(f"Video loaded with OpenCV: {num_real_frames} frames, FPS: {fps}")

        # Get native video resolution
        native_height, native_width = frames.shape[1], frames.shape[2]
        print(f"Native video resolution: {native_width}x{native_height}")
    
        # Use native resolution if height/width not specified, otherwise use provided values
        target_height = native_height if height is None else height
        target_width = native_width if width is None else width
    
        # Adjust to nearest bucket for model compatibility
        if not no_resize:
            target_height, target_width = find_nearest_bucket(target_height, target_width, resolution=resolution)
            print(f"Adjusted resolution: {target_width}x{target_height}")
        else:
            print(f"Using native resolution without resizing: {target_width}x{target_height}")

        # Preprocess frames to match original image processing
        processed_frames = []
        for i, frame in enumerate(frames):
            frame_np = resize_and_center_crop(frame, target_width=target_width, target_height=target_height)
            processed_frames.append(frame_np)
        processed_frames = np.stack(processed_frames)  # Shape: (num_real_frames, height, width, channels)
        print(f"Frames preprocessed: {processed_frames.shape}")

        # Save first frame for CLIP vision encoding
        input_image_np = processed_frames[0]

        # Convert to tensor and normalize to [-1, 1]
        print("Converting frames to tensor...")
        frames_pt = torch.from_numpy(processed_frames).float() / 127.5 - 1
        frames_pt = frames_pt.permute(0, 3, 1, 2)  # Shape: (num_real_frames, channels, height, width)
        frames_pt = frames_pt.unsqueeze(0)  # Shape: (1, num_real_frames, channels, height, width)
        frames_pt = frames_pt.permute(0, 2, 1, 3, 4)  # Shape: (1, channels, num_real_frames, height, width)
        print(f"Tensor shape: {frames_pt.shape}")
        
        # Save pixel frames for use in worker
        input_video_pixels = frames_pt.cpu()

        # Move to device
        print(f"Moving tensor to device: {device}")
        frames_pt = frames_pt.to(device)
        print("Tensor moved to device")

        # Move VAE to device
        print(f"Moving VAE to device: {device}")
        vae.to(device)
        print("VAE moved to device")

        # Encode frames in batches
        print(f"Encoding input video frames in VAE batch size {vae_batch_size} (reduce if memory issues here or if forcing video resolution)")
        latents = []
        vae.eval()
        with torch.no_grad():
            for i in tqdm(range(0, frames_pt.shape[2], vae_batch_size), desc="Encoding video frames", mininterval=0.1):
                batch = frames_pt[:, :, i:i + vae_batch_size]  # Shape: (1, channels, batch_size, height, width)
                try:
                    # Log GPU memory before encoding
                    if device == "cuda":
                        free_mem = torch.cuda.memory_allocated() / 1024**3
                    batch_latent = vae_encode(batch, vae)
                    # Synchronize CUDA to catch issues
                    if device == "cuda":
                        torch.cuda.synchronize()
                    latents.append(batch_latent)
                except RuntimeError as e:
                    print(f"Error during VAE encoding: {str(e)}")
                    if device == "cuda" and "out of memory" in str(e).lower():
                        print("CUDA out of memory, try reducing vae_batch_size or using CPU")
                    raise
        
        # Concatenate latents
        print("Concatenating latents...")
        history_latents = torch.cat(latents, dim=2)  # Shape: (1, channels, frames, height//8, width//8)
        print(f"History latents shape: {history_latents.shape}")

        # Get first frame's latent
        start_latent = history_latents[:, :, :1]  # Shape: (1, channels, 1, height//8, width//8)
        print(f"Start latent shape: {start_latent.shape}")

        # Move VAE back to CPU to free GPU memory
        if device == "cuda":
            vae.to(cpu)
            torch.cuda.empty_cache()
            print("VAE moved back to CPU, CUDA cache cleared")

        return start_latent, input_image_np, history_latents, fps, target_height, target_width, input_video_pixels

    except Exception as e:
        print(f"Error in video_encode: {str(e)}")
        raise


@torch.no_grad()
def extend_video_f1_style(
    input_video: Union[List[np.ndarray], str],
    prompt: str,
    models: dict,
    total_second_length: float = 5.0,
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
    progress_callback: Optional[Callable[[str, int], None]] = None
) -> str:
    """
    Extend video using F1 demo logic - treats the last frame as a starting image
    and generates new video content from there using sectional generation.
    
    This approach:
    1. Extracts the last frame from the input video as the "starting image"
    2. Uses F1's sectional generation logic to create new video content
    3. Concatenates the original video with the generated extension
    
    Args:
        input_video: Either a list of video frames or path to video file
        prompt: Text prompt describing the video extension
        models: Dictionary containing all required models
        total_second_length: Total length of the EXTENSION (not including original video)
        negative_prompt: Negative text prompt
        seed: Random seed for generation
        latent_window_size: Window size for latent generation (default: 9)
        steps: Number of inference steps
        cfg_scale: CFG scale (should be 1.0 for F1)
        distilled_cfg_scale: Distilled CFG scale
        cfg_rescale: CFG rescale factor
        gpu_memory_preservation: GPU memory to preserve (GB)
        use_teacache: Whether to use TeaCache for faster generation
        mp4_crf: Video compression quality
        output_dir: Output directory
        high_vram: Whether running in high VRAM mode
        progress_callback: Optional progress callback function
    
    Returns:
        Path to the extended video file
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
            progress_callback(message, percentage)
        else:
            print(f"[{percentage:3d}%] {message}")
    
    update_progress("Loading and processing input video...", 0)
    
    # Load video frames if path is provided
    if isinstance(input_video, str):
        original_frames = load_video_frames(input_video)
    else:
        original_frames = input_video
    
    if not original_frames:
        raise ValueError("No frames provided or loaded")
    
    # Extract last frame as starting image (F1 style)
    input_image = extract_last_frame_as_image(original_frames)
    print(f"✓ Loaded {len(original_frames)} original frames")
    print(f"✓ Using last frame as starting image for extension")
    
    # Calculate extension parameters (following F1 demo logic)
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))
    
    print(f"✓ Extension plan: {total_second_length}s = {total_latent_sections} sections")
    
    try:
        # Clean GPU (F1 style)
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Text encoding (F1 style)
        update_progress("Text encoding...", 10)

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

        # Processing input image (F1 style)
        update_progress("Image processing...", 20)

        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

        # Save the starting image for reference
        Image.fromarray(input_image_np).save(os.path.join(output_dir, f'{job_id}_start.png'))

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encoding (F1 style)
        update_progress("VAE encoding...", 30)

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        start_latent = vae_encode(input_image_pt, vae)

        # CLIP Vision (F1 style)
        update_progress("CLIP Vision encoding...", 35)

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Dtype conversion (F1 style)
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling initialization (F1 style)
        update_progress("Starting video generation...", 40)

        rnd = torch.Generator("cpu").manual_seed(seed)

        # Initialize history exactly like F1 demo
        history_latents = torch.zeros(size=(1, 16, 16 + 2 + 1, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None

        # Add start latent to history (F1 style)
        history_latents = torch.cat([history_latents, start_latent.to(history_latents)], dim=2)
        total_generated_latent_frames = 1

        print(f"✓ Initialized generation with start latent: {start_latent.shape}")
        print(f"✓ History buffer: {history_latents.shape}")

        # Main generation loop (F1 style)
        for section_index in range(total_latent_sections):
            section_progress = 40 + int(50 * section_index / total_latent_sections)
            update_progress(f"Generating section {section_index + 1}/{total_latent_sections}...", section_progress)
            
            print(f'🎬 === SECTION {section_index + 1}/{total_latent_sections} ===')

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
                total_frames_so_far = int(max(0, total_generated_latent_frames * 4 - 3))
                video_length_so_far = max(0, total_frames_so_far / 30)
                print(f"     Step {current_step}/{steps} ({step_progress}%) - Frames: {total_frames_so_far}, Length: {video_length_so_far:.2f}s")
                return

            # Set up indices (F1 style)
            indices = torch.arange(0, sum([1, 16, 2, 1, latent_window_size])).unsqueeze(0)
            clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split([1, 16, 2, 1, latent_window_size], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)

            # Get clean latents from history (F1 style)
            clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -sum([16, 2, 1]):, :, :].split([16, 2, 1], dim=2)
            clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)

            print(f"   ✓ Clean latents shape: {clean_latents.shape}")
            print(f"   ✓ Generating {latent_window_size * 4 - 3} frames...")

            # Generate new latents (F1 style)
            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=latent_window_size * 4 - 3,
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

            print(f"   ✓ Generated latents: {generated_latents.shape}")

            # Update history (F1 style)
            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)

            print(f"   ✓ Total latent frames: {total_generated_latent_frames}")

            # Decode pixels (F1 style)
            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :]

            if history_pixels is None:
                # First section - decode all
                print(f"   🎬 Decoding {total_generated_latent_frames} latent frames...")
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                # Subsequent sections - blend with overlap (F1 style)
                section_latent_frames = latent_window_size * 2
                overlapped_frames = latent_window_size * 4 - 3

                print(f"   🎬 Decoding and blending {section_latent_frames} latent frames...")
                current_pixels = vae_decode(real_history_latents[:, :, -section_latent_frames:], vae).cpu()
                history_pixels = soft_append_bcthw(history_pixels, current_pixels, overlapped_frames)

            if not high_vram:
                unload_complete_models()

            current_video_frames = history_pixels.shape[2]
            current_video_length = current_video_frames / 30
            print(f"   ✓ Current extension: {current_video_frames} frames ({current_video_length:.2f}s)")

            # Save intermediate result
            intermediate_filename = os.path.join(output_dir, f'{job_id}_section_{section_index + 1}.mp4')
            save_bcthw_as_mp4(history_pixels, intermediate_filename, fps=30, crf=mp4_crf)
            print(f"   ✓ Saved intermediate: {intermediate_filename}")

        update_progress("Combining original video with extension...", 90)

        # Now combine original video with generated extension
        print("🔗 Combining original video with generated extension...")
        
        # Process original frames to match the generated video dimensions and range
        processed_original_frames = []
        for frame in original_frames:
            processed_frame = resize_and_center_crop(frame, target_width=width, target_height=height)
            frame_tensor = torch.from_numpy(processed_frame).float() / 127.5 - 1.0  # Normalize to [-1, 1] to match VAE decoded frames
            frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC -> CHW
            processed_original_frames.append(frame_tensor)
        
        # Stack original frames: (C, H, W) -> (C, T, H, W) -> (1, C, T, H, W)
        original_video_tensor = torch.stack(processed_original_frames, dim=1).unsqueeze(0)
        
        print(f"   ✓ Original video tensor: {original_video_tensor.shape}")
        print(f"   ✓ Generated extension: {history_pixels.shape}")
        
        # Concatenate along time dimension
        # Note: Remove the first frame of extension since it's the same as the last frame of original
        extension_without_first = history_pixels[:, :, 1:, :, :]
        combined_video = torch.cat([original_video_tensor, extension_without_first], dim=2)
        
        print(f"   ✓ Combined video: {combined_video.shape}")
        
        # Save final combined video
        final_filename = os.path.join(output_dir, f'{job_id}_extended_final.mp4')
        save_bcthw_as_mp4(combined_video, final_filename, fps=30, crf=mp4_crf)

        update_progress("Video extension complete!", 100)

        total_original_frames = len(original_frames)
        total_extension_frames = history_pixels.shape[2] - 1  # Subtract 1 for the overlapping frame
        total_final_frames = combined_video.shape[2]
        
        print(f"\n=== F1-Style Video Extension Complete ===")
        print(f"Original video: {total_original_frames} frames ({total_original_frames/30:.2f}s)")
        print(f"Generated extension: {total_extension_frames} frames ({total_extension_frames/30:.2f}s)")
        print(f"Final combined video: {total_final_frames} frames ({total_final_frames/30:.2f}s)")
        print(f"Extension saved as: {final_filename}")
        
        return final_filename

    except Exception as e:
        print(f"❌ Error during video extension: {str(e)}")
        traceback.print_exc()

        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )
        
        raise e


@torch.no_grad()
def extend_video_f1_multiframe(
    input_video: Union[List[np.ndarray], str],
    prompt: str,
    models: dict,
    total_second_length: float = 5.0,
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
    progress_callback: Optional[Callable[[str, int], None]] = None,
    num_context_frames: int = 5,
    vae_batch_size: int = 16,
    resolution: int = 640,
    no_resize: bool = False
) -> str:
    """
    Extend video using F1 logic with multi-frame conditioning.
    
    This approach:
    1. Encodes the entire input video into latents
    2. Uses multiple frames as context for better temporal continuity
    3. Uses F1's sectional generation logic to create new video content
    4. Concatenates the original video with the generated extension
    
    Args:
        input_video: Either a list of video frames or path to video file
        prompt: Text prompt describing the video extension
        models: Dictionary containing all required models
        total_second_length: Total length of the EXTENSION (not including original video)
        negative_prompt: Negative text prompt
        seed: Random seed for generation
        latent_window_size: Window size for latent generation (default: 9)
        steps: Number of inference steps
        cfg_scale: CFG scale (should be 1.0 for F1)
        distilled_cfg_scale: Distilled CFG scale
        cfg_rescale: CFG rescale factor
        gpu_memory_preservation: GPU memory to preserve (GB)
        use_teacache: Whether to use TeaCache for faster generation
        mp4_crf: Video compression quality
        output_dir: Output directory
        high_vram: Whether running in high VRAM mode
        progress_callback: Optional progress callback function
        num_context_frames: Number of context frames to use (2-10)
        vae_batch_size: Batch size for VAE encoding
        resolution: Target resolution for bucket finding
        no_resize: Whether to skip resizing and use native resolution
    
    Returns:
        Path to the extended video file
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
            progress_callback(message, percentage)
        else:
            print(f"[{percentage:3d}%] {message}")
    
    update_progress("Loading and processing input video...", 0)
    
    # Encode entire input video to latents
    if isinstance(input_video, str):
        video_path = input_video
    else:
        # If frames are provided, we need to save them as a temporary video
        # For now, assume it's a video path
        video_path = input_video
    
    try:
        # Clean GPU
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Encode input video
        update_progress("Encoding input video to latents...", 5)
        start_latent, input_image_np, video_latents, fps, height, width, input_video_pixels = video_encode(
            video_path, resolution, no_resize, vae, vae_batch_size, device=gpu
        )
        
        print(f"✓ Encoded video: {video_latents.shape[2]} frames at {fps} FPS")
        print(f"✓ Resolution: {width}x{height}")

        # Text encoding
        update_progress("Text encoding...", 10)

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

        # CLIP Vision encoding
        update_progress("CLIP Vision encoding...", 15)

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Dtype conversion
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Calculate extension parameters
        total_latent_sections = (total_second_length * fps) / (latent_window_size * 4)
        total_latent_sections = int(max(round(total_latent_sections), 1))
        
        print(f"✓ Extension plan: {total_second_length}s = {total_latent_sections} sections")

        # Sampling initialization
        update_progress("Starting video generation...", 20)

        rnd = torch.Generator("cpu").manual_seed(seed)

        # Initialize history_latents with video latents (key innovation from your friend)
        history_latents = video_latents.cpu()
        total_generated_latent_frames = history_latents.shape[2]
        history_pixels = None

        print(f"✓ Initialized generation with {total_generated_latent_frames} latent frames from input video")
        print(f"✓ History buffer: {history_latents.shape}")

        # Main generation loop with multi-frame conditioning
        for section_index in range(total_latent_sections):
            section_progress = 20 + int(70 * section_index / total_latent_sections)
            update_progress(f"Generating section {section_index + 1}/{total_latent_sections}...", section_progress)
            
            print(f'🎬 === SECTION {section_index + 1}/{total_latent_sections} ===')

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
                total_frames_so_far = int(max(0, total_generated_latent_frames * 4 - 3))
                video_length_so_far = max(0, total_frames_so_far / fps)
                print(f"     Step {current_step}/{steps} ({step_progress}%) - Frames: {total_frames_so_far}, Length: {video_length_so_far:.2f}s")
                return

            # Dynamic context frame management (your friend's innovation)
            available_frames = history_latents.shape[2]
            max_pixel_frames = min(latent_window_size * 4 - 3, available_frames * 4)
            adjusted_latent_frames = max(1, (max_pixel_frames + 3) // 4)
            
            # Adjust num_context_frames to match original behavior
            effective_clean_frames = max(0, num_context_frames - 1) if num_context_frames > 1 else 0
            effective_clean_frames = min(effective_clean_frames, available_frames - 2) if available_frames > 2 else 0
            num_2x_frames = min(2, max(1, available_frames - effective_clean_frames - 1)) if available_frames > effective_clean_frames + 1 else 0
            num_4x_frames = min(16, max(1, available_frames - effective_clean_frames - num_2x_frames)) if available_frames > effective_clean_frames + num_2x_frames else 0
            
            total_context_frames = num_4x_frames + num_2x_frames + effective_clean_frames
            total_context_frames = min(total_context_frames, available_frames)

            # Dynamic index calculation (your friend's key insight)
            indices = torch.arange(0, sum([1, num_4x_frames, num_2x_frames, effective_clean_frames, adjusted_latent_frames])).unsqueeze(0)
            clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split(
                [1, num_4x_frames, num_2x_frames, effective_clean_frames, adjusted_latent_frames], dim=1
            )
            clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)

            # Split history_latents dynamically based on available frames
            fallback_frame_count = 2
            context_frames = history_latents[:, :, -total_context_frames:, :, :] if total_context_frames > 0 else history_latents[:, :, :fallback_frame_count, :, :]
            
            if total_context_frames > 0:
                split_sizes = [num_4x_frames, num_2x_frames, effective_clean_frames]
                split_sizes = [s for s in split_sizes if s > 0]
                if split_sizes:
                    splits = context_frames.split(split_sizes, dim=2)
                    split_idx = 0
                    clean_latents_4x = splits[split_idx] if num_4x_frames > 0 else history_latents[:, :, :fallback_frame_count, :, :]
                    if clean_latents_4x.shape[2] < 2:
                        clean_latents_4x = torch.cat([clean_latents_4x, clean_latents_4x[:, :, -1:, :, :]], dim=2)[:, :, :2, :, :]
                    split_idx += 1 if num_4x_frames > 0 else 0
                    clean_latents_2x = splits[split_idx] if num_2x_frames > 0 and split_idx < len(splits) else history_latents[:, :, :fallback_frame_count, :, :]
                    if clean_latents_2x.shape[2] < 2:
                        clean_latents_2x = torch.cat([clean_latents_2x, clean_latents_2x[:, :, -1:, :, :]], dim=2)[:, :, :2, :, :]
                    split_idx += 1 if num_2x_frames > 0 else 0
                    clean_latents_1x = splits[split_idx] if effective_clean_frames > 0 and split_idx < len(splits) else history_latents[:, :, :fallback_frame_count, :, :]
                else:
                    clean_latents_4x = clean_latents_2x = clean_latents_1x = history_latents[:, :, :fallback_frame_count, :, :]
            else:
                clean_latents_4x = clean_latents_2x = clean_latents_1x = history_latents[:, :, :fallback_frame_count, :, :]

            clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)

            print(f"   ✓ Clean latents shape: {clean_latents.shape}")
            print(f"   ✓ Context frames: 4x={num_4x_frames}, 2x={num_2x_frames}, 1x={effective_clean_frames}")
            print(f"   ✓ Generating {adjusted_latent_frames * 4 - 3} frames...")

            # Fix for short videos
            max_frames = min(latent_window_size * 4 - 3, history_latents.shape[2] * 4)

            # Generate new latents
            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=max_frames,
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

            print(f"   ✓ Generated latents: {generated_latents.shape}")

            # Update history
            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)

            print(f"   ✓ Total latent frames: {total_generated_latent_frames}")

            # Decode pixels
            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :]

            if history_pixels is None:
                # First section - decode all
                print(f"   🎬 Decoding {total_generated_latent_frames} latent frames...")
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                # Subsequent sections - blend with overlap
                section_latent_frames = latent_window_size * 2
                overlapped_frames = min(latent_window_size * 4 - 3, history_pixels.shape[2])

                print(f"   🎬 Decoding and blending {section_latent_frames} latent frames...")
                current_pixels = vae_decode(real_history_latents[:, :, -section_latent_frames:], vae).cpu()
                history_pixels = soft_append_bcthw(history_pixels, current_pixels, overlapped_frames)

            if not high_vram:
                unload_complete_models()

            current_video_frames = history_pixels.shape[2]
            current_video_length = current_video_frames / fps
            print(f"   ✓ Current extension: {current_video_frames} frames ({current_video_length:.2f}s)")

            # Save intermediate result
            intermediate_filename = os.path.join(output_dir, f'{job_id}_section_{section_index + 1}.mp4')
            save_bcthw_as_mp4(history_pixels, intermediate_filename, fps=fps, crf=mp4_crf)
            print(f"   ✓ Saved intermediate: {intermediate_filename}")

        update_progress("Video extension complete!", 100)

        # Final output
        final_filename = os.path.join(output_dir, f'{job_id}_multiframe_extended_final.mp4')
        save_bcthw_as_mp4(history_pixels, final_filename, fps=fps, crf=mp4_crf)

        total_original_frames = video_latents.shape[2] * 4  # Convert latent frames to pixel frames
        total_extension_frames = history_pixels.shape[2] - total_original_frames
        total_final_frames = history_pixels.shape[2]
        
        print(f"\n=== F1-Style Multi-Frame Video Extension Complete ===")
        print(f"Original video: {total_original_frames} frames ({total_original_frames/fps:.2f}s)")
        print(f"Generated extension: {total_extension_frames} frames ({total_extension_frames/fps:.2f}s)")
        print(f"Final combined video: {total_final_frames} frames ({total_final_frames/fps:.2f}s)")
        print(f"Context frames used: {num_context_frames}")
        print(f"Extension saved as: {final_filename}")
        
        return final_filename

    except Exception as e:
        print(f"❌ Error during multi-frame video extension: {str(e)}")
        traceback.print_exc()

        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )
        
        raise e


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


# Convenience function with simpler interface
@torch.no_grad()
def extend_video_simple(
    input_video_path: str,
    prompt: str,
    models: dict,
    extension_seconds: float = 3.0,
    output_dir: str = "./outputs",
    seed: int = 31337,
    steps: int = 25,
    high_vram: bool = False,
    progress_callback: Optional[Callable[[str, int], None]] = None
) -> str:
    """
    Simplified interface for video extension using F1 logic.
    
    Args:
        input_video_path: Path to input video file
        prompt: Text prompt describing the extension
        models: Dictionary containing all required models
        extension_seconds: Length of extension to generate (seconds)
        output_dir: Output directory
        seed: Random seed
        steps: Number of inference steps
        high_vram: Whether running in high VRAM mode
        progress_callback: Optional progress callback
    
    Returns:
        Path to the extended video file
    """
    return extend_video_f1_style(
        input_video=input_video_path,
        prompt=prompt,
        models=models,
        total_second_length=extension_seconds,
        negative_prompt="",
        seed=seed,
        latent_window_size=9,
        steps=steps,
        cfg_scale=1.0,
        distilled_cfg_scale=10.0,
        cfg_rescale=0.0,
        gpu_memory_preservation=6.0,
        use_teacache=True,
        mp4_crf=16,
        output_dir=output_dir,
        high_vram=high_vram,
        progress_callback=progress_callback
    ) 