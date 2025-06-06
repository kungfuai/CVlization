#!/usr/bin/env python3

import os
import sys
import argparse
import torch
import traceback
import einops
import numpy as np
from PIL import Image

# Set HF cache directory relative to script location
os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

# Add the parent directory to sys.path if needed for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from diffusers_helper.hf_login import login
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

# Import video extension functionality
from simple_video_extension import extend_video_conservative, create_models_dict
from f1_video_extension import extend_video_f1_style, extend_video_simple, extend_video_f1_multiframe


def parse_args():
    parser = argparse.ArgumentParser(description='FramePack CLI Enhanced - Generate videos from images or extend existing videos with multi-frame conditioning')
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['i2v', 'extend'], default='i2v',
                        help='Generation mode: i2v (image-to-video) or extend (video extension)')
    
    # Extension method selection
    parser.add_argument('--extension_method', type=str, choices=['f1', 'f1_multiframe', 'conservative'], default='f1_multiframe',
                        help='Video extension method: f1 (single frame), f1_multiframe (multi-frame, RECOMMENDED), or conservative (experimental)')
    
    # Input arguments (mode-dependent)
    parser.add_argument('--input_image', type=str, 
                        help='Path to input image file (required for i2v mode)')
    parser.add_argument('--input_video', type=str, 
                        help='Path to input video file (required for extend mode)')
    parser.add_argument('--output_dir', type=str, default='./outputs/', 
                        help='Directory to save output videos (default: ./outputs/)')
    
    # Generation parameters
    parser.add_argument('--prompt', type=str, default='', 
                        help='Text prompt for video generation')
    parser.add_argument('--negative_prompt', type=str, default='', 
                        help='Negative prompt (currently not used)')
    parser.add_argument('--seed', type=int, default=31337, 
                        help='Random seed for generation (default: 31337)')
    parser.add_argument('--total_seconds', type=float, default=5.0, 
                        help='Total video length in seconds for i2v mode (default: 5.0)')
    parser.add_argument('--extend_seconds', type=float, default=2.0,
                        help='Seconds to extend video in extend mode (default: 2.0)')
    parser.add_argument('--steps', type=int, default=25, 
                        help='Number of inference steps (default: 25)')
    
    # Advanced parameters
    parser.add_argument('--latent_window_size', type=int, default=9, 
                        help='Latent window size (default: 9, not recommended to change)')
    parser.add_argument('--cfg_scale', type=float, default=1.0, 
                        help='CFG scale (default: 1.0, not recommended to change)')
    parser.add_argument('--distilled_cfg_scale', type=float, default=10.0, 
                        help='Distilled CFG scale (default: 10.0)')
    parser.add_argument('--cfg_rescale', type=float, default=0.0, 
                        help='CFG rescale (default: 0.0, not recommended to change)')
    parser.add_argument('--gpu_memory_preservation', type=float, default=6.0, 
                        help='GPU memory to preserve in GB (default: 6.0)')
    parser.add_argument('--use_teacache', action='store_true', default=True, 
                        help='Use TeaCache for faster inference (default: True)')
    parser.add_argument('--no_teacache', action='store_true', 
                        help='Disable TeaCache')
    parser.add_argument('--mp4_crf', type=int, default=16, 
                        help='MP4 compression quality (0=uncompressed, 16=default)')
    
    # Multi-frame extension parameters
    parser.add_argument('--num_context_frames', type=int, default=5,
                        help='Number of context frames for f1_multiframe extension (2-10, default: 5)')
    parser.add_argument('--vae_batch_size', type=int, default=16,
                        help='VAE batch size for video encoding in f1_multiframe (default: 16)')
    parser.add_argument('--resolution', type=int, default=640,
                        help='Target resolution for bucket finding (default: 640)')
    parser.add_argument('--no_resize', action='store_true',
                        help='Skip resizing and use native video resolution (may require more VRAM)')
    
    # Conservative extension parameters
    parser.add_argument('--max_context_frames', type=int, default=9,
                        help='Maximum number of recent frames to use as context for conservative extension (default: 9)')
    
    args = parser.parse_args()
    
    # Handle teacache flags
    if args.no_teacache:
        args.use_teacache = False
    
    # Validate mode-specific arguments
    if args.mode == 'i2v':
        if not args.input_image:
            parser.error("--input_image is required for i2v mode")
    elif args.mode == 'extend':
        if not args.input_video:
            parser.error("--input_video is required for extend mode")
    
    return args


def load_models():
    """Load and initialize all required models"""
    print("Loading models...")
    
    # Check available VRAM
    free_mem_gb = get_cuda_free_memory_gb(gpu)
    high_vram = free_mem_gb > 60
    print(f'Free VRAM: {free_mem_gb} GB')
    print(f'High-VRAM Mode: {high_vram}')
    
    # Load models
    text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
    text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
    tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
    tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
    vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()
    
    feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
    image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()
    
    # Use F1 model for both i2v and extension
    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePack_F1_I2V_HY_20250503', torch_dtype=torch.bfloat16).cpu()
    
    # Set models to eval mode
    vae.eval()
    text_encoder.eval()
    text_encoder_2.eval()
    image_encoder.eval()
    transformer.eval()
    
    # Configure VAE for memory efficiency
    if not high_vram:
        vae.enable_slicing()
        vae.enable_tiling()
    
    # Set model properties
    transformer.high_quality_fp32_output_for_inference = True
    print('transformer.high_quality_fp32_output_for_inference = True')
    
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
    
    # Handle device placement based on VRAM
    if not high_vram:
        # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
        DynamicSwapInstaller.install_model(transformer, device=gpu)
        DynamicSwapInstaller.install_model(text_encoder, device=gpu)
    else:
        text_encoder.to(gpu)
        text_encoder_2.to(gpu)
        image_encoder.to(gpu)
        vae.to(gpu)
        transformer.to(gpu)
    
    models = {
        'text_encoder': text_encoder,
        'text_encoder_2': text_encoder_2,
        'tokenizer': tokenizer,
        'tokenizer_2': tokenizer_2,
        'vae': vae,
        'feature_extractor': feature_extractor,
        'image_encoder': image_encoder,
        'transformer': transformer,
        'high_vram': high_vram
    }
    
    print("Models loaded successfully!")
    return models


@torch.no_grad()
def generate_video(input_image_path, prompt, n_prompt, seed, total_second_length, 
                  latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, 
                  use_teacache, mp4_crf, output_dir, models):
    """Main video generation function (Image-to-Video)"""
    
    # Calculate sections
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))
    
    job_id = generate_timestamp()
    print(f"Starting I2V generation job: {job_id}")
    print(f"Total latent sections: {total_latent_sections}")
    
    # Load and process input image
    print("Loading input image...")
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Input image not found: {input_image_path}")
    
    input_image_pil = Image.open(input_image_path)
    if input_image_pil.mode != 'RGB':
        input_image_pil = input_image_pil.convert('RGB')
    input_image = np.array(input_image_pil)
    
    try:
        # Clean GPU
        if not models['high_vram']:
            unload_complete_models(
                models['text_encoder'], models['text_encoder_2'], 
                models['image_encoder'], models['vae'], models['transformer']
            )
        
        # Text encoding
        print("Encoding text prompts...")
        if not models['high_vram']:
            fake_diffusers_current_device(models['text_encoder'], gpu)
            load_model_as_complete(models['text_encoder_2'], target_device=gpu)
        
        llama_vec, clip_l_pooler = encode_prompt_conds(
            prompt, models['text_encoder'], models['text_encoder_2'], 
            models['tokenizer'], models['tokenizer_2']
        )
        
        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(
                n_prompt, models['text_encoder'], models['text_encoder_2'], 
                models['tokenizer'], models['tokenizer_2']
            )
        
        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
        
        # Processing input image
        print("Processing input image...")
        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
        
        Image.fromarray(input_image_np).save(os.path.join(output_dir, f'{job_id}.png'))
        
        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]
        
        # VAE encoding
        print("VAE encoding...")
        if not models['high_vram']:
            load_model_as_complete(models['vae'], target_device=gpu)
        
        start_latent = vae_encode(input_image_pt, models['vae'])
        
        # CLIP Vision
        print("CLIP Vision encoding...")
        if not models['high_vram']:
            load_model_as_complete(models['image_encoder'], target_device=gpu)
        
        image_encoder_output = hf_clip_vision_encode(input_image_np, models['feature_extractor'], models['image_encoder'])
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        
        # Dtype conversion
        llama_vec = llama_vec.to(models['transformer'].dtype)
        llama_vec_n = llama_vec_n.to(models['transformer'].dtype)
        clip_l_pooler = clip_l_pooler.to(models['transformer'].dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(models['transformer'].dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(models['transformer'].dtype)
        
        # Sampling
        print("Starting sampling...")
        rnd = torch.Generator("cpu").manual_seed(seed)
        
        history_latents = torch.zeros(size=(1, 16, 16 + 2 + 1, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        
        history_latents = torch.cat([history_latents, start_latent.to(history_latents)], dim=2)
        total_generated_latent_frames = 1
        
        for section_index in range(total_latent_sections):
            print(f'Section {section_index + 1}/{total_latent_sections}')
            
            if not models['high_vram']:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(models['transformer'], target_device=gpu, preserved_memory_gb=gpu_memory_preservation)
            
            if use_teacache:
                models['transformer'].initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                models['transformer'].initialize_teacache(enable_teacache=False)
            
            def callback(d):
                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                total_frames_so_far = int(max(0, total_generated_latent_frames * 4 - 3))
                video_length_so_far = max(0, total_frames_so_far / 30)
                print(f"  Step {current_step}/{steps} ({percentage}%) - Frames: {total_frames_so_far}, Length: {video_length_so_far:.2f}s")
                return
            
            indices = torch.arange(0, sum([1, 16, 2, 1, latent_window_size])).unsqueeze(0)
            clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split([1, 16, 2, 1, latent_window_size], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)
            
            clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -sum([16, 2, 1]):, :, :].split([16, 2, 1], dim=2)
            clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)
            
            generated_latents = sample_hunyuan(
                transformer=models['transformer'],
                sampler='unipc',
                width=width,
                height=height,
                frames=latent_window_size * 4 - 3,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
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
            
            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)
            
            if not models['high_vram']:
                offload_model_from_device_for_memory_preservation(models['transformer'], target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(models['vae'], target_device=gpu)
            
            real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :]
            
            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, models['vae']).cpu()
            else:
                section_latent_frames = latent_window_size * 2
                overlapped_frames = latent_window_size * 4 - 3
                
                current_pixels = vae_decode(real_history_latents[:, :, -section_latent_frames:], models['vae']).cpu()
                history_pixels = soft_append_bcthw(history_pixels, current_pixels, overlapped_frames)
            
            if not models['high_vram']:
                unload_complete_models()
            
            output_filename = os.path.join(output_dir, f'{job_id}_{total_generated_latent_frames}.mp4')
            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)
            
            print(f'  Saved intermediate: {output_filename}')
        
        final_output = os.path.join(output_dir, f'{job_id}_final.mp4')
        save_bcthw_as_mp4(history_pixels, final_output, fps=30, crf=mp4_crf)
        
        print(f"\n=== I2V Generation Complete ===")
        print(f"Job ID: {job_id}")
        print(f"Final video: {final_output}")
        
        return final_output
        
    except Exception as e:
        print(f"Error during generation: {e}")
        traceback.print_exc()
        
        if not models['high_vram']:
            unload_complete_models(
                models['text_encoder'], models['text_encoder_2'], 
                models['image_encoder'], models['vae'], models['transformer']
            )
        raise


@torch.no_grad()
def extend_video(input_video_path, prompt, n_prompt, seed, extend_seconds, 
                latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, 
                use_teacache, mp4_crf, output_dir, models, extension_method='f1_multiframe',
                num_context_frames=5, vae_batch_size=16, resolution=640, no_resize=False,
                max_context_frames=9):
    """Video extension function supporting F1-style, F1-multiframe, and conservative methods"""
    
    job_id = generate_timestamp()
    print(f"Starting video extension job: {job_id}")
    print(f"Extension method: {extension_method}")
    print(f"Extension length: {extend_seconds} seconds")
    
    # Check if input video exists
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video not found: {input_video_path}")
    
    try:
        # Convert models dict to the format expected by extension functions
        models_dict = create_models_dict(
            models['text_encoder'], models['text_encoder_2'], 
            models['tokenizer'], models['tokenizer_2'],
            models['vae'], models['transformer'], 
            models['feature_extractor'], models['image_encoder']
        )
        
        def progress_callback(message, percentage):
            print(f"[{percentage:3d}%] {message}")
        
        if extension_method == 'f1':
            print("🎬 Using F1-style video extension (single frame)")
            print("   - Treats last frame as starting image")
            print("   - Uses proven F1 demo logic")
            print("   - Good quality and reliability")
            print()
            
            # Use F1-style extension
            output_path = extend_video_f1_style(
                input_video=input_video_path,
                prompt=prompt,
                models=models_dict,
                total_second_length=extend_seconds,
                negative_prompt=n_prompt,
                seed=seed,
                latent_window_size=latent_window_size,
                steps=steps,
                cfg_scale=cfg,
                distilled_cfg_scale=gs,
                cfg_rescale=rs,
                gpu_memory_preservation=gpu_memory_preservation,
                use_teacache=use_teacache,
                mp4_crf=mp4_crf,
                output_dir=output_dir,
                high_vram=models['high_vram'],
                progress_callback=progress_callback
            )
            
            print(f"\n=== F1-Style Video Extension Complete ===")
            
        elif extension_method == 'f1_multiframe':
            print("🎬 Using F1-style multi-frame video extension (RECOMMENDED)")
            print("   - Uses multiple frames as context for better temporal continuity")
            print("   - Encodes entire input video to latents")
            print("   - Uses proven F1 sectional generation logic")
            print("   - Best quality and temporal consistency")
            print(f"   - Context frames: {num_context_frames}")
            print(f"   - VAE batch size: {vae_batch_size}")
            print()
            
            # Use F1-style multi-frame extension
            output_path = extend_video_f1_multiframe(
                input_video=input_video_path,
                prompt=prompt,
                models=models_dict,
                total_second_length=extend_seconds,
                negative_prompt=n_prompt,
                seed=seed,
                latent_window_size=latent_window_size,
                steps=steps,
                cfg_scale=cfg,
                distilled_cfg_scale=gs,
                cfg_rescale=rs,
                gpu_memory_preservation=gpu_memory_preservation,
                use_teacache=use_teacache,
                mp4_crf=mp4_crf,
                output_dir=output_dir,
                high_vram=models['high_vram'],
                progress_callback=progress_callback,
                num_context_frames=num_context_frames,
                vae_batch_size=vae_batch_size,
                resolution=resolution,
                no_resize=no_resize
            )
            
            print(f"\n=== F1-Style Multi-Frame Video Extension Complete ===")
            
        elif extension_method == 'conservative':
            print("⚠️  Using conservative video extension (EXPERIMENTAL)")
            print("   - Uses recent frames as context")
            print("   - Limited temporal continuity (~0.3s)")
            print("   - May not continue complex motions")
            print("   - Best for simple, consistent movements")
            print(f"   - Max context frames: {max_context_frames}")
            print()
            
            # Use conservative extension
            def conservative_progress_callback(percentage, message):
                print(f"[{percentage:3d}%] {message}")
            
            output_path = extend_video_conservative(
                existing_frames=input_video_path,
                prompt=prompt,
                models=models_dict,
                extend_seconds=extend_seconds,
                negative_prompt=n_prompt,
                seed=seed,
                latent_window_size=min(latent_window_size, max_context_frames),  # Respect context limit
                steps=steps,
                cfg_scale=cfg,
                distilled_cfg_scale=gs,
                cfg_rescale=rs,
                gpu_memory_preservation=gpu_memory_preservation,
                use_teacache=use_teacache,
                mp4_crf=mp4_crf,
                output_dir=output_dir,
                high_vram=models['high_vram'],
                progress_callback=conservative_progress_callback
            )
            
            print(f"\n=== Conservative Video Extension Complete ===")
        
        else:
            raise ValueError(f"Unknown extension method: {extension_method}")
        
        print(f"Job ID: {job_id}")
        print(f"Extended video: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"Error during video extension: {e}")
        traceback.print_exc()
        raise


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== FramePack CLI Enhanced ===")
    print(f"Mode: {args.mode}")
    
    if args.mode == 'i2v':
        print(f"Input image: {args.input_image}")
        print(f"Video length: {args.total_seconds} seconds")
    elif args.mode == 'extend':
        print(f"Input video: {args.input_video}")
        print(f"Extension method: {args.extension_method}")
        print(f"Extension length: {args.extend_seconds} seconds")
        if args.extension_method == 'f1_multiframe':
            print(f"Context frames: {args.num_context_frames}")
            print(f"VAE batch size: {args.vae_batch_size}")
            print(f"Resolution: {args.resolution}")
            print(f"No resize: {args.no_resize}")
        elif args.extension_method == 'conservative':
            print(f"Max context frames: {args.max_context_frames}")
    
    print(f"Prompt: '{args.prompt}'")
    print(f"Output directory: {args.output_dir}")
    print(f"Seed: {args.seed}")
    print(f"Steps: {args.steps}")
    print(f"Use TeaCache: {args.use_teacache}")
    print()
    
    # Load models
    models = load_models()
    
    # Route to appropriate function based on mode
    try:
        if args.mode == 'i2v':
            output_path = generate_video(
                input_image_path=args.input_image,
                prompt=args.prompt,
                n_prompt=args.negative_prompt,
                seed=args.seed,
                total_second_length=args.total_seconds,
                latent_window_size=args.latent_window_size,
                steps=args.steps,
                cfg=args.cfg_scale,
                gs=args.distilled_cfg_scale,
                rs=args.cfg_rescale,
                gpu_memory_preservation=args.gpu_memory_preservation,
                use_teacache=args.use_teacache,
                mp4_crf=args.mp4_crf,
                output_dir=args.output_dir,
                models=models
            )
        elif args.mode == 'extend':
            output_path = extend_video(
                input_video_path=args.input_video,
                prompt=args.prompt,
                n_prompt=args.negative_prompt,
                seed=args.seed,
                extend_seconds=args.extend_seconds,
                latent_window_size=args.latent_window_size,
                steps=args.steps,
                cfg=args.cfg_scale,
                gs=args.distilled_cfg_scale,
                rs=args.cfg_rescale,
                gpu_memory_preservation=args.gpu_memory_preservation,
                use_teacache=args.use_teacache,
                mp4_crf=args.mp4_crf,
                output_dir=args.output_dir,
                models=models,
                extension_method=args.extension_method,
                num_context_frames=args.num_context_frames,
                vae_batch_size=args.vae_batch_size,
                resolution=args.resolution,
                no_resize=args.no_resize,
                max_context_frames=args.max_context_frames
            )
        
        print(f"\nSUCCESS! Video saved to: {output_path}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    """
    # Image-to-Video (i2v) mode - Basic usage
    python predict_f1_enhanced.py --mode i2v --input_image /path/to/image.jpg --prompt "The girl dances gracefully"

    # Image-to-Video with custom parameters
    python predict_f1_enhanced.py \
        --mode i2v \
        --input_image /path/to/image.jpg \
        --prompt "A character doing some simple body movements" \
        --total_seconds 10.0 \
        --seed 42 \
        --steps 30 \
        --output_dir ./my_outputs/

    # Video Extension mode - F1 Multi-Frame (RECOMMENDED)
    python predict_f1_enhanced.py \
        --mode extend \
        --extension_method f1_multiframe \
        --input_video /path/to/video.mp4 \
        --prompt "The character continues dancing gracefully" \
        --extend_seconds 3.0 \
        --num_context_frames 5 \
        --vae_batch_size 16

    # Video Extension mode - F1 Single Frame
    python predict_f1_enhanced.py \
        --mode extend \
        --extension_method f1 \
        --input_video /path/to/video.mp4 \
        --prompt "The action continues" \
        --extend_seconds 3.0

    # Video Extension mode - Conservative (experimental)
    python predict_f1_enhanced.py \
        --mode extend \
        --extension_method conservative \
        --input_video /path/to/video.mp4 \
        --prompt "The action continues with graceful movements" \
        --extend_seconds 3.0 \
        --max_context_frames 9

    # High quality F1 multi-frame extension
    python predict_f1_enhanced.py \
        --mode extend \
        --extension_method f1_multiframe \
        --input_video /path/to/video.mp4 \
        --prompt "Extended dance sequence" \
        --extend_seconds 5.0 \
        --mp4_crf 0 \
        --steps 50 \
        --no_teacache \
        --num_context_frames 8 \
        --vae_batch_size 8

    # Native resolution extension (requires more VRAM)
    python predict_f1_enhanced.py \
        --mode extend \
        --extension_method f1_multiframe \
        --input_video /path/to/video.mp4 \
        --prompt "High quality extension" \
        --extend_seconds 3.0 \
        --no_resize \
        --vae_batch_size 4

    # Backward compatibility: i2v mode is default
    python predict_f1_enhanced.py --input_image /path/to/image.jpg --prompt "Dancing scene"
    """
    main() 