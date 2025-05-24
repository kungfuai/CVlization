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


def parse_args():
    parser = argparse.ArgumentParser(description='FramePack CLI - Generate videos from images')
    
    # Required arguments
    parser.add_argument('--input_image', type=str, required=True, 
                        help='Path to input image file')
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
                        help='Total video length in seconds (default: 5.0)')
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
    
    args = parser.parse_args()
    
    # Handle teacache flags
    if args.no_teacache:
        args.use_teacache = False
    
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
    
    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()
    
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
    """Main video generation function"""
    
    # Calculate sections
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))
    
    job_id = generate_timestamp()
    print(f"Starting generation job: {job_id}")
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
        
        # Save processed input image
        Image.fromarray(input_image_np).save(os.path.join(output_dir, f'{job_id}.png'))
        print(f"Processed image saved: {job_id}.png")
        
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
        
        image_encoder_output = hf_clip_vision_encode(
            input_image_np, models['feature_extractor'], models['image_encoder']
        )
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
        num_frames = latent_window_size * 4 - 3
        
        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0
        
        latent_paddings = reversed(range(total_latent_sections))
        
        if total_latent_sections > 4:
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
        
        for section_idx, latent_padding in enumerate(latent_paddings):
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size
            
            print(f"Section {section_idx + 1}/{len(list(latent_paddings))}: latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}")
            
            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
            
            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
            
            if not models['high_vram']:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(
                    models['transformer'], target_device=gpu, preserved_memory_gb=gpu_memory_preservation
                )
            
            if use_teacache:
                models['transformer'].initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                models['transformer'].initialize_teacache(enable_teacache=False)
            
            def callback(d):
                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                print(f"  Step {current_step}/{steps} ({percentage}%) - Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}")
                return
            
            generated_latents = sample_hunyuan(
                transformer=models['transformer'],
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
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
            
            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)
            
            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)
            
            if not models['high_vram']:
                offload_model_from_device_for_memory_preservation(
                    models['transformer'], target_device=gpu, preserved_memory_gb=8
                )
                load_model_as_complete(models['vae'], target_device=gpu)
            
            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]
            
            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, models['vae']).cpu()
            else:
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = latent_window_size * 4 - 3
                
                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], models['vae']).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)
            
            if not models['high_vram']:
                unload_complete_models()
            
            output_filename = os.path.join(output_dir, f'{job_id}_{total_generated_latent_frames}.mp4')
            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)
            
            current_video_length = max(0, (total_generated_latent_frames * 4 - 3) / 30)
            print(f"  Decoded section. Latent shape: {real_history_latents.shape}, Pixel shape: {history_pixels.shape}")
            print(f"  Current video length: {current_video_length:.2f} seconds")
            print(f"  Saved: {output_filename}")
            
            if is_last_section:
                break
        
        final_output = os.path.join(output_dir, f'{job_id}_final.mp4')
        save_bcthw_as_mp4(history_pixels, final_output, fps=30, crf=mp4_crf)
        
        print(f"\n=== Generation Complete ===")
        print(f"Job ID: {job_id}")
        print(f"Final video: {final_output}")
        print(f"Video length: {(total_generated_latent_frames * 4 - 3) / 30:.2f} seconds")
        print(f"Total frames: {total_generated_latent_frames * 4 - 3}")
        
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


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== FramePack CLI ===")
    print(f"Input image: {args.input_image}")
    print(f"Prompt: '{args.prompt}'")
    print(f"Output directory: {args.output_dir}")
    print(f"Video length: {args.total_seconds} seconds")
    print(f"Seed: {args.seed}")
    print(f"Steps: {args.steps}")
    print(f"Use TeaCache: {args.use_teacache}")
    print()
    
    # Load models
    models = load_models()
    
    # Generate video
    try:
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
        
        print(f"\nSUCCESS! Video saved to: {output_path}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    """
    # Basic usage
    python predict.py --input_image /path/to/image.jpg --prompt "The girl dances gracefully"

    # With custom parameters
    python predict.py \
        --input_image /path/to/image.jpg \
        --prompt "A character doing some simple body movements" \
        --total_seconds 10.0 \
        --seed 42 \
        --steps 30 \
        --output_dir ./my_outputs/

    # Disable TeaCache for better quality (slower)
    python predict.py \
        --input_image /path/to/image.jpg \
        --prompt "Dancing scene" \
        --no_teacache

    # High quality output
    python predict.py \
        --input_image /path/to/image.jpg \
        --prompt "Dance performance" \
        --mp4_crf 0 \
        --steps 50
    """
    main()