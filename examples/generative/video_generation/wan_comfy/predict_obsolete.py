import os
import torch
import numpy as np
from comfy import model_management
import comfy
from comfy.text_encoders.wan import WanT5Model, WanT5Tokenizer
from comfy.sd import CLIP
from comfy.ldm.wan.vae import WanVAE
from comfy.ldm.wan.model import WanModel
from comfy import clip_vision
from comfy import samplers
from comfy.model_base import WAN21, ModelType
from comfy.sd import load_state_dict_guess_config, load_checkpoint_guess_config, load_diffusion_model_state_dict
from comfy.model_patcher import ModelPatcher
from comfy.model_detection import detect_unet_config
from nodes_model_advanced import ModelSamplingSD3
from nodes_wan import WanImageToVideo
from nodes import CLIPLoader, CLIPVisionLoader, UNETLoader, VAELoader, CLIPVisionEncode, LoadImage, CLIPTextEncode, KSampler, VAEDecode
from safetensors.torch import load_file
import folder_paths
from PIL import Image


if __name__ == "__main__":
    # Implemented based on this I2V workflow: https://comfyanonymous.github.io/ComfyUI_examples/wan/
    CACHE_DIR = os.getenv("CACHE_DIR", "/root/.cache")

    height = 512
    width = 512
    
    text_prompt = "A beautiful cat"
    # image_prompt = Image.open("examples/video_gen/animate_x/data/images/1.jpg")
    # image_prompt = image_prompt.resize((width, height))

    image_prompt, image_mask = LoadImage().load_image(image="examples/video_gen/animate_x/data/images/1.jpg")
    

    with torch.inference_mode():    # load text encoder
        text_encoder_device = "cpu"
        text_encoder_clip_path = os.path.join(CACHE_DIR, "models/wan/umt5_xxl_fp8_e4m3fn_scaled.safetensors")
        clip_type = comfy.sd.CLIPType.WAN
        model_options = {}
        if text_encoder_device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
        # clip = comfy.sd.load_clip(ckpt_paths=[text_encoder_clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type, model_options=model_options)
        clip_loader = CLIPLoader()
        clip = clip_loader.load_clip(clip_name=text_encoder_clip_path, type="wan", device="default")[0] #, device=text_encoder_device)
        num_params = sum(p.numel() for p in clip.patcher.model.parameters())
        print(f"Text Encoder: {num_params / 1e9:.3f}B")

        # load clip vision model
        clip_vision_loader = CLIPVisionLoader()
        clip_vision_model_path = os.path.join(CACHE_DIR, "models/wan/clip_vision_h.safetensors")
        clip_vision_model = clip_vision_loader.load_clip(clip_name=clip_vision_model_path)[0] #, device=text_encoder_device)
        # clip_vision_model = clip_vision.load(os.path.join(CACHE_DIR, "models/wan/clip_vision_h.safetensors"))
        num_params = sum(p.numel() for p in clip_vision_model.model.parameters())
        print(f"Clip Vision Model: {num_params / 1e9:.3f}B")

        # load latent diffusion model (called UNET in ComfyUI)
        unet_loader = UNETLoader()
        diffusion_model_path = os.path.join(CACHE_DIR, "models/wan/wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors")
        patched_diffusion_model = unet_loader.load_unet(unet_name=diffusion_model_path, weight_dtype="fp8_e4m3fn")[0]
        # state_dict = load_file(os.path.join(CACHE_DIR, "models/wan/wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors"))
        # model_options["dtype"] = torch.float8_e4m3fn
        # patched_diffusion_model = load_diffusion_model_state_dict(state_dict, model_options=model_options)
        num_params = sum(p.numel() for p in patched_diffusion_model.model.parameters())
        print(f"WAN21 Model: {num_params / 1e9:.3f}B")

        
        free_memory = model_management.get_free_memory("cuda:0")
        print(f"***** After loading models, free_memory: {free_memory}")
        # Now we can generate a video
        # 1. encode image prompt
        clip_vision_encode = CLIPVisionEncode()
        image_encoded = clip_vision_encode.encode(clip_vision=clip_vision_model, image=image_prompt, crop=False)[0]
        free_memory = model_management.get_free_memory("cuda:0")
        print(f"***** After encoding image prompt, free_memory: {free_memory}")
        # image_encoded = clip_vision_model.encode_image(image_prompt, crop=None)
        """
        outputs = Output()
        outputs["last_hidden_state"] = out[0].to(comfy.model_management.intermediate_device())
        outputs["image_embeds"] = out[2].to(comfy.model_management.intermediate_device())
        outputs["penultimate_hidden_states"] = out[1].to(comfy.model_management.intermediate_device())
        """
        # 2. encode text prompt
        clip_text_encode = CLIPTextEncode()
        text_encoded = clip_text_encode.encode(clip=clip, text=text_prompt)[0]
        text_encoded_negative = clip_text_encode.encode(clip=clip, text="distorted")[0]
        free_memory = model_management.get_free_memory("cuda:0")
        print(f"***** After encoding text prompt, free_memory: {free_memory}")

        # Let's clear the memory.
        print(f"current loaded models: {len(model_management.current_loaded_models)}")
        for m in model_management.current_loaded_models:
            print(f"  {m.model.model.__class__.__name__}")
            m.model_unload(1e30)
            model_management.current_loaded_models.remove(m)
        # model_management.free_memory(1e30, "cuda:0")  # this did not work

        free_memory = model_management.get_free_memory("cuda:0")
        print(f"***** After clearing memory, free_memory: {free_memory}")
        print(f"current loaded models: {len(model_management.current_loaded_models)}")

        # 3. generate video

        # load vae
        vae_loader = VAELoader()
        vae_path = os.path.join(CACHE_DIR, "models/wan/wan_2.1_vae.safetensors")
        vae = vae_loader.load_vae(vae_name=vae_path)[0]

        model_sampling_sd3 = ModelSamplingSD3()
        patched_diffusion_model = model_sampling_sd3.patch(patched_diffusion_model, shift=8)[0]
        wan_image_to_video = WanImageToVideo()
        positive, negative, latent = wan_image_to_video.encode(
            positive=text_encoded,
            negative=text_encoded_negative,
            vae=vae,
            width=width,
            height=height,
            length=33,
            batch_size=1,
            clip_vision_output=image_encoded,
            start_image=image_prompt,
        )
        free_memory = model_management.get_free_memory("cuda:0")
        print(f"***** After encoding video, free_memory: {free_memory}")
        for m in model_management.current_loaded_models:
            print(f"  {m.model.model.__class__.__name__}")
            m.model_unload(1e30)
            model_management.current_loaded_models.remove(m)
        
        free_memory = model_management.get_free_memory("cuda:0")
        print(f"***** After clearing memory, free_memory: {free_memory}")

        print(f"latent: {latent['samples'].shape}, device: {latent['samples'].device}")
        # print(f"positive: {positive}")
        for pp in positive:
            print(f"positive: {pp[0].shape}, device: {pp[0].device}")
            for k, v in pp[1].items():
                if v is not None:
                    if hasattr(v, "shape"):
                        print(f"positive: {k}, {v.shape}, device: {v.device}")
                    elif hasattr(v, "keys"):
                        for kk, vv in v.items():
                            if vv is not None:
                                print(f"positive: {k}, {kk}, {vv.shape}, device: {vv.device}")

        noise = comfy.sample.prepare_noise(latent['samples'], seed=0)
        print(f"----- Done prepare_noise, noise: {noise.shape}, device: {noise.device}")

        for m in model_management.current_loaded_models:
            print(f"  {m.model.model.__class__.__name__}")
            m.model_unload(1e30)
            model_management.current_loaded_models.remove(m)
        
        free_memory = model_management.get_free_memory("cuda:0")
        print(f"***** After clearing memory, free_memory: {free_memory}")

        ksampler = KSampler()
        # sampler = samplers.KSampler(
        #     patched_diffusion_model,
        #     steps=20,
        #     device="cuda:0",
        #     sampler="uni_pc",
        #     scheduler="simple",
        #     denoise=1,
        #     model_options={'transformer_options': {}},
        # )
        print(f"device of patched_diffusion_model: {patched_diffusion_model.model.device}")
        video_latent = ksampler.sample(
            model=patched_diffusion_model,
            seed=0, steps=20, cfg=6,
            sampler_name="uni_pc", scheduler="simple",
            positive=positive, negative=negative,
            latent_image=latent,
            denoise=1
        )[0]
        # video_latent = sampler.sample(
        #     noise,
        #     positive=positive,
        #     negative=negative,
        #     latent_image=latent['samples'],
        #     cfg=6,
        # )

        # 4. decode video
        vae_decode = VAEDecode()
        video = vae.decode(video_latent)
        # 5. save video
        video.save("output.mp4")