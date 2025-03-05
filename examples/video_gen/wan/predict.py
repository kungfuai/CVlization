import os
import torch
import numpy as np
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
from nodes_wan import WanImageToVideo
from safetensors.torch import load_file
import folder_paths
from PIL import Image


if __name__ == "__main__":
    # Implemented based on this workflow: https://comfyui-wiki.com/_next/static/media/wan2.1-i2v-720p-workflow.a0cdedc2.jpg
    CACHE_DIR = os.getenv("CACHE_DIR", "/root/.cache")

    height = 512
    width = 512
    
    text_prompt = "A beautiful cat"
    image_prompt = Image.open("examples/video_gen/animate_x/data/images/1.jpg")
    image_prompt = image_prompt.resize((width, height))
    

    # load text encoder
    text_encoder_device = "cpu"
    text_encoder_clip_path = os.path.join(CACHE_DIR, "models/wan/umt5_xxl_fp8_e4m3fn_scaled.safetensors")
    clip_type = comfy.sd.CLIPType.WAN
    model_options = {}
    if text_encoder_device == "cpu":
        model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
    clip = comfy.sd.load_clip(ckpt_paths=[text_encoder_clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type, model_options=model_options)
    num_params = sum(p.numel() for p in clip.patcher.model.parameters())
    
    # text_encoder = WanT5Model(device="cpu")
    # state_dict = load_file(os.path.join(CACHE_DIR, "models/wan/umt5_xxl_fp8_e4m3fn_scaled.safetensors"))
    # text_encoder.load_state_dict(state_dict, strict=False)
    # text_encoder.eval()
    # tokenizer = WanT5Tokenizer(tokenizer_data=state_dict)
    # num_params = sum(p.numel() for p in text_encoder.parameters())
    print(f"Text Encoder: {num_params / 1e9:.3f}B")

    # load clip vision model
    clip_vision_model = clip_vision.load(os.path.join(CACHE_DIR, "models/wan/clip_vision_h.safetensors"))
    num_params = sum(p.numel() for p in clip_vision_model.model.parameters())
    print(f"Clip Vision Model: {num_params / 1e9:.3f}B")
    print(f"Clip Vision Model: {list(clip_vision_model.model.parameters())[5].device}")
    # load latent diffusion model (called UNET in ComfyUI)
    # diffusion_model = WanModel(
    #     device="cpu", model_type="i2v", dim=5120,
    #     ffn_dim=13824, in_dim=36,
    #     operations=ops.manual_cast,
    #     # operations=torch.nn
    # )
    state_dict = load_file(os.path.join(CACHE_DIR, "models/wan/wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors"))
    # model_patcher, clip_, vae_, clipvision_ = load_checkpoint_guess_config(
    #     os.path.join(CACHE_DIR, "models/wan/wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors"),
    #     output_model=True,
    # )
    # diffusion_model.load_state_dict(state_dict, strict=False)
    # diffusion_model.eval()
    # num_params = sum(p.numel() for p in diffusion_model.parameters())
    # print(f"Diffusion Model: {num_params / 1e9:.3f}B")
    # model_config = detect_unet_config(state_dict, key_prefix="")
    # wan_model = WAN21(
    #     model_config=model_config,
    #     model_type=ModelType.FLOW,
    #     image_to_video=True,
    #     device="cpu"
    # )
    model_options["dtype"] = torch.float8_e4m3fn
    patched_diffusion_model = load_diffusion_model_state_dict(state_dict, model_options=model_options)
    num_params = sum(p.numel() for p in patched_diffusion_model.model.parameters())
    print(f"WAN21 Model: {num_params / 1e9:.3f}B")

    # load vae
    vae_path = os.path.join(CACHE_DIR, "models/wan/wan_2.1_vae.safetensors")
    sd = comfy.utils.load_torch_file(vae_path)
    vae = comfy.sd.VAE(sd=sd)
    vae.first_stage_model.to("cpu")
    vae.patcher.model.to("cpu")

    # vae = WanVAE(dim=96, z_dim=16)
    # state_dict = load_file(os.path.join(CACHE_DIR, "models/wan/wan_2.1_vae.safetensors"))
    # vae.load_state_dict(state_dict, strict=False)
    # vae.eval()
    # num_params = sum(p.numel() for p in vae.parameters())
    # print(f"VAE: {num_params / 1e9:.3f}B")
    
    # Now we can generate a video
    # 1. encode image prompt
    image_prompt = torch.from_numpy(np.array(image_prompt)).unsqueeze(0).float()
    image_encoded = clip_vision_model.encode_image(image_prompt, crop=False)
    """
    outputs = Output()
    outputs["last_hidden_state"] = out[0].to(comfy.model_management.intermediate_device())
    outputs["image_embeds"] = out[2].to(comfy.model_management.intermediate_device())
    outputs["penultimate_hidden_states"] = out[1].to(comfy.model_management.intermediate_device())
    """
    # 2. encode text prompt
    ### ComfyUI's way    
    tokens = clip.tokenize(text_prompt)
    text_encoded = clip.encode_from_tokens_scheduled(tokens)
    # print(f"Text Prompt encoded: {text_encoded[0].shape}")

    ### ZZ's way (ignore)
    # tokenized = tokenizer.tokenize_with_weights(text_prompt)
    # text_clip_model = getattr(text_encoder, text_encoder.clip)
    # print(f"tokenized: {tokenized}")
    # print(f"text_clip_model: {text_encoder.clip}")
    # text_z, text_pooled_output = text_encoder.encode_token_weights(tokenized)
    

    # 3. generate video
    c_in = 36
    F = 16

    wan_image_to_video = WanImageToVideo()
    # image_mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
    positive, negative, latent = wan_image_to_video.encode(
        positive=text_encoded,
        negative=text_encoded,  # for debug
        vae=vae,
        width=width,
        height=height,
        length=7,
        batch_size=1,
        clip_vision_output=image_encoded,
        start_image=image_prompt,  # (image_prompt, image_mask),
    )
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
    print(f"noise: {noise.shape}, device: {noise.device}")

    sampler = samplers.KSampler(
        patched_diffusion_model,
        steps=20,
        device="cpu", sampler="uni_pc", scheduler="simple",
        # denoise=0
        denoise=1
    )
    print(f"device of patched_diffusion_model: {patched_diffusion_model.model.device}")
    video_latent = sampler.sample(
        noise,
        positive=positive,
        negative=negative,
        latent_image=latent['samples'],
        cfg=6
    )
    print(f"Video Latent: {video_latent.shape}")
    # 4. decode video
    video = vae.decode(video_latent)
    # 5. save video
    video.save("output.mp4")