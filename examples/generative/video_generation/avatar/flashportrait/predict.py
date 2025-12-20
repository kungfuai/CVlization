#!/usr/bin/env python3
"""
FlashPortrait inference wrapper for CVlization.

Generates portrait animation from a reference image and driving video.
Based on: https://github.com/Francis-Rings/FlashPortrait
"""

import argparse
import os
import sys

# Add vendor to path
VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
sys.path.insert(0, VENDOR_DIR)


def find_replacement(a):
    """Find the nearest valid frame count divisible by 4 after subtracting 1."""
    while a > 0:
        if (a - 1) % 4 == 0:
            return a
        a -= 1
    return 0


def download_models_if_needed(cache_dir):
    """Download FlashPortrait and Wan2.1 models if not present."""
    from huggingface_hub import snapshot_download

    print("Loading FlashPortrait weights...")
    fp_path = snapshot_download("FrancisRing/FlashPortrait", cache_dir=cache_dir)

    print("Loading Wan2.1-I2V-14B-720P model...")
    wan_path = snapshot_download("Wan-AI/Wan2.1-I2V-14B-720P", cache_dir=cache_dir)

    return fp_path, wan_path


def get_emo_feature(video_path, face_aligner, pd_fpg_motion, device):
    """Extract emotion features from driving video."""
    import cv2
    import torch
    from wan.models.pdf import det_landmarks, get_drive_expression_pd_fgc

    pd_fpg_motion = pd_fpg_motion.to(device)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_list = []
    ret, frame = cap.read()
    while ret:
        frame_list.append(frame.copy())
        ret, frame = cap.read()
    cap.release()

    num_frames = find_replacement(len(frame_list))
    frame_list = frame_list[:num_frames]

    landmark_list = det_landmarks(face_aligner, frame_list)[1]
    emo_list = get_drive_expression_pd_fgc(pd_fpg_motion, frame_list, landmark_list, device)

    emo_feat_list = []
    head_emo_feat_list = []
    for emo in emo_list:
        headpose_emb = emo["headpose_emb"]
        eye_embed = emo["eye_embed"]
        emo_embed = emo["emo_embed"]
        mouth_feat = emo["mouth_feat"]
        emo_feat = torch.cat([eye_embed, emo_embed, mouth_feat], dim=1)
        head_emo_feat = torch.cat([headpose_emb, emo_feat], dim=1)
        emo_feat_list.append(emo_feat)
        head_emo_feat_list.append(head_emo_feat)

    emo_feat_all = torch.cat(emo_feat_list, dim=0)
    head_emo_feat_all = torch.cat(head_emo_feat_list, dim=0)
    return emo_feat_all, head_emo_feat_all, fps, num_frames


def main():
    parser = argparse.ArgumentParser(description="FlashPortrait - Portrait Animation")

    # Required inputs
    parser.add_argument("--image", type=str, required=True, help="Reference portrait image")
    parser.add_argument("--video", type=str, required=True, help="Driving video for motion")
    parser.add_argument("--output", "-o", type=str, default="outputs/output.mp4", help="Output video path")

    # Fast mode (6x speedup)
    parser.add_argument("--fast", action="store_true",
                       help="Enable fast mode: 4-step inference with LoRA + tiny VAE (6x faster)")

    # Generation settings
    parser.add_argument("--prompt", type=str, default="A person talking", help="Text prompt")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--steps", type=int, default=None,
                       help="Inference steps (default: 4 with --fast, 30 otherwise)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Resolution
    parser.add_argument("--max_size", type=int, default=512, help="Max resolution (512, 720, 1280)")

    # CFG scales
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="Guidance scale")
    parser.add_argument("--text_cfg_scale", type=float, default=1.0, help="Text CFG scale")
    parser.add_argument("--emo_cfg_scale", type=float, default=4.0, help="Emotion CFG scale (2-5)")

    # Context settings
    parser.add_argument("--context_overlap", type=int, default=30, help="Context overlap (10-40)")
    parser.add_argument("--context_size", type=int, default=51, help="Context window size")
    parser.add_argument("--sub_num_frames", type=int, default=201, help="Frames per batch")
    parser.add_argument("--latents_num_frames", type=int, default=51, help="Latent frames per batch")

    # GPU settings
    parser.add_argument("--gpu_mode", type=str, default="model_cpu_offload",
                       choices=["model_full_load", "model_cpu_offload", "sequential_cpu_offload",
                               "model_full_load_and_qfloat8", "model_cpu_offload_and_qfloat8"],
                       help="GPU memory mode")

    # Model paths (optional, auto-downloaded if not specified)
    parser.add_argument("--cache_dir", type=str, default=None, help="HuggingFace cache directory")

    args = parser.parse_args()

    # Set default steps based on mode
    if args.steps is None:
        args.steps = 4 if args.fast else 30

    # Imports (after argparse to speed up --help)
    import numpy as np
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    from omegaconf import OmegaConf
    from PIL import Image
    from transformers import AutoTokenizer

    from wan.models.face_align import FaceAlignment
    from wan.models.face_model import FaceModel
    from wan.models.pdf import FanEncoder
    from wan.models.portrait_encoder import PortraitEncoder
    from wan.pipeline.pipeline_wan_long import WanI2VLongPipeline
    from wan.models import AutoencoderKLWan, CLIPModel, WanT5EncoderModel, WanTransformer3DModel
    from wan.utils.utils import filter_kwargs, save_videos_grid
    from wan.utils.fp8_optimization import convert_model_weight_to_float8, replace_parameters_by_name, convert_weight_dtype_wrapper

    # Setup paths
    cache_dir = args.cache_dir or os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    print("=" * 70)
    print("FlashPortrait - Portrait Animation")
    print("=" * 70)
    print(f"Image:      {args.image}")
    print(f"Video:      {args.video}")
    print(f"Output:     {args.output}")
    print(f"Max size:   {args.max_size}")
    print(f"GPU mode:   {args.gpu_mode}")
    print(f"Fast mode:  {args.fast} (steps: {args.steps})")
    print("=" * 70)

    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU detected, inference will be very slow")

    # Download models if needed
    fp_path, wan_path = download_models_if_needed(cache_dir)

    # Resolve model paths
    transformer_path = os.path.join(fp_path, "transformer.pt")
    portrait_encoder_path = os.path.join(fp_path, "portrait_encoder.pt")
    det_model_path = os.path.join(fp_path, "face_det.onnx")
    alignment_model_path = os.path.join(fp_path, "face_landmark.onnx")
    pd_fpg_model_path = os.path.join(fp_path, "pd_fpg.pth")

    # Fast mode paths
    fast_lora_path = os.path.join(fp_path, "fast_lora_rank64.safetensors")
    fast_vae_path = os.path.join(fp_path, "fast_vae.pth")

    print(f"\nModel paths:")
    print(f"  Wan2.1:           {wan_path}")
    print(f"  Transformer:      {transformer_path}")
    if args.fast:
        print(f"  Fast LoRA:        {fast_lora_path}")
        print(f"  Fast VAE:         {fast_vae_path}")

    # Load config
    config_path = os.path.join(VENDOR_DIR, "config/wan2.1/wan_civitai.yaml")
    config = OmegaConf.load(config_path)

    weight_dtype = torch.bfloat16

    # Load models
    print("\nLoading models...")

    # Transformer
    print("  Loading transformer...")
    transformer = WanTransformer3DModel.from_pretrained(
        os.path.join(wan_path, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )

    # Load FlashPortrait weights
    print(f"  Loading FlashPortrait weights...")
    transformer_state_dict = torch.load(transformer_path, map_location="cpu", weights_only=True)
    transformer_state_dict = transformer_state_dict.get("state_dict", transformer_state_dict)
    m, u = transformer.load_state_dict(transformer_state_dict, strict=False)
    print(f"    Missing keys: {len(m)}, Unexpected keys: {len(u)}")

    # VAE
    print("  Loading VAE...")
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(wan_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(weight_dtype)

    # Tokenizer
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(wan_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    # Text encoder
    print("  Loading text encoder...")
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(wan_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).eval()

    # CLIP encoder
    print("  Loading CLIP encoder...")
    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(wan_path, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
    ).to(weight_dtype).eval()

    # Face models
    print("  Loading face models...")
    face_aligner = FaceModel(
        face_alignment_module=FaceAlignment(
            gpu_id=None,
            alignment_model_path=alignment_model_path,
            det_model_path=det_model_path,
        ),
        reset=False,
    )

    pd_fpg_motion = FanEncoder()
    pd_fpg_checkpoint = torch.load(pd_fpg_model_path, map_location="cpu")
    pd_fpg_motion.load_state_dict(pd_fpg_checkpoint, strict=False)
    pd_fpg_motion = pd_fpg_motion.eval()

    # Portrait encoder
    print("  Loading portrait encoder...")
    portrait_encoder_state_dict = torch.load(portrait_encoder_path, map_location="cpu", weights_only=True)

    portrait_encoder_state_dict_sub_proj = {}
    portrait_encoder_state_dict_sub_mouth = {}
    portrait_encoder_state_dict_sub_emo = {}

    for k, v in portrait_encoder_state_dict.items():
        if k.startswith("proj_model."):
            portrait_encoder_state_dict_sub_proj[k[len("proj_model."):]] = v
        elif k.startswith("mouth_proj_model."):
            portrait_encoder_state_dict_sub_mouth[k[len("mouth_proj_model."):]] = v
        elif k.startswith("emo_proj_model."):
            portrait_encoder_state_dict_sub_emo[k[len("emo_proj_model."):]] = v

    portrait_encoder = PortraitEncoder(adapter_in_dim=768, adapter_proj_dim=2048)
    portrait_encoder.proj_model.load_state_dict(portrait_encoder_state_dict_sub_proj)
    portrait_encoder.mouth_proj_model.load_state_dict(portrait_encoder_state_dict_sub_mouth)
    portrait_encoder.emo_proj_model.load_state_dict(portrait_encoder_state_dict_sub_emo)
    portrait_encoder = portrait_encoder.eval()

    # Scheduler - use step distillation scheduler for fast mode
    print("  Creating scheduler...")
    if args.fast:
        from wan.utils.step_distill_scheduler import StepDistillScheduler
        scheduler = StepDistillScheduler(
            num_train_timesteps=1000,
            shift=5,
            denoising_step_list=[1000, 750, 500, 250],
        )
    else:
        scheduler = FlowMatchEulerDiscreteScheduler(
            **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
        )

    # Pipeline
    print("  Creating pipeline...")
    pipeline = WanI2VLongPipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        clip_image_encoder=clip_image_encoder,
        portrait_encoder=portrait_encoder,
    )

    # Apply GPU memory mode FIRST (before fast mode setup)
    print(f"  Applying GPU mode: {args.gpu_mode}")
    if args.gpu_mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer, ["modulation",], device=device)
        transformer.freqs = transformer.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif args.gpu_mode == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        pipeline.enable_model_cpu_offload(device=device)
        pipeline.portrait_encoder = pipeline.portrait_encoder.to(device)
    elif args.gpu_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
        pipeline.portrait_encoder = pipeline.portrait_encoder.to(device)
    elif args.gpu_mode == "model_full_load_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        pipeline.to(device=device)
    else:  # model_full_load
        pipeline.to(device=device)

    # Apply step distillation LoRA for fast mode (after GPU mode so models are on device)
    if args.fast and os.path.exists(fast_lora_path):
        print("  Applying step distillation LoRA...")
        from wan.utils.step_distill_lora import apply_step_distill_lora
        pipeline = apply_step_distill_lora(
            pipeline,
            lora_path=fast_lora_path,
            strength=1.0,
            dtype=torch.float32,
        )
        print(f"    4-step inference enabled (~7.5x speedup)")

    # Setup tiny VAE for fast mode (after GPU mode so VAE is on device)
    if args.fast and os.path.exists(fast_vae_path):
        print("  Setting up Tiny VAE...")
        try:
            from wan.models.wan_vae_tiny_pipeline import setup_tiny_vae
            pipeline = setup_tiny_vae(
                pipeline=pipeline,
                model_type="wan2.1",
                tiny_vae_path=fast_vae_path,
                parallel_decode=False,
                need_scaled=False,
            )
            print(f"    Tiny VAE enabled (~2-3x faster decoding)")
        except Exception as e:
            print(f"    Warning: Tiny VAE setup failed: {e}")

    print("\nModels loaded successfully!")

    # Process inputs
    print("\nProcessing inputs...")

    # Load and resize reference image
    image_start = clip_image = Image.open(args.image).convert("RGB")
    width, height = image_start.size
    scale = args.max_size / max(width, height)
    width, height = int(width * scale), int(height * scale)

    # Ensure divisibility by 16
    if height % 16 != 0:
        height = (height + 15) // 16 * 16
    if width % 16 != 0:
        width = (width + 15) // 16 * 16

    print(f"  Resolution: {width}x{height}")

    image_start = image_start.resize([width, height], Image.LANCZOS)
    clip_image = clip_image.resize([width, height], Image.LANCZOS)

    # Create input video tensor (first frame repeated)
    input_video = torch.tile(
        torch.from_numpy(np.array(image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0),
        [1, 1, args.sub_num_frames, 1, 1]
    ) / 255
    input_video_mask = torch.zeros_like(input_video[:, :1])
    input_video_mask[:, :, 1:] = 255

    # Extract emotion features from driving video
    print(f"  Extracting emotion features from: {args.video}")
    emo_feat_all, head_emo_feat_all, fps, num_frames = get_emo_feature(
        args.video, face_aligner, pd_fpg_motion, device=device
    )
    emo_feat_all = emo_feat_all.unsqueeze(0)
    head_emo_feat_all = head_emo_feat_all.unsqueeze(0)
    print(f"  Extracted {num_frames} frames @ {fps:.1f} fps")

    # Generate
    print(f"\nGenerating video ({args.steps} steps)...")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    with torch.no_grad():
        sample = pipeline(
            args.prompt,
            num_frames=num_frames,
            negative_prompt=args.negative_prompt,
            height=height,
            width=width,
            generator=generator,
            guidance_scale=args.cfg_scale,
            num_inference_steps=args.steps,
            video=input_video,
            mask_video=input_video_mask,
            clip_image=clip_image,
            shift=5,
            context_overlap=args.context_overlap,
            context_size=args.context_size,
            latents_num_frames=args.latents_num_frames,
            ip_scale=1.0,
            head_emo_feat_all=head_emo_feat_all.to(device),
            sub_num_frames=args.sub_num_frames,
            text_cfg_scale=args.text_cfg_scale,
            emo_cfg_scale=args.emo_cfg_scale,
        ).videos

    # Remove first frame (it's the reference)
    sample = sample[:, :, 1:]

    # Save output (use path relative to script directory for container compatibility)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.output):
        output_path = os.path.join(script_dir, args.output)
    else:
        output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_videos_grid(sample, output_path, fps=fps)

    print(f"\nOutput saved to: {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
