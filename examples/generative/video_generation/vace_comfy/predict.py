# This is a generated script from a ComfyUI workflow for VACE (Video Animation Control Extension).

import os
import random
import sys
import argparse
from typing import Sequence, Mapping, Any, Union
import torch
from nodes_wan import WanVaceToVideo, CreateFadeMaskAdvanced
from nodes_images import SaveAnimatedWEBP
from nodes_model_advanced import ModelSamplingSD3, CFGZeroStar, UNetTemporalAttentionMultiply, SkipLayerGuidanceDiT


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS
from nodes_extra import NODE_CLASS_MAPPINGS as EXTRA_NODE_CLASS_MAPPINGS
from comfyui_gguf import GGUF_NODE_CLASS_MAPPINGS

# Merge the node mappings
NODE_CLASS_MAPPINGS.update(EXTRA_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(GGUF_NODE_CLASS_MAPPINGS)


def parse_args():
    parser = argparse.ArgumentParser(description='WAN VACE Video Generation')
    parser.add_argument('-p', '--prompt', type=str, 
                      default="Animation of a young woman with light blonde hair, narrowed green eyes, and an expression that looks slightly annoyed or like she's contemplating something. A large, flowing curtain is also present. The wind is blowing from the left, causing her hair and the curtain to sway to the right. The linework is clean yet has a slightly sketchy touch, drawn in a modern animation style.",
                      help='Positive prompt for video generation')
    parser.add_argument('-n', '--negative-prompt', type=str, 
                      default="Overly vibrant colors, overexposed, static, blurry details, subtitles, style, work, artwork, painting, screen, frame, still, motionless, washed out, dull colors, grayish, worst quality, low quality, jpeg artifacts, compression artifacts, ugly, incomplete, mutilated, extra fingers, too many fingers, poorly drawn hands, bad hands, poorly drawn face, bad face, deformed, mutated, disfigured, malformed limbs, fused fingers, motionless scene, static image, cluttered background, messy background, three legs, too many people in background, crowded background, walking backward, backward movement",
                      help='Negative prompt for video generation')
    parser.add_argument('-i', '--input-images', type=str, nargs='+', 
                      default=["examples/video_gen/animate_x/data/images/1.jpg"],
                      help='Paths to input images for VACE control')
    parser.add_argument('-o', '--output-dir', type=str, default="output",
                      help='Directory to save output video')
    parser.add_argument('--fps', type=int, default=16,
                      help='Frames per second for output video')
    parser.add_argument('--cfg', type=float, default=4.0,
                      help='Classifier-free guidance scale')
    parser.add_argument('--steps', type=int, default=20,
                      help='Number of sampling steps')
    parser.add_argument('--width', type=int, default=720,
                      help='Output video width')
    parser.add_argument('--height', type=int, default=720,
                      help='Output video height')
    parser.add_argument('--length', type=int, default=49,
                      help='Number of frames to generate')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed for generation (default: random)')
    parser.add_argument('--model-shift', type=float, default=8.0,
                      help='Model sampling shift parameter')
    parser.add_argument('--fade-schedule', type=str, 
                      default="0:(0.0),\n1:(1.0),\n7:(1.0),\n15:(1.0),\n16:(0.0),\n17:(1.0),\n31:(1.0),\n32:(0.0),\n33:(1.0),\n47:(1.0),\n48:(0.0)",
                      help='Fade schedule for mask creation')
    parser.add_argument('--models_root_dir', default="/root/.cache/models/wan")
    return parser.parse_args()


def main():
    args = parse_args()
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    if args.seed is None:
        args.seed = random.randint(1, 2**64)
    
    with torch.inference_mode():
        # Load CLIP text encoder
        cliploader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
        cliploader_38 = cliploader.load_clip(
            clip_name=f"{args.models_root_dir}/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            type="wan",
            device="default",
        )
        print(f"CLIP Text Encoder loaded")

        # Encode positive prompt
        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_6 = cliptextencode.encode(
            text=args.prompt,
            clip=get_value_at_index(cliploader_38, 0),
        )

        # Encode negative prompt
        cliptextencode_7 = cliptextencode.encode(
            text=args.negative_prompt,
            clip=get_value_at_index(cliploader_38, 0),
        )

        # Load VAE
        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_39 = vaeloader.load_vae(vae_name=f"{args.models_root_dir}/wan_2.1_vae.safetensors")
        print(f"VAE loaded")

        # Load input images
        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        input_images = []
        for img_path in args.input_images:
            if os.path.exists(img_path):
                loaded_img = loadimage.load_image(image=img_path)
                input_images.append(get_value_at_index(loaded_img, 0))
                print(f"Loaded image: {img_path}")
            else:
                print(f"Warning: Image not found: {img_path}")

        # Get image size from first image
        if input_images:
            first_image = input_images[0]
            getimagesize = NODE_CLASS_MAPPINGS["GetImageSize+"]()
            size_info = getimagesize.get_size(image=first_image)
            image_width = get_value_at_index(size_info, 0)
            image_height = get_value_at_index(size_info, 1)
            print(f"Image size: {image_width}x{image_height}")
        else:
            image_width = args.width
            image_height = args.height

        # Create empty image for padding
        emptyimage = NODE_CLASS_MAPPINGS["EmptyImage"]()
        empty_img = emptyimage.generate(
            width=image_width,
            height=image_height,
            batch_size=1,
            color=16777215  # White
        )

        # Duplicate images to create video frames
        vhs_duplicate = NODE_CLASS_MAPPINGS["VHS_DuplicateImages"]()
        duplicated_frames = vhs_duplicate.duplicate_images(
            images=get_value_at_index(empty_img, 0),
            multiply_by=15
        )

        # Create image batch for control video
        impact_batch = NODE_CLASS_MAPPINGS["ImpactMakeImageBatch"]()
        control_video_batch = impact_batch.make_batch(
            image1=input_images[0] if len(input_images) > 0 else get_value_at_index(duplicated_frames, 0),
            image2=get_value_at_index(duplicated_frames, 0),
            image3=input_images[1] if len(input_images) > 1 else get_value_at_index(duplicated_frames, 0),
            image4=get_value_at_index(duplicated_frames, 0),
            image5=input_images[2] if len(input_images) > 2 else get_value_at_index(duplicated_frames, 0),
            image6=get_value_at_index(duplicated_frames, 0),
            image7=input_images[3] if len(input_images) > 3 else get_value_at_index(duplicated_frames, 0),
        )

        # Create fade mask
        create_fade_mask = CreateFadeMaskAdvanced()
        fade_mask = create_fade_mask.createfademask(
            invert=False,
            frames=args.length,
            width=image_width,
            height=image_height,
            points_string=args.fade_schedule,
            interpolation="linear"
        )

        # Load UNET model
        unetloader = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
        unetloader_147 = unetloader.load_unet(
            unet_name=f"{args.models_root_dir}/Wan2.1-VACE-14B-Q6_K.gguf"
        )
        print(f"UNET model loaded")

        # Apply model sampling
        modelsamplingsd3 = ModelSamplingSD3()
        model_with_sampling = modelsamplingsd3.patch(
            model=get_value_at_index(unetloader_147, 0),
            shift=args.model_shift
        )

        # Apply skip layer guidance
        skip_layer_guidance = SkipLayerGuidanceDiT()
        model_with_skip = skip_layer_guidance.patch(
            model=get_value_at_index(model_with_sampling, 0),
            skip_layers="9,10",
            skip_layers_cfg="9,10", 
            scale=3.0,
            start=0.01,
            end=0.8,
            cfg_scale_start=0.0
        )

        # Apply temporal attention multiply
        temporal_attn = UNetTemporalAttentionMultiply()
        model_with_temporal = temporal_attn.patch(
            model=get_value_at_index(model_with_skip, 0),
            self_attn_mult=1.0,
            cross_attn_mult=1.0,
            temporal_attn_mult=1.2,
            output_mult=1.3
        )

        # Apply CFG Zero Star
        cfg_zero_star = CFGZeroStar()
        final_model = cfg_zero_star.patch(
            model=get_value_at_index(model_with_temporal, 0)
        )

        # Setup VACE conditioning
        wan_vace_to_video = WanVaceToVideo()
        vace_conditioning = wan_vace_to_video.encode(
            positive=get_value_at_index(cliptextencode_6, 0),
            negative=get_value_at_index(cliptextencode_7, 0),
            vae=get_value_at_index(vaeloader_39, 0),
            control_video=get_value_at_index(control_video_batch, 0),
            control_masks=get_value_at_index(fade_mask, 0),
            reference_image=None,
            width=image_width,
            height=image_height,
            length=args.length,
            batch_size=1,
            trim_latent=1
        )

        # Sample with KSampler
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        print(f"Starting sampling with {args.steps} steps...")
        sampled_latent = ksampler.sample(
            seed=args.seed,
            steps=args.steps,
            cfg=args.cfg,
            sampler_name="uni_pc",
            scheduler="simple",
            denoise=1.0,
            model=get_value_at_index(final_model, 0),
            positive=get_value_at_index(vace_conditioning, 0),
            negative=get_value_at_index(vace_conditioning, 1),
            latent_image=get_value_at_index(vace_conditioning, 2),
        )
        print(f"Sampling complete. Decoding...")

        # Decode VAE
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        decoded_images = vaedecode.decode(
            samples=get_value_at_index(sampled_latent, 0),
            vae=get_value_at_index(vaeloader_39, 0),
        )

        # Save as video
        vhs_video_combine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
        video_output = vhs_video_combine.combine_video(
            images=get_value_at_index(decoded_images, 0),
            frame_rate=args.fps,
            loop_count=0,
            filename_prefix=os.path.join(args.output_dir, "Wan2.1_VACE"),
            format="video/h264-mp4",
            pix_fmt="yuv420p",
            crf=19,
            save_metadata=True,
            pingpong=False,
            save_output=True,
        )
        
        print(f"\nVACE video generation complete!")
        print(f"Output saved to: {args.output_dir}")
        print(f"Seed used: {args.seed}")


if __name__ == "__main__":
    main() 