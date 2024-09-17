import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

prompt = "Tracking shot,late afternoon light casting long shadows,a cyclist in athletic gear pedaling down a scenic mountain road,winding path with trees and a lake in the background,invigorating and adventurous atmosphere."

pipe = CogVideoXPipeline.from_pretrained(
    # "THUDM/CogVideoX-2b",
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16
)

pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator(device="cuda").manual_seed(0),
).frames[0]

export_to_video(video, "output.mp4", fps=8)