import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
# pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

# denoising
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()
pipe.enable_sequential_cpu_offload() # offloads modules to CPU on a submodule level (rather than model level)


prompt = "A cat holding a sign that says hello world"
prompt = """
A charismatic speaker is captured mid-speech. She has long smooth hair that's slightly messy on top. She has an angular face, clean shaven, adorned with circular glasses with red rims, is animated as she gestures with she left hand. She is holding a black microphone in her right hand, speaking passionately.

The lady is wearing a light grey sweater over a white t-shirt. She's also wearing a simple black lanyard hanging around his neck. The lanyard badge has the text "KUNGFU.AI", very visible. The photo includes the whole upperbody.

Behind her, there is a blurred background with a white banner containing logos and text (including kungfu.ai written in pink), a professional conference setting.
"""
image = pipe(
    prompt,
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256,
    height=1024,
    width=1024,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save("flux-schnell.png")