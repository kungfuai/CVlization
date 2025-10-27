from diffusers import StableDiffusionPipeline
import torch


model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]

# Save to outputs directory
import os
os.makedirs("outputs", exist_ok=True)
image.save("outputs/astronaut_rides_horse.png")
print("Image saved to outputs/astronaut_rides_horse.png")
