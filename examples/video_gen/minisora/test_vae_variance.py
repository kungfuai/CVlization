import torch
from diffusers.models import AutoencoderKL

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
vae.eval()

# create a sample batch
batch_size = 2
height = 64
width = 64
num_channels = 3
# x = torch.ones(batch_size, num_channels, height, width)
x = torch.randn(batch_size, num_channels, height, width) * 100

device = "cuda"
vae = vae.to(device)
x = x.to(device)
for _ in range(3):
    # run the model
    with torch.no_grad():
        out = vae.encode(x)
        z = out.latent_dist.sample()
        print(z.mean(), z.min(), z.max())
    
