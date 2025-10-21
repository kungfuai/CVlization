from PIL import Image
import torch
import numpy as np
from diffusers.models import AutoencoderKL
from cvlization.dataset.flying_mnist import FlyingMNISTDatasetBuilder

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
for i in range(3):
    # run the model
    with torch.no_grad():
        out = vae.encode(x)
        z = out.latent_dist.sample()
        print(f"repetition {i+1}", z.mean(), z.min(), z.max())
    

db = FlyingMNISTDatasetBuilder(resolution=256, max_frames_per_video=16, dataset_name="flying_mnist")
ds = db.training_dataset()
print(len(ds), "examples")
vid = ds[3]["video"]
i_frame = 10
vid = vid.unsqueeze(0).to(device)[:, :, i_frame, :, :] # take the i_frame-th frame
print(vid.shape, vid[0, 0].mean().item())
# save the first frame
frame = vid[0].detach().cpu()
print(f"frame mean: {frame.mean().item()}")
frame = frame.permute(1, 2, 0).numpy()
frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-6)
frame = (frame * 255).astype(np.uint8)
Image.fromarray(frame).save("data/tmp_frame.png")

with torch.no_grad():
    z = vae.encode(vid).latent_dist.sample()
    print(z.shape, z.mean(), z.min(), z.max())
    # corrupt the bottom half of the latent
    z[:, :, 26:, :] = torch.randn_like(z[:, :, 26:, :]) * 5
    decoded = vae.decode(z)
    print("decoded:", decoded.sample.shape)
    # save image to png
    recon_img = decoded.sample.detach().cpu()
    print("recon:", recon_img.shape, recon_img.mean().item())
    recon_img = recon_img[0].permute(1, 2, 0).numpy()
    recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min() + 1e-6)
    recon_img = (recon_img * 255).astype(np.uint8)
    Image.fromarray(recon_img).save("data/tmp_recon.png")
    print(f"reconstruction saved to data/tmp_recon.png")