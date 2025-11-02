import torch
import pytest

if not torch.cuda.is_available():
    pytest.skip("requires a CUDA-capable GPU", allow_module_level=True)

from PIL import Image
import numpy as np
from diffusers.models import AutoencoderKL
from cvlization.dataset.flying_mnist import FlyingMNISTDatasetBuilder
from cvlization.torch.net.vae.video_vqvae import VQVAE

vae = VQVAE.from_pretrained("zzsi_kungfu/videogpt/model-kbu39ped:v11")
vae.eval()

# create a sample batch
batch_size = 2
height = 64
width = 64
num_frames = 4
num_channels = 3
# x = torch.ones(batch_size, num_channels, height, width)
x = torch.randn(batch_size, num_channels, num_frames, height, width) * 100

device = "cuda"
vae = vae.to(device)
x = x.to(device)
for i in range(3):
    # run the model
    with torch.no_grad():
        out = vae.encode(x)
        z = out
        print(f"repetition {i+1}", z.mean(), z.min(), z.max())


db = FlyingMNISTDatasetBuilder(
    resolution=256, max_frames_per_video=16, dataset_name="flying_mnist"
)
ds = db.training_dataset()
print(len(ds), "examples")
vid = ds[3]["video"]
vid = vid.unsqueeze(0).to(device)[:, :, :, :, :]
print(vid.shape, vid[0, 0].mean().item())
# save the first frame
frame = vid[0].detach().cpu()

with torch.no_grad():
    z = vae.encode(vid)
    z = vae.vq(z)["z_recon"]
    # corrupt the latents
    std = z.std()
    print("std of z:", std.item())
    z[:, :, 1:, :, :] = torch.randn_like(z[:, :, 1:, :, :]) * std
    print(z.shape, z.mean(), z.min(), z.max())
    decoded = vae.decode(z)
    print("decoded:", decoded.shape)
    # save image to png
    recon_img = decoded.detach().cpu()[:, :, 0, :, :]
    print("recon:", recon_img.shape, recon_img.mean().item())
    recon_img = recon_img[0].permute(1, 2, 0).numpy()
    recon_img = (recon_img - recon_img.min()) / (
        recon_img.max() - recon_img.min() + 1e-6
    )
    recon_img = (recon_img * 255).astype(np.uint8)
    Image.fromarray(recon_img).save("data/tmp_recon.png")
    print(f"reconstruction saved to data/tmp_recon.png")
