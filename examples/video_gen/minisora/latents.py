from typing import Iterable
import torch
from torch.utils.data import DataLoader


def extract_latents(vae, dataset, batch_size: int = 32, output_device: str = "cpu") -> Iterable[torch.Tensor]:
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch in dl:
        x = batch["video"]
        assert x.ndim == 5, "videos must have 4 dimensions besides the batch dim"
        assert x.shape[1] == 3, "videos must have 3 channels at dim 1"
        x = x.to(vae.device)
        with torch.no_grad():
            z = vae.encoder(x)
            vq_output = vae.vq(z)
            if isinstance(vq_output, dict):
                z_q = vq_output["z_recon"]
            else:
                z_q = vq_output
            # unbatch
            for z_q_i in z_q.to(output_device):
                yield z_q_i


def extract_token_ids(vae, dataset, batch_size: int = 32, output_device: str = "cpu") -> Iterable[torch.IntTensor]:
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch in dl:
        x = batch["video"]
        # print("x:", x.mean())
        assert x.ndim == 5, "videos must have 4 dimensions besides the batch dim"
        assert x.shape[1] == 3, "videos must have 3 channels at dim 1"
        x = x.to(vae.device)
        with torch.no_grad():
            z = vae.encoder(x)
            # z shape is 
            # print("z:", z.mean())
            token_ids = vae.vq.vec_to_codes(z)
            # print("token_ids:", token_ids.float().mean())
            token_ids = token_ids.to(output_device)  # shape is (b, t * h * w)
            # unbatch
            for token_ids_i in token_ids:
                # print("token_ids_i:", token_ids_i.float().mean())
                yield token_ids_i