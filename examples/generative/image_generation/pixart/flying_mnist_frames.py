"""
An image dataset from Flying MNIST video frames.
"""
import numpy as np
import torch
from einops import rearrange
from cvlization.dataset.flying_mnist import FlyingMNISTDatasetBuilder

class FlyingMNISTImageLatentsBuilder:
    def __init__(self, latent_file: str):
        self.latents = np.load(latent_file)
        self.n_train = int(len(self.latents) * 0.8)
    
    def training_dataset(self):
        return FlyingMNISTImageLatents(self.latents[:self.n_train], video_ds=FlyingMNISTDatasetBuilder())
    
    def validation_dataset(self):
        return FlyingMNISTImageLatents(self.latents[self.n_train:], video_ds=FlyingMNISTDatasetBuilder())


class FlyingMNISTImageLatents:
    def __init__(self, latents: np.ndarray, video_ds, num_frames_per_video: int=32):
        self.video_ds = video_ds
        self.resolution = 32
        latents = torch.from_numpy(latents)
        latents = rearrange(latents, "b c t h w -> (b t) c h w")
        self.latents = latents
        self.load_vae_feat = True

    def __getitem__(self, idx):
        """
        Similar to SA.py
        """
        data_info = {'img_hw': torch.tensor([self.resolution, self.resolution], dtype=torch.float32),
                     'aspect_ratio': torch.tensor(1.)}

        img = self.latents[idx]
        # txt_fea = torch.from_numpy(npz_info['caption_feature'])
        # Use a constant text embedding from t5
        txt_fea = torch.ones(1, 120, 4096)
        attention_mask = torch.ones(1, 1, txt_fea.shape[1]).int()
        # if 'attention_mask' in npz_info.keys():
        #     attention_mask = torch.from_numpy(npz_info['attention_mask'])[None]

        data_info["mask_type"] = "null"

        return img, txt_fea, attention_mask, data_info
    
    def __len__(self):
        return len(self.latents)

    def _not_used_get_img(self, idx):
        video_idx = idx // self.video_ds.num_frames
        frame_idx = idx % self.video_ds.num_frames
        return self.video_ds[video_idx][frame_idx]


if __name__ == "__main__":
    db = FlyingMNISTImageLatentsBuilder("data/latents/flying_mnist__model-nilqq143_latents_32frames_train.npy")
    train_ds = db.training_dataset()
    print(len(train_ds))
    z, y, _, _ = train_ds[0]
    print("z:", z.shape, "y:", y.shape)