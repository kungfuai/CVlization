from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from typing import Callable
from einops import rearrange
import wandb
from mamba_ssm import Mamba
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig

# from ...net.mamba.mamba_simple import MambaLMHeadModel
from .data_utils import FlatTokenIds


class MambaTrainingPipeline:

    @dataclass
    class Config:
        # Decoder
        decoder: Callable = None
        max_length_to_generate: int = 200

        # Data
        block_size: int = 256
        vae_vocab_size: int = 5120
        vocab_size: int = 5120 + 20
        start_token: int = 5121
        position_shape: tuple = (8, 64, 64)

        # Optimizer
        lr: float = 1e-3
        batch_size: int = 64
        clip_grad: float = 1.0
        gradient_accumulation_steps: int = 1

        # Training loop
        epochs: int = 100
        max_iters: int = 10000
        log_interval: int = 100
        eval_interval: int = 1000
        eval_iters: int = 10
        sample_interval: int = 1000

        # Accelerator
        device: str = "cuda"

        # Logging
        output_dir: str = "logs/nanomamba"
        project: str = "mamba"
        track: bool = False

        # Model
        d_model: int = 512
        n_layer: int = 12
        pad_vocab_size_multiple: int = 8
        vae_model_name: str = None

    def __init__(self, config: Config):
        self.config = config
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.config.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.START_TOKEN = self.config.start_token
        torch.manual_seed(11)

    def fit(self, dataset_builder: FlatTokenIds):
        if self.config.track:
            wandb.init(project=self.config.project, config=self.config)
        self.create_dataloaders(dataset_builder)
        model = self.create_model()
        self.create_optimizer()
        self.training_loop()

    def get_batch(self, split):
        # generate targets and context
        if split == "train":
            data = self.train_data_flattened
        else:
            data = self.val_data_flattened
        block_size = self.config.block_size
        batch_size = self.config.batch_size
        device = self.config.device
        index = torch.randint(0, len(data) - block_size, (batch_size,))
        x = torch.stack([data[ind : ind + block_size] for ind in index])
        y = torch.stack([data[ind + 1 : ind + block_size + 1] for ind in index])
        return x.to(device), y.to(device)

    def create_dataloaders(self, dataset_builder):
        self.train_data = dataset_builder.training_dataset()
        self.val_data = dataset_builder.validation_dataset()
        assert isinstance(self.train_data, np.ndarray)
        assert isinstance(self.val_data, np.ndarray)
        assert self.train_data.dtype in [
            np.int32,
            np.int64,
            np.uint16,
            np.uint32,
            np.uint64,
        ]
        print(self.train_data.shape, self.val_data.shape)
        assert len(self.train_data.shape) == 1
        self.data_seq_len = self.train_data.shape[0]
        if len(self.train_data.shape) > 1:
            self.train_data_flattened = self.train_data.ravel()
            self.val_data_flattened = self.val_data.ravel()
        else:
            self.train_data_flattened = self.train_data
            self.val_data_flattened = self.val_data
        self.train_data_flattened = torch.tensor(
            self.train_data_flattened.astype(np.int32), dtype=torch.long
        )
        self.val_data_flattened = torch.tensor(
            self.val_data_flattened.astype(np.int32), dtype=torch.long
        )

    def create_model(self):
        """
            d_model: int = 2560
        n_layer: int = 64
        vocab_size: int = 50277
        ssm_cfg: dict = field(default_factory=dict)
        rms_norm: bool = True
        residual_in_fp32: bool = True
        fused_add_norm: bool = True
        pad_vocab_size_multiple: int = 8
        tie_embeddings: bool = True
        """
        self.model = MambaLMHeadModel(
            config=MambaConfig(
                d_model=self.config.d_model,
                n_layer=self.config.n_layer,
                vocab_size=self.config.vocab_size,
                ssm_cfg={},
                rms_norm=True,
                residual_in_fp32=True,
                fused_add_norm=True,
                pad_vocab_size_multiple=self.config.pad_vocab_size_multiple,
                tie_embeddings=True,
            ),
            initializer_cfg=None,
            device=None,
            dtype=None,
        )
        # self.model = MambaLM(
        #     vocab_size=self.config.vocab_size,
        #     block_size=self.config.block_size,
        #     n_embed=self.config.n_embed,
        #     n_heads=self.config.n_heads,
        #     n_layers=self.config.n_layers,
        #     device=self.config.device,
        # )
        self.model.to(self.config.device)
        if self.config.vae_model_name is not None:
            self.load_vae()
        print(self.model)
        print("Number of parameters:", sum(p.numel() for p in self.model.parameters()))
        return self.model

    def load_vae(self):
        if ":" in self.config.vae_model_name:
            # it is a wandb model
            from cvlization.torch.net.vae.video_vqvae import VQVAE

            vae = VQVAE.from_pretrained(self.config.vae_model_name)
        else:
            # it is a huggingface model
            from diffusers.models import AutoencoderKL

            vae = AutoencoderKL.from_pretrained(self.config.vae_model_name)

        vae.eval()
        vae = vae.to(self.config.device)
        self.vae = vae

    def create_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)

    def training_loop(self):
        optimizer = self.optimizer
        model = self.model

        output_dir = Path(self.config.output_dir)
        max_iters = self.config.max_iters
        eval_iters = self.config.eval_iters
        device = self.config.device

        print("Uses device " + device)
        losses_data = {"train": [], "test": []}

        t = self.config.position_shape[0]
        h = self.config.position_shape[1]
        w = self.config.position_shape[2]

        if self.config.vae_model_name is not None:
            vae = self.vae
            ground_truth_codes = (
                torch.from_numpy(self.val_data[0, 1:].astype(int)).long().to(device)
            )
            ground_truth_codes = rearrange(
                ground_truth_codes,
                "(b t h w) -> b t h w",
                b=1,
                t=int(t),
                h=int(h),
                w=int(w),
            )
            assert ground_truth_codes.shape == (1, t, h, w), ground_truth_codes.shape
            assert isinstance(
                ground_truth_codes, torch.Tensor
            ), f"expected torch.Tensor, got {type(ground_truth_codes)}"
            with torch.no_grad():
                z = vae.vq.codes_to_vec(ground_truth_codes)
                assert len(z.shape) == 5
                assert z.shape == (1, 4, t, h, w)
                video = vae.decoder(z)
                video = (video - video.min()) / (video.max() - video.min() + 1e-6)
                video = (video * 255).to(torch.uint8)
                video = rearrange(video, "b c t h w -> t c h (b w)")
                assert video.shape[1] == 3, f"shape of video is {video.shape}"
                display = wandb.Video(video.detach().cpu(), fps=5, format="mp4")
                print(f"video shape: {video.shape}")
                if self.config.track:
                    wandb.log({"sampled/ground_truth_decoded": display})

        for iter in tqdm(range(0, max_iters)):
            if iter % self.config.eval_interval == 0:
                losses = self.estimate_loss()
                # losses_data["train"].append(losses["train"].cpu().numpy())
                losses_data["test"].append(losses["test"].cpu().numpy())
                print(f"Step {iter}, val loss:{losses['test']:.4f}")
                if self.config.track:
                    wandb.log({"val/loss": losses["test"]})

            # Get data
            xb, yb = self.get_batch("train")

            # Evaluate loss
            # logits, loss = model(xb, yb)
            model_out = model(xb, yb)
            logits = model_out.logits
            # LM loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), yb.view(-1), ignore_index=-1
            )
            if self.config.track:
                wandb.log({"train/loss": loss.item()})

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.config.clip_grad:
                torch.nn.utils.clip_grad.clip_grad_norm_(
                    model.parameters(), self.config.clip_grad
                )
            optimizer.step()

            if iter % self.config.log_interval == 0:
                print(f"Step {iter}, loss:{loss.item():.4f}")
                if self.config.track:
                    wandb.log({"train/loss": loss.item()})

            if iter % self.config.sample_interval == 0:
                if self.config.vae_model_name is not None:
                    model.eval()
                    with torch.no_grad():
                        sampled_codes = model.generate(
                            idx=torch.Tensor(
                                np.ones((1, 1), dtype=np.int32) * self.START_TOKEN
                            )
                            .long()
                            .to(device),
                            max_length=self.config.max_length_to_generate,
                            # max_new_tokens=self.data_seq_len,
                            # temperature=1,
                            # top_k=100,
                            show_progress=True,
                        )
                        sampled_codes = sampled_codes[0, 1:]
                        violating_codes = (
                            (sampled_codes > self.config.vae_vocab_size - 1)
                            .float()
                            .mean()
                        )
                        print(f"violating codes: {violating_codes.item()}")
                        sampled_codes[
                            sampled_codes > self.config.vae_vocab_size - 1
                        ] = 0
                        print("sampled codes:", sampled_codes)
                        # print(sampled_codes.min(), sampled_codes.max())
                        sampled_codes = rearrange(
                            sampled_codes[: self.data_seq_len],
                            "(b t h w) 1 -> b t h w",
                            b=1,
                            t=int(t),
                            h=int(h),
                            w=int(w),
                        )
                        assert sampled_codes.shape == (1, t, h, w), sampled_codes.shape

                        z = vae.vq.codes_to_vec(sampled_codes)
                        assert len(z.shape) == 5
                        assert z.shape == (1, 4, t, h, w)
                        video = vae.decoder(z)
                        video = (video - video.min()) / (
                            video.max() - video.min() + 1e-6
                        )
                        video = (video * 255).to(torch.uint8)
                        video = rearrange(video, "b c t h w -> t c h (b w)")
                        video = video.detach().cpu()
                        display = wandb.Video(video, fps=5, format="mp4")
                        if self.config.track:
                            wandb.log(
                                {
                                    "sampled/generated_video": display,
                                    "sampled/violating_codes": violating_codes,
                                }
                            )
                else:
                    # Sample the tokens
                    # Log the tokens as a string.
                    model.eval()
                    with torch.no_grad():
                        sampled_tokens = model.generate(
                            torch.zeros((1, 1), dtype=torch.long).to(device),
                            max_length=self.config.max_length_to_generate,
                        )[0].tolist()
                        # decode
                        if self.config.decoder is not None:
                            sampled_str = self.config.decoder(sampled_tokens)
                            print("sampled_str:", sampled_str)
                            if self.config.track:
                                wandb.log({"sampled/generated_decoded": sampled_str})
                        else:
                            if self.config.track:
                                wandb.log({"sampled/generated": str(sampled_tokens)})

                model.train()

            if self.config.track:
                wandb.log({"train/lr": optimizer.param_groups[0]["lr"]})

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        model = self.model
        model.eval()
        eval_iters = self.config.eval_iters
        for split in ["test"]:  # ["train", "test"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                # X, Y: torch.Size([32, 512]) torch.Size([32, 512])
                X, Y = self.get_batch(split)
                # logits, loss = model(X, Y)
                model_out = model(X, Y)
                logits = model_out.logits
                # calculate LM loss
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-1
                )
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out


class SelfAttentionHead(nn.Module):
    def __init__(self, head_size, n_embed, block_size, device, dropout=0.2):
        super().__init__()
        self.keys = nn.Linear(n_embed, head_size)
        self.queries = nn.Linear(n_embed, head_size)
        self.values = nn.Linear(n_embed, head_size)
        self.head_size = head_size
        self.n_embed = n_embed
        self.register_buffer(
            "tril", torch.tril(torch.ones((block_size, block_size))).to(device)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.keys(x)  # (B,T,C_h)
        q = self.queries(x)  # (B,T,C_h)
        v = self.values(x)  # (B,T,C_h)
        wei = k @ q.transpose(-1, -2) * C ** (-0.5)  # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        # wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = torch.log(torch.exp(wei) + 1)  # (B,T,T)
        wei = self.dropout(wei)
        out = wei @ v  # (B,T,C_h)
        return out


class LayerNorm(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.eps = 1e-5
        # params
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        xmean = x.mean(dim=1, keepdim=True)
        xvar = ((x - xmean) ** 2).mean(dim=1, keepdim=True)
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class MultiHeadAttention(nn.Module):
    def __init__(
        self, n_heads, head_size, n_embed, block_size, device: str, dropout=0.2
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [
                SelfAttentionHead(
                    n_embed=n_embed,
                    head_size=head_size,
                    block_size=block_size,
                    device=device,
                    dropout=dropout,
                )
                for _ in range(n_heads)
            ]
        )
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout: float = 0.2) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ffn(x)


class Block(nn.Module):
    def __init__(
        self, n_embed, n_heads, device: str, d_state=16, d_conv=4, expand=2
    ) -> None:
        super().__init__()
        self.head_size = n_embed // n_heads
        # self.sa_head = MultiHeadAttention(n_heads, self.head_size)
        self.sa_head = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=n_embed,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        ).to(device)
        self.ffn = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
