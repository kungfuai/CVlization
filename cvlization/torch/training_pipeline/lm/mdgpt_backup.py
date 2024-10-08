"""
Multi-dimensional GPT.

TODO: use non-causal attention.
"""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Tuple
import os
import time
import math
import pickle
from einops import rearrange
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import init_process_group, destroy_process_group
import wandb
from .gpt import GPT, GPTConfig, LayerNorm, MLP, Block


class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        try:
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        except:
            # x.shape: torch.Size([8, 4096, 768]) this is the correct shape
            # x.shape: torch.Size([1, 1, 768]), n_embd: 768, n_head: 6, c_atten: Linear(in_features=768, out_features=2304, bias=True)
            print(
                f"x.shape: {x.shape}, n_embd: {self.n_embd}, n_head: {self.n_head}, c_atten: {self.c_attn}"
            )
            raise
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=False,
            )
        else:
            # manual implementation of attention
            raise NotImplementedError("regular self attention currently requires flash-attn")
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class BlockNonCausal(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class MDGPTTrainingPipeline:
    @dataclass
    class Config:

        log_dir: str = "logs/mdgpt"
        wandb_log: bool = False
        project: str = "mdgpt"
        init_from: str = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
        batch_size: int = 32

        # Context window
        block_size: int = 1024
        sparse_block_size: int = 128
        position_epsilon: float = 1e-2
        flatten_tokens: bool = False

        # Tokenizer
        vocab_size: int = 5120
        vae_vocab_size: int = 5120
        vae_model_name: str = None
        meta_vocab_size: int = None
        start_token: int = 5121
        ignore_token: int = 5122

        # Model
        causal: bool = False
        only_predict_last: bool = False
        n_layer: int = 12
        n_head: int = 12
        n_embd: int = 768
        dropout: float = 0.0
        gradient_accumulation_steps: int = 1
        bias: bool = True
        use_1d_pos_embedding: bool = False
        device: str = "cuda"

        learning_rate: float = 5e-4  # max learning rate
        max_iters: int = 600000  # total number of training iterations
        weight_decay: float = 1e-1
        beta1: float = 0.9
        beta2: float = 0.95
        grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0

        # learning rate decay settings
        decay_lr: bool = True  # whether to decay the learning rate
        warmup_iters: int = 2000  # how many steps to warm up for
        lr_decay_iters: int = 600000  # should be ~= max_iters per Chinchilla
        min_lr: float = (
            1e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
        )
        # DDP settings
        backend: str = "nccl"  # 'nccl', 'gloo', etc.

        eval_interval: int = 250  # keep frequent because we'll overfit
        eval_iters: int = 100
        log_interval: int = 10  # don't print too too often
        sample_interval: int = 500
        disable_sparse_context_window: bool = True

        # we expect to overfit on this small dataset, so only save when val improves
        always_save_checkpoint: bool = False
        compile: bool = False  # use PyTorch 2.0 to compile the model to be faster
        eval_only: bool = False  # if True, script exits right after the first eval

        debug: bool = False
        

    def __init__(self, config: Config):
        self.config = config
        self.out_dir = (
            f"{config.log_dir}/batch{config.batch_size}_block{config.block_size}"
        )
        # self.dtype = (
        #     "bfloat16"
        #     if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        #     else "float16"
        # )  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
        self.dtype = "float32"

        self.ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.dtype]
        device_type = (
            "cuda" if "cuda" in config.device else "cpu"
        )  # for later use in torch.autocast
        self.device_type = device_type
        self.ctx = (
            nullcontext()
            if device_type == "cpu"
            else torch.amp.autocast(device_type=device_type, dtype=self.ptdtype)
        )
        self._setup_io()
        if self.master_process:
            os.makedirs(self.out_dir, exist_ok=True)

        seed = 1337 + self.seed_offset
        print(f"setting random seed to {seed}")
        torch.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

        self.START_TOKEN = self.config.start_token
        self.IGNORE_TOKEN = self.config.ignore_token

    def fit(self, dataset_builder):
        if self.master_process and self.config.wandb_log:
            wandb.init(project=self.config.project, config=self.config)
        self.create_dataloaders(dataset_builder)
        # self.precompute_sparse_context_window()
        self.relative_pos_lut=None
        self.context_idx_lut=None
        self.position_shape = (8, 64, 64)
        self.position_dim = 3
        self.create_model()
        self.create_grad_scaler()
        self.create_optimizer()
        print(torch.randint(100, (1,)))
        self.training_loop()

    def _setup_io(self):
        """
        various inits, derived attributes, I/O setup
        """
        batch_size = self.config.batch_size
        block_size = self.config.block_size
        self.ddp = ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
        if ddp:
            init_process_group(backend=self.config.backend)
            ddp_rank = int(os.environ["RANK"])
            ddp_local_rank = int(os.environ["LOCAL_RANK"])
            ddp_world_size = int(os.environ["WORLD_SIZE"])
            device = f"cuda:{ddp_local_rank}"
            torch.cuda.set_device(device)
            self.master_process = (
                ddp_rank == 0
            )  # this process will do logging, checkpointing etc.
            self.seed_offset = ddp_rank  # each process gets a different seed
            # world_size number of processes will be training simultaneously, so we can scale
            # down the desired gradient accumulation iterations per process proportionally
            assert self.config.gradient_accumulation_steps % ddp_world_size == 0
            self.config.gradient_accumulation_steps //= ddp_world_size
        else:
            # if not ddp, we are running on a single gpu, and one process
            self.master_process = True
            self.seed_offset = 0
            ddp_world_size = 1
        tokens_per_iter = (
            self.config.gradient_accumulation_steps
            * ddp_world_size
            * batch_size
            * block_size
        )
        print(f"tokens per iteration will be: {tokens_per_iter:,}")
        self.tokens_per_iter = tokens_per_iter

    def get_batch(self, split: str):
        if self.config.flatten_tokens:
            raise NotImplementedError("flatten_tokens not implemented")
        else:
            train_data = self.train_data
            val_data = self.val_data
            assert len(train_data.shape) == 2, f"Expected 2D array for training data, got {train_data.shape}"
            # if len(train_data.shape) > 2:
            #     train_data = train_data.reshape(train_data.shape[0], -1)
            #     val_data = val_data.reshape(val_data.shape[0], -1)
        block_size = self.config.block_size
        batch_size = self.config.batch_size
        device = self.config.device

        data = train_data if split == "train" else val_data
        if len(data.shape) == 2:
            # batch x sequence len
            irow = torch.randint(data.shape[0], (batch_size,))
            ix = torch.randint(data.shape[1] - block_size, (batch_size,))
            # print(ix)
            x = torch.stack([
                torch.from_numpy(data[i, i1 : i1 + block_size].astype(np.int64)) for i, i1 in zip(irow, ix)
            ])
            y = torch.stack([
                torch.from_numpy(data[i, i1 + 1 : i1 + 1 + block_size].astype(np.int64)) for i, i1 in zip(irow, ix)
            ])
        else:
            raise ValueError(f"should not be here, train_data.shape={train_data.shape}")
            ix = torch.randint(len(data) - block_size, (batch_size,))
            x = torch.stack(
                [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
            )
            y = torch.stack(
                [
                    torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
                    for i in ix
                ]
            )
        if self.device_type == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
                device, non_blocking=True
            )
        else:
            x, y = x.to(device), y.to(device)
        assert x.shape == (self.config.batch_size, self.config.block_size), f"x.shape: {x.shape}"
        assert y.shape == (self.config.batch_size, self.config.block_size), f"y.shape: {y.shape}"

        return x, y

    def get_batch2(self, split: str):
        """
        Get a batch of data for training or validation.

        Args:
            split (str): The split to get the data from. Can be either "train" or "val".

        Returns:
            tuple: A tuple containing two tensors: x and y.
                - x: The input tensor of shape (batch_size, block_size).
                - y: The target tensor of shape (batch_size, block_size).

        """
        if self.config.flatten_tokens:
            raise NotImplementedError("flatten_tokens not implemented")
        else:
            train_data = self.train_data
            val_data = self.val_data
            assert len(train_data.shape) == 2, f"Expected 2D array for training data, got {train_data.shape}"
            # if len(train_data.shape) > 2:
            #     train_data = train_data.reshape(train_data.shape[0], -1)
            #     val_data = val_data.reshape(val_data.shape[0], -1)
        block_size = self.config.block_size
        batch_size = self.config.batch_size
        device = self.config.device

        data = train_data if split == "train" else val_data
        if len(data.shape) == 2:
            # batch x sequence len
            irow = torch.randint(data.shape[0], (batch_size,))
            ix = torch.randint(data.shape[1] - block_size, (batch_size,))
            # print(ix)
            x = torch.stack([
                torch.from_numpy(data[i, i1 : i1 + block_size].astype(np.int64)) for i, i1 in zip(irow, ix)
            ])
            y = torch.stack([
                torch.from_numpy(data[i, i1 + 1 : i1 + 1 + block_size].astype(np.int64)) for i, i1 in zip(irow, ix)
            ])
        else:
            raise ValueError(f"should not be here, train_data.shape={train_data.shape}")
            ix = torch.randint(len(data) - block_size, (batch_size,))
            x = torch.stack(
                [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
            )
            y = torch.stack(
                [
                    torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
                    for i in ix
                ]
            )
        if self.device_type == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
                device, non_blocking=True
            )
        else:
            x, y = x.to(device), y.to(device)
        assert x.shape == (self.config.batch_size, self.config.block_size), f"x.shape: {x.shape}"
        assert y.shape == (self.config.batch_size, self.config.block_size), f"y.shape: {y.shape}"

        x_pos = None
        y_pos = None
        return x_pos, x, y_pos, y
    
    def get_batch_md(self, split: str):
        """
        Get a batch of data for training or validation.

        Args:
            split (str): The split to get the data from. Can be either "train" or "val".

        Returns:
            - x_pos: The positional indices of the input tokens. The shape is (batch_size, block_size, position_dim).
            - x: The input tensor of shape (batch_size, block_size).
            - y_pos: The positional indices of the target tokens. The shape is (batch_size, position_dim).
            - y: The target tensor of shape (batch_size, 1).

        """
        train_data = self.train_data
        position_dim = self.position_dim
        sparse_block_size = self.config.sparse_block_size
        batch_size = self.config.batch_size
        device = self.config.device

        data = train_data if split == "train" else self.val_data
        # TODO: apply a random "video latent frame offset" to have the video
        #  start from a random frame.

        # TODO: the positions only need to be computed once.
        # TODO: position_dim == 1 vs. > 1 should be handled in different functions
        # Coordinates of tokens. Shape is the same as data.
        if position_dim == 1:
            positions = np.arange(len(data), dtype=np.int64)
        else:
            meshgrid_args = [np.arange(s) for s in train_data.shape[1:]]
            positions = np.array(
                np.meshgrid(*meshgrid_args, indexing="ij"),
            )
            orig_shape = tuple(range(len(positions.shape)))
            transposed_shape = orig_shape[1:] + (orig_shape[0],)
            positions = positions.transpose(*transposed_shape)
            # flatten:
            data = data.reshape(data.shape[0], -1)  # this is the whole video
            positions_flattened = positions.reshape(-1, positions.shape[-1])
            # print(f"data shape: {data.shape}")
            # print(f"positions_flattened shape: {positions_flattened.shape}")
            # print("relative_pos_lut:", self.relative_pos_lut.shape)
            # print("context_idx_lut:", self.context_idx_lut.shape)

        # Problem formulation: x_pos, x, y_pos, y?
        # Randomly sample a target index for each example in the batch
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        assert (
            len(data.shape) == 2
        ), f"Expected 2D array for training data, got {data.shape}"

        if self.config.disable_sparse_context_window:
            block_size = self.config.block_size
            irow = np.random.randint(data.shape[0], size=batch_size)
            ix = np.random.randint(data.shape[1] - block_size, size=batch_size)
            x = torch.stack([
                torch.from_numpy(data[i, i1 : i1 + block_size].astype(np.int64)) for i, i1 in zip(irow, ix)
            ])
            y = torch.stack([
                torch.from_numpy(data[i, i1 + 1 : i1 + 1 + block_size].astype(np.int64)) for i, i1 in zip(irow, ix)
            ])
            x_pos = None
            y_pos = None
            if self.device_type == "cuda":
                # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
                x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
                    device, non_blocking=True
                )
            else:
                x, y = x.to(device), y.to(device)
            return x_pos, x, y_pos, y

        # Insert the ignore token at the beginning
        # raise ValueError("should not be here")
        data_with_ignore_token = np.concatenate(
            [np.ones((data.shape[0], 1), dtype=data.dtype) * self.IGNORE_TOKEN, data],
            axis=1,
        )

        irow = np.random.randint(data.shape[0], size=batch_size)
        u = float(np.random.rand(1))
        coldstart_prob = 0.8
        if u < 0.1 * coldstart_prob:
            # very cold start
            ix = torch.randint(16, (1,)) * torch.ones(batch_size, dtype=torch.long)
        elif u < coldstart_prob:
            # cold start
            ix = torch.randint(self.config.sparse_block_size, (batch_size,))
        else:
            ix = torch.randint(positions_flattened.shape[0], (batch_size,))
        target_pos = [
            torch.from_numpy(positions_flattened[col, :]) for row, col in zip(irow, ix)
        ]
        # for debug
        if False:
            print("*" * 60)
            print("irow:", irow[0], "ix:", ix[0])
            assert len(data.shape) == 2
            print("x (last 5):", data[irow[0], ix[0]-5:ix[0]], "y:", data[irow[0], ix[0]])
        y_pos = target_pos = torch.stack(target_pos).unsqueeze(1).int()
        if self.relative_pos_lut is not None:
            assert (
                ix.max() < self.relative_pos_lut.shape[0]
            ), f"ix.max()={ix.max()}, data.shape={data.shape}, relative_pos_lut.shape={self.relative_pos_lut.shape}"
            relative_pos = [self.relative_pos_lut[i, ...] for i in ix]
        else:
            # This means the data is probably 1D.
            # TODO: randomly set the first few context tokens to IGNORE_TOKEN.
            #   This is to simulate cold-start: very few context tokens.
            relative_pos = [
                sparse_block_size - torch.from_numpy(np.arange(0, sparse_block_size).reshape(-1, 1))
                for i in ix
            ]
        relative_pos = torch.stack(relative_pos)
        # print("relative_pos shape:", relative_pos.shape)
        # print("target_pos shape:", target_pos.shape)
        if len(target_pos.shape) == 2:
            target_pos = target_pos.unsqueeze(1)
        x_pos = (target_pos - relative_pos).int()
        sparse_block_size = self.config.sparse_block_size
        if self.context_idx_lut is not None:
            assert data_with_ignore_token.shape == (data.shape[0], data.shape[1] + 1), f"data_with_ignore_token.shape={data_with_ignore_token.shape}, data.shape={data.shape}"
            x = [
                data_with_ignore_token[row][self.context_idx_lut[col] + 1]
                for (row, col) in zip(irow, ix)
            ]
        else:
            x = []
            for i, (row, col) in enumerate(zip(irow, ix)):
                context_idx = np.arange(col - sparse_block_size, col)
                context_idx[context_idx < 0] = -1
                x_ = data_with_ignore_token[row][context_idx + 1]
                # print("y_pos:", y_pos[i].numpy().ravel())
                # print("context_idx for x:", context_idx[-10:] + 1)
                # print("x_pos:", x_pos[i].numpy().ravel()[-10:])
                x.append(x_)

        x = torch.stack([torch.from_numpy(x_.astype(np.int32)) for x_ in x]).long()
        y = [data[row : row + 1, col] for (row, col) in zip(irow, ix)]
        y = torch.stack([torch.from_numpy(y_.astype(np.int32)) for y_ in y]).long()
        if len(y_pos.shape) == 3:
            y_pos = y_pos.squeeze(1)

        # assert the shapes
        assert x.shape == (batch_size, sparse_block_size), x.shape
        assert y.shape == (batch_size, 1), y.shape
        assert x_pos.shape == (batch_size, sparse_block_size, position_dim), x_pos.shape
        assert y_pos.shape == (batch_size, position_dim), f"y_pos.shape={y_pos.shape}, position_dim={position_dim}, batch_size={batch_size}"
        assert x.dtype == torch.long, f"expected long tensor, got {x.dtype}"

        # for debug, print first 5 tokens
        if self.config.debug:
            print("===========================")
            print("x:", x[0, :5])
            print("y:", y[0])
            print("y_pos:", y_pos[0])
            print("x_pos:", x_pos[0, :5])

        if self.device_type == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
                device, non_blocking=True
            )
            x_pos, y_pos = x_pos.pin_memory().to(
                device, non_blocking=True
            ), y_pos.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
            x_pos, y_pos = x_pos.to(device), y_pos.to(device)

        # print(
        #     f"max relative_pos={(y_pos.unsqueeze(1) - x_pos).max()}, x={x.max()}, y={y.max()}"
        # )
        # print(
        #     f"min relative_pos={(y_pos.unsqueeze(1) - x_pos).min()}, x={x.min()}, y={y.min()}"
        # )
        assert (y_pos.unsqueeze(1) - x_pos).max() <= self.config.block_size, f"max relative_pos={(y_pos.unsqueeze(1) - x_pos).max()}, but block_size is {self.config.block_size}"
        assert x.max() < self.config.vocab_size, f"x.max()={x.max()}, vocab_size={self.config.vocab_size}"
        assert y.max() < self.config.vocab_size, f"y.max()={y.max()}, vocab_size={self.config.vocab_size}"
        return x_pos, x, y_pos, y

    def create_dataloaders(self, dataset_builder):
        self.train_data = dataset_builder.training_dataset()  # (B, T, H, W)
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
        # assert len(self.train_data.shape) == 2, f"Expected 2D array for training data, got {self.train_data.shape}"
        # self.train_data_flattened = self.train_data.ravel()
        # self.val_data_flattened = self.val_data.ravel()
        print(f"block size:", self.config.block_size)
        # position_shape determines the size of embeddings for positional indices
        
    def precompute_sparse_context_window(self):
        # precompute the sparse context window for each position
        # this is a one-time operation that will speed up training
        block_size = self.config.block_size
        sparse_block_size = self.config.sparse_block_size
        train_data = self.train_data

        if len(train_data.shape) == 1:
            if len(train_data) > 1e3:
                print(
                    "Not precomputing sparse context window for 1D dataset with long sequences. This will be slow."
                )
                self.relative_pos_lut = None
                self.context_idx_lut = None
                return
            train_data = train_data.reshape(1, -1)

        if len(train_data) in [1, 2]:  # TODO: revisit this
            # This means the training data is either a long sequence or
            # of a batch of 1D sequences.
            positions = np.arange(train_data.shape[-1])
            self.position_dim = 1
        else:
            if len(train_data.shape) == 1:
                meshgrid_args = [np.arange(train_data.shape[0])]
            else:
                # This assumes train_data has a shape like (B, T, H, W)
                meshgrid_args = [np.arange(s) for s in train_data.shape[1:]]
            # print("train_data:", train_data.shape, "position_dim:", position_dim)
            # print("meshgrid_args:", meshgrid_args)
            positions = np.array(
                np.meshgrid(*meshgrid_args, indexing="ij"),
            )
            orig_shape = tuple(range(len(positions.shape)))
            transposed_shape = orig_shape[1:] + (orig_shape[0],)
            positions = positions.transpose(*transposed_shape)
            self.position_shape = positions.shape[:-1]
            position_dim = self.position_dim = len(self.position_shape)

        print("positions shape:", positions.shape, "position_dim:", position_dim)
        positions_flat = positions.reshape(-1, position_dim)
        self.relative_pos_lut, self.context_idx_lut = precompute_context_windows(
            positions_flat=positions_flat,
            block_size=block_size,
            sparse_block_size=sparse_block_size,
            position_epsilon=self.config.position_epsilon,
        )
        self.relative_pos_lut = torch.from_numpy(self.relative_pos_lut)
        self.context_idx_lut = torch.from_numpy(self.context_idx_lut)
        print(
            "Done precomputing context windows. relative_pos_lut shape:",
            self.relative_pos_lut.shape,
            "context_idx_lut shape:",
            self.context_idx_lut.shape,
        )

    def _try_to_infer_vocab_size(self):
        # attempt to derive vocab_size from the dataset
        data_dir = os.path.join("data", self.config.dataset)
        meta_path = os.path.join(data_dir, "meta.pkl")
        meta_vocab_size = None
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            meta_vocab_size = meta["vocab_size"]
            print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    def create_model(self):
        n_layer = self.config.n_layer
        n_head = self.config.n_head
        n_embd = self.config.n_embd
        block_size = self.config.block_size
        bias = self.config.bias
        dropout = self.config.dropout
        device = self.config.device
        init_from = self.config.init_from
        if self.config.meta_vocab_size is not None:
            meta_vocab_size = self.config.meta_vocab_size
        else:
            meta_vocab_size = self.config.vocab_size
        out_dir = self.out_dir

        # model init
        self.model_args = dict(
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            block_size=block_size,
            bias=bias,
            vocab_size=None,
            dropout=dropout,
        )  # start with model_args from command line
        if init_from == "scratch":
            # init a new model from scratch
            print("Initializing a new model from scratch")
            # determine the vocab size we'll use for from-scratch training
            if meta_vocab_size is None:
                print(
                    "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)"
                )
            self.model_args["vocab_size"] = (
                meta_vocab_size if meta_vocab_size is not None else 50304
            )
            print("***** model args:", self.model_args)
            gptconf = GPTConfig(**(self.model_args))
            model = GPT(gptconf)
        elif init_from == "resume":
            print(f"Resuming training from {out_dir}")
            # resume training from a checkpoint.
            ckpt_path = os.path.join(out_dir, "ckpt.pt")
            checkpoint = torch.load(ckpt_path, map_location=device)
            checkpoint_model_args = checkpoint["model_args"]
            # force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired from command line
            for k in [
                "n_layer",
                "n_head",
                "n_embd",
                "block_size",
                "bias",
                "vocab_size",
            ]:
                self.model_args[k] = checkpoint_model_args[k]
            # create the model
            gptconf = GPTConfig(**(self.model_args))
            model = GPT(gptconf)
            state_dict = checkpoint["model"]
            # fix the keys of the state dictionary :(
            # honestly no idea how checkpoints sometimes get this prefix, have to debug more
            unwanted_prefix = "_orig_mod."
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
            self.iter_num = checkpoint["iter_num"]
            self.best_val_loss = checkpoint["best_val_loss"]
        elif init_from.startswith("gpt2"):
            print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
            # initialize from OpenAI GPT-2 weights
            override_args = dict(dropout=dropout)
            model = GPT.from_pretrained(init_from, override_args)
            # read off the created config params, so we can store them into checkpoint correctly
            for k in [
                "n_layer",
                "n_head",
                "n_embd",
                "block_size",
                "bias",
                "vocab_size",
            ]:
                self.model_args[k] = getattr(model.config, k)
        # crop down the model block size if desired, using model surgery
        if block_size < model.config.block_size:
            print(
                f"cropping model block size from {model.config.block_size} to {block_size}"
            )
            model.crop_block_size(block_size)
            self.model_args["block_size"] = (
                block_size  # so that the checkpoint will have the right value
            )
        print(model)
        model.to(device)
        if self.config.compile:
            print("compiling the model...")
            model = torch.compile(model)
        self.model = model

        if self.master_process and self.config.vae_model_name is not None:
            self.load_vae()


    def create_model_md(self):
        n_layer = self.config.n_layer
        n_head = self.config.n_head
        n_embd = self.config.n_embd
        block_size = self.config.block_size
        bias = self.config.bias
        dropout = self.config.dropout
        device = self.config.device
        init_from = self.config.init_from
        if self.config.meta_vocab_size is not None:
            meta_vocab_size = self.config.meta_vocab_size
        else:
            meta_vocab_size = self.config.vocab_size
        out_dir = self.out_dir

        # model init
        self.model_args = dict(
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            block_size=block_size,
            bias=bias,
            vocab_size=None,
            dropout=dropout,
            position_shape=self.position_shape,
            use_1d_pos_embedding=self.config.use_1d_pos_embedding,
            causal=self.config.causal,
            only_predict_last=self.config.only_predict_last,
            disable_sparse_context_window=self.config.disable_sparse_context_window,
        )  # start with model_args from command line
        if init_from == "scratch":
            # init a new model from scratch
            print("Initializing a new model from scratch")
            # determine the vocab size we'll use for from-scratch training
            if meta_vocab_size is None:
                print(
                    "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)"
                )
            self.model_args["vocab_size"] = (
                meta_vocab_size if meta_vocab_size is not None else 50304
            )
            self.model_args["sparse_block_size"] = self.config.sparse_block_size
            self.model_args["start_token"] = self.START_TOKEN
            self.model_args["ignore_token"] = self.IGNORE_TOKEN
            self.model_args["position_shape"] = self.position_shape
            print("***** model args:", self.model_args)
            gptconf = MDGPTConfig(**(self.model_args))
            model = MDGPT(gptconf)
        elif init_from == "resume":
            print(f"Resuming training from {out_dir}")
            # resume training from a checkpoint.
            ckpt_path = os.path.join(out_dir, "ckpt.pt")
            checkpoint = torch.load(ckpt_path, map_location=device)
            checkpoint_model_args = checkpoint["model_args"]
            # force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired from command line
            for k in [
                "n_layer",
                "n_head",
                "n_embd",
                "block_size",
                "bias",
                "vocab_size",
            ]:
                self.model_args[k] = checkpoint_model_args[k]
            # create the model
            gptconf = GPTConfig(**(self.model_args))
            model = GPT(gptconf)
            state_dict = checkpoint["model"]
            # fix the keys of the state dictionary :(
            # honestly no idea how checkpoints sometimes get this prefix, have to debug more
            unwanted_prefix = "_orig_mod."
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
            self.iter_num = checkpoint["iter_num"]
            self.best_val_loss = checkpoint["best_val_loss"]
        elif init_from.startswith("gpt2"):
            print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
            # initialize from OpenAI GPT-2 weights
            override_args = dict(dropout=dropout)
            model = GPT.from_pretrained(init_from, override_args)
            # read off the created config params, so we can store them into checkpoint correctly
            for k in [
                "n_layer",
                "n_head",
                "n_embd",
                "block_size",
                "bias",
                "vocab_size",
            ]:
                self.model_args[k] = getattr(model.config, k)
        # crop down the model block size if desired, using model surgery
        if block_size < model.config.block_size:
            model.crop_block_size(block_size)
            self.model_args["block_size"] = (
                block_size  # so that the checkpoint will have the right value
            )
        print(model)
        model.to(device)
        self.model = model
        self.model.set_lut(relative_pos_lut=self.relative_pos_lut, context_idx_lut=self.context_idx_lut)

        if self.master_process and self.config.vae_model_name is not None:
            self.load_vae()

    def load_vae(self):
        if ":" in self.config.vae_model_name:
            # it is a wandb model
            wandb_model_name = self.config.vae_model_name
            vae = self._load_model_from_wandb(wandb_model_name)
        else:
            # it is a huggingface model
            from diffusers.models import AutoencoderKL

            vae = AutoencoderKL.from_pretrained(self.config.vae_model_name)

        vae.eval()
        vae = vae.to(self.config.device)
        self.vae = vae

    def _load_model_from_wandb(self, model_full_name: str):
        # TODO: move this to a separate module
        from cvlization.torch.net.vae.video_vqvae import VQVAE

        api = wandb.Api()
        # skip if the file already exists
        artifact_dir = f"artifacts/{model_full_name.split('/')[-1]}"
        if os.path.exists(artifact_dir):
            print(f"Model already exists at {artifact_dir}")
        else:
            artifact_dir = api.artifact(model_full_name).download()
        # The file is model.ckpt.
        # print(list(state_dict.keys()))
        model = VQVAE.load_from_checkpoint(artifact_dir + "/model.ckpt")
        return model

    def create_grad_scaler(self):
        # initialize a GradScaler. If enabled=False scaler is a no-op
        dtype = self.dtype
        self.scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
        print(f"GradScaler enabled: {dtype == 'float16'}")

    def create_optimizer(self):
        model = self.model
        weight_decay = self.config.weight_decay
        learning_rate = self.config.learning_rate
        beta1 = self.config.beta1
        beta2 = self.config.beta2
        device_type = self.device_type
        init_from = self.config.init_from
        # optimizer
        print(f"creating optimizer with lr {learning_rate}")
        optimizer = model.configure_optimizers(
            weight_decay, learning_rate, (beta1, beta2), device_type
        )
        if init_from == "resume":
            optimizer.load_state_dict(self.checkpoint["optimizer"])
        self.checkpoint = None  # free up memory
        self.optimizer = optimizer
        return optimizer

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self, it):
        learning_rate = self.config.learning_rate
        warmup_iters = self.config.warmup_iters
        lr_decay_iters = self.config.lr_decay_iters
        min_lr = self.config.min_lr

        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

    @torch.no_grad()
    def estimate_loss(self):
        model = self.model
        eval_iters = self.config.eval_iters
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X_pos, X, Y_pos, Y = self.get_batch(split)
                # print("X_pos", X_pos.shape, X_pos.max())
                # print("X", X.shape, X.max())
                # print("Y_pos", Y_pos.shape, Y_pos.max())
                # print("Y", Y.shape, Y.max())
                with self.ctx:
                    logits, loss = model(X_pos, X, Y_pos, Y)

                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    def training_loop(self):
        model = self.model
        optimizer = self.optimizer
        scaler = self.scaler
        ddp = self.ddp
        decay_lr = self.config.decay_lr
        learning_rate = self.config.learning_rate
        max_iters = self.config.max_iters
        grad_clip = self.config.grad_clip
        eval_interval = self.config.eval_interval
        log_interval = self.config.log_interval
        sample_interval = self.config.sample_interval
        always_save_checkpoint = self.config.always_save_checkpoint
        out_dir = self.out_dir
        wandb_log = self.config.wandb_log
        eval_only = self.config.eval_only
        batch_size = self.config.batch_size
        gradient_accumulation_steps = self.config.gradient_accumulation_steps
        master_process = self.master_process
        ctx = self.ctx
        print("master_process:", master_process)
        print("batch_size:", batch_size)
        print("gradient_accumulation_steps:", gradient_accumulation_steps)

        # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
        iter_num = 0
        best_val_loss = 1e9

        # _, X, _, Y = self.get_batch("train")  # fetch the very first batch
        X, Y = self.get_batch("train")  # fetch the very first batch
        print(f"X.shape: {X.shape}, Y.shape: {Y.shape}")
        print(f"X: {X[0, :5]}")
        # import sys; sys.exit(0)
        t0 = time.time()
        local_iter_num = 0  # number of iterations in the lifetime of this process
        raw_model = model.module if ddp else model  # unwrap DDP container if needed
        running_mfu = -1.0
        while True:
            # determine and set the learning rate for this iteration
            lr = self.get_lr(iter_num) if decay_lr else learning_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if (iter_num + 1) % eval_interval == 0 and self.master_process:
                losses = self.estimate_loss()
                print(
                    f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )
                # print("batch_size:", batch_size)
                if wandb_log:
                    wandb.log(
                        {
                            "iter": iter_num,
                            "train/loss": losses["train"],
                            "val/loss": losses["val"],
                            "lr": lr,
                            "mfu": running_mfu * 100,  # convert to percentage
                        }
                    )
                if losses["val"] < best_val_loss or always_save_checkpoint:
                    best_val_loss = losses["val"]
                    if iter_num > 0:
                        model_args = self.model_args
                        checkpoint = {
                            "model": raw_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "model_args": model_args,
                            "iter_num": iter_num,
                            "best_val_loss": best_val_loss,
                            "config": self.config.__dict__,
                        }
                        print(f"saving checkpoint to {out_dir}")
                        torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
            if iter_num == 0 and eval_only:
                break

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(gradient_accumulation_steps):
                if ddp:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code
                    # looking at the source of that context manager, it just toggles this variable
                    model.require_backward_grad_sync = (
                        micro_step == gradient_accumulation_steps - 1
                    )
                with ctx:
                    start_time = time.time()
                    logits, loss = model(X, Y)
                    end_time = time.time()
                    # print(f"forward pass time: {end_time - start_time:.3f}s")
                    loss = (
                        loss / gradient_accumulation_steps
                    )  # scale the loss to account for gradient accumulation
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                # _, X, _, Y = self.get_batch("train")
                X, Y = self.get_batch("train")
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()
            # clip the gradient
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % log_interval == 0 and master_process:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * gradient_accumulation_steps
                if local_iter_num >= 5:  # let the training loop settle a bit
                    mfu = raw_model.estimate_mfu(
                        batch_size * gradient_accumulation_steps, dt
                    )
                    running_mfu = (
                        mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                    )
                print(
                    f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
                )

            # log the decoded ground truth codes
            # TODO: this is hardcoded for now
            t = 32 / 4
            h = 256 / 4
            w = 256 / 4

            if iter_num == 0:
                # Decode from the ground truth token ids
                if master_process and self.config.vae_model_name is not None:
                    device = self.config.device
                    vae = self.vae
                    ground_truth_codes = (
                        torch.Tensor(self.val_data[0, 1:].astype(np.int64))
                        .long()
                        .to(device)
                    )  # this is hard coded
                    ground_truth_codes = rearrange(
                        ground_truth_codes,
                        "(b t h w) -> b t h w",
                        b=1,
                        t=int(t),
                        h=int(h),
                        w=int(w),
                    )
                    assert ground_truth_codes.shape == (
                        1,
                        t,
                        h,
                        w,
                    ), ground_truth_codes.shape
                    assert isinstance(
                        ground_truth_codes, torch.Tensor
                    ), f"expected torch.Tensor, got {type(ground_truth_codes)}"
                    with torch.no_grad():
                        z = vae.vq.codes_to_vec(ground_truth_codes)
                        assert len(z.shape) == 5
                        assert z.shape == (1, 4, t, h, w)
                        video = vae.decoder(z)
                        video = (video - video.min()) / (
                            video.max() - video.min() + 1e-6
                        )
                        video = (video * 255).to(torch.uint8)
                        video = rearrange(video, "b c t h w -> t c h (b w)")
                        assert video.shape[1] == 3, f"shape of video is {video.shape}"
                        display = wandb.Video(video.detach().cpu(), fps=5, format="mp4")
                        if wandb_log:
                            wandb.log({"sampled/ground_truth_decoded": display})

            if (iter_num + 1) % sample_interval == 0 and master_process:
                if self.config.vae_model_name is not None:
                    # sample from the model
                    model.eval()
                    with torch.no_grad():
                        sampled_codes = model.generate(
                            idx=torch.Tensor(
                                np.ones((1, 1), dtype=np.int32) * self.START_TOKEN
                            )
                            .long()
                            .to(device),
                            max_new_tokens=self.data_seq_len,
                            temperature=1,
                            top_k=100,
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
                        # force_cudnn_initialization()
                        # sampled_codes = torch.ones(1, 32768, dtype=torch.long).to(device)
                        print("sampled codes:", sampled_codes)
                        # print(sampled_codes.min(), sampled_codes.max())
                        assert (
                            self.data_seq_len >= t * h * w
                        ), f"{self.data_seq_len} < {t*h*w} not enough tokens sampled"
                        n_ = int(t * h * w)
                        sampled_codes = rearrange(
                            sampled_codes[:n_],
                            "(b t h w) -> b t h w",
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
                        display = wandb.Video(video.cpu(), fps=5, format="mp4")
                        if wandb_log:
                            wandb.log(
                                {
                                    "sampled/generated_video": display,
                                    "sampled/violating_codes": violating_codes,
                                }
                            )

                    model.train()

            iter_num += 1
            local_iter_num += 1

            # termination conditions
            if iter_num > max_iters:
                break

        if ddp:
            destroy_process_group()


    def training_loop_md(self):
        model = self.model
        optimizer = self.optimizer
        scaler = self.scaler
        ddp = self.ddp
        decay_lr = self.config.decay_lr
        learning_rate = self.config.learning_rate
        max_iters = self.config.max_iters
        grad_clip = self.config.grad_clip
        eval_interval = self.config.eval_interval
        log_interval = self.config.log_interval
        sample_interval = self.config.sample_interval
        always_save_checkpoint = self.config.always_save_checkpoint
        out_dir = self.out_dir
        wandb_log = self.config.wandb_log
        eval_only = self.config.eval_only
        batch_size = self.config.batch_size
        gradient_accumulation_steps = self.config.gradient_accumulation_steps
        master_process = self.master_process
        ctx = self.ctx
        actual_context_window_without_ignore_tokens = 0
        print("master_process:", master_process)
        print("batch size:", batch_size)

        # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
        iter_num = 0
        best_val_loss = 1e9

        X_pos, X, Y_pos, Y = self.get_batch("train")  # fetch the very first batch
        print(f"X shape: {X.shape}, Y shape: {Y.shape}")
        print(f"X: {X[0, :5]}")
        # import sys; sys.exit(0)
        t0 = time.time()
        local_iter_num = 0  # number of iterations in the lifetime of this process
        raw_model = model.module if ddp else model  # unwrap DDP container if needed
        running_mfu = -1.0
        tokens_trained = 0
        while True:
            # determine and set the learning rate for this iteration
            lr = self.get_lr(iter_num) if decay_lr else learning_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if (iter_num + 1) % eval_interval == 0 and self.master_process:
                losses = self.estimate_loss()
                print(
                    f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )
                # print("batch_size:", batch_size)
                if wandb_log:
                    wandb.log(
                        {
                            "iter": iter_num,
                            "train/loss": losses["train"],
                            "val/loss": losses["val"],
                            "lr": lr,
                            "mfu": running_mfu * 100,  # convert to percentage
                            "train/tokens": tokens_trained,
                            "train/actual_context_window": actual_context_window_without_ignore_tokens
                        }
                    )
                if losses["val"] < best_val_loss or always_save_checkpoint:
                    best_val_loss = losses["val"]
                    if iter_num > 0:
                        model_args = self.model_args
                        checkpoint = {
                            "model": raw_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "model_args": model_args,
                            "iter_num": iter_num,
                            "best_val_loss": best_val_loss,
                            "config": self.config.__dict__,
                        }
                        print(f"saving checkpoint to {out_dir}")
                        torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))

            if iter_num == 0 and eval_only:
                break

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(gradient_accumulation_steps):
                if ddp:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code
                    # looking at the source of that context manager, it just toggles this variable
                    model.require_backward_grad_sync = (
                        micro_step == gradient_accumulation_steps - 1
                    )
                with ctx:
                    start_time = time.time()
                    # logits, loss = model(x_pos=X_pos, x=X, y_pos=Y_pos, y=Y)
                    logits, loss = model(X, Y)
                    end_time = time.time()
                    # print(f"forward pass time: {end_time - start_time:.3f} seconds")
                    loss = (
                        loss / gradient_accumulation_steps
                    )  # scale the loss to account for gradient accumulation

                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X_pos, X, Y_pos, Y = self.get_batch("train")

                # count tokens and other stats
                tokens_trained += X.shape[0] * X.shape[1]
                actual_context_window_without_ignore_tokens = (X != self.IGNORE_TOKEN).sum() / X.shape[0]

                # backward pass, with gradient scaling if training in fp16
                # if self.config.device == "cuda":
                scaler.scale(loss).backward()
                # else:
                #     loss.backward()
            # clip the gradient
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % log_interval == 0 and master_process:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * gradient_accumulation_steps
                if local_iter_num >= 5:  # let the training loop settle a bit
                    mfu = raw_model.estimate_mfu(
                        batch_size * gradient_accumulation_steps, dt
                    )
                    running_mfu = (
                        mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                    )
                print(
                    f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
                )

            # log the decoded ground truth codes
            # TODO: this is hardcoded for now
            if self.position_dim == 3:
                t = self.position_shape[0]
                h = self.position_shape[1]
                w = self.position_shape[2]

            if iter_num == 0:
                # Decode from the ground truth token ids
                if master_process and self.config.vae_model_name is not None:
                    device = self.config.device
                    vae = self.vae
                    # TODO: this is hard coded for 3D positions (time, height, width)
                    t = 8
                    h = 64
                    w = 64
                    offset = 1
                    ground_truth_codes = (
                        torch.Tensor(self.val_data[0:1, offset:].astype(np.int64))
                        .long()
                        .to(device)
                    )  # this is hard coded
                    
                    if len(ground_truth_codes.shape) == 2:
                        ground_truth_codes = rearrange(
                            ground_truth_codes,
                            "b (t h w) -> b t h w",
                            b=1,
                            t=int(t),
                            h=int(h),
                            w=int(w),
                        )
                    if self.position_dim == 3:
                        assert ground_truth_codes.shape == (
                            1,
                            t,
                            h,
                            w,
                        ), ground_truth_codes.shape
                    assert isinstance(
                        ground_truth_codes, torch.Tensor
                    ), f"expected torch.Tensor, got {type(ground_truth_codes)}"
                    with torch.no_grad():
                        z = vae.vq.codes_to_vec(ground_truth_codes)
                        assert len(z.shape) == 5
                        assert z.shape == (1, 4, t, h, w)
                        video = vae.decoder(z)
                        video = (video - video.min()) / (
                            video.max() - video.min() + 1e-6
                        )
                        video = (video * 255).to(torch.uint8)
                        video = rearrange(video, "b c t h w -> t c h (b w)")
                        assert video.shape[1] == 3, f"shape of video is {video.shape}"
                        display = wandb.Video(video.detach().cpu(), fps=5, format="mp4")
                        if wandb_log:
                            wandb.log({"sampled/ground_truth_decoded": display})

            if ((iter_num + 1) % sample_interval == 0) and master_process:
                # sample from the model
                model.eval()
                # TODO: this is hard coded for 3D positions (time, height, width)
                t = 2
                max_new_tokens = int(t) * int(h) * int(w)
                with torch.no_grad():
                    sampled_codes = model.generate(
                        # [],
                        torch.Tensor(
                            np.ones((1, 1), dtype=np.int32) * self.config.start_token
                        ).long().to(device),
                        max_new_tokens,
                        temperature=1.0,
                        top_k=100,
                        show_progress=True,
                    )
                    sampled_codes = sampled_codes[0, 1:]

                if self.config.vae_model_name is not None:
                    idx_violating = sampled_codes > (self.config.vae_vocab_size - 1)
                    violating_codes = idx_violating.float().mean()
                    print(f"violating codes: {violating_codes.item()}. vae_vocab_size={self.config.vae_vocab_size}")
                    sampled_codes[idx_violating] = 0
                    # force_cudnn_initialization()
                    # sampled_codes = torch.ones(1, 32768, dtype=torch.long).to(device)
                    print("sampled codes:", sampled_codes)
                    # print(sampled_codes.min(), sampled_codes.max())

                    # This is harded coded for video generation.
                    assert (
                        self.position_dim == 3
                    ), "Only 3D positions are supported for decoding"
                    sampled_codes = rearrange(
                        sampled_codes[:max_new_tokens],
                        "(b t h w) -> b t h w",
                        b=1,
                        t=int(t),
                        h=int(h),
                        w=int(w),
                    )
                    assert sampled_codes.shape == (1, t, h, w), sampled_codes.shape
                    print(f"sampled_codes max={sampled_codes.max()}, min={sampled_codes.min()}")
                    z = vae.vq.codes_to_vec(sampled_codes)
                    assert len(z.shape) == 5
                    assert z.shape == (1, 4, t, h, w)
                    video = vae.decoder(z)
                    video = (video - video.min()) / (video.max() - video.min() + 1e-6)
                    video = (video * 255).to(torch.uint8)
                    print("video before rearange:", video.shape)
                    video = rearrange(video, "b c t h w -> t c h (b w)", c=3, b=1)
                    display = wandb.Video(video.cpu(), fps=5, format="mp4")
                    if wandb_log:
                        wandb.log(
                            {
                                "sampled/generated_video": display,
                                "sampled/violating_codes": violating_codes,
                            }
                        )

                    model.train()

            iter_num += 1
            local_iter_num += 1

            # termination conditions
            if iter_num > max_iters:
                break

        if ddp:
            destroy_process_group()


@dataclass
class MDGPTConfig(GPTConfig):
    sparse_block_size: int = 128
    start_token: int = 5121
    ignore_token: int = 5122
    # Position shape is the shape of the positional indices.
    position_shape: Tuple[int, int, int] = None
    use_1d_pos_embedding: bool = False
    causal: bool = False
    only_predict_last: bool = False
    disable_sparse_context_window: bool = False


def find_sparse_context_window(
    target_idx: int, positions, block_size, sparse_block_size, position_epsilon=1e-1
):
    """
    Given a target index, find up to `sparse_block_size` closest points to the target index.
    The neighborhood is defined on the position grid of N points.

    Args:
        target_idx: int, an integer between 0 and N-1 (inclusive). For a 3D position (t, r, c),
        inside the grid of size (T, H, W), the target_idx would be t * H * W + r * W + c.
        positions: (N, position_dim)
        block_size: int, the maximum number of points to look back.
        sparse_block_size: int, the number of points to return.
        position_epsilon: float, the noise to add to the position to break ties.
    
    Returns:
        context_token_idx: (sparse_block_size,), the indices of the closest points to the target_idx.
            Each idx can take values between -1 and N-1 (inclusive). -1 indicates an invalid idx.
        relative_pos: (sparse_block_size, position_dim), the relative position of the context_token_idx
            with respect to the target_idx. When target_idx is 0, the relative_pos is 0 for all
            context_token_idx. relative_pos can be determined by context_token_idx, target_idx and positions.
    """
    if target_idx == 0:
        # This is the very first position. There is no prior positions to look back on.
        position_dim = positions.shape[-1]
        assert position_dim < 5, f"position_dim={position_dim}, positions.shape={positions.shape}"
        context_token_idx = -1 * np.ones((sparse_block_size,))
        relative_pos = 0 * np.ones((sparse_block_size, position_dim))
        return context_token_idx, relative_pos
    
    # The look-back context window is from target_idx - block_size to target_idx.
    # When target_idx - block_size is negative, we start from 0.
    # From the context window up to size `block_size`, we will pick `sparse_block_size` closest positions.
    # So the actual sparse context window will be of size `sparse_block_size`.
    start_idx = max(0, target_idx - block_size)
    target_position = positions[target_idx]
    # print("target pos:", target_position)
    distances = np.linalg.norm(
        positions[start_idx:target_idx] - target_position, axis=1
    )
    distances_exponential = np.exp(-distances)
    position_dim = positions.shape[1]
    if position_dim > 1:
        # For multi-dimensional positions, add some noise to break ties.
        # This is because there are multiple ways to order the points in multi-dimensional space.
        # For 1-D positions, this is not necessary.
        distances_exponential += (
            np.random.rand(*distances_exponential.shape) * position_epsilon
        )

    # Now, order the distances from smallest to largest.
    # print(distances.shape)
    # print("distances:", distances_exponential)

    # TODO: for debug
    # distances_exponential = -np.arange(start_idx, target_idx)

    sorted_indices = np.argsort(-distances_exponential)
    # print("sorted indices:", sorted_indices)
    

    try:
        context_token_idx = np.arange(start_idx, target_idx)[
            sorted_indices[:sparse_block_size]
        ]
        # TODO: just do context_token_idx = sorted_indices[:sparse_block_size] + start_idx
        assert np.all(context_token_idx == sorted_indices[:sparse_block_size] + start_idx)
        
    except:
        print(f"sorted_indices={sorted_indices[:sparse_block_size]}, start_idx={start_idx}, target_idx={target_idx}")
        raise

    # Determine the relative position of the context_token_idx with respect to the target_idx.
    relative_pos = target_position - positions[context_token_idx]
    assert len(relative_pos.shape) in [2, 3], f"relative_pos.shape={relative_pos.shape}, target_position.shape={target_position.shape}, positions.shape={positions.shape}, context_token_idx.shape={context_token_idx.shape}, start_idx={start_idx}, target_idx={target_idx}, block_size={block_size}, sparse_block_size={sparse_block_size}"
    return context_token_idx, relative_pos


def pad_seq(context_idx, sparse_block_size, value=-1):
    # context_idx: (sparse_block_size,)
    # returns: (sparse_block_size,)
    # pad the context_idx to the sparse_block_size
    padded = np.ones((sparse_block_size,), dtype=np.int64) * value
    padded[: len(context_idx)] = context_idx
    return padded


def precompute_context_windows(positions_flat, block_size, sparse_block_size, position_epsilon=1e-1):
    """
    Note! To use context_idx, you need to first insert an "ignore token" at the start of the positions and tokens for each example.
    And you need to use context_idx + 1.
    This way, a token idx of 0 is reserved for the ignore token at the beginning of the sequence.
    The real idx starts from 1.

    Returns:
        relative_pos_lut: (N, sparse_block_size, position_dim), the relative position of the context_token_idx
            with respect to the target_idx. When target_idx is 0, the relative_pos is 0 for all
            context_token_idx. relative_pos can be determined by context_token_idx, target_idx and positions.
        context_token_idx_lut: (N, sparse_block_size), the indices of the closest points to the target_idx.
            Each idx can take values between -1 and N-1 (inclusive). -1 indicates an invalid idx.
    """

    relative_pos_lut = []
    context_token_idx_lut = []
    position_dim = positions_flat.shape[1]
    for target_idx in tqdm(range(0, len(positions_flat))):
        context_token_idx, relative_pos = find_sparse_context_window(
            target_idx,
            positions_flat,
            block_size,
            sparse_block_size,
            position_epsilon=position_epsilon,
        )
        # print(f"context_token_idx.shape={context_token_idx.shape}, relative_pos.shape={relative_pos.shape}")
        relative_pos_padded = np.concatenate(
            [
                relative_pos,
                np.zeros((sparse_block_size - len(relative_pos), position_dim)),
            ],
            axis=0,
        )
        # print(f"relative_pos_padded.shape={relative_pos_padded.shape}, sparse_block_size={sparse_block_size}")
        context_token_idx_padded = pad_seq(context_token_idx, sparse_block_size, value=-1)

        relative_pos_lut.append(relative_pos_padded)
        context_token_idx_lut.append(context_token_idx_padded)

    relative_pos_lut = np.stack(relative_pos_lut, axis=0)
    context_token_idx_lut = np.stack(context_token_idx_lut, axis=0)
    return relative_pos_lut, context_token_idx_lut


class MDGPT(GPT):

    def __init__(self, config: MDGPTConfig):
        super(GPT, self).__init__()  # Use the grandparent class.
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert (
            config.start_token < config.vocab_size
        ), f"start_token: {config.start_token}, vocab_size: {config.vocab_size}"
        assert (
            config.ignore_token < config.vocab_size
        ), f"ignore_token: {config.ignore_token}, vocab_size: {config.vocab_size}"
        print("position_shape:", config.position_shape)
        position_embedding_input_sizes = [2 * s + 1 for s in config.position_shape]
        print("position_embedding_input_sizes:", position_embedding_input_sizes)
        self.config = config
        BlockCls = Block if config.causal else BlockNonCausal
        if config.use_1d_pos_embedding:
            self.transformer = nn.ModuleDict(
                dict(
                    wte=nn.Embedding(config.vocab_size, config.n_embd),
                    wpe=nn.Embedding(config.sparse_block_size, config.n_embd),
                    drop=nn.Dropout(config.dropout),
                    h=nn.ModuleList([BlockCls(config) for _ in range(config.n_layer)]),
                    ln_f=LayerNorm(config.n_embd, bias=config.bias),
                )
            )
        else:
            self.transformer = nn.ModuleDict(
                dict(
                    wte=nn.Embedding(config.vocab_size, config.n_embd),
                    wpe=nn.ModuleList(
                        [
                            nn.Embedding(s, config.n_embd)
                            for s in position_embedding_input_sizes
                        ]
                    ),
                    drop=nn.Dropout(config.dropout),
                    h=nn.ModuleList([BlockCls(config) for _ in range(config.n_layer)]),
                    ln_f=LayerNorm(config.n_embd, bias=config.bias),
                )
            )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)

        # Check the embedding weights
        if not config.use_1d_pos_embedding:
            for i, wpe in enumerate(self.transformer.wpe):
                print(
                    f"embedding {i} weight mean, std:", wpe.weight.mean(), wpe.weight.std()
                )

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def set_lut(self, relative_pos_lut, context_idx_lut):
        self.context_idx_lut = context_idx_lut
        self.relative_pos_lut = relative_pos_lut

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            if self.config.use_1d_pos_embedding:
                n_params -= self.transformer.wpe.weight.numel()
            else:
                n_params -= sum([m.weight.numel() for m in self.transformer.wpe])
        return n_params

    def forward(self, x_pos, x, y_pos, y=None):
        # print(f"x_mean: {x.float().mean()},")
        idx = x
        targets = y
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)

            # TODO: for debug
            # targets[:, :-1] = -1

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def forward_md(self, x_pos, x, y_pos, y=None):
        """
        Args:
            x_pos (torch.Tensor): The positional indices of the input tokens. The shape is (batch_size, block_size, position_dim).
            x (torch.Tensor): The input tokens. The shape is (batch_size, block_size).
            y_pos (torch.Tensor): The positional indices of the target tokens. The shape is (batch_size, position_dim).
            y (torch.Tensor): The target tokens. The shape is (batch_size, 1).

        Returns:
            torch.Tensor: The logits of the model. The shape is (batch_size, block_size, vocab_size).
            torch.Tensor: The loss of the model.

        Context window sparsification:
            x and x_pos will be re-ordered in axis 1. The re-ordering is based on the
            distance between y_pos and x_pos. Tie is broken arbitrarily with randomness.
            The distance should saturate after a certain value, to ensure the furthest tokens
            have a chance to be included in the context window.
            In re-ordering, the context tokens closest to y_pos will be placed first.
            Note that this will often reverse the original order of the tokens in x.
            The context tokens and their positions are then pruned to the
            first `sparse_block_size` tokens. Resulting in a speed up of (block_size / sparse_block_size) ^ 2.
        """
        # First, reverse the order such that the nearest tokens are last.
        if self.config.disable_sparse_context_window:
            pass
        else:
            x = x.flip(1)
            x_pos = x_pos.flip(1)
            assert len(y_pos.shape) == 2
            relative_pos = y_pos.unsqueeze(1) - x_pos
            # The L2 norm of relative_pos should decrease.
            l2_relative_pos = torch.norm(relative_pos.float(), dim=-1)
            assert l2_relative_pos[0, -1] <= max(1, l2_relative_pos[0, -2]), f"l2_relative_pos: {l2_relative_pos}, {l2_relative_pos.shape}, x_pos: {x_pos}, y_pos: {y_pos}"

        # print("relative_pos:", relative_pos.shape, "x:", x.shape)
        assert (
            x.max() < self.config.vocab_size
        ), f"x.max(): {x.max()}, vocab_size: {self.config.vocab_size}"

        input_x = x
        b, t = input_x.size()

        if self.config.disable_sparse_context_window:
            assert self.config.use_1d_pos_embedding, "disable_sparse_context_window only supported with 1D pos embedding"
        x = self.transformer.wte(x)
        if self.config.use_1d_pos_embedding:
            pos = torch.arange(0, t, dtype=torch.long, device=x.device)
            if self.config.disable_sparse_context_window:
                assert t <= self.config.block_size, f"t: {t}, block_size: {self.config.block_size}"
            else:
                assert t <= self.config.sparse_block_size
            try:
                pos_emb = self.transformer.wpe(pos)
                embed_dim = self.transformer.wte.embedding_dim
                assert pos_emb.shape == (x.shape[1], embed_dim), f"pos_emb: {pos_emb.shape}, x: {x.shape}"
                x += pos_emb.unsqueeze(0)
            except:
                print("pos:", pos.shape)
                raise
        else:
            position_dim = len(self.config.position_shape)
            for d in range(position_dim):
                offset = self.config.position_shape[d]
                # TODO: try absolute value, instead of offsetting
                assert (
                    relative_pos[:, :, d].max() < self.transformer.wpe[d].num_embeddings
                ), f"relative_pos.max(): {relative_pos.max()}, block_size: {self.config.block_size}, {self.transformer.wpe[d]}, num_embeddings: {self.transformer.wpe[d].num_embeddings}"
                relative_pos_with_offset = relative_pos[:, :, d] + offset
                # print(f"x: {x.shape}, relative_pos_with_offset: {relative_pos_with_offset.shape}, position_dim: {position_dim}")
                x += self.transformer.wpe[d](relative_pos_with_offset) / float(
                    position_dim
                )
        
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if y is not None:
            if self.config.only_predict_last:
                # Only predict using the last token
                logits = self.lm_head(
                    x[:, -1, :]
                )
                y[y==self.config.ignore_token] = -1
                loss = F.cross_entropy(
                    logits,
                    y.squeeze(-1),
                    ignore_index=-1,
                )
                if loss.isnan():
                    print("Nan loss detected")
                    print(f"input_x: {input_x.detach().cpu().numpy().ravel()}")
                    print(f"x: {x.detach().cpu().numpy().ravel()}")
                    print(
                        f"relative_pos: {relative_pos.detach().cpu().numpy().ravel()}"
                    )
            else:
                # predict all tokens
                y[y==self.config.ignore_token] = -1
                # repeat y on the time dimension
                if self.config.disable_sparse_context_window:
                    assert len(y.shape) == 2, f"{y}"
                    pass
                else:
                    if len(y.shape) == 2:
                        y = y.expand(-1, x.shape[1])
                    else:
                        raise ValueError(f"y shape: {y.shape} not supported in loss calculation")
                    y = y.contiguous()
                logits = self.lm_head(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=-1,
                )

        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            # TODO: -1 is not necessarily the last position, use a mask to find the last position.
            if x.shape[0] == 1:
                # TODO: prune ignored tokens
                # x = x[x != self.config.ignore_token]
                pass
            else:
                if (x == self.config.ignore_token).sum() > 0:
                    print(f"Warning: batch size is not 1, some ignored tokens.")
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(
        self, idx, max_new_tokens, temperature=1.0, top_k=None, show_progress=False
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        if show_progress:
            from tqdm import tqdm

            range_to_iterate = tqdm(range(max_new_tokens), mininterval=2)
        else:
            range_to_iterate = range(max_new_tokens)
        for _ in range_to_iterate:
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )

            # import sys
            # print("idx_cond max:", idx_cond.max())
            # sys.exit(0)

            # forward the model to get the logits for the index in the sequence
            logits, _ = self(x_pos=None, x=idx_cond, y_pos=None)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @torch.no_grad()
    def generate_md(
        self, x, max_new_tokens=None, temperature=1.0, top_k=None, show_progress=False
    ):
        batch_size = 1  # TODO: this is hardcoded
        device = self.transformer.wte.weight.device

        target_pos = len(x)  # target_pos should be 0 if x is empty
                             # this should happen before the start token is inserted
        
        if len(x) == 0:
            x = torch.from_numpy(np.array([self.config.start_token])).to(device).unsqueeze(0)

        if len(self.config.position_shape) == 1:
            assert (
                max_new_tokens is not None
            ), f"max_new_tokens should not be None for 1D sequence"

        target_indices_to_print = [0, 9]
        iterated = range(target_pos, target_pos + max_new_tokens)
        if show_progress:
            iterated = tqdm(iterated, mininterval=2)
        for target_idx in iterated:
            if target_idx in target_indices_to_print:
                print("=" * 80)
                print(
                    f"Generating target_idx: {target_idx}. ignore_token={self.config.ignore_token}, start_token={self.config.start_token}"
                )
            context_token_idx = self.context_idx_lut[target_idx]
            relative_pos = self.relative_pos_lut[target_idx]
            # print("context_token_idx:", context_token_idx)
            # print("relative_pos:", relative_pos.shape)
            assert len(relative_pos.shape) in [2, 3], f"relative_pos shape: {relative_pos.shape}"
            # assert context_token_idx.min() >= -1, f"context_token_idx.min(): {context_token_idx.min()}"

            # valid_idx = context_token_idx >= 0
            # context_token_idx = context_token_idx[valid_idx]
            # relative_pos = relative_pos[valid_idx]

            if len(self.config.position_shape) == 1:
                y_pos = torch.from_numpy(np.array([target_idx])).unsqueeze(0)
            elif len(self.config.position_shape) == 3:
                y_pos = torch.from_numpy(
                    np.array(
                        [
                            target_idx
                            // (
                                self.config.position_shape[1]
                                * self.config.position_shape[2]
                            ),
                            (target_idx // self.config.position_shape[2])
                            % self.config.position_shape[1],
                            target_idx % self.config.position_shape[2],
                        ]
                    )
                )
            cond_y_pos = y_pos

            # print(
            #     f"context_token_idx: {context_token_idx}, {context_token_idx.shape}, {type(context_token_idx)} x: {x}"
            # )
            if len(context_token_idx) == 0:
                cond_x = torch.concatenate(x, dim=0)
                relative_pos = np.zeros((1, len(self.config.position_shape)))
            else:
                context_token_idx = (
                    context_token_idx.int() + 1
                )  # position 0 is reserved for ignore token

                x_concat = x
                if (target_idx in target_indices_to_print):
                    print(f"x_concat: {x_concat}, shape is {x_concat.shape}")

                # assert context_token_idx.min() >= 0, f"context_token_idx.min(): {context_token_idx.min()}"
                # print("context_token_idx:", context_token_idx)
                if self.config.disable_sparse_context_window:
                    if x_concat.shape[1] < self.config.block_size:
                        cond_x = x_concat
                    else:
                        cond_x = x_concat[:, -self.config.block_size:]
                else:
                    cond_x = x_concat[:, context_token_idx]
                # print("cond_x:", cond_x.shape, "x_concat:", x_concat.shape, "context_token_idx:", context_token_idx.shape)

            # cond_x = rearrange(cond_x, "(b t) -> b t", b=batch_size)
            if isinstance(relative_pos, np.ndarray):
                relative_pos = torch.from_numpy(relative_pos)
            relative_pos = relative_pos.unsqueeze(0)  # add batch dimension
            assert (
                len(relative_pos.shape) == 3
            ), f"relative_pos shape: {relative_pos.shape}"
            if len(y_pos.shape) == 1:
                y_pos = y_pos.unsqueeze(0)
            assert len(y_pos.shape) == 2, f"y_pos shape: {y_pos.shape}"
            # print(
            #     f"cond_x: {cond_x.shape}, cond_y_pos: {cond_y_pos.shape}, relative_pos: {relative_pos.shape}"
            # )
            cond_x_pos = cond_y_pos - relative_pos

            if (target_idx in target_indices_to_print):
                print(f"cond_x (first 5): {cond_x[:, :5, ...]}")
                print(f"cond_x_pos (first 5): {cond_x_pos[:, :5, ...]}")
                print(f"relative pos (first 5):", relative_pos[:, :5, ...])
                print(f"y_pos: {y_pos}")
            
            sparse_block_size = len(context_token_idx)
            # print("sparse block_size:", sparse_block_size)
            if not self.config.disable_sparse_context_window:
                assert cond_x.shape == (batch_size, sparse_block_size), f"cond_x shape: {cond_x.shape}"
            logits, _ = self.forward(
                x_pos=cond_x_pos.long().to(device),
                x=cond_x.long().to(device),
                y_pos=y_pos.long().to(device),
            )
            logits = logits[0, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx_next = idx_next.unsqueeze(0)
            # print(f"x:", x.shape, "idx_next:", idx_next.shape)
            x_shape = x.shape
            x = torch.cat((x, idx_next), dim=1)
            assert len(x.shape) == len(x_shape), f"before: {x_shape}, after: {x.shape}"
            # x.append(idx_next.detach())
            if target_idx in target_indices_to_print:
                print(f"idx_next: {idx_next.detach().cpu().numpy()}")

        # x = torch.cat(x, dim=0)
        # x = x.unsqueeze(0)
        return x

