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
from .gpt import GPT, GPTConfig, Block, CausalSelfAttention, LayerNorm, MLP


class MDGPTTrainingPipeline:
    @dataclass
    class Config:

        log_dir: str = "logs/mdgpt"
        wandb_log: bool = False
        project: str = "mdgpt"
        init_from: str = "scratch"  # 'scratch' or 'resume' or 'gpt2*'

        block_size: int = 1024
        sparse_block_size: int = 128
        vocab_size: int = 5120
        batch_size: int = 32
        n_layer: int = 12
        n_head: int = 12
        n_embd: int = 768
        dropout: float = 0.0
        gradient_accumulation_steps: int = 1
        bias: bool = True
        device: str = "cuda"

        learning_rate: float = 6e-4  # max learning rate
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
        vae_model_name: str = None
        vocab_size: int = 5123
        meta_vocab_size: int = None
        start_token: int = 5121
        ignore_token: int = 5122

        # we expect to overfit on this small dataset, so only save when val improves
        always_save_checkpoint: bool = False
        compile: bool = True  # use PyTorch 2.0 to compile the model to be faster
        eval_only = False  # if True, script exits right after the first eval

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

        torch.manual_seed(1337 + self.seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

        self.START_TOKEN = self.config.start_token
        self.IGNORE_TOKEN = self.config.ignore_token

    def fit(self, dataset_builder):
        if self.master_process and self.config.wandb_log:
            wandb.init(project=self.config.project, config=self.config)
        self.create_dataloaders(dataset_builder)
        self.precompute_sparse_context_window()
        self.create_model()
        self.create_grad_scaler()
        self.create_optimizer()
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
        block_size = self.config.block_size
        sparse_block_size = self.config.sparse_block_size
        batch_size = self.config.batch_size
        device = self.config.device

        data = train_data if split == "train" else self.val_data
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
            positions = positions.reshape(-1, positions.shape[-1])
            # print(f"data shape: {data.shape}")
            # print(f"positions shape: {positions.shape}")

        # Problem formulation: x_pos, x, y_pos, y?
        # Randomly sample a target index for each example in the batch
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        assert (
            len(data.shape) == 2
        ), f"Expected 2D array for training data, got {data.shape}"

        # # Insert the start token (optional)
        # data = np.concatenate(
        #     [np.ones((data.shape[0], 1), dtype=data.dtype) * self.START_TOKEN, data],
        #     axis=1,
        # )

        # Insert the ignore token at the beginning
        data_with_ignore_token = np.concatenate(
            [np.ones((data.shape[0], 1), dtype=data.dtype) * self.IGNORE_TOKEN, data],
            axis=1,
        )

        irow = np.random.randint(data.shape[0], size=batch_size)
        ix = torch.randint(data.shape[1] - block_size - 2, (batch_size,)) + 1
        # print("positions:", positions.shape)
        target_pos = [
            torch.from_numpy(positions[col, ...]) for row, col in zip(irow, ix)
        ]
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
        x_pos = (target_pos - relative_pos).int()
        sparse_block_size = self.config.sparse_block_size
        if self.context_idx_lut is not None:
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
        y_pos = y_pos.squeeze(1)

        # assert the shapes
        assert x.shape == (batch_size, sparse_block_size), x.shape
        assert y.shape == (batch_size, 1), y.shape
        assert x_pos.shape == (batch_size, sparse_block_size, position_dim), x_pos.shape
        assert y_pos.shape == (batch_size, position_dim), y_pos.shape
        assert x.dtype == torch.long, f"expected long tensor, got {x.dtype}"

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
        assert (y_pos.unsqueeze(1) - x_pos).max() <= self.config.sparse_block_size
        assert x.max() < self.config.vocab_size
        assert y.max() < self.config.vocab_size
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
        if len(self.train_data.shape) == 1:
            self.position_shape = (self.config.sparse_block_size,)
        else:
            self.position_shape = self.train_data.shape[1:]
        self.position_dim = len(self.position_shape)

    def precompute_sparse_context_window(self):
        # precompute the sparse context window for each position
        # this is a one-time operation that will speed up training
        block_size = self.config.block_size
        sparse_block_size = self.config.sparse_block_size
        train_data = self.train_data
        position_dim = self.position_dim

        if len(train_data.shape) == 1:
            if len(train_data) > 1e3:
                print(
                    "Not precomputing sparse context window for 1D dataset with long sequences. This will be slow."
                )
                self.relative_pos_lut = None
                self.context_idx_lut = None
                return
            train_data = train_data.reshape(1, -1)

        if position_dim == 1:
            positions = np.arange(train_data.shape[-1])
        else:
            if len(train_data.shape) == 1:
                meshgrid_args = [np.arange(train_data.shape[0])]
            else:
                meshgrid_args = [np.arange(s) for s in train_data.shape[1:]]
            positions = np.array(
                np.meshgrid(*meshgrid_args, indexing="ij"),
            )
            orig_shape = tuple(range(len(positions.shape)))
            transposed_shape = orig_shape[1:] + (orig_shape[0],)
            positions = positions.transpose(*transposed_shape)

        print("positions shape:", positions.shape)
        positions_flat = positions.reshape(-1, position_dim)
        self.relative_pos_lut, self.context_idx_lut = precompute_context_windows(
            positions_flat=positions_flat,
            block_size=block_size,
            sparse_block_size=sparse_block_size,
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
            meta_vocab_size = (
                self.config.vocab_size + 20
            )  # 20 extra tokens for special tokens
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

        # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
        iter_num = 0
        best_val_loss = 1e9

        X_pos, X, Y_pos, Y = self.get_batch("train")  # fetch the very first batch
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
            if iter_num % eval_interval == 0 and self.master_process:
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
                    logits, loss = model(X_pos, X, Y_pos, Y)
                    loss = (
                        loss / gradient_accumulation_steps
                    )  # scale the loss to account for gradient accumulation
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X_pos, X, Y_pos, Y = self.get_batch("train")
                # backward pass, with gradient scaling if training in fp16
                if self.config.device == "cuda":
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
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
                    f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, y_pos_min {Y_pos.min()}, y_pos_max {Y_pos.max()}"
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
                    ground_truth_codes = (
                        torch.Tensor(self.val_data[0:1, :].astype(np.int64))
                        .long()
                        .to(device)
                    )  # this is hard coded
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
                max_new_tokens = self.config.block_size
                with torch.no_grad():
                    sampled_codes = model.generate(
                        [],
                        max_new_tokens,
                    )
                    sampled_codes = sampled_codes[0, 1:]

                if self.config.vae_model_name is not None:
                    idx_violating = sampled_codes > (self.config.vae_vocab_size - 1)
                    violating_codes = idx_violating.float().mean()
                    print(f"violating codes: {violating_codes.item()}")
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

                    z = vae.vq.codes_to_vec(sampled_codes)
                    assert len(z.shape) == 5
                    assert z.shape == (1, 4, t, h, w)
                    video = vae.decoder(z)
                    video = (video - video.min()) / (video.max() - video.min() + 1e-6)
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


@dataclass
class MDGPTConfig(GPTConfig):
    sparse_block_size: int = 128
    start_token: int = 5121
    ignore_token: int = 5122
    position_shape: Tuple[int, int, int] = (8, 64, 64)


def find_sparse_context_window(
    target_idx: int, positions, block_size, sparse_block_size, position_epsilon=1e-1
):
    # positions: (N, 3)
    # target_idx: (3,)
    # block_size: int
    # returns: (block_size, 3)
    # find the block_size closest points to the target_idx
    # using the L2 distance
    # look back at most block_size points
    start_idx = max(0, target_idx - block_size)
    target_position = positions[target_idx]
    # print("target pos:", target_position)
    distances = np.linalg.norm(
        positions[start_idx:target_idx] - target_position, axis=1
    )
    distances_exponential = np.exp(-distances)
    position_dim = positions.shape[1]
    if position_dim > 1:
        distances_exponential += (
            np.random.rand(*distances_exponential.shape) * position_epsilon
        )

    # print(distances.shape)
    # print("distances:", distances_exponential)
    sorted_indices = np.argsort(-distances_exponential)
    # print("sorted indices:", sorted_indices)
    context_token_idx = np.arange(start_idx, target_idx)[
        sorted_indices[:sparse_block_size]
    ]
    relative_pos = target_position - positions[context_token_idx]
    return context_token_idx, relative_pos


def pad_seq(context_idx, sparse_block_size, value=-1):
    # context_idx: (sparse_block_size,)
    # returns: (sparse_block_size,)
    # pad the context_idx to the sparse_block_size
    padded = np.ones((sparse_block_size,), dtype=np.int64) * value
    padded[: len(context_idx)] = context_idx
    return padded


def precompute_context_windows(positions_flat, block_size, sparse_block_size):
    """
    Note! To use context_idx, you need to first insert an "ignore token" at the start of the positions and tokens for each example.
    And you need to use context_idx + 1.
    This way, a token idx of 0 is reserved for the ignore token at the beginning of the sequence.
    The real idx starts from 1.
    """
    # positions_flat_padded = np.concatenate(
    #     [np.ones((1, 3), dtype=np.int64) * 0, positions_flat],
    #     axis=0,
    # )

    relative_pos_lut = []
    context_token_idx_lut = []
    position_dim = positions_flat.shape[1]
    for target_idx in tqdm(range(len(positions_flat))):
        context_token_idx, relative_pos = find_sparse_context_window(
            target_idx,
            positions_flat,
            block_size,
            sparse_block_size,
        )
        relative_pos_padded = np.concatenate(
            [
                relative_pos,
                np.zeros((sparse_block_size - len(relative_pos), position_dim)),
            ],
            axis=0,
        )
        context_token_idx_padded = pad_seq(context_token_idx, sparse_block_size)  # + 1

        relative_pos_lut.append(relative_pos_padded)
        context_token_idx_lut.append(context_token_idx_padded)

    relative_pos_lut = np.stack(relative_pos_lut, axis=0)
    context_token_idx_lut = np.stack(context_token_idx_lut, axis=0)
    return relative_pos_lut, context_token_idx_lut


class MDGPT(GPT):

    def __init__(self, config):
        super(GPT, self).__init__()  # Use the grandparent class.
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert (
            config.start_token < config.vocab_size
        ), f"start_token: {config.start_token}, vocab_size: {config.vocab_size}"
        assert (
            config.ignore_token < config.vocab_size
        ), f"ignore_token: {config.ignore_token}, vocab_size: {config.vocab_size}"
        position_dim = len(config.position_shape)
        print("position_shape:", config.position_shape)
        position_embedding_input_sizes = [2 * s + 1 for s in config.position_shape]
        print("position_embedding_input_sizes:", position_embedding_input_sizes)
        self.config = config
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
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
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

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= sum([m.weight.numel() for m in self.transformer.wpe])
        return n_params

    def forward(self, x_pos, x, y_pos, y=None):
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

        if x.shape[0] == 1:
            # Perform this trick to save some computation when batch_size is 1
            idx_to_ignore = x == self.config.ignore_token
            if (~idx_to_ignore).sum() > 0:
                x = x[~idx_to_ignore]
                x_pos = x_pos[~idx_to_ignore]
                if len(x.shape) == 1:
                    x = x.unsqueeze(0)
                if len(x_pos.shape) == 2:
                    x_pos = x_pos.unsqueeze(0)

        relative_pos = y_pos.unsqueeze(1) - x_pos
        assert (
            x.max() < self.config.vocab_size
        ), f"x.max(): {x.max()}, vocab_size: {self.config.vocab_size}"

        input_x = x

        # if y is not None:
        #     print(
        #         f"x={x[0].cpu().numpy().ravel()},\ny={y[0].cpu().numpy().ravel()},\nrelative_pos={relative_pos[0].cpu().numpy().ravel()}"
        #     )

        # if self.transformer.wte.weight.isnan().sum() > 0:
        #     print(
        #         "Nan detected in wte:",
        #         self.transformer.wte.weight.isnan().sum().item(),
        #         "nan values",
        #     )
        #     raise ValueError("Nan detected in wte weights")
        x = self.transformer.wte(x)
        # if x.isnan().sum() > 0:
        #     print("Nan detected in x after wte:", x.isnan().sum().item(), "nan values")
        #     print(f"x: {x}")
        #     raise ValueError("Nan detected in x after wte")
        position_dim = len(self.config.position_shape)
        for d in range(position_dim):
            offset = self.config.position_shape[d]
            # print(f"d={d}, offset={offset}, relative_pos_pruned shape: {relative_pos_pruned.shape}")
            # print("embedding model:", self.transformer.wpe[d])
            # print(f"input to embedding:", relative_pos_pruned[:, :, d] + offset)
            # TODO: try absolute value, instead of offsetting
            assert (
                relative_pos[:, :, d].max() < self.transformer.wpe[d].num_embeddings
            ), f"x_pos.max(): {relative_pos.max()}, block_size: {self.config.block_size}, {self.transformer.wpe[0]}"
            x += self.transformer.wpe[d](relative_pos[:, :, d] + offset) / float(
                position_dim
            )
        # if x.isnan().sum() > 0:
        #     print("Nan detected in x after wpe")
        #     print(f"x: {x}")
        #     raise ValueError("Nan detected in x after wpe")
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        # if x.isnan().sum() > 0:
        #     print("Nan detected in x after block")
        #     print(f"x: {x}")
        #     raise ValueError("Nan detected in x after block")
        x = self.transformer.ln_f(x)

        # print(f"after linear, x shape: {x.shape}")
        if y is not None:
            # Only predict using the last token
            logits = self.lm_head(
                x[:, -1, :]
            )
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

            if False:
                # ===========================================
                # if we are given some desired targets also calculate the loss
                idx_to_ignore = input_x == self.config.ignore_token
                # print(f"idx_to_ignore shape: {idx_to_ignore.shape}, {idx_to_ignore.sum()}")
                logits = self.lm_head(x)
                # print(f"logits shape: {logits.shape}")
                # y needs to be broadcasted
                # print(f"y shape 0: {y.shape}")
                y = y.expand(-1, logits.shape[1])
                y = y.unsqueeze(-1)
                # print(f"y shape 1: {y.shape}")

                logits_selected = logits
                y_selected = y.clone()
                y_selected[idx_to_ignore] = -1

                # print(f"logits_selected shape: {logits_selected.shape}")
                # print(f"y_selected shape: {y_selected.shape}")
                try:
                    logits_selected = logits_selected.view(-1, logits_selected.size(-1))
                    y_selected = y_selected.view(-1)
                    # print(
                    #     f"y_selected shape: {y_selected.shape}, logits_selected shape: {logits_selected.shape}"
                    # )
                    # print(f"y_selected: {y_selected.cpu().numpy().ravel()[:5]}")
                    # print(f"logits_selected: {logits_selected.cpu().numpy()[:2, :5]}")
                    loss = F.cross_entropy(
                        logits_selected,
                        y_selected,
                        ignore_index=-1,
                    )
                    if loss.isnan():
                        print("Nan loss detected")
                        print(f"input_x: {input_x.detach().cpu().numpy().ravel()}")
                        print(f"x: {x.detach().cpu().numpy().ravel()}")
                        print(
                            f"relative_pos: {relative_pos.detach().cpu().numpy().ravel()}"
                        )
                        print(f"y_selected: {y_selected.detach().cpu().numpy()}")
                        print(f"logits_selected: {logits_selected.detach().cpu().numpy()}")
                        raise ValueError("Nan loss detected")
                except:
                    print(f"y shape: {y.shape}")
                    print(f"logits shape: {logits.shape}")
                    # print(f"loss_weight shape: {loss_weight.shape}")
                    raise
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
        self, x, max_new_tokens=None, temperature=1.0, top_k=None, show_progress=False
    ):
        batch_size = 1  # TODO: this is hardcoded
        device = self.transformer.wte.weight.device

        target_pos = len(x)
        if len(x) == 0:
            x.append(torch.from_numpy(np.array([self.config.start_token])).to(device))

        whole_seq_len = len(x) + max_new_tokens

        if len(self.config.position_shape) == 1:
            assert (
                max_new_tokens is not None
            ), f"max_new_tokens should not be None for 1D sequence"
            positions = np.arange(whole_seq_len).reshape(-1, 1)
        else:
            print("Ignoring max_new_tokens for generating multi-dimensional data")
            meshgrid_args = [np.arange(s) for s in self.config.position_shape]
            positions = np.array(
                np.meshgrid(
                    *meshgrid_args,
                    indexing="ij",
                )
            )

        target_indices_to_print = [2, 9]
        for target_idx in range(target_pos, target_pos + max_new_tokens):
            if target_idx in target_indices_to_print:
                print("=" * 80)
                print(
                    f"Generating target_idx: {target_idx}. ignore_token={self.config.ignore_token}, start_token={self.config.start_token}"
                )
            context_token_idx, relative_pos = find_sparse_context_window(
                target_idx=target_idx,
                positions=positions,
                block_size=self.config.block_size,
                sparse_block_size=self.config.sparse_block_size,
            )
            assert len(relative_pos.shape) == 2
            valid_idx = context_token_idx >= 0
            context_token_idx = context_token_idx[valid_idx]
            relative_pos = relative_pos[valid_idx]

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
                    context_token_idx.astype(int) + 1
                )  # position 0 is reserved for ignore token
                x_concat = torch.cat(x, dim=0)
                if target_idx in target_indices_to_print:
                    print(f"x: {x_concat.cpu().numpy()}")
                cond_x = x_concat[context_token_idx]

            cond_x = rearrange(cond_x, "(b t) -> b t", b=batch_size)
            # print(f"cond_x: {cond_x.cpu().numpy()}")
            relative_pos = torch.from_numpy(relative_pos)
            relative_pos = relative_pos.unsqueeze(0)  # add batch dimension
            assert (
                len(relative_pos.shape) == 3
            ), f"relative_pos shape: {relative_pos.shape}"
            assert len(y_pos.shape) == 2
            cond_x_pos = cond_y_pos - relative_pos

            # print(
            #     f"cond_x_pos: {cond_x_pos.shape}, cond_x: {cond_x.shape}, y_pos: {y_pos.shape}"
            # )
            if target_idx in target_indices_to_print:
                print(f"cond_x: {cond_x.cpu().numpy()}")
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
            x.append(idx_next.detach())
            if target_idx in target_indices_to_print:
                print(f"idx_next: {idx_next.detach().cpu().numpy()}")

        x = torch.cat(x, dim=0)
        x = x.unsqueeze(0)
        return x

    @torch.no_grad()
    def generate_2(
        self, x_pos, x, new_pos, temperature=1.0, top_k=None, show_progress=False
    ):
        """
        Take a conditioning sequence of indices x and their positions x_pos, to complete
        the sequence len(new_pos) times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.

        Args:
            x_pos (torch.Tensor): The positional indices of the input tokens. The shape is (batch_size, block_size, position_dim).
            x (torch.Tensor): The input tokens. The shape is (batch_size, block_size).
            new_pos (torch.Tensor): The positional indices of the new tokens. The shape is (batch_size, max_new_tokens, position_dim).
            temperature (float): The temperature of the softmax.
            top_k (int): The number of top-k tokens to sample from.
            show_progress (bool): Whether to show a progress bar.
        """
        assert len(x_pos.shape) == 3, f"expected 3D tensor for x_pos, got {x_pos.shape}"
        assert (
            len(new_pos.shape) == 3
        ), f"expected 3D tensor for new_pos, got {new_pos.shape}"
        max_new_tokens = new_pos.shape[1]
        if show_progress:
            from tqdm import tqdm

            range_to_iterate = tqdm(range(max_new_tokens), mininterval=2)
        else:
            range_to_iterate = range(max_new_tokens)

        x_pos_updated = x_pos
        x_updated = x
        for j in range_to_iterate:
            y_pos = new_pos[:, j, :].to(x.device)
            # if the sequence context is growing too long we must crop it at block_size
            # idx_cond = (
            #     idx
            #     if idx.size(1) <= self.config.block_size
            #     else idx[:, -self.config.block_size :]
            # )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(x=x_updated, x_pos=x_pos_updated, y_pos=y_pos)
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
            # idx = torch.cat((idx, idx_next), dim=1)
            # update the context
            x_pos_updated = torch.cat((x_pos_updated, y_pos.unsqueeze(1)), dim=1)
            x_updated = torch.cat((x_updated, idx_next), dim=1)

        return x_pos_updated, x_updated
