import math
import time
import pickle
import numpy as np
from typing import Tuple
import inspect
import os
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from einops import rearrange
import wandb
from .gpt import GPT, GPTConfig, Block, MLP, CausalSelfAttention, LayerNorm

class MDGPTTrainingPipeline:
    # TODO: compile the model.
    @dataclass
    class Config:

        log_dir: str = "logs/nanogpt"
        wandb_log: bool = False
        project: str = "nano-gpt"
        init_from: str = "scratch"  # 'scratch' or 'resume' or 'gpt2*'

        # Data
        block_size: int = 1024
        sparse_block_size: int = 128
        vocab_size: int = 5120
        batch_size: int = 32
        flatten_tokens: bool = False
        ignore_token: int = -1
        start_token: int = 5121
        use_1d_pos_embedding: bool = False
        causal: bool = False
        only_predict_last: bool = False

        # Model
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
        sample_interval: int = 250
        vae_model_name: str = None
        vae_vocab_size: int = 5120
        vocab_size: int = 5120 + 3
        start_token: int = 5121
        meta_vocab_size: int = None
        max_tokens_to_sample: int = 128

        # sparse context window
        sparse_context_window: bool = False
        context_stride: int = 2
        context_stride_start: int = (
            32  # only do sparse context window before (block_size - context_stride_start)
        )

        # we expect to overfit on this small dataset, so only save when val improves
        always_save_checkpoint: bool = False
        compile: bool = False  # use PyTorch 2.0 to compile the model to be faster
        eval_only = False  # if True, script exits right after the first eval

    def __init__(self, config: Config):
        self.config = config
        self.out_dir = (
            f"{config.log_dir}/batch{config.batch_size}_block{config.block_size}"
        )
        self.dtype = (
            "bfloat16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "float16"
        )  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

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

    def fit(self, dataset_builder):
        if self.master_process and self.config.wandb_log:
            wandb.init(project=self.config.project, config=self.config)
        self.create_dataloaders(dataset_builder)
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
        """
        Get a batch of data for training or validation.

        Args:
            split (str): The split to get the data from. Can be either "train" or "val".

        Returns:
            tuple: A tuple containing two tensors: x and y.
                - x: The input tensor of shape (batch_size, block_size).
                - y: The target tensor of shape (batch_size, block_size).

        """
        if self.config.only_predict_last:
            return self.get_batch_for_predict_only_last(split)
        
        if self.config.flatten_tokens:
            train_data = self.train_data_flattened
            val_data = self.val_data_flattened
        else:
            train_data = self.train_data
            val_data = self.val_data
        
        block_size = self.config.block_size
        batch_size = self.config.batch_size
        device = self.config.device

        data = train_data if split == "train" else val_data
        x_pos = None
        y_pos = None
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
            x_pos = torch.stack([
                torch.from_numpy(self.positions_flattened[i1 : i1 + block_size]) for i1 in ix
            ])
        else:
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
            if x_pos is not None:
                x_pos = x_pos.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
            if x_pos is not None:
                x_pos = x_pos.to(device)
                assert len(x_pos.shape) == 2, f"x_pos.shape: {x_pos.shape}"
        assert x.shape == (self.config.batch_size, self.config.block_size), f"x.shape: {x.shape}"
        assert y.shape == (self.config.batch_size, self.config.block_size), f"y.shape: {y.shape}"

        return x_pos, x, y_pos, y

    def get_batch_for_predict_only_last(self, split: str):
        """
        Want to generate batches that train the model with:
        x1 -> x2
        x1 x2 -> x3
        x1 x2 x3 -> x4

        Use a class variable to keep track of where the target is.
        """
        data = self.train_data if split == "train" else self.val_data
        if not hasattr(self, "target_idx"):
            self.target_idx = 1
            self.pos_offset = 0
            # TODO: consider the start token, then idx starts from 1
        
        assert len(data.shape) == 2, f"expected 2D data, got {data.shape}"
        # insert the start token
        data = np.concatenate([np.ones((data.shape[0], 1), dtype=data.dtype) * self.START_TOKEN, data], axis=1)

        block_size = self.config.block_size
        irow = torch.randint(data.shape[0], (self.config.batch_size,))
        # In this batch, the target idx is shared.
        target_idx = self.target_idx
        context_start_idx = max(target_idx - block_size, 0)
        if target_idx == 1:
            # only with start token
            x = torch.stack([
                torch.from_numpy(data[i, :1].astype(np.int64)) for i in irow
            ])
        else:
            x = torch.stack([
                torch.from_numpy(data[i, (self.pos_offset+context_start_idx):(self.pos_offset+target_idx)].astype(np.int64)) for i in irow
            ])
        y = torch.stack([
            torch.from_numpy(data[i, (self.pos_offset+target_idx):(self.pos_offset+target_idx+1)].astype(np.int64)) for i in irow
        ])
        x_pos = None
        y_pos = None
        if self.device_type == "cuda":
            x, y = x.pin_memory().to(self.config.device, non_blocking=True), y.pin_memory().to(
                self.config.device, non_blocking=True
            )
            if x_pos is not None:
                x_pos = x_pos.pin_memory().to(self.config.device, non_blocking=True)
        else:
            x, y = x.to(self.config.device), y.to(self.config.device)
            if x_pos is not None:
                x_pos = x_pos.to(self.config.device)
                assert len(x_pos.shape) == 2, f"x_pos.shape: {x_pos.shape}"
        # update the target idx
        self.target_idx += 1
        if self.target_idx >= self.config.block_size:
            self.target_idx = 1
            # TODO: make pos_offset less random. It should only fall on the beginning of each frame.
            tokens_in_each_frame = self.train_data_orig.shape[2] * self.train_data_orig.shape[3]
            frame_offset = torch.randint(self.train_data_orig.shape[1] - block_size // tokens_in_each_frame, (1,)).item()
            self.pos_offset = frame_offset * tokens_in_each_frame
            # self.pos_offset = torch.randint(data.shape[1] - block_size, (1,)).item()
        return x_pos, x, y_pos, y

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

        if self.config.flatten_tokens:
            self.train_data_flattened = self.train_data.ravel()
            self.val_data_flattened = self.val_data.ravel()

        print(f"block size:", self.config.block_size)
        # This is the sequence length (max new tokens) to sample.
        if len(self.train_data.shape) == 2:
            # TODO: use max_tokens_to_sample from config
            self.data_seq_len = self.train_data.shape[-1]
        else:
            self.data_seq_len = self.config.max_tokens_to_sample

        self.train_data_orig = self.train_data
        self.val_data_orig = self.val_data
        if len(self.train_data.shape) > 2:
            self.train_data = self.train_data.reshape(len(self.train_data), -1)
            self.val_data = self.val_data.reshape(len(self.val_data), -1)
        
        if self.config.use_1d_pos_embedding:
            assert len(self.train_data.shape) in [
                1,
                2,
            ], f"Expected 1D or 2D array for training data, got {self.train_data.shape}"
        else:
            assert len(self.train_data_orig.shape) > 2, f"expected multi-dim data, got training data shape {self.train_data.shape}"
        
        meshgrid_args = [np.arange(s) for s in self.train_data_orig.shape[1:]]
        positions = np.array(
            np.meshgrid(*meshgrid_args, indexing="ij"),
        )
        orig_shape = tuple(range(len(positions.shape)))
        transposed_shape = orig_shape[1:] + (orig_shape[0],)
        self.positions = positions.transpose(*transposed_shape)
        self.position_shape = self.positions.shape[:-1]
        print(f"position_shape: {self.position_shape}")
        self.position_dim = len(self.position_shape)
        self.positions_flattened = self.positions.reshape(-1, self.position_dim)


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
            position_shape=self.position_shape,
            use_1d_pos_embedding=self.config.use_1d_pos_embedding,
            causal=self.config.causal,
            only_predict_last=self.config.only_predict_last,
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
                # print(f"estimate_loss(): k={k}, X.shape: {X.shape}, Y.shape: {Y.shape}")
                with self.ctx:
                    logits, loss = model(x=X, y=Y, x_pos=X_pos, y_pos=Y_pos)
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
                        },
                        step=iter_num,
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
                    logits, loss = model(X_pos, X, Y_pos, Y)
                    end_time = time.time()
                    # print(f"forward pass time: {end_time - start_time:.3f}s")
                    loss = (
                        loss / gradient_accumulation_steps
                    )  # scale the loss to account for gradient accumulation
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X_pos, X, Y_pos, Y = self.get_batch("train")
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
                        torch.Tensor(self.val_data[0, 0:].astype(np.int64))
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
                    t = 2
                    n_ = int(t * h * w)
                    with torch.no_grad():
                        positions_flattened = None if self.config.use_1d_pos_embedding else self.positions_flattened
                        sampled_codes = model.generate(
                            idx=torch.Tensor(
                                np.ones((1, 1), dtype=np.int32) * self.START_TOKEN
                            )
                            .long()
                            .to(device),
                            max_new_tokens=n_, # self.data_seq_len,
                            positions_flattened=positions_flattened,
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
                                },
                                step=iter_num,
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
        BlockCls = Block
        if config.use_1d_pos_embedding:
            self.transformer = nn.ModuleDict(
                dict(
                    wte=nn.Embedding(config.vocab_size, config.n_embd),
                    wpe=nn.Embedding(config.block_size, config.n_embd),
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
        idx = x
        targets = y
        # print(f"x_mean: {idx.float().mean()}")
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        if self.config.use_1d_pos_embedding:
            assert pos.max() < self.transformer.wpe.num_embeddings, f"pos.max(): {pos.max()}, block_size: {self.config.block_size}, {self.transformer.wpe}, num_embeddings: {self.transformer.wpe.num_embeddings}"
            pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
            x = tok_emb + pos_emb
        else:
            x = tok_emb
            position_dim = len(self.config.position_shape)
            for d in range(position_dim):
                offset = self.config.position_shape[d]
                # TODO: try absolute value, instead of offsetting
                # assert (
                #     relative_pos[:, :, d].max() < self.transformer.wpe[d].num_embeddings
                # ), f"relative_pos.max(): {relative_pos.max()}, block_size: {self.config.block_size}, {self.transformer.wpe[d]}, num_embeddings: {self.transformer.wpe[d].num_embeddings}"
                # relative_pos_with_offset = relative_pos[:, :, d] + offset
                assert len(x_pos.shape) == 3, f"expected 3D tensor for x_pos, got {x_pos.shape}"
                relative_pos_with_offset = x_pos[:, :, d]
                # print("d:", d, "relative_pos_with_offset:", relative_pos_with_offset.shape, "max:", relative_pos_with_offset.max(), "embedding:", self.transformer.wpe[d],
                #       "position_dim:", position_dim, "x:", x.shape)
                # import sys; sys.exit(0)
                # print(f"x: {x.shape}, relative_pos_with_offset: {relative_pos_with_offset.shape}, position_dim: {position_dim}")
                x += self.transformer.wpe[d](relative_pos_with_offset) / float(
                    position_dim
                )
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            if self.config.only_predict_last:
                logits = self.lm_head(
                    x[:, [-1], :]
                )  # note: using list [-1] to preserve the time dim
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-1,
                )
            else:
                logits = self.lm_head(x)
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
    
    @torch.no_grad()
    def generate(
        self, idx, max_new_tokens, positions_flattened=None, temperature=1.0, top_k=None, show_progress=False
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
            if positions_flattened is not None:
                pos_idx = np.arange(idx_cond.size(1))
                x_pos = torch.from_numpy(positions_flattened[pos_idx])
                x_pos = x_pos.unsqueeze(0).expand(idx_cond.size(0), -1, -1).to(idx.device)
            else:
                x_pos = None
            logits, _ = self(x=idx_cond, x_pos=x_pos, y_pos=None, y=None)
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