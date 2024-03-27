"""
Multi-dimensional GPT.
"""
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Tuple
import os
import time
import math
import pickle
from einops import rearrange
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
        min_lr: float = 1e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
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
        self.out_dir = f"{config.log_dir}/batch{config.batch_size}_block{config.block_size}"
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
        device_type = "cuda" if "cuda" in config.device else "cpu"  # for later use in torch.autocast
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
            self.master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
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
        tokens_per_iter = self.config.gradient_accumulation_steps * ddp_world_size * batch_size * block_size
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
        batch_size = self.config.batch_size
        device = self.config.device
        
        data = train_data if split == "train" else self.val_data
        # TODO: position_dim == 1 vs. > 1 should be handled in different functions
        # Coordinates of tokens. Shape is the same as data.
        if position_dim == 1:
            positions = np.arange(len(data))
        else:
            if len(data.shape) == 1:
                meshgrid_args = [np.arange(data.shape[0])]
            else:
                meshgrid_args = [np.arange(s) for s in data.shape[1:]]
            positions = np.array(
                np.meshgrid(
                    *meshgrid_args,
                    indexing="ij"
                ),
            )
            if len(positions.shape) == 4:
                positions.transpose(1, 2, 3, 0)  # positions[t, i, j] == [t, i, j]
            elif len(positions.shape) == 3:
                positions.transpose(1, 2, 0)
            elif len(positions.shape) == 2:
                positions.transpose(1, 0)
            elif len(positions.shape) == 1:
                positions = positions[np.newaxis]
            else:
                raise ValueError(f"Dimension of positions is {len(positions.shape)}, not supported.")

            # flatten:
            data = data.reshape(data.shape[0], -1)  # this is the whole video
            positions = positions.reshape(-1, positions.shape[-1])
            # print(f"data shape: {data.shape}")
            # print(f"positions shape: {positions.shape}")

        # Problem formulation: x_pos, x, y_pos, y?
        if position_dim == 1:
            ix = torch.randint(len(data) - block_size, (batch_size,))
            x = torch.stack(
                [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
            )
        else:
            # pad the data with ignore tokens if needed
            end_irow = torch.randint(data.shape[0], (batch_size,))
            end_ix = torch.randint(data.shape[1], (batch_size,))
            # use a look back window of block_size, but not exceeding the beginning of the data
            start_ix = end_ix - block_size
            x = np.ones((len(start_ix), block_size), dtype=np.int64) * self.IGNORE_TOKEN
            for i, (s, e) in enumerate(zip(start_ix, end_ix)):
                src_start = max(0, s)
                src_end = e
                dst_start = max(0, -s)
                dst_end = block_size
                x[i, dst_start:dst_end] = data[end_irow[i], src_start:src_end]
            # If x is [28, 3, 556, 132, ...]
            # x can look like this:
            # [[IGNORE_TOKEN, 1, 2, 3, 4, 5],
            #  [IGNORE_TOKEN, 6, 7, 8, 9, 10],
            #  [IGNORE_TOKEN, IGNORE_TOKEN, 11, 12, 13, 14]]
            # ]
            x = torch.from_numpy(x)

        if position_dim == 1:
            x_pos = torch.stack(
                [torch.from_numpy((positions[i : i + block_size]).astype(np.int64)) for i in ix]
            )
            x_pos = x_pos.unsqueeze(-1)
        else:
            x_pos = np.ones((len(start_ix), block_size, position_dim), dtype=np.int64) * 0
            for i, (s, e) in enumerate(zip(start_ix, end_ix)):
                src_start = max(0, s)
                src_end = e
                dst_start = max(0, -s)
                dst_end = block_size
                x_pos[i, dst_start:dst_end, :] = positions[src_start:src_end, :]
            x_pos = torch.from_numpy(x_pos)

        if position_dim == 1:
            y = torch.stack(
                [
                    torch.from_numpy((
                        data[i + block_size, ...]
                    ).astype(np.int64))
                    for i in ix
                ]
            )
            y = y.unsqueeze(1)
        else:
            y = torch.stack(
                [
                    torch.from_numpy((
                        data[[row], [e]]
                    ).astype(np.int64))
                    for row, e in zip(end_irow, end_ix)
                ]
            )
        if position_dim == 1:
            y_pos = torch.stack(
                [
                    torch.from_numpy((
                        positions[i + block_size : i + 1 + block_size, ...]
                    ).astype(np.int64))
                    for i in ix
                ]
            )
        else:
            y_pos = torch.stack(
                [
                    torch.from_numpy((
                        positions[[e]]
                    ).astype(np.int64))
                    for e in end_ix
                ]
            ).squeeze(1)
        
        # assert the shapes
        assert x.shape == (batch_size, block_size), x.shape
        assert y.shape == (batch_size, 1), y.shape
        assert x_pos.shape == (batch_size, block_size, position_dim), x_pos.shape
        assert y_pos.shape == (batch_size, position_dim), y_pos.shape
        
        if self.device_type == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
                device, non_blocking=True
            )
            x_pos, y_pos = x_pos.pin_memory().to(device, non_blocking=True), y_pos.pin_memory().to(
                device, non_blocking=True
            )
        else:
            x, y = x.to(device), y.to(device)
            x_pos, y_pos = x_pos.to(device), y_pos.to(device)
        return x_pos, x, y_pos, y

    def create_dataloaders(self, dataset_builder):
        self.train_data = dataset_builder.training_dataset()  # (B, T, H, W)
        self.val_data = dataset_builder.validation_dataset()
        assert isinstance(self.train_data, np.ndarray)
        assert isinstance(self.val_data, np.ndarray)
        assert self.train_data.dtype in [np.int32, np.int64, np.uint16, np.uint32, np.uint64]
        # assert len(self.train_data.shape) == 2, f"Expected 2D array for training data, got {self.train_data.shape}"
        # self.train_data_flattened = self.train_data.ravel()
        # self.val_data_flattened = self.val_data.ravel()
        print(f"block size:", self.config.block_size)
        if len(self.train_data.shape) == 1:
            self.position_shape = (self.train_data.shape[0],)
        else:
            self.position_shape = self.train_data.shape[1:]
        self.position_dim = len(self.position_shape)

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
            meta_vocab_size = self.config.vocab_size + 20  # 20 extra tokens for special tokens
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
            self.model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
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
            for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
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
            for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
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

    def create_optimizer(self):
        model = self.model
        weight_decay = self.config.weight_decay
        learning_rate = self.config.learning_rate
        beta1 = self.config.beta1
        beta2 = self.config.beta2
        device_type = self.device_type
        init_from = self.config.init_from
        # optimizer
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
                    mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
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
                    ground_truth_codes = (
                        torch.Tensor(self.val_data[0:1, :].astype(np.int64)).long().to(device)
                    )  # this is hard coded
                    if self.position_dim == 3:
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
                        if wandb_log:
                            wandb.log({"sampled/ground_truth_decoded": display})

            if iter_num % sample_interval == 0 and master_process:
                if self.config.vae_model_name is not None:
                    # sample from the model
                    model.eval()
                    with torch.no_grad():
                        meshgrid_args = [np.arange(s) for s in self.position_shape]
                        cond_y_pos = np.array(
                            np.meshgrid(
                                *meshgrid_args,
                                indexing="ij"
                            ),
                        ).transpose(1, 2, 3, 0)  # positions[t, i, j] == [t, i, j]
                        cond_y_pos = cond_y_pos.reshape(1, -1, self.position_dim)
                        cond_y_pos = torch.from_numpy(cond_y_pos).long().to(device)
                        sampled_codes = model.generate(
                            x_pos=torch.zeros(1, 1, len(self.position_shape)).long().to(device),
                            x=torch.Tensor(np.ones((1, 1), dtype=np.int32) * self.START_TOKEN)
                            .long()
                            .to(device),
                            new_pos=cond_y_pos,
                            temperature=1,
                            top_k=20,
                            show_progress=True,
                        )
                        sampled_codes = sampled_codes[0, 1:]
                        violating_codes = (sampled_codes > 5119).float().mean()
                        print(f"violating codes: {violating_codes.item()}")
                        sampled_codes[sampled_codes > 5119] = 0
                        # force_cudnn_initialization()
                        # sampled_codes = torch.ones(1, 32768, dtype=torch.long).to(device)
                        print("sampled codes:", sampled_codes)
                        # print(sampled_codes.min(), sampled_codes.max())
                        sampled_codes = rearrange(
                            sampled_codes[:32768],
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


class MDGPT(GPT):

    def __init__(self, config):
        super(GPT, self).__init__()  # Use the grandparent class.
        assert config.vocab_size is not None
        assert config.block_size is not None
        position_dim = len(config.position_shape)
        if position_dim == 1:
            position_embedding_input_sizes = [1 + config.block_size]
        else:
            position_embedding_input_sizes = [2 * s + 1 for s in config.position_shape]
        print("position_embedding_input_sizes:", position_embedding_input_sizes)
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.ModuleList(
                    [nn.Embedding(s, config.n_embd) for s in position_embedding_input_sizes]
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
        
        #
        ## Pruning begin
        #
        idx_to_ignore = (x == self.config.ignore_token)
        #  to the loss function to ignore these tokens
        relative_pos = y_pos.unsqueeze(1) - x_pos
        # print(f"relative_pos shape: {relative_pos.shape}")
        # TODO: consider weighted Euclidean distance
        euclidean_dist = torch.norm(relative_pos.float(), dim=-1, keepdim=False)
        euclidean_dist_exp = torch.exp(-euclidean_dist)
        # add random perturbation to break ties and to allow for saturation
        euclidean_dist_exp[euclidean_dist_exp < 1e-6] = 0
        euclidean_dist_exp += torch.rand_like(euclidean_dist_exp) * 1e-6
        # For tokens to be ignored, set their similarity to -1
        euclidean_dist_exp[idx_to_ignore] = -1
        sorted_idx = torch.argsort(euclidean_dist_exp, dim=1, descending=True)
        # re-order
        x_reordered = torch.gather(x, 1, sorted_idx)
        x_pos_reordered = torch.gather(x_pos, 1, sorted_idx.unsqueeze(-1).expand(-1, -1, x_pos.shape[-1]))
        relative_pos_reordered = torch.gather(relative_pos, 1, sorted_idx.unsqueeze(-1).expand(-1, -1, relative_pos.shape[-1]))
        # insert a start token at the beginning of each sequence
        x_reordered = torch.cat([torch.ones_like(x_reordered[:, :1]) * self.config.start_token, x_reordered], dim=1)
        x_pos_reordered = torch.cat([torch.zeros_like(x_pos_reordered[:, :1]), x_pos_reordered], dim=1)
        relative_pos_reordered = torch.cat([torch.zeros_like(relative_pos_reordered[:, :1]), relative_pos_reordered], dim=1)
        # prune
        # [3, 4, 10, ignore, ignore, ....]
        # [logits0 (nearest neighbor), logits1 (1st 2 closest neighbors), logits2, logits3_should_ignore, ...]
        # All logits are predicting the same future token.
        x_pruned = x_reordered[:, : self.config.sparse_block_size]
        relative_pos_pruned = relative_pos_reordered[:, : self.config.sparse_block_size]
        idx_to_ignore = idx_to_ignore[:, : self.config.sparse_block_size]
        x_original = x
        x_pos_original = x_pos
        position_dim = len(self.config.position_shape)
        # print(f"position_shape: {self.config.position_shape}")
        #
        ## Pruning end
        #

        x = self.transformer.wte(x_pruned)
        for d in range(position_dim):
            offset = self.config.position_shape[d]
            # print(f"d={d}, offset={offset}, relative_pos_pruned shape: {relative_pos_pruned.shape}")
            # print("embedding model:", self.transformer.wpe[d])
            # print(f"input to embedding:", relative_pos_pruned[:, :, d] + offset)
            # TODO: try absolute value, instead of offsetting
            x += self.transformer.wpe[d](
                relative_pos_pruned[:, :, d] + offset
            )
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        # print(f"after linear, x shape: {x.shape}")
        if y is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            # y needs to be broadcasted
            y = y.expand(-1, logits.shape[1])
            y = y.unsqueeze(-1)
            logits_selected = logits[~idx_to_ignore]
            y_selected = y[~idx_to_ignore]
            # loss_weight = 1 - idx_to_ignore.float()
            try:
                loss = F.cross_entropy(
                    logits_selected.view(-1, logits_selected.size(-1)),
                    y_selected.reshape(-1),
                    # weight=loss_weight.view(-1),
                    ignore_index=-1
                )
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
    def generate(self, x_pos, x, new_pos, temperature=1.0, top_k=None, show_progress=False):
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
        assert len(new_pos.shape) == 3, f"expected 3D tensor for new_pos, got {new_pos.shape}"
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

        return x_updated