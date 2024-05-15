"""
Adapted from Andrej's NanoGPT.

Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import time
import pickle
import numpy as np
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


class NanoGPTTrainingPipeline:
    # TODO: compile the model.
    @dataclass
    class Config:

        log_dir: str = "logs/nanogpt"
        wandb_log: bool = False
        project: str = "nano-gpt"
        init_from: str = "scratch"  # 'scratch' or 'resume' or 'gpt2*'

        # Data
        block_size: int = 1024
        vocab_size: int = 5120
        batch_size: int = 32
        flatten_tokens: bool = False

        # Model
        n_layer: int = 12
        n_head: int = 12
        n_embd: int = 768
        dropout: float = 0.0
        gradient_accumulation_steps: int = 1
        bias: bool = True
        device: str = "cuda"
        use_mamba_mixer: bool = False

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

        # make the context window sparse in the beginning of the sequence
        if self.config.sparse_context_window:
            sparse_idx = np.concatenate(
                [
                    np.arange(
                        0,
                        block_size - self.config.context_stride_start,
                        self.config.context_stride,
                    ),
                    np.arange(
                        block_size - self.config.context_stride_start, block_size
                    ),
                ]
            )
            x = x[:, sparse_idx]
            y = y[:, sparse_idx]

        # For debugging.
        # Treatment 1
        # This is so that the model can learn to predict the final token, using varying context window lengths:
        #  with only the nearest neighbor, and then the 2 nearest neighbors, then 3, ...
        # x = torch.flip(
        #     x, [1]
        # )  # This combined with the following line (treatment 2) will make the loss nan.
        # Treatment 2
        # y[:, :-1] = -1  # This alone will make the training inefficient but still works.
        # Treatment 3
        # x[:, -1] = x[:, 0]  # This combined with treatment 1 and 2 fixes the nan issue.
        return x, y

    def get_batch_sparse(self, split: str):
        """
        Get a batch of data for training or validation.

        Args:
            split (str): The split to get the data from. Can be either "train" or "val".

        Returns:
            tuple: A tuple containing two tensors: x and y.
                - x: The input tensor of shape (batch_size, block_size).
                - y: The target tensor of shape (batch_size, block_size).

        """
        train_data = self.train_data_flattened
        val_data = self.val_data_flattened
        block_size = self.config.block_size
        batch_size = self.config.batch_size
        device = self.config.device

        data = train_data if split == "train" else val_data
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

        # make the context window sparse in the beginning of the sequence
        sparse_idx = np.concatenate(
            [np.arange(0, block_size - 32, 1), np.arange(block_size - 32, block_size)]
        )
        x = x[:, sparse_idx]
        y = y[:, sparse_idx]
        # assert x.shape == (self.config.batch_size, self.config.block_size)
        # assert y.shape == (self.config.batch_size, self.config.block_size)
        return x, y

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
        assert len(self.train_data.shape) in [
            1,
            2,
        ], f"Expected 1D or 2D array for training data, got {self.train_data.shape}"

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
            if self.config.use_mamba_mixer:
                model = GPTMamba(gptconf)
            else:
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
                X, Y = self.get_batch(split)
                # print(f"estimate_loss(): k={k}, X.shape: {X.shape}, Y.shape: {Y.shape}")
                with self.ctx:
                    logits, loss = model(X, Y)
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


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

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
                is_causal=True,
            )
        else:
            # manual implementation of attention
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


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = (
        50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    )
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = (
        True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    )

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
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
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # print(f"x_mean: {idx.float().mean()}")
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

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == "dropout" for k in override_args)
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        config_args["bias"] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if "dropout" in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

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
            logits, _ = self(idx_cond)
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


class MambaBackbone(nn.Module):
    def __init__(self, config: GPTConfig = None):
        super().__init__()
        from mamba_ssm.models.mixer_seq_simple import create_block

        norm_epsilon: float = 1e-5
        n_layer = config.n_layer
        rms_norm: bool = False
        initializer_cfg = None
        fused_add_norm = True
        residual_in_fp32 = True
        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model=config.n_embd,
                    ssm_cfg={}, #ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                )
                for i in range(n_layer)
            ]
        )
        self.norm_f = nn.LayerNorm(
            config.n_embd, eps=norm_epsilon,
        )
        self.fused_add_norm = False

    def forward(self, x):
        hidden_states = x
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        # else:
        #     # Set prenorm=False here since we don't need the residual
        #     fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
        #     hidden_states = fused_add_norm_fn(
        #         hidden_states,
        #         self.norm_f.weight,
        #         self.norm_f.bias,
        #         eps=self.norm_f.eps,
        #         residual=residual,
        #         prenorm=False,
        #         residual_in_fp32=self.residual_in_fp32,
        #     )
        return hidden_states

class GPTMamba(GPT):
    """
    GPT model with Mamba support.
    
    transformer.h:
        Use MambaLayer instead of Block.
    """
    def __init__(self, config):
        super().__init__(config)
        
        self.transformer.h = nn.ModuleList([MambaBackbone(config)])