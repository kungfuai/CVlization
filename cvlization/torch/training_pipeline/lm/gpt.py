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
from typing import Dict, Optional

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
        use_program_augmentation: bool = False
        program_offset: int = None
        program_nil_id: int = None
        program_vocab_size: int = 0
        program_nil_loss_weight: float = 1.0
        itos: Optional[Dict[int, str]] = None
        program_pos_vocab: Optional[list] = None

        # Model
        n_layer: int = 12
        n_head: int = 12
        n_embd: int = 768
        dropout: float = 0.0
        gradient_accumulation_steps: int = 1
        bias: bool = True
        device: str = "cuda"  # e.g. 'cuda', 'cpu', or 'mps'
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
        requested_device = (config.device or "cpu").lower()
        if "cuda" in requested_device:
            self.dtype = (
                "bfloat16"
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else "float16"
            )
        elif "mps" in requested_device:
            self.dtype = "float32"
        else:
            self.dtype = "float32"
        # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

        self.ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.dtype]
        if "cuda" in requested_device:
            device_type = "cuda"
        elif "mps" in requested_device:
            device_type = "mps"
        else:
            device_type = "cpu"
        # for later use in torch.autocast
        self.device_type = device_type
        if device_type == "cuda":
            self.ctx = torch.amp.autocast(device_type=device_type, dtype=self.ptdtype)
        else:
            self.ctx = nullcontext()
        self._setup_io()
        if self.master_process:
            os.makedirs(self.out_dir, exist_ok=True)

        seed = 1337 + self.seed_offset
        print(f"setting random seed to {seed}")
        torch.manual_seed(seed)
        if device_type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
            torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

        self.START_TOKEN = self.config.start_token
        self.program_offset = None
        self.program_nil_id = None
        self.program_nil_local_id = None
        self.program_vocab_size = None
        self.itos = self.config.itos
        self.program_pos_vocab = self.config.program_pos_vocab

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
            backend = self.config.backend
            if backend == "nccl" and self.device_type != "cuda":
                backend = "gloo"
            init_process_group(backend=backend)
            ddp_rank = int(os.environ["RANK"])
            ddp_local_rank = int(os.environ["LOCAL_RANK"])
            ddp_world_size = int(os.environ["WORLD_SIZE"])
            if self.device_type == "cuda":
                device = f"cuda:{ddp_local_rank}"
                torch.cuda.set_device(device)
            else:
                device = self.config.device
            self.master_process = (
                ddp_rank == 0
            )  # this process will do logging, checkpointing etc.
            self.seed_offset = ddp_rank  # each process gets a different seed
            # world_size number of processes will be training simultaneously, so we can scale
            # down the desired gradient accumulation iterations per process proportionally
            assert self.config.gradient_accumulation_steps % ddp_world_size == 0
            self.config.gradient_accumulation_steps //= ddp_world_size
            self.config.device = device
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

    def _sample_batch(self, split: str):
        if self.config.flatten_tokens:
            data = self.train_data_flattened if split == "train" else self.val_data_flattened
        else:
            data = self.train_data if split == "train" else self.val_data

        block_size = self.config.block_size
        batch_size = self.config.batch_size
        device = self.config.device

        if len(data.shape) == 2:
            irow = torch.randint(data.shape[0], (batch_size,))
            ix = torch.randint(data.shape[1] - block_size, (batch_size,))

            def _slice_rows(arr, offset=0):
                return torch.stack(
                    [
                        torch.from_numpy(
                            arr[i, j + offset : j + offset + block_size].astype(np.int64)
                        )
                        for i, j in zip(irow, ix)
                    ]
                )

            x = _slice_rows(data)
            y = _slice_rows(data, offset=1)
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
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)

        assert x.shape == (self.config.batch_size, self.config.block_size), f"x.shape: {x.shape}"
        assert y.shape == (self.config.batch_size, self.config.block_size), f"y.shape: {y.shape}"

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

        return x, y

    def get_batch(self, split: str):
        x, y = self._sample_batch(split)
        batch = {"input_ids": x, "targets": y}
        if not self.config.use_program_augmentation:
            return batch

        text_targets = y.clone()
        program_targets = torch.full_like(y, self.program_nil_local_id)
        program_mask = y >= self.program_offset
        if program_mask.any():
            program_targets[program_mask] = y[program_mask] - self.program_offset
            text_targets[program_mask] = -1

        batch.update(
            {
                "targets_text": text_targets,
                "targets_program": program_targets,
            }
        )
        return batch

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

        if self.config.use_program_augmentation:
            assert (
                self.config.program_offset is not None
            ), "program_offset must be set when program augmentation is enabled"
            assert (
                self.config.program_nil_id is not None
            ), "program_nil_id must be set when program augmentation is enabled"
            assert (
                self.config.program_vocab_size > 0
            ), "program_vocab_size must be > 0 when program augmentation is enabled"
            self.program_offset = self.config.program_offset
            self.program_nil_id = self.config.program_nil_id
            self.program_vocab_size = self.config.program_vocab_size
            self.program_nil_local_id = self.program_nil_id - self.program_offset

    def _decode_tokens(self, tokens: torch.Tensor) -> str:
        if self.itos is None:
            return "".join(str(int(tok)) for tok in tokens.tolist())
        pieces = []
        for tok in tokens.tolist():
            tok = int(tok)
            if (
                self.program_offset is not None
                and tok >= self.program_offset
                and self.program_pos_vocab is not None
            ):
                idx = tok - self.program_offset
                if 0 <= idx < len(self.program_pos_vocab):
                    pieces.append(f"<{self.program_pos_vocab[idx]}>")
                    continue
            pieces.append(self.itos.get(tok, "?"))
        return "".join(pieces)

    @torch.no_grad()
    def _sampled_text_ce(self, batch: Dict[str, torch.Tensor]) -> Optional[float]:
        model = self.model
        if not self.config.use_program_augmentation or not isinstance(
            model, ProgramAugmentedGPT
        ):
            return None

        input_ids = batch["input_ids"]
        targets_text = batch["targets_text"]
        targets_program = batch["targets_program"]

        # pass 1: get program predictions under teacher-forced context
        features = model.backbone.forward_features(input_ids)
        prog_logits = model.program_head(features)
        sampled_local = torch.argmax(prog_logits, dim=-1)
        sampled_tokens = sampled_local + model.program_offset

        # replace program tokens with sampled ones
        generated = input_ids.clone()
        program_mask = targets_program != model.nil_local_id
        generated[program_mask] = sampled_tokens[program_mask]

        # pass 2: compute text CE under sampled program context
        features_new = model.backbone.forward_features(generated)
        logits_text = model.backbone.lm_head(features_new)
        text_mask = targets_text != -1
        if not text_mask.any():
            return None
        logits_masked = logits_text[text_mask]
        targets_masked = targets_text[text_mask].long()
        loss = F.cross_entropy(logits_masked, targets_masked, reduction="mean")
        return loss.item()

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

        if meta_vocab_size is None:
            meta_vocab_size = self.config.vocab_size
        vocab_for_model = meta_vocab_size
        if self.config.use_program_augmentation:
            vocab_for_model = max(vocab_for_model, self.config.program_nil_id + 1)

        # model init
        self.model_args = dict(
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            block_size=block_size,
            bias=bias,
            vocab_size=vocab_for_model,
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
            self.model_args["vocab_size"] = vocab_for_model
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
        if self.config.use_program_augmentation:
            model = ProgramAugmentedGPT(
                backbone=model,
                program_vocab_size=self.config.program_vocab_size,
                program_offset=self.program_offset,
                nil_local_id=self.program_nil_local_id,
                nil_loss_weight=self.config.program_nil_loss_weight,
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
        if self.device_type == "cuda":
            enabled = dtype == "float16"
            self.scaler = torch.cuda.amp.GradScaler(enabled=enabled)
        else:
            class _IdentityScaler:
                def scale(self, loss):
                    return loss

                def unscale_(self, optimizer):
                    return None

                def step(self, optimizer):
                    optimizer.step()

                def update(self):
                    return None

            self.scaler = _IdentityScaler()
            enabled = False
        print(f"GradScaler enabled: {enabled}")

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
        eval_iters_val = min(self.config.eval_iters, 100) if self.master_process else self.config.eval_iters
        out = {}
        model.eval()
        split = "val"
        losses = torch.zeros(eval_iters_val)
        text_losses = [] if self.config.use_program_augmentation else None
        sampled_text_losses = [] if self.config.use_program_augmentation else None
        if self.master_process:
            print(f"[eval] split={split} running {eval_iters_val} iters...")
        for k in range(eval_iters_val):
            batch = self.get_batch(split)
            with self.ctx:
                if self.config.use_program_augmentation:
                    _, loss, metrics = model(
                        batch["input_ids"],
                        targets_text=batch["targets_text"],
                        targets_program=batch["targets_program"],
                        return_logits=False,
                    )
                    if "loss_text" in metrics:
                        text_losses.append(metrics["loss_text"].item())
                    sampled_ce = self._sampled_text_ce(batch)
                    if sampled_ce is not None:
                        sampled_text_losses.append(sampled_ce)
                else:
                    _, loss = model(
                        batch["input_ids"], batch["targets"]
                    )
            losses[k] = loss.item()
        out[split] = losses.mean().item()
        if text_losses:
            out[f"{split}_text_ce"] = float(sum(text_losses) / len(text_losses))
        if sampled_text_losses:
            out[f"{split}_sampled_text_ce"] = float(
                sum(sampled_text_losses) / len(sampled_text_losses)
            )
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
        sample_interval = (
            self.config.sample_interval
            if self.config.sample_interval is not None
            else self.config.eval_interval
        )
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

        batch = self.get_batch("train")  # fetch the very first batch
        print(f"input_ids shape: {batch['input_ids'].shape}")
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
                message_parts = [f"val loss {losses['val']:.4f}"]
                if "train" in losses:
                    message_parts.insert(0, f"train loss {losses['train']:.4f}")
                if "val_text_ce" in losses:
                    message_parts.append(
                        f"val text CE {losses['val_text_ce']:.4f}"
                    )
                if "val_sampled_text_ce" in losses:
                    message_parts.append(
                        f"val sampled text CE {losses['val_sampled_text_ce']:.4f}"
                    )
                print(f"step {iter_num}: " + ", ".join(message_parts))
                if wandb_log:
                    log_payload = {
                        "iter": iter_num,
                        "val/loss": losses["val"],
                        "lr": lr,
                        "mfu": running_mfu * 100,
                    }
                    if "train" in losses:
                        log_payload["train/loss"] = losses["train"]
                    if "val_text_ce" in losses:
                        log_payload["val/text_ce"] = losses["val_text_ce"]
                    if "train_text_ce" in losses:
                        log_payload["train/text_ce"] = losses["train_text_ce"]
                    if "val_sampled_text_ce" in losses:
                        log_payload["val/text_ce_sampled"] = losses["val_sampled_text_ce"]
                    if "train_sampled_text_ce" in losses:
                        log_payload["train/text_ce_sampled"] = losses[
                            "train_sampled_text_ce"
                        ]
                    wandb.log(log_payload)
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
            last_metrics = {}
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
                    if self.config.use_program_augmentation:
                        _, loss, metrics = model(
                            batch["input_ids"],
                            targets_text=batch["targets_text"],
                            targets_program=batch["targets_program"],
                            return_logits=False,
                        )
                    else:
                        _, loss = model(batch["input_ids"], batch["targets"])
                        metrics = {}
                    end_time = time.time()
                    loss = loss / gradient_accumulation_steps
                next_batch = self.get_batch("train")
                scaler.scale(loss).backward()
                batch = next_batch
                last_metrics = metrics
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
                log_line = f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
                if self.config.use_program_augmentation:
                    if "loss_text" in last_metrics:
                        log_line += f", text CE {last_metrics['loss_text'].item():.4f}"
                    if "loss_prog" in last_metrics:
                        log_line += f", prog CE {last_metrics['loss_prog'].item():.4f}"
                    if "loss_nil" in last_metrics:
                        log_line += f", nil CE {last_metrics['loss_nil'].item():.4f}"
                print(log_line)
                if wandb_log and self.config.use_program_augmentation:
                    wandb.log(
                        {
                            "train/text_ce_step": last_metrics["loss_text"].item()
                            if "loss_text" in last_metrics
                            else float("nan"),
                            "train/prog_ce_step": last_metrics["loss_prog"].item()
                            if "loss_prog" in last_metrics
                            else float("nan"),
                            "train/nil_ce_step": last_metrics["loss_nil"].item()
                            if "loss_nil" in last_metrics
                            else float("nan"),
                        }
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
                elif self.itos is not None:
                    model.eval()
                    try:
                        with torch.no_grad():
                            context = batch["input_ids"][:1, :1]
                            sample = model.generate(
                                context,
                                max_new_tokens=self.config.block_size,
                                temperature=0.8,
                                top_k=40,
                                show_progress=False,
                            )
                            decoded = self._decode_tokens(sample[0])
                            preview_len = min(len(decoded), 200)
                            preview = decoded[:preview_len]
                            if preview_len < len(decoded):
                                preview += "..."
                            print("---- sampled text ----")
                            print(preview)
                            if wandb_log:
                                table = wandb.Table(columns=["preview"])
                                table.add_data(preview)
                                wandb.log({"sampled/text": table})
                            if self.master_process:
                                print(
                                    f"[sample] iter={iter_num+1} generated preview (len={len(preview)})"
                                )
                    finally:
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

    def forward_features(self, idx: torch.Tensor) -> torch.Tensor:
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return x

    def forward(self, idx, targets=None):
        x = self.forward_features(idx)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            return logits, loss

        logits = self.lm_head(x[:, [-1], :])
        loss = None
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
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
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        return flops_achieved / flops_promised

    @torch.no_grad()
    def generate(
        self, idx, max_new_tokens, temperature=1.0, top_k=None, show_progress=False
    ):
        if show_progress:
            from tqdm import tqdm

            range_iter = tqdm(range(max_new_tokens), mininterval=2)
        else:
            range_iter = range(max_new_tokens)
        for _ in range_iter:
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


class ProgramAugmentedGPT(nn.Module):
    def __init__(
        self,
        backbone: GPT,
        program_vocab_size: int,
        program_offset: int,
        nil_local_id: int,
        nil_loss_weight: float,
    ):
        super().__init__()
        self.backbone = backbone
        self.program_head = nn.Linear(
            backbone.config.n_embd, program_vocab_size + 1, bias=False
        )
        torch.nn.init.normal_(self.program_head.weight, mean=0.0, std=0.02)
        self.program_vocab_size = program_vocab_size
        self.program_offset = program_offset
        self.nil_local_id = nil_local_id
        self.nil_loss_weight = nil_loss_weight
        self.config = backbone.config
        self.sample_program_inference = True
        self.sample_program_argmax = False

    def forward(
        self,
        input_ids,
        targets=None,
        targets_text=None,
        targets_program=None,
        return_logits: bool = False,
    ):
        if targets_text is None or targets_program is None:
            logits, loss = self.backbone(input_ids, targets=targets)
            if return_logits:
                return logits, loss, {}
            return logits, loss

        features = self.backbone.forward_features(input_ids)
        logits_text = self.backbone.lm_head(features)
        logits_prog = self.program_head(features)

        text_mask = targets_text.ne(-1)
        if text_mask.any():
            loss_text = F.cross_entropy(
                logits_text[text_mask], targets_text[text_mask]
            )
            nil_targets = torch.full_like(
                targets_text[text_mask], self.nil_local_id
            )
            loss_nil = F.cross_entropy(logits_prog[text_mask], nil_targets)
        else:
            loss_text = logits_text.new_zeros(())
            loss_nil = logits_prog.new_zeros(())

        program_mask = targets_program.ne(self.nil_local_id)
        if program_mask.any():
            loss_prog = F.cross_entropy(
                logits_prog[program_mask], targets_program[program_mask]
            )
        else:
            loss_prog = logits_prog.new_zeros(())

        total_loss = loss_prog + loss_text + self.nil_loss_weight * loss_nil

        metrics: Dict[str, torch.Tensor] = {
            "loss_prog": loss_prog.detach(),
            "loss_text": loss_text.detach(),
            "loss_nil": loss_nil.detach(),
        }

        if return_logits:
            return {"text": logits_text, "program": logits_prog}, total_loss, metrics
        return None, total_loss, metrics

    def generate(
        self,
        idx,
        max_new_tokens,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        show_progress: bool = False,
    ):
        if not self.sample_program_inference:
            return self.backbone.generate(
                idx,
                max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                show_progress=show_progress,
            )

        if show_progress:
            from tqdm import tqdm

            range_iter = tqdm(range(max_new_tokens), mininterval=2)
        else:
            range_iter = range(max_new_tokens)

        def _top_k_filter(logits: torch.Tensor) -> torch.Tensor:
            if top_k is None:
                return logits
            k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, k)
            threshold = values[:, [-1]]
            logits = logits.clone()
            logits[logits < threshold] = -float("Inf")
            return logits

        for _ in range_iter:
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )

            features = self.backbone.forward_features(idx_cond)
            last_hidden = features[:, [-1], :]

            prog_logits = self.program_head(last_hidden)[:, 0, :] / temperature
            prog_logits = _top_k_filter(prog_logits)
            if self.sample_program_argmax:
                prog_samples = torch.argmax(prog_logits, dim=-1, keepdim=True)
            else:
                prog_probs = F.softmax(prog_logits, dim=-1)
                prog_samples = torch.multinomial(prog_probs, num_samples=1)
            nil_mask = prog_samples.squeeze(-1).eq(self.nil_local_id)

            text_logits = self.backbone.lm_head(last_hidden)[:, 0, :] / temperature
            text_logits = _top_k_filter(text_logits)
            text_probs = F.softmax(text_logits, dim=-1)
            text_samples = torch.multinomial(text_probs, num_samples=1)

            prog_tokens = prog_samples + self.program_offset
            idx_next = torch.where(
                nil_mask.unsqueeze(-1), text_samples, prog_tokens
            )
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def set_program_sampling(self, enabled: bool, deterministic: bool = False):
        self.sample_program_inference = bool(enabled)
        self.sample_program_argmax = bool(deterministic)

    def forward_features(self, idx: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(idx)

    def configure_optimizers(self, *args, **kwargs):
        return self.backbone.configure_optimizers(*args, **kwargs)

    def estimate_mfu(self, *args, **kwargs):
        return self.backbone.estimate_mfu(*args, **kwargs)

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
