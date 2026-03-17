#!/usr/bin/env python3
"""
DeepSeek-OCR-2 (3B) fine-tuning for Optical Music Recognition (OMR).

Uses a custom DeepSeekOCR2DataCollator and base Trainer (not SFTTrainer),
following the unsloth DeepSeek-OCR-2 fine-tuning notebook pattern.

Usage:
  python train_deepseek_ocr2.py                         # uses config_deepseek_ocr2.yaml
  python train_deepseek_ocr2.py --max-samples 100       # smoke test
  python train_deepseek_ocr2.py --epochs 2              # full training run
"""

import argparse
import glob
import io
import math
import os
import re
import sys
import yaml
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageOps
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import (
    Trainer, TrainingArguments, TrainerCallback, AutoModel, TextStreamer
)
from unsloth import FastVisionModel, is_bf16_supported
from huggingface_hub import snapshot_download

INSTRUCTION = "Transcribe this sheet music page to MusicXML."
DEFAULT_CONFIG = "config_deepseek_ocr2.yaml"


# ── MusicXML cleanup ───────────────────────────────────────────────────────────

def strip_musicxml_header(xml: str) -> str:
    xml = re.sub(r'<\?xml[^?]*\?>\s*', '', xml)
    xml = re.sub(r'<!DOCTYPE[^>]*>\s*', '', xml)
    xml = re.sub(r'\s*<identification>.*?</identification>', '', xml, flags=re.DOTALL)
    xml = re.sub(r'\s*<defaults>.*?</defaults>', '', xml, flags=re.DOTALL)
    xml = re.sub(r'\s*<movement-title>tmp[^<]*</movement-title>', '', xml)
    return xml.strip()


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(model_config):
    """Download, patch, and load DeepSeek-OCR-2 with unsloth."""
    model_name = model_config["name"]
    print(f"Pre-downloading {model_name} ...")
    local_path = snapshot_download(model_name)
    print(f"  Downloaded to: {local_path}")

    # Patch snapshot files (from_pretrained copies these to modules cache)
    for py_file in glob.glob(f"{local_path}/*.py"):
        content = open(py_file).read()
        patched = content
        # Fix 1: transformers 5.x renamed DeepseekV2MoE → DeepseekV2Moe
        if "DeepseekV2MoE" in patched:
            patched = patched.replace("DeepseekV2MoE", "DeepseekV2Moe")
        # Fix 2: keep logits in bfloat16 to avoid 50+ GB float32 allocation
        if "logits = logits.float()" in patched:
            patched = patched.replace(
                "logits = logits.float()",
                "logits = logits  # keep bfloat16; .float() OOMs on large vocab"
            )
        if patched != content:
            open(py_file, "w").write(patched)
            print(f"  Patched snapshot {os.path.basename(py_file)}")

    # Make model utilities importable: use the transformers_modules cache
    # (the snapshot uses relative imports, so sys.path insert won't work directly;
    # transformers caches modules as a package after trust_remote_code loading)
    hf_modules = os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface", "modules", "transformers_modules"
    )
    snapshot_hash = os.path.basename(local_path)
    modules_path  = os.path.join(hf_modules, snapshot_hash)
    if os.path.isdir(modules_path) and modules_path not in sys.path:
        sys.path.insert(0, hf_modules)  # enables: import <hash>.modeling_deepseekocr2

    # Also patch files in modules cache
    for py_file in glob.glob(f"{modules_path}/*.py"):
        content = open(py_file).read()
        patched = content
        if "DeepseekV2MoE" in patched:
            patched = patched.replace("DeepseekV2MoE", "DeepseekV2Moe")
        # Keep logits in bfloat16 to avoid 50+ GB float32 allocation
        if "logits = logits.float()" in patched:
            patched = patched.replace(
                "logits = logits.float()",
                "logits = logits  # keep bfloat16; .float() causes OOM on large vocab"
            )
        if patched != content:
            open(py_file, "w").write(patched)
            print(f"  Patched {os.path.basename(py_file)}")
            # Clear pycache so transformers re-copies & recompiles from snapshot
            pycache_dir = os.path.join(os.path.dirname(py_file), "__pycache__")
            stem = os.path.splitext(os.path.basename(py_file))[0]
            for pyc in glob.glob(os.path.join(pycache_dir, f"{stem}*.pyc")):
                os.remove(pyc)

    if modules_path not in sys.path:
        sys.path.insert(0, modules_path)

    model, tokenizer = FastVisionModel.from_pretrained(
        local_path,
        load_in_4bit=model_config.get("load_in_4bit", True),
        auto_model=AutoModel,
        trust_remote_code=True,
        unsloth_force_compile=True,
        use_gradient_checkpointing=model_config.get("use_gradient_checkpointing", "unsloth"),
    )
    return model, tokenizer, local_path


# ── Data collator (adapted from unsloth DeepSeek-OCR-2 notebook) ──────────────

@dataclass
class DeepSeekOCR2DataCollator:
    """Custom collator for DeepSeek-OCR-2 vision-language training.

    Handles image cropping/tiling (DeepEncoder V2 dynamic resolution),
    token interleaving, and label masking (train on responses only).
    """
    tokenizer: Any
    model: Any
    image_size: int = 768
    base_size: int = 1024
    crop_mode: bool = True
    train_on_responses_only: bool = True
    max_length: int = None

    def __post_init__(self):
        self.image_token_id = 128815
        self.dtype = self.model.dtype
        self.patch_size = 16
        self.downsample_ratio = 4
        self.bos_id = getattr(self.tokenizer, "bos_token_id", 0) or 0

        # After FastVisionModel.from_pretrained(trust_remote_code=True), the model's
        # custom module is already loaded into sys.modules — find it there.
        mod = None
        for key, m in sys.modules.items():
            if "modeling_deepseekocr2" in key:
                mod = m
                break
        if mod is None:
            raise ImportError(
                "modeling_deepseekocr2 not found in sys.modules. "
                "Call load_model() before instantiating DeepSeekOCR2DataCollator."
            )
        self._BasicImageTransform = mod.BasicImageTransform
        self._dynamic_preprocess  = mod.dynamic_preprocess
        self._text_encode         = mod.text_encode

        self.image_transform = self._BasicImageTransform(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            normalize=True,
        )

    def deserialize_image(self, image_data) -> Image.Image:
        if isinstance(image_data, Image.Image):
            return image_data.convert("RGB")
        elif isinstance(image_data, dict) and "bytes" in image_data:
            return Image.open(io.BytesIO(image_data["bytes"])).convert("RGB")
        else:
            raise ValueError(f"Unsupported image format: {type(image_data)}")

    def process_image(self, image: Image.Image):
        images_list, images_crop_list, images_spatial_crop = [], [], []

        if self.crop_mode:
            if image.size[0] <= 768 and image.size[1] <= 768:
                crop_ratio = (1, 1)
                images_crop_raw = []
            else:
                images_crop_raw, crop_ratio = self._dynamic_preprocess(
                    image, min_num=2, max_num=6,
                    image_size=self.image_size, use_thumbnail=False,
                )

            pad_color = tuple(int(x * 255) for x in self.image_transform.mean)
            global_view = ImageOps.pad(image, (self.base_size, self.base_size), color=pad_color)
            images_list.append(self.image_transform(global_view).to(self.dtype))

            w_crop, h_crop = crop_ratio
            images_spatial_crop.append([w_crop, h_crop])

            if w_crop > 1 or h_crop > 1:
                for c in images_crop_raw:
                    images_crop_list.append(self.image_transform(c).to(self.dtype))

            nq  = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
            nqb = math.ceil((self.base_size  // self.patch_size) / self.downsample_ratio)
            tokenized_image = [self.image_token_id] * (nqb * nqb) + [self.image_token_id]
            if w_crop > 1 or h_crop > 1:
                tokenized_image += [self.image_token_id] * (nq * w_crop * nq * h_crop)

        else:
            crop_ratio = (1, 1)
            images_spatial_crop.append([1, 1])
            pad_color = tuple(int(x * 255) for x in self.image_transform.mean)
            global_view = ImageOps.pad(image, (self.base_size, self.base_size), color=pad_color)
            images_list.append(self.image_transform(global_view).to(self.dtype))
            nq = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)
            tokenized_image = [self.image_token_id] * (nq * nq) + [self.image_token_id]

        return images_list, images_crop_list, images_spatial_crop, tokenized_image

    def process_single_sample(self, messages: List[Dict]) -> Dict[str, Any]:
        images = []
        for msg in messages:
            for img in msg.get("images", []):
                if img is not None:
                    images.append(self.deserialize_image(img))

        tokenized_str, images_seq_mask = [self.bos_id], [False]
        images_list, images_crop_list, images_spatial_crop = [], [], []
        prompt_token_count = -1
        assistant_started = False
        image_idx = 0

        for msg in messages:
            role, content = msg["role"], msg["content"]

            if role == "<|Assistant|>" and not assistant_started:
                prompt_token_count = len(tokenized_str)
                assistant_started = True
                content = f"{content.strip()} {self.tokenizer.eos_token}"

            for i, text_seg in enumerate(content.split("<image>")):
                toks = self._text_encode(self.tokenizer, text_seg, bos=False, eos=False)
                tokenized_str.extend(toks)
                images_seq_mask.extend([False] * len(toks))

                if i < content.count("<image>"):
                    img_list, crop_list, spatial, tok_img = self.process_image(images[image_idx])
                    images_list.extend(img_list)
                    images_crop_list.extend(crop_list)
                    images_spatial_crop.extend(spatial)
                    tokenized_str.extend(tok_img)
                    images_seq_mask.extend([True] * len(tok_img))
                    image_idx += 1

        if not assistant_started:
            prompt_token_count = len(tokenized_str)

        images_ori  = torch.stack(images_list, dim=0)
        spatial_t   = torch.tensor(images_spatial_crop, dtype=torch.long)
        images_crop = (
            torch.stack(images_crop_list, dim=0) if images_crop_list
            else torch.zeros((1, 3, self.base_size, self.base_size), dtype=self.dtype)
        )

        return {
            "input_ids":          torch.tensor(tokenized_str, dtype=torch.long),
            "images_seq_mask":    torch.tensor(images_seq_mask, dtype=torch.bool),
            "images_ori":         images_ori,
            "images_crop":        images_crop,
            "images_spatial_crop": spatial_t,
            "prompt_token_count": prompt_token_count,
        }

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_data = []
        for f in features:
            try:
                batch_data.append(self.process_single_sample(f["messages"]))
            except Exception as e:
                print(f"Warning: skipped sample — {e}")

        if not batch_data:
            raise ValueError("No valid samples in batch")

        # Truncate to max_length if set
        if self.max_length is not None:
            for d in batch_data:
                if len(d["input_ids"]) > self.max_length:
                    d["input_ids"]       = d["input_ids"][:self.max_length]
                    d["images_seq_mask"] = d["images_seq_mask"][:self.max_length]
                    d["prompt_token_count"] = min(d["prompt_token_count"], self.max_length)

        input_ids  = pad_sequence(
            [d["input_ids"] for d in batch_data],
            batch_first=True, padding_value=self.tokenizer.pad_token_id,
        )
        img_seq_mask = pad_sequence(
            [d["images_seq_mask"] for d in batch_data],
            batch_first=True, padding_value=False,
        )
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[img_seq_mask] = -100

        if self.train_on_responses_only:
            for i, d in enumerate(batch_data):
                pc = d["prompt_token_count"]
                if pc > 0:
                    labels[i, :pc] = -100

        attention_mask   = (input_ids != self.tokenizer.pad_token_id).long()
        images_batch     = [(d["images_crop"], d["images_ori"]) for d in batch_data]
        images_spatial   = torch.cat([d["images_spatial_crop"] for d in batch_data], dim=0)

        return {
            "input_ids":          input_ids,
            "attention_mask":     attention_mask,
            "labels":             labels,
            "images":             images_batch,
            "images_seq_mask":    img_seq_mask,
            "images_spatial_crop": images_spatial,
        }


# ── WandB inference callback ───────────────────────────────────────────────────

class WandbInferenceCallback(TrainerCallback):
    """Logs held-out inference samples to WandB every N steps."""

    def __init__(self, model, tokenizer, data_collator, samples, every_n_steps=100, max_new_tokens=512):
        self.model          = model
        self.tokenizer      = tokenizer
        self.data_collator  = data_collator
        self.samples        = samples
        self.every_n_steps  = every_n_steps
        self.max_new_tokens = max_new_tokens
        self._images_logged = False

    def _run_inference(self, step):
        import wandb, time, html as _html
        if wandb.run is None:
            return

        FastVisionModel.for_inference(self.model)
        self.model.eval()
        t0 = time.time()
        rows = []

        for sample in self.samples:
            try:
                feature = {
                    "messages": [
                        {"role": "<|User|>", "content": f"<image>\n{INSTRUCTION}",
                         "images": [sample["image"]]},
                        {"role": "<|Assistant|>", "content": ""},
                    ]
                }
                batch = self.data_collator([feature])
                # Use only the prompt tokens (labels == -100 means prompt or image)
                prompt_len = int((batch["labels"][0] == -100).sum())
                input_ids      = batch["input_ids"][:, :prompt_len].to("cuda")
                attention_mask = batch["attention_mask"][:, :prompt_len].to("cuda")
                images         = batch["images"]
                images_seq_mask = batch["images_seq_mask"][:, :prompt_len].to("cuda")
                images_spatial  = batch["images_spatial_crop"].to("cuda")
                with torch.no_grad():
                    out = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        images=images,
                        images_seq_mask=images_seq_mask,
                        images_spatial_crop=images_spatial,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                    )
                prediction = self.tokenizer.decode(
                    out[0][input_ids.shape[1]:], skip_special_tokens=True
                ).strip()
            except Exception as e:
                prediction = f"[inference error: {e}]"

            rows.append([
                step,
                wandb.Image(sample["image"], caption=f"{sample['score_id']} p{sample['page']}"),
                sample["score_id"],
                sample["corpus"],
                sample["page"],
                f"{sample['bar_start']}-{sample['bar_end']}",
                strip_musicxml_header(sample["musicxml"]),
                prediction,
            ])

        table = wandb.Table(
            columns=["step", "image", "score_id", "corpus", "page", "bars", "reference", "prediction"],
            data=rows,
        )
        log_dict = {"inference_examples": table, "inference/duration_sec": time.time() - t0}

        n_panels = min(3, len(self.samples))
        for i in range(n_panels):
            s = self.samples[i]
            if not self._images_logged:
                log_dict[f"sample_{i}/image"]        = wandb.Image(s["image"])
                log_dict[f"sample_{i}/ground_truth"] = wandb.Html(
                    f"<pre>{_html.escape(rows[i][6])}</pre>")
            log_dict[f"sample_{i}/prediction"] = wandb.Html(
                f"<pre>{_html.escape(rows[i][7])}</pre>")

        self._images_logged = True
        wandb.log(log_dict)
        FastVisionModel.for_training(self.model)

    def on_train_begin(self, args, state, control, **kwargs):
        self._run_inference(step=0)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.every_n_steps != 0:
            return
        self._run_inference(step=state.global_step)


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--corpus", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading configuration...")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_config  = config["dataset"]
    model_config    = config["model"]
    lora_config     = config["lora"]
    training_config = config["training"]
    wandb_config    = config.get("wandb", {})

    corpora = args.corpus or dataset_config.get("corpora", None)
    if args.epochs:
        training_config["num_train_epochs"] = args.epochs
        training_config["max_steps"] = -1
    if args.max_samples:
        dataset_config["max_samples"] = args.max_samples

    # ── WandB ──────────────────────────────────────────────────────────────────
    use_wandb = bool(os.environ.get("WANDB_API_KEY") and wandb_config.get("project"))
    wandb_run_id = None
    if use_wandb:
        import wandb
        wandb.init(
            project=wandb_config["project"],
            name=wandb_config.get("run_name"),
            config={
                "model":    model_config,
                "lora":     lora_config,
                "dataset":  {**dataset_config, "corpora": corpora},
                "training": training_config,
            },
        )
        print(f"WandB run: {wandb.run.url}")
        wandb_run_id = wandb.run.id
    else:
        if wandb_config.get("project"):
            print("WARN: wandb.project set but WANDB_API_KEY not found — logging disabled.")
        print("WandB logging disabled.")

    # ── Model ──────────────────────────────────────────────────────────────────
    model, tokenizer, local_path = load_model(model_config)

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=lora_config["finetune_vision_layers"],
        finetune_language_layers=lora_config["finetune_language_layers"],
        finetune_attention_modules=lora_config["finetune_attention_modules"],
        finetune_mlp_modules=lora_config["finetune_mlp_modules"],
        r=lora_config["r"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        bias="none",
        random_state=training_config["seed"],
        use_rslora=False,
        loftq_config=None,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # ── Dataset ────────────────────────────────────────────────────────────────
    repo  = dataset_config["repo"]
    cfg   = dataset_config["config"]
    split = dataset_config.get("split", "train")
    print(f"Loading dataset: {repo} (config={cfg}, split={split}) ...")
    dataset = load_dataset(repo, cfg, split=split)

    if corpora:
        print(f"Filtering to corpora: {corpora} ...")
        dataset = dataset.filter(lambda r: r["corpus"] in set(corpora))
        print(f"  {len(dataset)} rows after filter")

    if "max_samples" in dataset_config:
        n = dataset_config["max_samples"]
        dataset = dataset.select(range(min(n, len(dataset))))
        print(f"  Limited to {len(dataset)} samples")

    print(f"Dataset size: {len(dataset)}")
    print(f"Sample — score_id={dataset[0]['score_id']} "
          f"corpus={dataset[0]['corpus']} "
          f"page={dataset[0]['page']}/{dataset[0]['n_pages']} "
          f"bars={dataset[0]['bar_start']}-{dataset[0]['bar_end']}")

    if dataset_config.get("shuffle", False):
        dataset = dataset.shuffle(seed=training_config["seed"])
        print(f"Dataset shuffled (seed={training_config['seed']})")

    def convert_to_conversation(sample):
        return {
            "messages": [
                {
                    "role": "<|User|>",
                    "content": f"<image>\n{INSTRUCTION}",
                    "images": [sample["image"]],
                },
                {
                    "role": "<|Assistant|>",
                    "content": strip_musicxml_header(sample["musicxml"]),
                },
            ]
        }

    print("Converting to conversation format ...")
    train_data = [convert_to_conversation(s) for s in dataset]

    # ── Validation split ───────────────────────────────────────────────────────
    val_data = None
    val_raw  = None
    dev_split = load_dataset(repo, cfg, split="dev")
    if corpora:
        dev_split = dev_split.filter(lambda r: r["corpus"] in set(corpora))
    if len(dev_split) > 0:
        val_data = [convert_to_conversation(s) for s in dev_split]
        val_raw  = dev_split
        print(f"Validation: {len(val_data)} rows from HF dev split")
    else:
        print("No dev split rows — skipping validation")

    print(f"Train: {len(train_data)}  Val: {len(val_data) if val_data else 0}")

    # ── Held-out inference samples for WandB ───────────────────────────────────
    n_examples = wandb_config.get("n_examples", 4)
    src   = val_raw if val_raw is not None and len(val_raw) >= n_examples else dataset
    total = len(src)
    inference_samples = [src[int(i * total / n_examples)] for i in range(n_examples)]

    # ── Training ───────────────────────────────────────────────────────────────
    FastVisionModel.for_training(model)

    data_collator = DeepSeekOCR2DataCollator(
        tokenizer=tokenizer,
        model=model,
        image_size=768,
        base_size=1024,
        crop_mode=True,
        train_on_responses_only=True,
        max_length=model_config.get("max_length", 2048),
    )

    # Reduce memory fragmentation for large logits.float() allocation
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    gpu_stats = torch.cuda.get_device_properties(0)
    start_mem = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
    max_mem   = round(gpu_stats.total_memory / 1024 ** 3, 3)
    print(f"\nGPU: {gpu_stats.name}  |  {max_mem} GB total  |  {start_mem} GB reserved")

    training_args = TrainingArguments(
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=training_config.get("max_grad_norm", 0.3),
        warmup_ratio=training_config.get("warmup_ratio", 0.03),
        max_steps=training_config.get("max_steps", -1),
        num_train_epochs=(
            training_config.get("num_train_epochs", 1)
            if training_config.get("max_steps", -1) == -1 else 1
        ),
        learning_rate=training_config["learning_rate"],
        logging_steps=training_config["logging_steps"],
        save_strategy="steps",
        save_steps=training_config["save_steps"],
        eval_strategy="steps" if val_data else "no",
        eval_steps=(
            training_config.get("val_steps", training_config["save_steps"])
            if val_data else None
        ),
        optim=training_config["optim"],
        weight_decay=training_config["weight_decay"],
        lr_scheduler_type=training_config["lr_scheduler_type"],
        seed=training_config["seed"],
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        output_dir=training_config["output_dir"],
        report_to="wandb" if use_wandb else "none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    callbacks = []
    if use_wandb:
        callbacks.append(WandbInferenceCallback(
            model=model,
            tokenizer=tokenizer,
            data_collator=data_collator,
            samples=inference_samples,
            every_n_steps=wandb_config.get("log_examples_every_n_steps", 100),
            max_new_tokens=wandb_config.get("inference_max_new_tokens", 512),
        ))

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        callbacks=callbacks,
    )

    print("\nStarting training ...")
    stats = trainer.train()

    used_mem = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"  Time   : {round(stats.metrics['train_runtime'] / 60, 2)} min")
    print(f"  Memory : {used_mem} GB peak ({round(used_mem / max_mem * 100, 1)}%)")
    print('='*70)

    # ── Save ───────────────────────────────────────────────────────────────────
    output_dir = training_config["output_dir"]
    final_dir  = f"{output_dir}/final_model"
    print(f"\nSaving model to {final_dir} ...")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("Model saved.")

    if use_wandb and wandb_run_id:
        import wandb
        if wandb.run is None:
            wandb.init(project=wandb_config["project"], id=wandb_run_id, resume="must")
        artifact = wandb.Artifact(
            name=f"lora-adapter-{wandb_run_id}",
            type="model",
            description=f"LoRA adapter for {model_config['name']} trained on {corpora}",
            metadata={
                "model":       model_config["name"],
                "corpora":     corpora,
                "epochs":      training_config.get("num_train_epochs"),
                "train_loss":  round(stats.metrics.get("train_loss", 0), 4),
                "peak_vram_gb": used_mem,
            },
        )
        artifact.add_dir(final_dir)
        wandb.log_artifact(artifact)
        print(f"WandB artifact logged: {artifact.name}")
        wandb.finish()
    elif use_wandb:
        import wandb
        if wandb.run is not None:
            wandb.finish()

    # ── Post-training inference test ───────────────────────────────────────────
    if training_config.get("test_after_training", True) and len(dataset) >= 2:
        print("\nPost-training inference test ...")
        FastVisionModel.for_inference(model)
        sample = inference_samples[0]
        # Build token sequence via collator (single sample, no labels needed)
        test_feature = {
            "messages": [
                {"role": "<|User|>", "content": f"<image>\n{INSTRUCTION}",
                 "images": [sample["image"]]},
                {"role": "<|Assistant|>", "content": ""},
            ]
        }
        try:
            batch = data_collator([test_feature])
            input_ids     = batch["input_ids"][:, :batch["labels"][0].eq(-100).sum()].to("cuda")
            attention_mask = batch["attention_mask"][:, :input_ids.shape[1]].to("cuda")
            images        = batch["images"]
            images_seq_mask = batch["images_seq_mask"][:, :input_ids.shape[1]].to("cuda")
            images_spatial  = batch["images_spatial_crop"].to("cuda")
            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images,
                    images_seq_mask=images_seq_mask,
                    images_spatial_crop=images_spatial,
                    max_new_tokens=512,
                    do_sample=False,
                    use_cache=True,
                )
            pred = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
            print(f"\n--- Sample (score_id={sample['score_id']}, page={sample['page']}) ---")
            print(f"Reference (first 200 chars): {sample['musicxml'][:200].strip()}")
            print(f"Prediction:\n{pred[:500]}")
            print("--- End ---")
        except Exception as e:
            print(f"Inference test failed: {e}")


if __name__ == "__main__":
    main()
