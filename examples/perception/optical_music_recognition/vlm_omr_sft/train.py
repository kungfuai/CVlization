#!/usr/bin/env python3
"""
Gemma-3 Vision fine-tuning for Optical Music Recognition (OMR).

Fine-tunes Gemma-3 on the zzsi/openscore pages_transcribed dataset:
  input  — full-page sheet music image (LilyPond render)
  output — per-page MusicXML (bars on that page)

Based on the gemma3_vision_sft example; adapted for OMR with the openscore dataset.

Usage:
  python train.py                           # uses config.yaml defaults (smoke test)
  python train.py --corpus lieder quartets  # override corpus filter
  python train.py --epochs 2               # full training run
"""

import argparse
import datetime
import os
import yaml
import torch
from datasets import load_dataset
from transformers import TrainerCallback, AutoModel
from unsloth import FastVisionModel, get_chat_template
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

from mxc import xml_to_mxc

INSTRUCTION_XML = "Transcribe this sheet music page to MusicXML."
INSTRUCTION_MXC = "Transcribe this sheet music page to MXC (compact MusicXML)."


def prepare_inference_inputs(processor, image, instruction=None):
    """Prepare model inputs for inference.

    Works for Gemma-3, Qwen3-VL, and other unsloth-supported VLMs.
    For Qwen3-VL: ensure get_chat_template is NOT called (set chat_template: null
    in config), so the model's built-in vision-aware template is preserved.
    """
    if instruction is None:
        instruction = INSTRUCTION_XML
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": instruction},
    ]}]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    return processor(image, input_text, add_special_tokens=False, return_tensors="pt")


def strip_musicxml_header(xml: str) -> str:
    """Remove non-visible boilerplate from MusicXML, keeping only content
    that is visible on the sheet music image (notes, rests, stems, beams,
    lyrics, accidentals, barlines, directions with visible text, etc.).
    """
    import re
    # Strip XML declaration and DOCTYPE
    xml = re.sub(r'<\?xml[^?]*\?>\s*', '', xml)
    xml = re.sub(r'<!DOCTYPE[^>]*>\s*', '', xml)
    # Strip <identification> but keep composer/lyricist (visible on page)
    xml = re.sub(r'\s*<rights>[^<]*</rights>', '', xml)
    xml = re.sub(r'\s*<encoding>.*?</encoding>', '', xml, flags=re.DOTALL)
    xml = re.sub(r'\s*<creator type="arranger">[^<]*</creator>', '', xml)
    # Remove empty <identification> tags left behind
    xml = re.sub(r'\s*<identification>\s*</identification>', '', xml, flags=re.DOTALL)
    # Strip <defaults>...</defaults> block
    xml = re.sub(r'\s*<defaults>.*?</defaults>', '', xml, flags=re.DOTALL)
    # Strip <movement-title> if it's a temp filename (e.g. tmp6abc.xml)
    xml = re.sub(r'\s*<movement-title>tmp[^<]*</movement-title>', '', xml)
    # Strip non-musical metadata: <score-instrument>, <midi-instrument>, <midi-device>
    xml = re.sub(r'\s*<score-instrument[^>]*>.*?</score-instrument>', '', xml, flags=re.DOTALL)
    xml = re.sub(r'\s*<midi-instrument[^>]*>.*?</midi-instrument>', '', xml, flags=re.DOTALL)
    xml = re.sub(r'\s*<midi-device[^>]*/?>', '', xml)
    # Strip XML comments (e.g. <!--=== Part 1 ===-->)
    xml = re.sub(r'\s*<!--.*?-->', '', xml, flags=re.DOTALL)
    # Strip <sound .../> elements (invisible numeric metadata like tempo="92")
    xml = re.sub(r'\s*<sound\b[^/]*/>', '', xml)
    # Strip <direction> blocks that contain only <sound> or empty <words/>
    xml = re.sub(
        r'\s*<direction[^>]*>\s*<direction-type>\s*<words\s*/>\s*</direction-type>\s*(?:<sound\b[^/]*/>\s*)?</direction>',
        '', xml, flags=re.DOTALL)
    xml = re.sub(
        r'\s*<direction[^>]*>\s*<direction-type>\s*</direction-type>\s*<sound\b[^/]*/>\s*</direction>',
        '', xml, flags=re.DOTALL)
    # Strip implicit="no" (always "no", adds nothing)
    xml = xml.replace(' implicit="no"', '')
    return xml.strip()


# ── WandB inference callback ───────────────────────────────────────────────────

class WandbInferenceCallback(TrainerCallback):
    """Runs inference on fixed held-out samples every N steps and logs to WandB.

    Logs a wandb.Table with columns:
      step | image | score_id | corpus | page | bars | reference | prediction
    """

    def __init__(self, model, processor, samples, every_n_steps=100, max_new_tokens=512,
                 target_format="xml", col=None):
        self.model          = model
        self.processor      = processor
        self.samples        = samples       # list of raw dataset rows (PIL images + metadata)
        self.every_n_steps  = every_n_steps
        self.max_new_tokens = max_new_tokens
        self.target_format  = target_format
        self.col            = col or {"image": "image", "musicxml": "musicxml", "id": "score_id", "label": "page"}
        self._images_logged = False         # log images + ground truth only once

    def _run_inference(self, step):
        import wandb
        if wandb.run is None:
            return

        import time
        wandb.log({"inference/started_at_step": step})
        t0 = time.time()

        FastVisionModel.for_inference(self.model)
        self.model.eval()

        rows = []
        for sample in self.samples:
            image = sample[self.col["image"]]
            instr = INSTRUCTION_MXC if self.target_format == "mxc" else INSTRUCTION_XML
            inputs = prepare_inference_inputs(self.processor, image, instr).to("cuda")

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True,
                    do_sample=False,
                )
            pred_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            prediction  = self.processor.decode(pred_tokens, skip_special_tokens=True).strip()

            ref = strip_musicxml_header(sample[self.col["musicxml"]])
            if self.target_format == "mxc":
                try:
                    ref = xml_to_mxc(ref)
                except Exception as e:
                    print(f"WARN: xml_to_mxc failed for inference sample "
                          f"{sample.get(self.col['id'], '?')}: {e}")

            sample_id = sample.get(self.col["id"], "")
            sample_label = sample.get(self.col["label"], "")
            rows.append([
                step,
                wandb.Image(image, caption=f"{sample_id} {sample_label}"),
                sample_id,
                sample.get(self.col.get("corpus", "corpus"), ""),
                sample_label,
                "",
                ref,
                prediction,
            ])

        table = wandb.Table(
            columns=["step", "image", "score_id", "corpus", "page",
                     "bars", "reference", "prediction"],
            data=rows,
        )

        import html as _html
        log_dict = {
            "inference_examples":     table,
            "inference/duration_sec": time.time() - t0,
        }

        # Log images + ground truth once; predictions every step
        n_panels = min(3, len(self.samples))
        for i in range(n_panels):
            s = self.samples[i]
            ref_i = rows[i][6]
            pred_i = rows[i][7]
            caption = f"{s.get(self.col['id'], '?')} {s.get(self.col['label'], '')}"
            if not self._images_logged:
                log_dict[f"sample_{i}/image"]        = wandb.Image(s["image"], caption=caption)
                log_dict[f"sample_{i}/ground_truth"] = wandb.Html(f"<pre>{_html.escape(ref_i)}</pre>")
            log_dict[f"sample_{i}/prediction"] = wandb.Html(f"<pre>{_html.escape(pred_i)}</pre>")

        self._images_logged = True
        wandb.log(log_dict)

        FastVisionModel.for_training(self.model)

    def on_train_begin(self, args, state, control, **kwargs):
        self._run_inference(step=0)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.every_n_steps != 0:
            return
        self._run_inference(step=state.global_step)


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--corpus", nargs="+", default=None,
                        help="Override corpus filter (e.g. lieder quartets orchestra)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override num_train_epochs (disables max_steps)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Override max_samples")
    parser.add_argument("--model", default=None,
                        help="Override model name (e.g. unsloth/Qwen2-VL-7B-Instruct)")
    parser.add_argument("--chat-template", default=None,
                        help="Override chat template (e.g. qwen-2.5, llama-3.2)")
    parser.add_argument("--resume-adapter", default=None,
                        help="Path to a LoRA adapter to load before training (resets scheduler/optimizer)")
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

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

    # CLI overrides
    corpora = args.corpus or dataset_config.get("corpora", None)
    if args.epochs is not None:
        training_config["num_train_epochs"] = args.epochs
        training_config["max_steps"] = -1
    if args.max_samples is not None:
        dataset_config["max_samples"] = args.max_samples
    if args.model is not None:
        model_config["name"] = args.model
    if args.chat_template is not None:
        model_config["chat_template"] = args.chat_template

    # ── WandB init ─────────────────────────────────────────────────────────────
    use_wandb = bool(os.environ.get("WANDB_API_KEY") and wandb_config.get("project"))
    if use_wandb:
        import wandb
        wandb.init(
            project=wandb_config["project"],
            name=wandb_config.get("run_name", None),
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
        wandb_run_id = None
        if wandb_config.get("project"):
            print("WARN: wandb.project set in config but WANDB_API_KEY not found — logging disabled.")
        print("WandB logging disabled.")

    # ── Model ──────────────────────────────────────────────────────────────────
    model_name = model_config["name"]
    print(f"Loading model: {model_name} ...")

    # Some models (e.g. DeepSeek-OCR-2) require snapshot_download to a local path first
    if model_config.get("snapshot_download", False):
        from huggingface_hub import snapshot_download
        import glob as _glob
        print(f"  Pre-downloading {model_name} to HF cache ...")
        model_name = snapshot_download(model_name)
        print(f"  Downloaded to: {model_name}")
        # Patch compatibility: transformers 5.x renamed DeepseekV2MoE → DeepseekV2Moe
        for py_file in _glob.glob(f"{model_name}/*.py"):
            with open(py_file, "r") as f:
                content = f.read()
            if "DeepseekV2MoE" in content:
                patched = content.replace("DeepseekV2MoE", "DeepseekV2Moe")
                with open(py_file, "w") as f:
                    f.write(patched)
                print(f"  Patched {py_file}: DeepseekV2MoE → DeepseekV2Moe")

    # Quantization: 4-bit, 8-bit, or bf16 (none).
    # For Qwen3.5, Unsloth recommends bf16 instead of 4-bit (4-bit has known issues).
    load_in_4bit = model_config.get("load_in_4bit", True)
    load_in_8bit = model_config.get("load_in_8bit", False)
    if load_in_4bit and load_in_8bit:
        raise ValueError("Cannot set both load_in_4bit and load_in_8bit")
    quant_label = "4-bit" if load_in_4bit else "8-bit" if load_in_8bit else "bf16 (no quantization)"
    print(f"Quantization: {quant_label}")

    extra_kwargs = {
        "load_in_4bit": load_in_4bit,
        "load_in_8bit": load_in_8bit,
        "use_gradient_checkpointing": model_config.get("use_gradient_checkpointing", "unsloth"),
    }
    if model_config.get("trust_remote_code", False):
        extra_kwargs["trust_remote_code"] = True
    if model_config.get("unsloth_force_compile", False):
        extra_kwargs["unsloth_force_compile"] = True
    if model_config.get("auto_model", False):
        extra_kwargs["auto_model"] = AutoModel
    if model_config.get("unsloth_tiled_mlp", False):
        extra_kwargs["unsloth_tiled_mlp"] = True

    model, processor = FastVisionModel.from_pretrained(model_name, **extra_kwargs)

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
        target_modules="all-linear",
    )

    # Optionally load pre-trained LoRA weights (fresh scheduler/optimizer)
    if args.resume_adapter:
        from peft import set_peft_model_state_dict
        import safetensors.torch
        adapter_path = args.resume_adapter
        print(f"Loading LoRA adapter from {adapter_path} ...")
        adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
        if os.path.exists(adapter_file):
            state_dict = safetensors.torch.load_file(adapter_file)
            set_peft_model_state_dict(model, state_dict)
            print(f"  Loaded {len(state_dict)} tensors (scheduler/optimizer will be fresh)")
        else:
            print(f"  WARNING: {adapter_file} not found, starting from scratch")

    # ── Dataset ────────────────────────────────────────────────────────────────
    repo   = dataset_config["repo"]
    cfg    = dataset_config.get("config", "default")
    split  = dataset_config.get("split", "train")
    dev_split_name = dataset_config.get("dev_split", "dev")

    # Column mapping: config can override column names for different datasets
    col = {
        "image":    "image",
        "musicxml": "musicxml",
        "id":       "score_id",
        "label":    "page",       # used for display captions
        "corpus":   "corpus",     # optional, for filtering
    }
    col.update(dataset_config.get("columns", {}))

    print(f"Loading dataset: {repo} (config={cfg}, split={split}) ...")
    dataset = load_dataset(repo, cfg, split=split)

    if corpora and col["corpus"] in dataset.column_names:
        print(f"Filtering to corpora: {corpora} ...")
        corpus_col = col["corpus"]
        corpus_set = set(corpora)
        dataset = dataset.filter(lambda r: r[corpus_col] in corpus_set)
        print(f"  {len(dataset)} rows after filter")

    if "max_samples" in dataset_config:
        n = dataset_config["max_samples"]
        dataset = dataset.select(range(min(n, len(dataset))))
        print(f"  Limited to {len(dataset)} samples")

    print(f"Dataset size: {len(dataset)}")
    s0 = dataset[0]
    print(f"Sample — id={s0.get(col['id'], '?')} label={s0.get(col['label'], '?')}")

    # ── Chat format ────────────────────────────────────────────────────────────
    target_format = dataset_config.get("target_format", "xml")
    if target_format == "mxc":
        instruction = INSTRUCTION_MXC
        print(f"Target format: MXC (compact)")
    else:
        instruction = INSTRUCTION_XML
        print(f"Target format: XML")

    strict_mxc = dataset_config.get("strict_mxc", False)

    class MxcFailureTracker:
        __slots__ = ("count", "first_id", "first_err")
        def __init__(self):
            self.count = 0
            self.first_id = None
            self.first_err = None
        def record(self, sample_id, err):
            if self.count == 0:
                self.first_id = sample_id
                self.first_err = str(err)
            self.count += 1

    mxc_failures = MxcFailureTracker()

    def convert_to_conversation(sample):
        text = strip_musicxml_header(sample[col["musicxml"]])
        if target_format == "mxc":
            try:
                text = xml_to_mxc(text)
            except Exception as e:
                sample_id = sample.get(col["id"], "?")
                if strict_mxc:
                    raise RuntimeError(
                        f"xml_to_mxc failed on sample {sample_id}: {e}. "
                        f"Set dataset.strict_mxc=false to fall back to cleaned XML."
                    ) from e
                mxc_failures.record(sample_id, e)
                # fall back to cleaned XML for malformed samples
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text",  "text": instruction},
                        {"type": "image", "image": sample[col["image"]]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": text}],
                },
            ]
        }

    if dataset_config.get("shuffle", False):
        dataset = dataset.shuffle(seed=training_config["seed"])
        print(f"Dataset shuffled (seed={training_config['seed']})")

    print("Converting to conversation format ...")
    train_data = [convert_to_conversation(s) for s in dataset]

    # ── Validation split: prefer HF dev split, fall back to manual carve-out ──
    val_data = None
    val_raw  = None
    try:
        dev_split = load_dataset(repo, cfg, split=dev_split_name)
        if corpora and col["corpus"] in dev_split.column_names:
            corpus_col = col["corpus"]
            dev_split = dev_split.filter(lambda r: r[corpus_col] in set(corpora))
        if len(dev_split) > 0:
            val_data = [convert_to_conversation(s) for s in dev_split]
            val_raw  = dev_split
            print(f"Validation: {len(val_data)} rows from HF dev split")
        else:
            print("No dev split rows found — skipping validation")
    except Exception:
        print("No dev split available — skipping validation")

    print(f"Train: {len(train_data)}  Val: {len(val_data) if val_data else 0}")

    if target_format == "mxc" and mxc_failures.count > 0:
        # convert_to_conversation runs on both train and val splits, so the
        # failure count covers both halves.
        total = len(train_data) + (len(val_data) if val_data else 0)
        pct = 100 * mxc_failures.count / total
        print(
            f"WARNING: xml_to_mxc failed on {mxc_failures.count}/{total} samples "
            f"({pct:.1f}%). First failure: {mxc_failures.first_id} — "
            f"{mxc_failures.first_err}. Those samples trained on cleaned XML "
            f"instead of MXC. Set dataset.strict_mxc=true to fail fast."
        )

    chat_template = model_config.get("chat_template", "gemma-3")
    if chat_template:
        print(f"Setting up {chat_template} chat template ...")
        processor = get_chat_template(processor, chat_template)
    else:
        print("Using model's built-in chat template (skipping get_chat_template)")

    # ── Held-out inference samples for WandB ───────────────────────────────────
    n_examples = wandb_config.get("n_examples", 4)
    if val_raw is not None and len(val_raw) >= n_examples:
        src = val_raw
    else:
        src = dataset
    total = len(src)
    example_idxs = [int(i * total / n_examples) for i in range(n_examples)]
    inference_samples = [src[i] for i in example_idxs]

    # ── Training ───────────────────────────────────────────────────────────────
    FastVisionModel.for_training(model)

    # Per-run output subdirectory to prevent checkpoint collisions across runs.
    # Uses wandb run_id if available, otherwise a timestamp.
    base_output_dir = training_config["output_dir"]
    if wandb_run_id:
        run_subdir = wandb_run_id
    else:
        run_subdir = f"run-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_output_dir = os.path.join(base_output_dir, run_subdir)
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {run_output_dir}")

    training_args = SFTConfig(
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
        load_best_model_at_end=True if val_data else False,
        metric_for_best_model="eval_loss",
        optim=training_config["optim"],
        weight_decay=training_config["weight_decay"],
        lr_scheduler_type=training_config["lr_scheduler_type"],
        seed=training_config["seed"],
        output_dir=run_output_dir,
        report_to="wandb" if use_wandb else "none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=model_config.get("max_length", 4096),
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_mem = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
    max_mem   = round(gpu_stats.total_memory / 1024 ** 3, 3)
    print(f"\nGPU: {gpu_stats.name}  |  {max_mem} GB total  |  {start_mem} GB reserved")

    callbacks = []
    if use_wandb:
        callbacks.append(WandbInferenceCallback(
            model=model,
            processor=processor,
            samples=inference_samples,
            every_n_steps=wandb_config.get("log_examples_every_n_steps", 100),
            max_new_tokens=wandb_config.get("inference_max_new_tokens", 512),
            target_format=target_format,
            col=col,
        ))

    # Handle models where FastVisionModel returns tokenizer directly (e.g. DeepSeek-OCR-2)
    processing_class = getattr(processor, "tokenizer", processor)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=processing_class,
        data_collator=UnslothVisionDataCollator(model, processor),
        args=training_args,
        callbacks=callbacks,
    )

    print("\nStarting training ...")
    # Use unsloth_train(trainer) instead of trainer.train() to apply Unsloth's
    # gradient accumulation bug fix (fixes incorrect loss denominator when
    # sequence lengths vary across grad accumulation steps).
    # See: https://unsloth.ai/blog/gradient
    if training_config.get("use_unsloth_train", False):
        from unsloth import unsloth_train
        print("  Using unsloth_train() for fixed gradient accumulation")
        stats = unsloth_train(trainer)
    else:
        stats = trainer.train()

    used_mem = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"  Time   : {round(stats.metrics['train_runtime'] / 60, 2)} min")
    print(f"  Memory : {used_mem} GB peak ({round(used_mem / max_mem * 100, 1)}%)")
    print('='*70)

    # ── Save ───────────────────────────────────────────────────────────────────
    # With load_best_model_at_end=True, the model is already the best checkpoint
    final_dir = f"{run_output_dir}/final_model"
    best_step = getattr(trainer.state, "best_global_step", None)
    best_metric = getattr(trainer.state, "best_metric", None)
    if best_step:
        print(f"\nBest checkpoint was step {best_step} (eval_loss={best_metric:.5f})")
    print(f"Saving model to {final_dir} ...")
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print("Model saved.")

    if use_wandb and wandb_run_id:
        import wandb
        # Re-open run if Trainer already closed it (report_to="wandb" calls finish internally)
        if wandb.run is None:
            wandb.init(project=wandb_config["project"], id=wandb_run_id, resume="must")
        # Upload final LoRA adapter as a WandB artifact
        best_info = f" (best step {best_step}, eval_loss={best_metric:.5f})" if best_step else ""
        artifact = wandb.Artifact(
            name=f"lora-adapter-{wandb_run_id}",
            type="model",
            description=f"LoRA adapter for {model_config['name']} trained on {corpora}{best_info}",
            metadata={
                "model": model_config["name"],
                "corpora": corpora,
                "epochs": training_config.get("num_train_epochs"),
                "train_loss": round(stats.metrics.get("train_loss", 0), 4),
                "best_step": best_step,
                "best_eval_loss": round(best_metric, 5) if best_metric else None,
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
        instr = INSTRUCTION_MXC if target_format == "mxc" else INSTRUCTION_XML
        inputs = prepare_inference_inputs(processor, sample[col["image"]], instr).to("cuda")

        from transformers import TextStreamer
        streamer = TextStreamer(processor, skip_prompt=True)
        print(f"\n--- Sample (id={sample.get(col['id'], '?')}, label={sample.get(col['label'], '?')}) ---")
        print(f"Reference (first 200 chars): {sample[col['musicxml']][:200].strip()}")
        print("Prediction:")
        model.generate(**inputs, streamer=streamer, max_new_tokens=512,
                       use_cache=True, temperature=1.0, top_p=0.95)
        print("\n--- End ---")


if __name__ == "__main__":
    main()
