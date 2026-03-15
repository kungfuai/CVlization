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
import os
import yaml
import torch
from datasets import load_dataset
from transformers import TrainerCallback, AutoModel
from unsloth import FastVisionModel, get_chat_template
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

INSTRUCTION = "Transcribe this sheet music page to MusicXML."


def prepare_inference_inputs(processor, image):
    """Prepare model inputs for inference.

    Works for Gemma-3, Qwen3-VL, and other unsloth-supported VLMs.
    For Qwen3-VL: ensure get_chat_template is NOT called (set chat_template: null
    in config), so the model's built-in vision-aware template is preserved.
    """
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": INSTRUCTION},
    ]}]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    return processor(image, input_text, add_special_tokens=False, return_tensors="pt")


def strip_musicxml_header(xml: str) -> str:
    """Remove noisy boilerplate from MusicXML, keeping only musically useful content.

    Strips:
      - XML declaration and DOCTYPE
      - <movement-title> when it looks like a temp filename (tmp*.xml)
      - <identification> block (IMSLP URLs, transcriber names, encoding software)
      - <defaults> block (scaling params, always identical)

    Keeps:
      - <work><work-title> (actual piece title)
      - <movement-number> and real <movement-title> (orchestra movements)
      - <part-list> (instrument names)
      - <part> elements (actual music)
    """
    import re
    # Strip XML declaration and DOCTYPE
    xml = re.sub(r'<\?xml[^?]*\?>\s*', '', xml)
    xml = re.sub(r'<!DOCTYPE[^>]*>\s*', '', xml)
    # Strip <identification>...</identification> block
    xml = re.sub(r'\s*<identification>.*?</identification>', '', xml, flags=re.DOTALL)
    # Strip <defaults>...</defaults> block
    xml = re.sub(r'\s*<defaults>.*?</defaults>', '', xml, flags=re.DOTALL)
    # Strip <movement-title> if it's a temp filename (e.g. tmp6abc.xml)
    xml = re.sub(r'\s*<movement-title>tmp[^<]*</movement-title>', '', xml)
    return xml.strip()


# ── WandB inference callback ───────────────────────────────────────────────────

class WandbInferenceCallback(TrainerCallback):
    """Runs inference on fixed held-out samples every N steps and logs to WandB.

    Logs a wandb.Table with columns:
      step | image | score_id | corpus | page | bars | reference | prediction
    """

    def __init__(self, model, processor, samples, every_n_steps=100, max_new_tokens=512):
        self.model          = model
        self.processor      = processor
        self.samples        = samples       # list of raw dataset rows (PIL images + metadata)
        self.every_n_steps  = every_n_steps
        self.max_new_tokens = max_new_tokens
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
            image = sample["image"]
            inputs = prepare_inference_inputs(self.processor, image).to("cuda")

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True,
                    do_sample=False,
                )
            pred_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            prediction  = self.processor.decode(pred_tokens, skip_special_tokens=True).strip()

            ref = strip_musicxml_header(sample["musicxml"])

            rows.append([
                step,
                wandb.Image(image, caption=f"{sample['score_id']} p{sample['page']}"),
                sample["score_id"],
                sample["corpus"],
                sample["page"],
                f"{sample['bar_start']}-{sample['bar_end']}",
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
            caption = f"{s['score_id']} p{s['page']} bars {s['bar_start']}-{s['bar_end']}"
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

    extra_kwargs = {}
    if model_config.get("trust_remote_code", False):
        extra_kwargs["trust_remote_code"] = True
    if model_config.get("unsloth_force_compile", False):
        extra_kwargs["unsloth_force_compile"] = True
    if model_config.get("auto_model", False):
        extra_kwargs["auto_model"] = AutoModel

    model, processor = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit=model_config["load_in_4bit"],
        use_gradient_checkpointing=model_config.get("use_gradient_checkpointing", "unsloth"),
        **extra_kwargs,
    )

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

    # ── Dataset ────────────────────────────────────────────────────────────────
    repo   = dataset_config["repo"]
    cfg    = dataset_config["config"]
    split  = dataset_config.get("split", "train")
    print(f"Loading dataset: {repo} (config={cfg}, split={split}) ...")
    dataset = load_dataset(repo, cfg, split=split)

    if corpora:
        print(f"Filtering to corpora: {corpora} ...")
        corpus_set = set(corpora)
        dataset = dataset.filter(lambda r: r["corpus"] in corpus_set)
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

    # ── Chat format ────────────────────────────────────────────────────────────
    def convert_to_conversation(sample):
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text",  "text": INSTRUCTION},
                        {"type": "image", "image": sample["image"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": strip_musicxml_header(sample["musicxml"])}],
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
    dev_split = load_dataset(repo, cfg, split="dev")
    if corpora:
        dev_split = dev_split.filter(lambda r: r["corpus"] in set(corpora))
    if len(dev_split) > 0:
        val_data = [convert_to_conversation(s) for s in dev_split]
        val_raw  = dev_split
        print(f"Validation: {len(val_data)} rows from HF dev split")
    else:
        print("No dev split rows found for selected corpora — skipping validation")

    print(f"Train: {len(train_data)}  Val: {len(val_data) if val_data else 0}")

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
        optim=training_config["optim"],
        weight_decay=training_config["weight_decay"],
        lr_scheduler_type=training_config["lr_scheduler_type"],
        seed=training_config["seed"],
        output_dir=training_config["output_dir"],
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
    processor.save_pretrained(final_dir)
    print("Model saved.")

    if use_wandb and wandb_run_id:
        import wandb
        # Re-open run if Trainer already closed it (report_to="wandb" calls finish internally)
        if wandb.run is None:
            wandb.init(project=wandb_config["project"], id=wandb_run_id, resume="must")
        # Upload final LoRA adapter as a WandB artifact
        artifact = wandb.Artifact(
            name=f"lora-adapter-{wandb_run_id}",
            type="model",
            description=f"LoRA adapter for {model_config['name']} trained on {corpora}",
            metadata={
                "model": model_config["name"],
                "corpora": corpora,
                "epochs": training_config.get("num_train_epochs"),
                "train_loss": round(stats.metrics.get("train_loss", 0), 4),
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
        inputs = prepare_inference_inputs(processor, sample["image"]).to("cuda")

        from transformers import TextStreamer
        streamer = TextStreamer(processor, skip_prompt=True)
        print(f"\n--- Sample (score_id={sample['score_id']}, page={sample['page']}) ---")
        print(f"Reference (first 200 chars): {sample['musicxml'][:200].strip()}")
        print("Prediction:")
        model.generate(**inputs, streamer=streamer, max_new_tokens=512,
                       use_cache=True, temperature=1.0, top_p=0.95)
        print("\n--- End ---")


if __name__ == "__main__":
    main()
