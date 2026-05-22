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
from mxc2 import xml_to_mxc2

INSTRUCTION_XML = "Transcribe this sheet music page to MusicXML."
INSTRUCTION_MXC = "Transcribe this sheet music page to MXC (compact MusicXML)."
INSTRUCTION_MXC2 = "Transcribe this sheet music page to MXC2 (compact MusicXML)."


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
                 target_format="xml", drop_beams=False, col=None):
        self.model          = model
        self.processor      = processor
        self.samples        = samples
        self.every_n_steps  = every_n_steps
        self.max_new_tokens = max_new_tokens
        self.target_format  = target_format
        self.drop_beams     = drop_beams
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
            instr = (INSTRUCTION_MXC2 if self.target_format == "mxc2"
                     else INSTRUCTION_MXC if self.target_format == "mxc"
                     else INSTRUCTION_XML)
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
            if self.target_format in ("mxc", "mxc2"):
                try:
                    if self.target_format == "mxc2":
                        ref = xml_to_mxc2(ref, drop_beams=self.drop_beams)
                    else:
                        ref = xml_to_mxc(ref)
                except Exception as e:
                    print(f"WARN: {self.target_format} failed for inference sample "
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

    # Optionally append more synthetic configs (e.g. level9 in addition to
    # level7a) — useful for the key classifier where we want broader key /
    # rendering variety than a single level provides.
    extra_cfgs = dataset_config.get("extra_configs", [])
    if extra_cfgs:
        from datasets import concatenate_datasets, Value
        parts = [dataset]
        for ec in extra_cfgs:
            print(f"  + appending {repo} (config={ec}, split={split}) ...")
            parts.append(load_dataset(repo, ec, split=split))
        # Reduce to a common schema (only the columns we need) — different
        # synthetic configs have inconsistent dtypes on metadata columns.
        keep = list({col["image"], col["musicxml"], col["id"]})
        for i, p in enumerate(parts):
            drop = [c for c in p.column_names if c not in keep]
            if drop:
                parts[i] = p.remove_columns(drop)
        dataset = concatenate_datasets(parts)
        print(f"  combined size: {len(dataset)} samples")

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
    drop_beams = dataset_config.get("drop_beams", False)
    if target_format == "mxc2":
        instruction = INSTRUCTION_MXC2
        print(f"Target format: MXC2 (compact v2, drop_beams={drop_beams})")
    elif target_format == "mxc":
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

    def _convert_xml_to_target(text):
        if target_format == "mxc2":
            return xml_to_mxc2(text, drop_beams=drop_beams)
        elif target_format == "mxc":
            return xml_to_mxc(text)
        return text

    # When hints are injected, also pad all images to a uniform size so the
    # image-token count is constant across the dataset. Different image
    # heights produce different image-token counts, which can drift out of
    # sync with the chat template's text-side count → trainer mismatch error.
    pad_to_uniform = bool(globals().get("_INJECTED_HINTS_TRAIN"))
    PAD_W, PAD_H = 1240, 1792   # safe size for synthetic-scores Level 9

    def _pad_image(img):
        from PIL import Image
        w, h = img.size
        if w == PAD_W and h == PAD_H:
            return img
        # Inscribe if larger than the canvas (preserves aspect ratio)
        if w > PAD_W or h > PAD_H:
            scale = min(PAD_W / w, PAD_H / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            w, h = img.size
        # Pad with white, original at top-left
        canvas = Image.new("RGB", (PAD_W, PAD_H), "white")
        canvas.paste(img.convert("RGB") if img.mode != "RGB" else img, (0, 0))
        return canvas

    # Optional hint injection (set by train_with_hints.py before calling main)
    hints_train = globals().get("_INJECTED_HINTS_TRAIN") or {}
    hints_dev = globals().get("_INJECTED_HINTS_DEV") or {}
    hint_instruction = globals().get("_INJECTED_INSTRUCTION_WITH_HINT")
    if hints_train or hints_dev:
        # Compress hints (drops Audiveris noise: articulations, dynamics, voice
        # markers, stem directions, etc.) — ~25% size reduction with no info loss.
        from hint_compress import compress_hint
        hints_train = {k: compress_hint(v) for k, v in hints_train.items()}
        hints_dev = {k: compress_hint(v) for k, v in hints_dev.items()}
        train_lens = [len(v) for v in hints_train.values()]
        if train_lens:
            print(f"Hint injection enabled: {len(hints_train)} train, "
                  f"{len(hints_dev)} dev. Compressed hint chars: "
                  f"avg={sum(train_lens)/len(train_lens):.0f} max={max(train_lens)}")

    def convert_to_conversation(sample, _hint_pool=None):
        text = strip_musicxml_header(sample[col["musicxml"]])
        if target_format in ("mxc", "mxc2"):
            try:
                text = _convert_xml_to_target(text)
            except Exception as e:
                sample_id = sample.get(col["id"], "?")
                if strict_mxc:
                    raise RuntimeError(
                        f"{target_format} conversion failed on sample {sample_id}: {e}. "
                        f"Set dataset.strict_mxc=false to fall back to cleaned XML."
                    ) from e
                mxc_failures.record(sample_id, e)
                # fall back to cleaned XML for malformed samples

        # Build instruction (with hint if available)
        instr = instruction
        if _hint_pool is not None and hint_instruction:
            sample_id = sample.get(col["id"])
            hint = _hint_pool.get(sample_id) if sample_id else None
            if hint:
                instr = hint_instruction.format(hint=hint[:3500])

        # Optionally inject the ground-truth key signature into the prompt.
        # Tests whether transcription becomes reliable when the model is
        # handed the key instead of having to read it. Enable via
        # dataset.inject_gt_key.
        if dataset_config.get("inject_gt_key", False):
            import re as _re_k
            mk = _re_k.search(r"<fifths>(-?\d+)</fifths>",
                              sample[col["musicxml"]])
            if mk:
                instr = (f"{instr} The key signature is key={mk.group(1)} "
                         f"(N = sharps if positive, flats if negative).")

        # Text-only mode: skip the image entirely. Used when the input is
        # purely an Audiveris-generated transcription (no image). Sidesteps
        # the unsloth multimodal tokenizer mismatch bug.
        text_only = dataset_config.get("text_only", False)
        if text_only:
            user_content = [{"type": "text", "text": instr}]
        else:
            img = sample[col["image"]]
            if pad_to_uniform:
                img = _pad_image(img)
            user_content = [
                {"type": "text",  "text": instr},
                {"type": "image", "image": img},
            ]

        return {
            "messages": [
                {
                    "role": "user",
                    "content": user_content,
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
    # Per-measure mode: expand each page into per-measure (image, prompt, slice)
    # rows using the stateless MXC2 slicer. The model is trained to localize
    # and transcribe a specific measure given the page image.
    if dataset_config.get("per_measure", False):
        from mxc2_slice import iter_measures
        from mxc2 import xml_to_mxc2 as _xml_to_mxc2
        import random as _random_pm
        sample_k = dataset_config.get("per_measure_sample_k", 0)
        rng_pm = _random_pm.Random(training_config["seed"])
        train_data = []
        for s in dataset:
            try:
                mxc2_full = _xml_to_mxc2(
                    strip_musicxml_header(s[col["musicxml"]]),
                    drop_beams=dataset_config.get("drop_beams", False),
                )
                slices = list(iter_measures(mxc2_full))
                if sample_k and len(slices) > sample_k:
                    slices = rng_pm.sample(slices, sample_k)
                img = s[col["image"]]
                if pad_to_uniform:
                    img = _pad_image(img)
                for p_idx, m_num, slice_text in slices:
                    prompt = (f"Transcribe measure {m_num} of part {p_idx} "
                              f"on this page.")
                    train_data.append({
                        "messages": [
                            {"role": "user", "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image", "image": img},
                            ]},
                            {"role": "assistant",
                             "content": [{"type": "text", "text": slice_text}]},
                        ]
                    })
            except Exception as e:
                print(f"  per_measure: skipping sample (error: {e})")
        print(f"  Per-measure expansion: {len(train_data)} measure-level "
              f"training rows (sample_k={sample_k or 'all'})")
    else:
        train_data = [convert_to_conversation(s, _hint_pool=hints_train) for s in dataset]

    # Pre-filter samples that would exceed max_length (image tokens cannot be
    # truncated safely — they cause runtime errors). Conservative estimate:
    # ~975 image tokens + 200 instruction + 0.4 chars/token for text content.
    max_len = model_config.get("max_length", 8192)
    if hints_train:  # only when hint augmentation is active
        IMG_TOK = 1000
        WRAP_TOK = 250
        MARGIN = 500
        max_text_chars = (max_len - IMG_TOK - WRAP_TOK - MARGIN) // 0.4
        before = len(train_data)
        filtered = []
        for item in train_data:
            # Total text content = user text + assistant text
            text_chars = 0
            for msg in item["messages"]:
                for chunk in msg["content"]:
                    if chunk.get("type") == "text":
                        text_chars += len(chunk["text"])
            if text_chars <= max_text_chars:
                filtered.append(item)
        train_data = filtered
        dropped = before - len(train_data)
        print(f"  Length filter: kept {len(train_data)}/{before} "
              f"(dropped {dropped} samples >{int(max_text_chars)} chars)")

    # ── Auxiliary key-signature task (optional data mixing) ──────────────────
    # Mixes in dedicated "what is the key signature?" examples to give the
    # model direct gradient pressure on the key=N token. The token is a
    # negligible fraction of a full transcription's loss, so the model
    # otherwise never learns to read key signatures reliably. On a key-only
    # example, key=N IS the whole target — large fraction of that example's
    # loss. Enable via dataset.key_aux_ratio (0 = off).
    key_aux_ratio = dataset_config.get("key_aux_ratio", 0.0)
    if key_aux_ratio > 0:
        import random as _random
        import re as _re
        KEY_QUESTION = (
            "What is the key signature of this score? Answer in the form "
            "key=N, where N is the number of sharps (positive) or flats "
            "(negative), or key=0 for no sharps or flats."
        )
        # Chain-of-thought target — forces the model to enumerate accidentals
        # before naming the key, breaking the +3/+4 cluster the simple
        # "key=N" target collapsed into.
        _SHARP_ORDER = ["F", "C", "G", "D", "A", "E", "B"]
        _FLAT_ORDER = ["B", "E", "A", "D", "G", "C", "F"]
        _NUM_WORDS = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four",
                      5: "five", 6: "six", 7: "seven"}

        def _key_to_cot(fifths: int) -> str:
            if fifths == 0:
                return "No sharps or flats. key=0"
            if fifths > 0:
                names = _SHARP_ORDER[:fifths]
                return (f"Sharps: {' '.join(names)}. "
                        f"{_NUM_WORDS[fifths].capitalize()} sharps. "
                        f"key={fifths}")
            n = -fifths
            names = _FLAT_ORDER[:n]
            return (f"Flats: {' '.join(names)}. "
                    f"{_NUM_WORDS[n].capitalize()} flats. "
                    f"key={fifths}")

        use_cot = dataset_config.get("key_cot", False)
        rng = _random.Random(training_config["seed"])
        aux = []
        for sample in dataset:
            if rng.random() >= key_aux_ratio:
                continue
            m = _re.search(r"<fifths>(-?\d+)</fifths>", sample[col["musicxml"]])
            if not m:
                continue
            img = sample[col["image"]]
            if pad_to_uniform:
                img = _pad_image(img)
            fifths_val = int(m.group(1))
            target_text = (_key_to_cot(fifths_val) if use_cot
                           else f"key={m.group(1)}")
            aux.append({
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": KEY_QUESTION},
                        {"type": "image", "image": img},
                    ]},
                    {"role": "assistant",
                     "content": [{"type": "text", "text": target_text}]},
                ]
            })
        # `key_aux_only`: replace transcription examples with key-only.
        # Used for training a focused image→key=N classifier as Stage 1
        # of the re-spell pipeline (see respell.py).
        if dataset_config.get("key_aux_only", False):
            train_data = aux
            print(f"  Key-aux ONLY mode: replaced transcription with "
                  f"{len(aux)} key-only examples")
        else:
            train_data = train_data + aux
            print(f"  Key-aux mixing: added {len(aux)} key-only examples "
                  f"(ratio={key_aux_ratio}); train_data now {len(train_data)}")
        if dataset_config.get("shuffle", False):
            _random.Random(training_config["seed"]).shuffle(train_data)

    # ── Validation split: prefer HF dev split, fall back to manual carve-out ──
    val_data = None
    val_raw  = None
    try:
        dev_split = load_dataset(repo, cfg, split=dev_split_name)
        if corpora and col["corpus"] in dev_split.column_names:
            corpus_col = col["corpus"]
            dev_split = dev_split.filter(lambda r: r[corpus_col] in set(corpora))
        if len(dev_split) > 0:
            val_data = [convert_to_conversation(s, _hint_pool=hints_dev) for s in dev_split]
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
            drop_beams=drop_beams,
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
