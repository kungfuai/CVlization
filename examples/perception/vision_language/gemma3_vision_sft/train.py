#!/usr/bin/env python3
"""
Gemma-3 Vision Fine-tuning with Unsloth

Fine-tune Gemma-3 vision models with Unsloth for efficient vision-language training.
Based on:
- https://github.com/unslothai/notebooks/blob/main/nb/Gemma3_(4B)-Vision.ipynb
- https://github.com/unslothai/notebooks/blob/main/nb/Gemma3N_(4B)-Vision.ipynb

Key features:
- Vision + language fine-tuning with LoRA/PEFT
- 4-bit quantization for memory efficiency
- Support for both Gemma-3 and Gemma-3N variants
- LaTeX OCR example (image-to-text)
"""

from unsloth import FastVisionModel, get_chat_template
from unsloth.trainer import UnslothVisionDataCollator
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import torch
import yaml

# Load configuration
print("Loading configuration...")
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Extract config
dataset_config = config["dataset"]
model_config = config["model"]
lora_config = config["lora"]
training_config = config["training"]

# Load model and processor
print(f"Loading model: {model_config['name']}...")
model, processor = FastVisionModel.from_pretrained(
    model_config["name"],
    load_in_4bit=model_config["load_in_4bit"],
    use_gradient_checkpointing=model_config.get("use_gradient_checkpointing", "unsloth"),
)

# Prepare model for fine-tuning with LoRA
print("Preparing model for fine-tuning with LoRA...")
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

# Load dataset
print(f"Loading dataset: {dataset_config['path']}...")
dataset = load_dataset(dataset_config["path"], split=dataset_config.get("split", "train"))

# Limit dataset size if specified
if "max_samples" in dataset_config:
    max_samples = dataset_config["max_samples"]
    dataset = dataset.select(range(min(max_samples, len(dataset))))
    print(f"Limited dataset to {len(dataset)} samples")

print(f"Dataset size: {len(dataset)}")
print(f"Sample image size: {dataset[0]['image'].size}")
print(f"Sample text: {dataset[0]['text'][:100]}...")

# Convert to conversation format
instruction = dataset_config.get("instruction", "Write the LaTeX representation for this image.")

def convert_to_conversation(sample):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": sample["image"]},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": sample["text"]}]},
    ]
    return {"messages": conversation}

print("Converting dataset to conversation format...")
converted_dataset = [convert_to_conversation(sample) for sample in dataset]

# Split for validation if specified
val_split_ratio = dataset_config.get("val_split_ratio", 0.0)
if val_split_ratio > 0:
    split_idx = int(len(converted_dataset) * (1 - val_split_ratio))
    train_dataset = converted_dataset[:split_idx]
    val_dataset = converted_dataset[split_idx:]
    print(f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} val")
else:
    train_dataset = converted_dataset
    val_dataset = None
    print(f"Training dataset size: {len(train_dataset)}")

# Set up chat template
chat_template = model_config.get("chat_template", "gemma-3")
print(f"Setting up {chat_template} chat template...")
processor = get_chat_template(processor, chat_template)

# Test inference before training (optional)
if training_config.get("test_before_training", False):
    print("\nTesting inference before training...")
    FastVisionModel.for_inference(model)

    image = dataset[2]["image"]
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction}],
        }
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")

    from transformers import TextStreamer
    text_streamer = TextStreamer(processor, skip_prompt=True)
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128,
                      use_cache=True, temperature=1.0, top_p=0.95, top_k=64)
    print("\nPre-training test complete.\n")

# Enable training mode
FastVisionModel.for_training(model)

# Training configuration
training_args = SFTConfig(
    per_device_train_batch_size=training_config["per_device_train_batch_size"],
    gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_grad_norm=training_config.get("max_grad_norm", 0.3),
    warmup_ratio=training_config.get("warmup_ratio", 0.03),
    max_steps=training_config.get("max_steps", -1),
    num_train_epochs=training_config.get("num_train_epochs", 1) if training_config.get("max_steps", -1) == -1 else 1,
    learning_rate=training_config["learning_rate"],
    logging_steps=training_config["logging_steps"],
    save_strategy="steps",
    save_steps=training_config["save_steps"],
    eval_strategy="steps" if val_dataset else "no",
    eval_steps=training_config.get("val_steps", training_config["save_steps"]) if val_dataset else None,
    optim=training_config["optim"],
    weight_decay=training_config["weight_decay"],
    lr_scheduler_type=training_config["lr_scheduler_type"],
    seed=training_config["seed"],
    output_dir=training_config["output_dir"],
    report_to="none",
    # Vision-specific requirements
    remove_unused_columns=False,
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
    max_length=model_config.get("max_length", 2048),
)

# Create trainer
print("Initializing SFT trainer with vision data collator...")
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=processor.tokenizer,
    data_collator=UnslothVisionDataCollator(model, processor),
    args=training_args,
)

# Show memory stats before training
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"\nGPU: {gpu_stats.name}")
print(f"Max memory: {max_memory} GB")
print(f"Reserved memory: {start_gpu_memory} GB\n")

# Train the model
print("Starting training...")
trainer_stats = trainer.train()

# Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_training = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
training_percentage = round(used_memory_for_training / max_memory * 100, 3)

print(f"\n{'='*80}")
print("TRAINING COMPLETE")
print('='*80)
print(f"Time: {round(trainer_stats.metrics['train_runtime']/60, 2)} minutes")
print(f"Peak reserved memory: {used_memory} GB ({used_percentage}%)")
print(f"Training memory: {used_memory_for_training} GB ({training_percentage}%)")
print('='*80 + '\n')

# Test inference after training
if training_config.get("test_after_training", True):
    print("Testing inference after training...")
    FastVisionModel.for_inference(model)

    image = dataset[10]["image"] if len(dataset) > 10 else dataset[0]["image"]
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction}],
        }
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")

    from transformers import TextStreamer
    text_streamer = TextStreamer(processor, skip_prompt=True)
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128,
                      use_cache=True, temperature=1.0, top_p=0.95, top_k=64)
    print("\nPost-training test complete.\n")

# Save the final model
output_dir = training_config["output_dir"]
final_model_dir = f"{output_dir}/final_model"
print(f"Saving model to {final_model_dir}...")
model.save_pretrained(final_model_dir)
processor.save_pretrained(final_model_dir)

print("\nModel saved successfully!")
print(f"To use: FastVisionModel.from_pretrained('{final_model_dir}')")

# Post-training evaluation metrics on held-out test set
if len(dataset) >= 20:  # Only evaluate if dataset is large enough for meaningful metrics
    print(f"\n{'='*80}")
    print("EVALUATION METRICS (Test Set)")
    print('='*80)

    try:
        import Levenshtein
        from PIL import Image

        # Use last 10% as test set (not used in training)
        test_size = max(20, int(len(dataset) * 0.1))
        test_dataset = dataset.select(range(len(dataset) - test_size, len(dataset)))
        print(f"Evaluating on {len(test_dataset)} held-out samples...")

        exact_matches = 0
        char_errors = []
        predictions = []
        references = []

        for example in test_dataset:
            image = example['image']
            reference = example['text'].strip()

            # Generate prediction
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Convert this image to LaTeX:"}
            ]}]

            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(
                input_text,
                [image],
                add_special_tokens=False,
                return_tensors="pt"
            ).to("cuda")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    use_cache=True,
                    temperature=0.1,  # Lower temperature for more deterministic output
                )
            prediction = processor.decode(outputs[0], skip_special_tokens=True)
            prediction = prediction.split("<start_of_turn>model\n")[-1].split("<end_of_turn>")[0].strip()

            predictions.append(prediction)
            references.append(reference)

            # Exact match
            if prediction == reference:
                exact_matches += 1

            # Character Error Rate (edit distance / length)
            edit_distance = Levenshtein.distance(prediction, reference)
            cer = edit_distance / max(len(reference), 1)
            char_errors.append(cer)

        # Compute metrics
        exact_match_accuracy = exact_matches / len(test_dataset)
        avg_cer = sum(char_errors) / len(char_errors)

        print(f"\nExact Match Accuracy: {exact_match_accuracy:.4f} ({exact_matches}/{len(test_dataset)})")
        print(f"Character Error Rate (CER): {avg_cer:.4f}")

        # Show a few examples
        print("\nSample predictions:")
        for i in range(min(3, len(predictions))):
            print(f"\n  Reference: {references[i][:80]}...")
            print(f"  Prediction: {predictions[i][:80]}...")
            print(f"  Match: {'✓' if predictions[i] == references[i] else '✗'}")

    except ImportError:
        print("\nEvaluation skipped (install python-Levenshtein: pip install python-Levenshtein)")
    except Exception as e:
        print(f"\nEvaluation failed: {e}")

    print('='*80)
