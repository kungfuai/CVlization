#!/usr/bin/env python3
"""
Gemma-3 4B Text Fine-tuning with Unsloth

Fine-tune Gemma-3 4B with Unsloth for 2x faster training.
Based on: https://github.com/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb

Key features:
- 4-bit quantization for memory efficiency
- LoRA/PEFT for parameter-efficient fine-tuning
- Train on responses only (mask prompts)
- Gemma-3 chat template support
"""

from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only, standardize_data_formats
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import torch
import yaml
import os

# Load configuration
print("Loading configuration...")
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Extract config
dataset_config = config["dataset"]
model_config = config["model"]
lora_config = config["lora"]
training_config = config["training"]

# Load model and tokenizer
print(f"Loading model: {model_config['name']}...")
model, tokenizer = FastModel.from_pretrained(
    model_name=model_config["name"],
    max_seq_length=model_config["max_seq_length"],
    load_in_4bit=model_config["load_in_4bit"],
    load_in_8bit=model_config.get("load_in_8bit", False),
    full_finetuning=False,
)

# Prepare model for fine-tuning with LoRA
print("Preparing model for fine-tuning with LoRA...")
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,  # Text-only
    finetune_language_layers=True,
    finetune_attention_modules=lora_config["finetune_attention_modules"],
    finetune_mlp_modules=lora_config["finetune_mlp_modules"],
    r=lora_config["r"],
    lora_alpha=lora_config["alpha"],
    lora_dropout=lora_config["dropout"],
    bias="none",
    random_state=training_config["seed"],
)

# Set up Gemma-3 chat template
print("Setting up Gemma-3 chat template...")
tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma-3",
)

# Load dataset
print(f"Loading dataset: {dataset_config['path']}...")
dataset = load_dataset(dataset_config["path"], split=dataset_config.get("split", "train"))

# Limit dataset size if specified
if "max_samples" in dataset_config:
    max_samples = dataset_config["max_samples"]
    dataset = dataset.select(range(min(max_samples, len(dataset))))
    print(f"Limited dataset to {len(dataset)} samples")

# Standardize data formats (handle various conversation formats)
print("Standardizing dataset format...")
dataset = standardize_data_formats(dataset)

# Format prompts using Gemma-3 chat template
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = []
    for convo in convos:
        text = tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False
        ).removeprefix('<bos>')
        texts.append(text)
    return {"text": texts}

print("Formatting prompts...")
dataset = dataset.map(formatting_prompts_func, batched=True)

# Split for validation if needed
if training_config.get("do_eval", False):
    val_ratio = training_config.get("val_split_ratio", 0.1)
    dataset = dataset.train_test_split(test_size=val_ratio, seed=training_config["seed"])
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]
    print(f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} val")
else:
    train_dataset = dataset
    val_dataset = None

# Training configuration
training_args = SFTConfig(
    dataset_text_field="text",
    per_device_train_batch_size=training_config["per_device_train_batch_size"],
    gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
    warmup_steps=training_config["warmup_steps"],
    max_steps=training_config.get("max_steps", -1),
    num_train_epochs=training_config.get("num_train_epochs", 1) if training_config.get("max_steps", -1) == -1 else 1,
    learning_rate=training_config["learning_rate"],
    logging_steps=training_config["logging_steps"],
    optim=training_config["optim"],
    weight_decay=training_config["weight_decay"],
    lr_scheduler_type=training_config["lr_scheduler_type"],
    seed=training_config["seed"],
    output_dir=training_config["output_dir"],
    save_strategy="steps",
    save_steps=training_config["save_steps"],
    eval_strategy="steps" if val_dataset else "no",
    eval_steps=training_config.get("val_steps", training_config["save_steps"]) if val_dataset else None,
    report_to="none",  # Can be changed to "wandb", "tensorboard", etc.
)

# Create trainer
print("Initializing SFT trainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=training_args,
)

# Enable training on responses only (mask instruction/input parts)
print("Configuring train_on_responses_only...")
trainer = train_on_responses_only(
    trainer,
    instruction_part="<start_of_turn>user\n",
    response_part="<start_of_turn>model\n",
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

# Save the final model
output_dir = training_config["output_dir"]
final_model_dir = f"{output_dir}/final_model"
print(f"Saving model to {final_model_dir}...")
model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)

print("\nModel saved successfully!")
print(f"To use: FastModel.from_pretrained('{final_model_dir}')")

# Post-training evaluation metrics
if val_dataset:
    print(f"\n{'='*80}")
    print("EVALUATION METRICS")
    print('='*80)

    # Validation loss and perplexity (from trainer)
    if hasattr(trainer.state, 'log_history'):
        eval_logs = [log for log in trainer.state.log_history if 'eval_loss' in log]
        if eval_logs:
            final_eval = eval_logs[-1]
            eval_loss = final_eval.get('eval_loss', None)
            if eval_loss:
                perplexity = torch.exp(torch.tensor(eval_loss)).item()
                print(f"Validation Loss: {eval_loss:.4f}")
                print(f"Perplexity: {perplexity:.2f}")

    # ROUGE-L computation (optional, requires rouge_score package)
    try:
        from rouge_score import rouge_scorer
        import numpy as np

        print("\nComputing ROUGE-L scores on validation set (sample)...")
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        # Sample a subset for faster evaluation
        eval_sample_size = min(100, len(val_dataset))
        eval_sample = val_dataset.select(range(eval_sample_size))

        rouge_scores = []
        for example in eval_sample:
            # Extract reference text (ground truth response)
            convo = example['conversations']
            reference = next((turn['value'] for turn in convo if turn['from'] == 'assistant'), "")

            # Generate prediction
            prompt = tokenizer.apply_chat_template(
                [turn for turn in convo if turn['from'] != 'assistant'],
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to("cuda")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                )
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            prediction = prediction.split("<start_of_turn>model\n")[-1].split("<end_of_turn>")[0].strip()

            # Compute ROUGE-L
            if reference and prediction:
                score = scorer.score(reference, prediction)['rougeL'].fmeasure
                rouge_scores.append(score)

        if rouge_scores:
            avg_rouge = np.mean(rouge_scores)
            print(f"ROUGE-L (n={len(rouge_scores)}): {avg_rouge:.4f}")

    except ImportError:
        print("\nROUGE-L computation skipped (install rouge_score: pip install rouge-score)")
    except Exception as e:
        print(f"\nROUGE-L computation failed: {e}")

    print('='*80)
