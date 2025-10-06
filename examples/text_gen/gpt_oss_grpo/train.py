from unsloth import FastLanguageModel
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
import torch
import yaml
from reward_functions import (
    function_works,
    no_cheating,
    correctness_check,
    speed_check
)

# Load configuration
print("Loading configuration...")
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Extract config values
model_config = config["model"]
lora_config = config["lora"]
grpo_config = config["grpo"]
task_config = config["task"]

# Load model and tokenizer
print(f"Loading model: {model_config['name']}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_config["name"],
    max_seq_length=model_config["max_seq_length"],
    dtype=None,  # Auto-detect
    load_in_4bit=model_config["load_in_4bit"],
    offload_embedding=True,  # Reduces VRAM by 1GB
)

# Prepare model for GRPO with LoRA
print("Preparing model for GRPO training...")
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_config["r"],
    target_modules=lora_config["target_modules"],
    lora_alpha=lora_config["alpha"],
    lora_dropout=lora_config["dropout"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=grpo_config["seed"],
)

# Prepare dataset from config
print("Preparing dataset...")
prompt_text = task_config["prompt"].strip()

# Create dataset with single prompt repeated (matching exact Unsloth notebook format)
dataset_examples = [{
    "prompt": [{"role": "user", "content": prompt_text}],
    "answer": 0,
    "reasoning_effort": "low"  # Required for GPT-OSS GRPO
}] * 1000  # Replicate single prompt 1000 times

dataset = Dataset.from_list(dataset_examples)

print(f"Created dataset with {len(dataset)} examples")

# Reward functions are used directly from reward_functions.py
# No need for a combined reward function - pass list of functions to GRPOTrainer

# GRPO training configuration
print("Setting up GRPO trainer...")
# Calculate max prompt and completion lengths (required for GRPO)
max_seq_length = model_config["max_seq_length"]
max_prompt_length = max_seq_length // 2  # Reserve half for prompt
max_completion_length = max_seq_length - max_prompt_length  # Rest for completion

training_args = GRPOConfig(
    output_dir=grpo_config["output_dir"],
    beta=grpo_config.get("beta", 0.1),  # KL penalty coefficient
    temperature=grpo_config["temperature"],
    learning_rate=grpo_config["learning_rate"],
    weight_decay=grpo_config["weight_decay"],
    num_generations=grpo_config["num_generations"],
    max_steps=grpo_config["max_steps"],
    per_device_train_batch_size=grpo_config["per_device_train_batch_size"],
    gradient_accumulation_steps=grpo_config["gradient_accumulation_steps"],
    warmup_steps=grpo_config["warmup_steps"],
    logging_steps=grpo_config["logging_steps"],
    save_steps=grpo_config["save_steps"],
    seed=grpo_config["seed"],
    max_prompt_length=max_prompt_length,  # Required for GRPO
    max_completion_length=max_completion_length,  # Required for GRPO
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
)

# Create GRPO trainer
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        function_works,
        no_cheating,
        correctness_check,
        speed_check,
    ],
    args=training_args,
    train_dataset=dataset,
)

# Train the model
print("Starting GRPO training...")
print(f"Training on {len(dataset)} examples for {grpo_config['max_steps']} steps")
print("Reward functions: function_works, no_cheating, correctness_check, speed_check")
trainer.train()

# Save the model
output_dir = grpo_config["output_dir"]
final_model_dir = f"{output_dir}/final_model"
print(f"Saving model to {final_model_dir}...")
model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)

print("GRPO training complete!")
