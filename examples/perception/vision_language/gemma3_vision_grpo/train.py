#!/usr/bin/env python3
"""
Gemma-3 Vision GRPO (Reinforcement Learning)

Fine-tune Gemma-3 vision model with GRPO for math visual reasoning.
Based on: https://github.com/unslothai/notebooks/blob/main/nb/Gemma3_(4B)-Vision-GRPO.ipynb

Key features:
- Group Relative Policy Optimization (GRPO) for RL training
- Math visual reasoning with MathVista dataset
- Custom reward functions (formatting + correctness)
- Does NOT finetune vision layers (only language)
- Requires vLLM for faster inference
"""

from unsloth import FastVisionModel
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import torch
import yaml
import re
import os

# Set environment variable for vLLM standby mode
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"  # Extra 30% context lengths

# Load configuration
print("Loading configuration...")
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Extract config
dataset_config = config["dataset"]
model_config = config["model"]
lora_config = config["lora"]
training_config = config["training"]
reward_config = config["reward"]

# Load model and tokenizer
print(f"Loading model: {model_config['name']}...")
model, tokenizer = FastVisionModel.from_pretrained(
    model_config["name"],
    load_in_4bit=model_config["load_in_4bit"],
    use_gradient_checkpointing=model_config.get("use_gradient_checkpointing", "unsloth"),
)

# Prepare model for fine-tuning with LoRA
# NOTE: For GRPO, we do NOT finetune vision layers
print("Preparing model for fine-tuning with LoRA (language only)...")
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=lora_config["finetune_vision_layers"],  # False for GRPO
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
    use_gradient_checkpointing=model_config.get("use_gradient_checkpointing", "unsloth"),
)

# Load dataset
print(f"Loading dataset: {dataset_config['path']}...")
dataset = load_dataset(dataset_config["path"], split=dataset_config.get("split", "testmini"))

# Filter for numeric answers only
def is_numeric_answer(example):
    try:
        float(example["answer"])
        return True
    except:
        return False

print("Filtering for numeric answers...")
dataset = dataset.filter(is_numeric_answer)

# Resize images to reduce memory
def resize_images(example):
    image = example["decoded_image"]
    target_size = dataset_config.get("image_size", 512)
    image = image.resize((target_size, target_size))
    example["decoded_image"] = image
    return example

print(f"Resizing images to {dataset_config.get('image_size', 512)}x{dataset_config.get('image_size', 512)}...")
dataset = dataset.map(resize_images)

# Convert to RGB
def convert_to_rgb(example):
    image = example["decoded_image"]
    if image.mode != "RGB":
        image = image.convert("RGB")
    example["decoded_image"] = image
    return example

dataset = dataset.map(convert_to_rgb)

# Limit dataset size if specified
if "max_samples" in dataset_config:
    max_samples = dataset_config["max_samples"]
    dataset = dataset.select(range(min(max_samples, len(dataset))))
    print(f"Limited dataset to {len(dataset)} samples")

print(f"Final dataset size: {len(dataset)}")

# Define delimiter variables for reasoning
REASONING_START = reward_config["reasoning_start"]
REASONING_END = reward_config["reasoning_end"]
SOLUTION_START = reward_config["solution_start"]
SOLUTION_END = reward_config["solution_end"]

# Convert to conversation format with reasoning delimiters
def make_conversation(example):
    text_content = (
        f"{example['question']}, provide your reasoning between {REASONING_START} and {REASONING_END} "
        f"and then your final answer between {SOLUTION_START} and (put a float here) {SOLUTION_END}"
    )

    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text_content},
            ],
        },
    ]

    return {"prompt": prompt, "image": example["decoded_image"], "answer": example["answer"]}

print("Converting dataset to conversation format...")
train_dataset = dataset.map(make_conversation)

# Rename image column
train_dataset = train_dataset.remove_columns("image")
train_dataset = train_dataset.rename_column("decoded_image", "image")

# Apply chat template to prompts
train_dataset = train_dataset.map(
    lambda example: {
        "prompt": tokenizer.apply_chat_template(
            example["prompt"],
            tokenize=False,
            add_generation_prompt=False
        )
    }
)

# Define reward functions
def formatting_reward_func(completions, **kwargs):
    """Reward for proper formatting with reasoning and solution delimiters."""
    thinking_pattern = f'{REASONING_START}(.*?){REASONING_END}'
    answer_pattern = f'{SOLUTION_START}(.*?){SOLUTION_END}'

    scores = []
    for completion in completions:
        score = 0
        thinking_matches = re.findall(thinking_pattern, completion, re.DOTALL)
        answer_matches = re.findall(answer_pattern, completion, re.DOTALL)

        # Reward for exactly one reasoning block
        if len(thinking_matches) == 1:
            score += reward_config["formatting_reward"]

        # Reward for exactly one solution block
        if len(answer_matches) == 1:
            score += reward_config["formatting_reward"]

        scores.append(score)

    return scores

def correctness_reward_func(prompts, completions, answer, **kwargs):
    """Reward for correct answers."""
    answer_pattern = f'{SOLUTION_START}(.*?){SOLUTION_END}'

    responses = [re.findall(answer_pattern, completion, re.DOTALL) for completion in completions]
    q = prompts[0]

    # Print first example for debugging
    print('-' * 20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:{completions[0]}")

    # Reward for correct answer
    scores = []
    for r, a in zip(responses, answer):
        if len(r) == 1 and a == r[0].replace('\n', ''):
            scores.append(reward_config["correctness_reward"])
        else:
            scores.append(0.0)

    return scores

# Training configuration
print("Initializing GRPO trainer...")
training_args = GRPOConfig(
    learning_rate=training_config["learning_rate"],
    adam_beta1=training_config.get("adam_beta1", 0.9),
    adam_beta2=training_config.get("adam_beta2", 0.99),
    weight_decay=training_config["weight_decay"],
    warmup_ratio=training_config.get("warmup_ratio", 0.1),
    lr_scheduler_type=training_config["lr_scheduler_type"],
    optim=training_config["optim"],
    logging_steps=training_config["logging_steps"],
    per_device_train_batch_size=training_config["per_device_train_batch_size"],
    gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
    num_generations=training_config.get("num_generations", 4),
    max_prompt_length=training_config.get("max_prompt_length", 1024),
    max_completion_length=training_config.get("max_completion_length", 1024),
    importance_sampling_level=training_config.get("importance_sampling_level", "sequence"),
    mask_truncated_completions=training_config.get("mask_truncated_completions", False),
    loss_type=training_config.get("loss_type", "dr_grpo"),
    max_steps=training_config.get("max_steps", -1),
    save_steps=training_config["save_steps"],
    max_grad_norm=training_config.get("max_grad_norm", 0.1),
    report_to="none",
    output_dir=training_config["output_dir"],
)

# Create trainer
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    reward_funcs=[formatting_reward_func, correctness_reward_func],
    train_dataset=train_dataset,
)

# Show memory stats before training
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"\nGPU: {gpu_stats.name}")
print(f"Max memory: {max_memory} GB")
print(f"Reserved memory: {start_gpu_memory} GB\n")

# Train the model
print("Starting GRPO training...")
trainer.train()

print(f"\n{'='*80}")
print("TRAINING COMPLETE")
print('='*80)

# Test inference after training
if training_config.get("test_after_training", True):
    print("Testing inference after training...")
    FastVisionModel.for_inference(model)

    test_idx = min(100, len(dataset) - 1)
    image = dataset[test_idx]["decoded_image"]
    instruction = (
        f"{dataset[test_idx]['question']}, provide your reasoning between {REASONING_START} and {REASONING_END} "
        f"and then your final answer between {SOLUTION_START} and (put a float here) {SOLUTION_END}"
    )

    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction}],
        }
    ]

    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(input_text, add_special_tokens=False, return_tensors="pt").to("cuda")

    from transformers import TextStreamer
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=256,
                      use_cache=True, temperature=1.0, top_p=0.95, top_k=64)
    print("\nPost-training test complete.\n")

# Save the final model
output_dir = training_config["output_dir"]
final_model_dir = f"{output_dir}/final_model"
print(f"Saving model to {final_model_dir}...")
model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)

print("\nModel saved successfully!")
print(f"To use: FastVisionModel.from_pretrained('{final_model_dir}')")
