from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
import torch
import yaml

# Load configuration
print("Loading configuration...")
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

dataset_config = config["dataset"]
model_config = config["model"]
lora_config = config["lora"]
training_config = config["training"]

# Load model with 4-bit quantization
print(f"Loading model: {model_config['name']}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=model_config["load_in_4bit"],
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_config["name"],
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_config["name"],
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load dataset
print(f"Loading dataset: {dataset_config['path']}...")
dataset = load_dataset(dataset_config["path"], split=dataset_config["split"])

# Limit dataset size if specified
if "max_samples" in dataset_config:
    dataset = dataset.select(range(min(dataset_config["max_samples"], len(dataset))))

# Format dataset based on format type
print(f"Formatting dataset as {dataset_config['format']}...")

def format_alpaca(examples):
    """Format Alpaca-style dataset."""
    texts = []
    for instruction, input_text, output in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        texts.append(prompt)
    return {"text": texts}

def format_sharegpt(examples):
    """Format ShareGPT-style dataset."""
    texts = []
    for conversations in examples["conversations"]:
        formatted = tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(formatted)
    return {"text": texts}

# Apply formatting
if dataset_config["format"] == "alpaca":
    dataset = dataset.map(
        format_alpaca,
        batched=True,
        remove_columns=dataset.column_names,
    )
elif dataset_config["format"] == "sharegpt":
    dataset = dataset.map(
        format_sharegpt,
        batched=True,
        remove_columns=dataset.column_names,
    )
# For "custom" format, assume dataset already has "text" column

print(f"Dataset prepared with {len(dataset)} examples")

# LoRA configuration
print("Setting up LoRA...")
peft_config = LoraConfig(
    r=lora_config["r"],
    lora_alpha=lora_config["alpha"],
    lora_dropout=lora_config["dropout"],
    target_modules=lora_config["target_modules"],
    bias="none",
    task_type="CAUSAL_LM",
)

# Training configuration
print("Setting up trainer...")
training_args = SFTConfig(
    output_dir=training_config["output_dir"],
    max_steps=training_config["max_steps"],
    num_train_epochs=training_config["num_epochs"],
    per_device_train_batch_size=training_config["per_device_train_batch_size"],
    gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
    learning_rate=training_config["learning_rate"],
    warmup_steps=training_config["warmup_steps"],
    lr_scheduler_type=training_config["lr_scheduler_type"],
    optim=training_config["optim"],
    weight_decay=training_config["weight_decay"],
    logging_steps=training_config["logging_steps"],
    save_steps=training_config["save_steps"],
    seed=training_config["seed"],
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    dataset_text_field="text",
    packing=False,
)

# Create SFT trainer
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
)

# Train the model
print("Starting training...")
trainer.train()

# Save the model
output_dir = training_config["output_dir"]
final_model_dir = f"{output_dir}/final_model"
print(f"Saving model to {final_model_dir}...")
trainer.save_model(final_model_dir)

print("Training complete!")
