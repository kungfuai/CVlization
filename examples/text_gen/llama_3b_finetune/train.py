from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
import torch
import yaml
import sys

# Load configuration
print("Loading configuration...")
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Extract config values
dataset_config = config["dataset"]
model_config = config["model"]
lora_config = config["lora"]
training_config = config["training"]

# Load model and tokenizer
print(f"Loading model: {model_config['name']}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_config["name"],
    max_seq_length=model_config["max_seq_length"],
    dtype=None,  # Auto-detect
    load_in_4bit=model_config["load_in_4bit"],
)

# Load dataset
print(f"Loading dataset: {dataset_config['path']}...")
dataset_path = dataset_config["path"]
dataset_split = dataset_config.get("split", "train")
dataset = load_dataset(dataset_path, split=dataset_split)

# Limit dataset size if specified
if "max_samples" in dataset_config:
    max_samples = dataset_config["max_samples"]
    dataset = dataset.select(range(min(max_samples, len(dataset))))
    print(f"Limited dataset to {len(dataset)} samples")

# Format dataset based on format type
dataset_format = dataset_config["format"]

if dataset_format == "alpaca":
    print("Formatting dataset in Alpaca format...")
    def format_alpaca(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]

        texts = []
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            texts.append(prompt)

        return {"text": texts}

    dataset = dataset.map(format_alpaca, batched=True, remove_columns=dataset.column_names)

elif dataset_format == "sharegpt":
    print("Formatting dataset in ShareGPT format...")
    def format_sharegpt(examples):
        conversations = examples["conversations"]
        texts = []
        for convo in conversations:
            text = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(format_sharegpt, batched=True, remove_columns=dataset.column_names)

elif dataset_format == "custom":
    print("Using custom format (expecting 'text' column)...")
    if "text" not in dataset.column_names:
        raise ValueError("Custom format requires a 'text' column in the dataset")
else:
    raise ValueError(f"Unknown dataset format: {dataset_format}")

# Split dataset for training and validation if eval is enabled
if training_config.get("do_eval", False):
    eval_split_ratio = training_config.get("eval_split_ratio", 0.1)
    dataset = dataset.train_test_split(test_size=eval_split_ratio, seed=training_config["seed"])
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
else:
    train_dataset = dataset
    eval_dataset = None

# Prepare model for fine-tuning with LoRA
print("Preparing model for fine-tuning...")
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_config["r"],
    target_modules=lora_config["target_modules"],
    lora_alpha=lora_config["alpha"],
    lora_dropout=lora_config["dropout"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=training_config["seed"],
    use_rslora=False,
    loftq_config=None,
)

# Determine max_steps or num_train_epochs
max_steps = training_config.get("max_steps", -1)
num_train_epochs = 1 if max_steps != -1 else training_config.get("num_epochs", 1)

# Training arguments
training_args = TrainingArguments(
    output_dir=training_config["output_dir"],
    per_device_train_batch_size=training_config["per_device_train_batch_size"],
    gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
    warmup_steps=training_config["warmup_steps"],
    max_steps=max_steps,
    num_train_epochs=num_train_epochs,
    learning_rate=training_config["learning_rate"],
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=training_config["logging_steps"],
    optim=training_config["optim"],
    weight_decay=training_config["weight_decay"],
    lr_scheduler_type=training_config["lr_scheduler_type"],
    seed=training_config["seed"],
    save_strategy="steps",
    save_steps=training_config["save_steps"],
    eval_strategy="steps" if eval_dataset else "no",
    eval_steps=training_config.get("eval_steps", training_config["save_steps"]) if eval_dataset else None,
    do_eval=eval_dataset is not None,
)

# Create trainer
print("Initializing trainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=model_config["max_seq_length"],
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    args=training_args,
)

# Train the model
print("Starting training...")
trainer.train()

# Save the model
output_dir = training_config["output_dir"]
final_model_dir = f"{output_dir}/final_model"
print(f"Saving model to {final_model_dir}...")
model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)

print("Training complete!")
