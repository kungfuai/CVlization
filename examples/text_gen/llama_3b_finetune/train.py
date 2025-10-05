from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
import torch

# Model configuration
max_seq_length = 2048
dtype = None  # Auto-detect dtype
load_in_4bit = True

# Load model and tokenizer
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Load dataset - using Alpaca format for simplicity
print("Loading dataset...")
dataset = load_dataset("yahma/alpaca-cleaned", split="train")

# Format dataset for instruction tuning
def format_prompt(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]

    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        # Combine instruction and input
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        texts.append(prompt)

    return {"text": texts}

# Process dataset
dataset = dataset.map(format_prompt, batched=True, remove_columns=dataset.column_names)

# Split dataset for training and validation
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# Prepare model for fine-tuning with LoRA
print("Preparing model for fine-tuning...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # LoRA rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 42,
    use_rslora = False,
    loftq_config = None,
)

# Training arguments
training_args = TrainingArguments(
    output_dir = "./llama-alpaca-finetune",
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    warmup_steps = 2,
    max_steps = 20,  # Small number for quick test
    learning_rate = 2e-4,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 1,  # Log every step to observe loss
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = 42,
    save_strategy = "steps",
    save_steps = 10,
    eval_strategy = "steps",
    eval_steps = 10,
    do_eval = True,
)

# Create trainer
print("Initializing trainer...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset = dataset["test"],
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    args = training_args,
)

# Train the model
print("Starting training...")
trainer.train()

# Save the model
print("Saving model...")
model.save_pretrained("./llama-alpaca-finetune/final_model")
tokenizer.save_pretrained("./llama-alpaca-finetune/final_model")

print("Training complete!")
