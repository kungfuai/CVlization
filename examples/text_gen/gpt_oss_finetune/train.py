from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import torch

# Model configuration
max_seq_length = 1024  # GPT-OSS supports up to 128k, but start with 1k for testing
dtype = None  # Auto-detect dtype
load_in_4bit = True

# Load model and tokenizer
print("Loading GPT-OSS model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gpt-oss-20b",  # 20B parameter model
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    full_finetuning = False,  # Use LoRA
)

# Load dataset - using Multilingual-Thinking for reasoning tasks
print("Loading dataset...")
dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")

# Format dataset for chat template
def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in convos
    ]
    return {"text": texts}

# Process dataset
print("Processing dataset...")
dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=dataset.column_names)

# Split dataset for training and validation
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# Prepare model for fine-tuning with LoRA
print("Preparing model for fine-tuning...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 8,  # LoRA rank - smaller than Llama due to larger base model
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 42,
)

# Training arguments
training_args = SFTConfig(
    output_dir = "./gpt-oss-finetune",
    per_device_train_batch_size = 1,  # Smaller batch for 20B model on A10
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
    args = training_args,
)

# Train the model
print("Starting training...")
trainer.train()

# Save the model
print("Saving model...")
model.save_pretrained("./gpt-oss-finetune/final_model")
tokenizer.save_pretrained("./gpt-oss-finetune/final_model")

print("Training complete!")
