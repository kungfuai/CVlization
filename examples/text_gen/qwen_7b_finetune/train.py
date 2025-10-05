from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import torch

# Model configuration
max_seq_length = 2048  # Qwen supports up to 40960
dtype = None  # Auto-detect dtype
load_in_4bit = True

# Load Qwen 2.5 7B model
print("Loading Qwen 2.5 7B model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-7B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    full_finetuning = False,
)

# Add LoRA adapters
print("Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # LoRA rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# Load dataset - using Alpaca for consistency with other examples
print("Loading dataset...")
dataset = load_dataset("yahma/alpaca-cleaned", split="train")

# Format dataset for Qwen's chat template
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []

    for instruction, input_text, output in zip(instructions, inputs, outputs):
        # Combine instruction and input
        if input_text:
            message = f"{instruction}\n{input_text}"
        else:
            message = instruction

        # Format as chat messages for Qwen
        messages = [
            {"role": "user", "content": message},
            {"role": "assistant", "content": output}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)

    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

# Training arguments
print("Setting up training...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 20,  # Quick test - increase for real training
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "qwen-7b-finetune",
        report_to = "none",
        save_steps = 10,
    ),
)

# Train
print("Starting training...")
trainer.train()

# Save model
print("Saving model...")
model.save_pretrained("qwen-7b-finetune/final_model")
tokenizer.save_pretrained("qwen-7b-finetune/final_model")

print("Training complete!")
