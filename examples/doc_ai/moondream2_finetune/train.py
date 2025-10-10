"""
Fine-tune Moondream2 for document understanding.
Based on official Moondream finetune_text.py approach.
"""

import argparse
import json
import math
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def lr_schedule(step, max_steps, base_lr):
    """Cosine learning rate schedule with warmup."""
    x = step / max_steps
    if x < 0.1:
        return 0.1 * base_lr + 0.9 * base_lr * x / 0.1
    else:
        return 0.1 * base_lr + 0.9 * base_lr * (1 + math.cos(math.pi * (x - 0.1))) / 2


class DocumentQADataset(Dataset):
    """Dataset for document Q&A fine-tuning."""

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.samples = []

        # Load JSONL data
        if self.data_path.suffix == '.jsonl':
            with open(self.data_path, 'r') as f:
                for line in f:
                    self.samples.append(json.loads(line))
        else:
            with open(self.data_path, 'r') as f:
                self.samples = json.load(f)

        print(f"Loaded {len(self.samples)} samples from {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        # Load image
        image_path = sample.get('image') or sample.get('image_path')
        image = Image.open(image_path).convert('RGB')

        # Get Q&A
        question = sample.get('question') or sample.get('prompt', '')
        answer = sample.get('answer') or sample.get('response', '')

        return {
            'image': image,
            'question': question,
            'answer': answer
        }


def evaluate_accuracy(model, tokenizer, dataset, device, num_samples=None, use_bidirectional_image_attn=False):
    """
    Evaluate accuracy by generating answers and comparing to ground truth.
    This is slow but gives true accuracy.
    """
    model.eval()

    if num_samples is None:
        num_samples = len(dataset)
    else:
        num_samples = min(num_samples, len(dataset))

    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(num_samples):
            sample = dataset[i]

            # Encode image
            img_emb = model.encode_image(sample['image'])
            if img_emb.device != device:
                img_emb = img_emb.to(device)

            # Create prompt
            question_text = f"\n\nQuestion: {sample['question']}\n\nAnswer:"
            question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
            bos_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
            question_ids = torch.tensor([question_tokens], device=device)

            # Get embeddings
            if hasattr(model.text_model, 'model'):
                text_embed_layer = model.text_model.model.embed_tokens
            else:
                text_embed_layer = model.text_model.get_input_embeddings()

            bos_emb = text_embed_layer(bos_ids)
            question_emb = text_embed_layer(question_ids)

            # Concatenate: [BOS] + [IMG] + [QUESTION]
            inputs_embeds = torch.cat([
                bos_emb,
                img_emb.unsqueeze(0) if img_emb.dim() == 2 else img_emb,
                question_emb
            ], dim=1)

            # Generate tokens (simple greedy decoding, max 20 tokens)
            generated_ids = []

            if use_bidirectional_image_attn:
                # Setup for custom attention mask with 730-token boundary
                seq_len = inputs_embeds.shape[1]
                img_len = min(730, img_emb.shape[0] if img_emb.dim() == 2 else img_emb.shape[1])
                boundary = 1 + img_len

            for _ in range(20):
                if use_bidirectional_image_attn:
                    # Build attention mask for current sequence length
                    curr_seq_len = inputs_embeds.shape[1]
                    attn_mask_2d = torch.zeros(curr_seq_len, curr_seq_len, dtype=torch.float16, device=device)
                    attn_mask_2d[:boundary, :boundary] = 1
                    for i in range(boundary, curr_seq_len):
                        attn_mask_2d[i, :i + 1] = 1
                    attn_mask = attn_mask_2d.unsqueeze(0).unsqueeze(0)

                    outputs = model.text_model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attn_mask,
                        use_cache=False
                    )
                else:
                    outputs = model.text_model(inputs_embeds=inputs_embeds, use_cache=False)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = next_token_logits.argmax(dim=-1)

                # Stop if EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break

                generated_ids.append(next_token.item())

                # Add token embedding for next iteration
                next_token_emb = text_embed_layer(next_token.unsqueeze(0))
                inputs_embeds = torch.cat([inputs_embeds, next_token_emb], dim=1)

            # Decode generated answer
            generated = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            expected = sample['answer'].strip()

            if generated == expected:
                correct += 1

            total += 1

            # Print first few examples
            if i < 3:
                match = "✓" if generated == expected else "✗"
                print(f"  {match} Q: {sample['question'][:50]}...")
                print(f"    Expected: '{expected}' | Got: '{generated}'")

    accuracy = correct / total if total > 0 else 0
    return accuracy


def train(model, tokenizer, train_dataset, val_dataset, optimizer, args):
    """Training loop with accuracy tracking."""

    total_steps = args.epochs * len(train_dataset) // args.grad_accum
    pbar = tqdm(total=total_steps, desc="Training")

    # Track metrics
    running_loss = 0
    running_token_acc = 0
    num_updates = 0

    # Smoothed metrics (exponential moving average)
    smooth_loss = None
    smooth_token_acc = None
    smoothing = 0.9  # EMA coefficient

    step = 0
    for epoch in range(args.epochs):
        model.text_model.train()
        model.vision_encoder.eval()

        for i, sample in enumerate(train_dataset):
            step += 1

            # Encode image (no gradients for vision encoder)
            with torch.no_grad():
                img_emb = model.encode_image(sample['image'])
                if img_emb.device != args.device:
                    img_emb = img_emb.to(args.device)

            # Tokenize question and answer
            question_text = f"\n\nQuestion: {sample['question']}\n\nAnswer:"
            answer_text = f"{sample['answer']}<|endoftext|>"

            question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
            answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False)

            # Convert to tensors
            question_ids = torch.tensor([question_tokens], device=args.device)
            answer_ids = torch.tensor([answer_tokens], device=args.device)
            bos_ids = torch.tensor([[tokenizer.bos_token_id]], device=args.device)

            # Get embeddings from text model
            if hasattr(model.text_model, 'model'):
                text_embed_layer = model.text_model.model.embed_tokens
            else:
                text_embed_layer = model.text_model.get_input_embeddings()

            bos_emb = text_embed_layer(bos_ids)
            question_emb = text_embed_layer(question_ids)
            answer_emb = text_embed_layer(answer_ids)

            # Concatenate: [BOS] + [IMG] + [QUESTION] + [ANSWER]
            inputs_embeds = torch.cat([
                bos_emb,
                img_emb.unsqueeze(0) if img_emb.dim() == 2 else img_emb,
                question_emb,
                answer_emb
            ], dim=1)

            # Forward through text model
            if args.use_bidirectional_image_attn:
                # Create custom attention mask with 730-token boundary
                # First 730 tokens (image embeddings) use bidirectional attention
                # Remaining tokens (text) use causal attention
                seq_len = inputs_embeds.shape[1]

                # Bidirectional for first 730 tokens (or actual image length if smaller)
                img_len = min(730, img_emb.shape[0] if img_emb.dim() == 2 else img_emb.shape[1])
                boundary = 1 + img_len  # 1 for BOS + image tokens

                # Build 2D attention mask then expand to 4D
                attn_mask_2d = torch.zeros(seq_len, seq_len, dtype=torch.float16, device=args.device)
                attn_mask_2d[:boundary, :boundary] = 1  # Bidirectional for image
                for i in range(boundary, seq_len):
                    attn_mask_2d[i, :i + 1] = 1  # Causal for text

                # HuggingFace expects shape [batch_size, 1, seq_len, seq_len]
                attn_mask = attn_mask_2d.unsqueeze(0).unsqueeze(0)

                outputs = model.text_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attn_mask,
                    use_cache=False
                )
            else:
                # Standard causal attention for all tokens
                outputs = model.text_model(inputs_embeds=inputs_embeds, use_cache=False)
            lm_logits = outputs.logits

            # Compute loss: only on answer tokens
            seq_len = inputs_embeds.shape[1]
            answer_len = answer_emb.shape[1]
            shift_index = seq_len - answer_len - 1

            shifted_logits = lm_logits[..., shift_index:-1, :].contiguous()
            shifted_labels = answer_ids.contiguous()

            loss = nn.CrossEntropyLoss()(
                shifted_logits.view(-1, shifted_logits.size(-1)),
                shifted_labels.view(-1)
            )

            # Calculate token-level accuracy
            with torch.no_grad():
                predicted_tokens = shifted_logits.argmax(dim=-1)
                token_correct = (predicted_tokens == shifted_labels).float().sum()
                token_total = shifted_labels.numel()
                token_acc = token_correct / token_total if token_total > 0 else 0

            # Backward pass
            loss.backward()

            # Track metrics
            running_loss += loss.item()
            running_token_acc += token_acc.item()

            # Update weights with gradient accumulation
            if step % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                num_updates += 1

                # Update learning rate
                lr = lr_schedule(step // args.grad_accum, total_steps, args.lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                # Average metrics
                avg_loss = running_loss / args.grad_accum
                avg_token_acc = running_token_acc / args.grad_accum

                # Update smoothed metrics (exponential moving average)
                if smooth_loss is None:
                    smooth_loss = avg_loss
                    smooth_token_acc = avg_token_acc
                else:
                    smooth_loss = smoothing * smooth_loss + (1 - smoothing) * avg_loss
                    smooth_token_acc = smoothing * smooth_token_acc + (1 - smoothing) * avg_token_acc

                pbar.set_postfix({
                    'epoch': epoch + 1,
                    'loss': f'{smooth_loss:.4f}',
                    'token_acc': f'{smooth_token_acc:.3f}',
                    'lr': f'{lr:.2e}'
                })
                pbar.update(1)

                # Reset running metrics
                running_loss = 0
                running_token_acc = 0

                # Validate every N steps
                if args.eval_steps > 0 and num_updates % args.eval_steps == 0:
                    print(f"\n=== Validation at step {num_updates} ===")
                    val_acc = evaluate_accuracy(
                        model, tokenizer, val_dataset, args.device,
                        num_samples=args.eval_samples,
                        use_bidirectional_image_attn=args.use_bidirectional_image_attn
                    )
                    print(f"Validation Accuracy: {val_acc:.1%} ({int(val_acc * min(args.eval_samples, len(val_dataset)))}/{min(args.eval_samples, len(val_dataset))} correct)")

                    # Back to training mode
                    model.text_model.train()

    pbar.close()

    # Final validation
    if val_dataset and len(val_dataset) > 0:
        print(f"\n=== Final Validation ===")
        val_acc = evaluate_accuracy(
            model, tokenizer, val_dataset, args.device,
            num_samples=args.eval_samples,
            use_bidirectional_image_attn=args.use_bidirectional_image_attn
        )
        print(f"Final Validation Accuracy: {val_acc:.1%}")
        return val_acc

    return 0.0


def main():
    parser = argparse.ArgumentParser(description='Fine-tune Moondream2 for document understanding')
    parser.add_argument('--model_id', default='vikhyatk/moondream2', help='Moondream model ID')
    parser.add_argument('--data', required=True, help='Path to training data (JSONL)')
    parser.add_argument('--val_data', default=None, help='Path to validation data (JSONL)')
    parser.add_argument('--output_dir', default='./outputs/moondream2_ft', help='Output directory')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs (official: 3)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (always 1)')
    parser.add_argument('--lr', type=float, default=3e-6, help='Learning rate (official: 3e-6)')
    parser.add_argument('--grad_accum', type=int, default=8, help='Gradient accumulation steps (tuned for dataset size)')
    parser.add_argument('--eval_steps', type=int, default=50, help='Evaluate every N steps (0 to disable)')
    parser.add_argument('--eval_samples', type=int, default=20, help='Number of validation samples to evaluate')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_length', type=int, default=2048, help='Max sequence length')
    parser.add_argument('--use_bidirectional_image_attn', action='store_true',
                        help='Use 730-token bidirectional attention for image tokens (experimental)')

    args = parser.parse_args()

    print("Training Moondream2 fine-tuning")
    print(f"Model: {args.model_id}")
    print(f"Training data: {args.data}")
    print(f"Validation data: {args.val_data or 'None'}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Gradient accumulation: {args.grad_accum}")
    print(f"Eval every: {args.eval_steps} steps")
    print(f"Bidirectional image attention: {args.use_bidirectional_image_attn}")
    print()

    # Set default device
    if args.device == 'cuda' and torch.cuda.is_available():
        torch.set_default_device('cuda')
        args.device = 'cuda'
    else:
        args.device = 'cpu'

    # Load model and tokenizer
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        revision='2024-07-23',  # Stable revision
        torch_dtype=torch.float16  # Use FP16 for memory efficiency
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Freeze vision encoder, only train text model
    print("Setting up optimizer...")
    for param in model.vision_encoder.parameters():
        param.requires_grad = False

    trainable_params = [p for p in model.text_model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-6
    )

    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Load datasets
    print("Loading datasets...")
    train_dataset = DocumentQADataset(args.data)

    val_dataset = None
    if args.val_data:
        val_dataset = DocumentQADataset(args.val_data)

    print()

    # Train
    final_acc = train(model, tokenizer, train_dataset, val_dataset, optimizer, args)

    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n✓ Training complete! Model saved to:", output_dir)
    if final_acc > 0:
        print(f"  Final validation accuracy: {final_acc:.1%}")
    print("\nTo use the fine-tuned model:")
    print(f'  model = AutoModelForCausalLM.from_pretrained("{output_dir}", trust_remote_code=True)')


if __name__ == "__main__":
    main()
