#!/usr/bin/env python3
"""
OpenVLA Single-Step Inference Demo

Loads the OpenVLA model and runs inference on static images,
visualizing the predicted robot actions.

Usage:
    python inference.py --image path/to/image.jpg --instruction "pick up the red block"
    python inference.py --interactive  # Interactive mode with sample images
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from PIL import Image


def load_model(model_path: str = "openvla/openvla-7b", device: str = "cuda:0"):
    """Load the OpenVLA model and processor."""
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print(f"Loading model: {model_path}")
    print("This may take a few minutes on first run (downloading ~14GB)...")

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    # Try flash attention, fall back to standard if not available
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(device)
        print("Using Flash Attention 2")
    except Exception as e:
        print(f"Flash Attention not available ({e}), using standard attention")
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(device)

    model.eval()
    print("Model loaded successfully!")
    return model, processor


def predict_action(
    model,
    processor,
    image: Image.Image,
    instruction: str,
    unnorm_key: str = "bridge_orig",
    device: str = "cuda:0"
) -> np.ndarray:
    """
    Run single-step inference to predict robot action.

    Args:
        model: OpenVLA model
        processor: OpenVLA processor
        image: PIL Image from camera/file
        instruction: Natural language task instruction
        unnorm_key: Dataset key for action unnormalization
        device: CUDA device

    Returns:
        action: 7-DoF action array [dx, dy, dz, droll, dpitch, dyaw, gripper]
    """
    # Format prompt according to OpenVLA convention
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"

    # Process inputs
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

    # Run inference
    with torch.no_grad():
        action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)

    return action


def visualize_action(
    image: Image.Image,
    action: np.ndarray,
    instruction: str,
    output_path: str = None,
    show: bool = True
):
    """
    Visualize the predicted action overlaid on the input image.

    Args:
        image: Input PIL Image
        action: 7-DoF action array
        instruction: Task instruction (for title)
        output_path: Optional path to save visualization
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Original image with motion arrows
    ax1 = axes[0]
    ax1.imshow(image)
    ax1.set_title(f'Input: "{instruction}"', fontsize=11)
    ax1.axis('off')

    # Draw arrow representing XY motion (centered on image)
    img_w, img_h = image.size
    center_x, center_y = img_w // 2, img_h // 2

    # Scale action for visualization (actions are typically small deltas)
    scale = 200  # pixels per unit action
    dx, dy = action[0] * scale, -action[1] * scale  # flip Y for image coords

    if abs(dx) > 5 or abs(dy) > 5:  # Only draw if motion is significant
        ax1.arrow(
            center_x, center_y, dx, dy,
            head_width=15, head_length=10,
            fc='lime', ec='darkgreen', linewidth=2
        )

    # Add gripper state indicator
    gripper_state = "OPEN" if action[6] > 0.5 else "CLOSED"
    gripper_color = "green" if action[6] > 0.5 else "red"
    ax1.text(
        10, img_h - 10, f"Gripper: {gripper_state}",
        fontsize=12, color='white', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor=gripper_color, alpha=0.8)
    )

    # Right: Action breakdown bar chart
    ax2 = axes[1]
    action_labels = ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'gripper']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']

    bars = ax2.barh(action_labels, action, color=colors, edgecolor='black')
    ax2.set_xlabel('Action Value', fontsize=11)
    ax2.set_title('Predicted 7-DoF Action', fontsize=12)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlim(-1.5, 1.5)

    # Add value labels on bars
    for bar, val in zip(bars, action):
        x_pos = val + 0.05 if val >= 0 else val - 0.15
        ax2.text(x_pos, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def run_interactive(model, processor, sample_dir: str = "sample_images"):
    """Run interactive demo with sample images."""

    sample_instructions = [
        "pick up the coke can",
        "move the spoon to the towel",
        "open the drawer",
        "stack the blocks",
        "put the object in the bin"
    ]

    # Check for sample images
    sample_path = Path(sample_dir)
    if sample_path.exists():
        images = list(sample_path.glob("*.jpg")) + list(sample_path.glob("*.png"))
    else:
        images = []

    if not images:
        print("\nNo sample images found. Creating a placeholder image...")
        # Create a simple placeholder image
        placeholder = Image.new('RGB', (256, 256), color=(100, 100, 100))
        images = [placeholder]

    print("\n" + "="*60)
    print("OpenVLA Interactive Demo")
    print("="*60)
    print("\nSample instructions to try:")
    for i, inst in enumerate(sample_instructions, 1):
        print(f"  {i}. {inst}")

    print("\nCommands: 'q' to quit, 'n' for next image, or type custom instruction")
    print("="*60 + "\n")

    img_idx = 0
    while True:
        # Get current image
        if isinstance(images[img_idx], Image.Image):
            image = images[img_idx]
            img_name = "placeholder"
        else:
            image = Image.open(images[img_idx]).convert("RGB")
            img_name = images[img_idx].name

        print(f"\nCurrent image: {img_name}")
        instruction = input("Enter instruction (or command): ").strip()

        if instruction.lower() == 'q':
            print("Exiting...")
            break
        elif instruction.lower() == 'n':
            img_idx = (img_idx + 1) % len(images)
            continue
        elif instruction.isdigit() and 1 <= int(instruction) <= len(sample_instructions):
            instruction = sample_instructions[int(instruction) - 1]
            print(f"Using: {instruction}")
        elif not instruction:
            instruction = sample_instructions[0]
            print(f"Using default: {instruction}")

        print("Running inference...")
        try:
            action = predict_action(model, processor, image, instruction)
            print(f"\nPredicted action: {action}")
            visualize_action(image, action, instruction)
        except Exception as e:
            print(f"Error during inference: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="OpenVLA Single-Step Inference Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on a single image
  python inference.py --image robot_view.jpg --instruction "pick up the red cup"

  # Interactive mode
  python inference.py --interactive

  # Save visualization without displaying
  python inference.py --image img.jpg --instruction "open drawer" --output result.png --no-show
        """
    )

    parser.add_argument(
        "--image", "-i",
        type=str,
        help="Path to input image"
    )
    parser.add_argument(
        "--instruction", "-t",
        type=str,
        default="pick up the object",
        help="Task instruction for the robot"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openvla/openvla-7b",
        help="HuggingFace model path"
    )
    parser.add_argument(
        "--unnorm-key",
        type=str,
        default="bridge_orig",
        help="Dataset key for action unnormalization"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output path for visualization"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display visualization (useful for headless servers)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run on"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.interactive and not args.image:
        parser.print_help()
        print("\nError: Must specify --image or --interactive")
        sys.exit(1)

    # Check CUDA availability
    if "cuda" in args.device and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU (will be slow!)")
        args.device = "cpu"

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f}GB)")

    # Load model
    model, processor = load_model(args.model, args.device)

    if args.interactive:
        run_interactive(model, processor)
    else:
        # Single image inference
        if not os.path.exists(args.image):
            print(f"Error: Image not found: {args.image}")
            sys.exit(1)

        image = Image.open(args.image).convert("RGB")
        print(f"Image: {args.image} ({image.size[0]}x{image.size[1]})")
        print(f"Instruction: {args.instruction}")

        action = predict_action(
            model, processor, image, args.instruction,
            unnorm_key=args.unnorm_key, device=args.device
        )

        print(f"\nPredicted action (7-DoF):")
        labels = ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'gripper']
        for label, val in zip(labels, action):
            print(f"  {label:8s}: {val:+.4f}")

        visualize_action(
            image, action, args.instruction,
            output_path=args.output,
            show=not args.no_show
        )


if __name__ == "__main__":
    main()
