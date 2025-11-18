#!/usr/bin/env python3
"""
Test if MolmoE-1B supports native multi-image input.
"""

import os
from unittest.mock import patch
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from transformers.dynamic_module_utils import get_imports

MODEL_ID = "allenai/MolmoE-1B-0924"

# Workaround for transformers' conditional import checking bug
def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Remove conditional imports that aren't actually needed."""
    imports = get_imports(filename)
    if "tensorflow" in imports:
        imports.remove("tensorflow")
    return imports

print("Loading model and processor...")
with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        low_cpu_mem_usage=True
    )

# Create test images
print("\nCreating test images...")
img1 = Image.new('RGB', (800, 600), color='red')
img2 = Image.new('RGB', (800, 600), color='blue')
img3 = Image.new('RGB', (800, 600), color='green')

prompt = "Describe what you see in these images."

# Test 1: Single image (baseline - should work)
print("\n" + "="*60)
print("Test 1: Single image (baseline)")
print("="*60)
try:
    inputs = processor.process(images=[img1], text=prompt)
    print(f"✓ Processor accepted single image")
    print(f"  Input keys: {inputs.keys()}")
    for k, v in inputs.items():
        if hasattr(v, 'shape'):
            print(f"  {k}.shape: {v.shape}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Multiple images (native multi-image)
print("\n" + "="*60)
print("Test 2: Multiple images (native multi-image)")
print("="*60)
try:
    inputs = processor.process(images=[img1, img2, img3], text=prompt)
    print(f"✓ Processor accepted multiple images!")
    print(f"  Input keys: {inputs.keys()}")
    for k, v in inputs.items():
        if hasattr(v, 'shape'):
            print(f"  {k}.shape: {v.shape}")

    # Try inference
    print("\n  Testing inference...")
    inputs_batch = {
        k: v.to(model.device).unsqueeze(0).to(model.dtype)
        if v.dtype.is_floating_point
        else v.to(model.device).unsqueeze(0)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        output = model.generate_from_batch(
            inputs_batch,
            GenerationConfig(max_new_tokens=50, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )

    generated_tokens = output[0, inputs_batch['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print(f"✓ Inference successful!")
    print(f"  Generated: {generated_text[:200]}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Vertical concatenation (fallback approach)
print("\n" + "="*60)
print("Test 3: Vertical concatenation")
print("="*60)
try:
    # Concatenate vertically
    total_height = img1.height + img2.height + img3.height
    concatenated = Image.new('RGB', (800, total_height))
    concatenated.paste(img1, (0, 0))
    concatenated.paste(img2, (0, img1.height))
    concatenated.paste(img3, (0, img1.height + img2.height))

    print(f"  Concatenated size: {concatenated.size}")

    inputs = processor.process(images=[concatenated], text=prompt)
    print(f"✓ Processor accepted concatenated image")
    for k, v in inputs.items():
        if hasattr(v, 'shape'):
            print(f"  {k}.shape: {v.shape}")

except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "="*60)
print("Summary")
print("="*60)
print("Check results above to determine best multi-image strategy")
