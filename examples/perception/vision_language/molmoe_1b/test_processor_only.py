#!/usr/bin/env python3
"""
Quick test of MolmoE processor multi-image support (no model loading).
"""

import os
from unittest.mock import patch
from PIL import Image
from transformers import AutoProcessor
from transformers.dynamic_module_utils import get_imports

MODEL_ID = "allenai/MolmoE-1B-0924"

# Workaround for transformers' conditional import checking bug
def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Remove conditional imports that aren't actually needed."""
    imports = get_imports(filename)
    if "tensorflow" in imports:
        imports.remove("tensorflow")
    return imports

print("="*60)
print("MolmoE Processor Multi-Image Test")
print("="*60)

print("\nLoading processor...")
with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

print("✓ Processor loaded\n")

# Create test images
img1 = Image.new('RGB', (800, 600), color='red')
img2 = Image.new('RGB', (800, 600), color='blue')
img3 = Image.new('RGB', (800, 600), color='green')

prompt = "Describe what you see in these images."

# Test 1: Single image
print("Test 1: Single image")
try:
    inputs = processor.process(images=[img1], text=prompt)
    print(f"✓ Single image works")
    print(f"  Input keys: {inputs.keys()}")
    for k, v in inputs.items():
        if hasattr(v, 'shape'):
            print(f"  {k}.shape: {v.shape}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Multiple images (native multi-image)
print("\nTest 2: Multiple images (native)")
try:
    inputs = processor.process(images=[img1, img2, img3], text=prompt)
    print(f"✓ Multiple images accepted!")
    print(f"  Input keys: {inputs.keys()}")
    for k, v in inputs.items():
        if hasattr(v, 'shape'):
            print(f"  {k}.shape: {v.shape}")

    # Check if image tensors are larger (indicating multiple images processed)
    single_img_size = processor.process(images=[img1], text=prompt)['images'].shape
    multi_img_size = inputs['images'].shape

    if multi_img_size != single_img_size:
        print(f"\n✓ Native multi-image SUPPORTED!")
        print(f"  Single image tensor: {single_img_size}")
        print(f"  Multi image tensor: {multi_img_size}")
    else:
        print(f"\n⚠ Images processed but tensor size unchanged - likely only using first image")

except Exception as e:
    print(f"✗ Native multi-image NOT supported")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Vertical concatenation
print("\nTest 3: Vertical concatenation")
try:
    total_height = img1.height + img2.height + img3.height
    concatenated = Image.new('RGB', (800, total_height))
    concatenated.paste(img1, (0, 0))
    concatenated.paste(img2, (0, img1.height))
    concatenated.paste(img3, (0, img1.height + img2.height))

    print(f"  Concatenated size: {concatenated.size}")

    inputs = processor.process(images=[concatenated], text=prompt)
    print(f"✓ Concatenated image works")
    for k, v in inputs.items():
        if hasattr(v, 'shape'):
            print(f"  {k}.shape: {v.shape}")

except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "="*60)
print("Done!")
print("="*60)
