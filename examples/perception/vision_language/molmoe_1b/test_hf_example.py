#!/usr/bin/env python3
"""
Minimal test based on HuggingFace official example
"""
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

def fixed_get_imports(filename):
    """Workaround: Remove tensorflow from imports."""
    imports = get_imports(filename)
    if "tensorflow" in imports:
        imports.remove("tensorflow")
    return imports

print("Loading processor...")
with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    processor = AutoProcessor.from_pretrained(
        'allenai/MolmoE-1B-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        'allenai/MolmoE-1B-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='cuda:0'  # Force GPU
    )

print("Model loaded! Processing image...")
inputs = processor.process(
    images=[Image.open(requests.get("https://picsum.photos/id/237/536/354", stream=True).raw)],
    text="Describe this image."
)

print("Moving to device...")
inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

print("Generating...")
output = model.generate_from_batch(
    inputs,
    GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
    tokenizer=processor.tokenizer
)

generated_tokens = output[0,inputs['input_ids'].size(1):]
generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

print("\n" + "="*60)
print("Generated text:")
print("="*60)
print(generated_text)
print("="*60)
