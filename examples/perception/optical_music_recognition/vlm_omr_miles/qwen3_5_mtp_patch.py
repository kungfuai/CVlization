"""Patch for Miles qwen3_5 mbridge MTP layer naming.

Bug: slime issue #1894 — Megatron-LM upstream renamed `transformer_layer` to
`mtp_model_layer` in MTP submodule. Miles' qwen3_5.py mbridge still checks
for the old name, causing AttributeError: 'NoneType' object has no attribute
'megatron_module' during HF→Megatron weight conversion.

Applied at container build time. Replaces in-place:
  - The "transformer_layer" check on line ~272 to accept both names
  - The string-substitution to handle both naming conventions
"""

import re
import sys

TARGET = "/root/miles/miles_plugins/mbridge/qwen3_5.py"

OLD_BLOCK = '''        if "transformer_layer" in name:
            proxy_name = name.replace(
                f"mtp.layers.{mtp_layer_idx}.transformer_layer",
                f"decoder.layers.{mtp_layer_idx}",
            )'''

NEW_BLOCK = '''        if "transformer_layer" in name or "mtp_model_layer" in name:
            mtp_sublayer = "mtp_model_layer" if "mtp_model_layer" in name else "transformer_layer"
            proxy_name = name.replace(
                f"mtp.layers.{mtp_layer_idx}.{mtp_sublayer}",
                f"decoder.layers.{mtp_layer_idx}",
            )'''

with open(TARGET) as f:
    src = f.read()

if NEW_BLOCK in src:
    print(f"Patch already applied to {TARGET}")
    sys.exit(0)

if OLD_BLOCK not in src:
    print(f"FAIL: target block not found in {TARGET}", file=sys.stderr)
    print("Miles version may have changed; patch needs update.", file=sys.stderr)
    sys.exit(1)

patched = src.replace(OLD_BLOCK, NEW_BLOCK)
with open(TARGET, "w") as f:
    f.write(patched)

print(f"Patched {TARGET}: accept both transformer_layer and mtp_model_layer")
