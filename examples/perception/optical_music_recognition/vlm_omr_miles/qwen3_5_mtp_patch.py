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

OLD_BLOCK = '''        if "mtp.layers." not in name:
            raise NotImplementedError(f"Invalid MTP parameter name: {name}")

        parts = name.split(".")
        mtp_layer_idx = parts[2]  # mtp.layers.{idx}'''

NEW_BLOCK = '''        if "mtp.layers." not in name:
            raise NotImplementedError(f"Invalid MTP parameter name: {name}")

        # VLM checkpoints prefix params with "language_model." — locate the
        # "mtp.layers" segment and read the actual layer index after it.
        parts = name.split(".")
        mtp_idx_pos = parts.index("layers", parts.index("mtp")) + 1
        mtp_layer_idx = parts[mtp_idx_pos]'''

DUAL_NAME_OLD = '''        if "transformer_layer" in name:
            proxy_name = name.replace(
                f"mtp.layers.{mtp_layer_idx}.transformer_layer",
                f"decoder.layers.{mtp_layer_idx}",
            )'''

DUAL_NAME_NEW = '''        if "transformer_layer" in name or "mtp_model_layer" in name:
            mtp_sublayer = "mtp_model_layer" if "mtp_model_layer" in name else "transformer_layer"
            proxy_name = name.replace(
                f"mtp.layers.{mtp_layer_idx}.{mtp_sublayer}",
                f"decoder.layers.{mtp_layer_idx}",
            )'''

with open(TARGET) as f:
    src = f.read()

if NEW_BLOCK in src and DUAL_NAME_NEW in src:
    print(f"Patch already fully applied to {TARGET}")
    sys.exit(0)

changes = 0
if OLD_BLOCK in src:
    src = src.replace(OLD_BLOCK, NEW_BLOCK)
    changes += 1
    print(f"  Applied VLM prefix fix (parts[mtp_idx_pos] lookup)")
elif NEW_BLOCK not in src:
    print(f"FAIL: VLM prefix patch target not found in {TARGET}", file=sys.stderr)
    sys.exit(1)

if DUAL_NAME_OLD in src:
    src = src.replace(DUAL_NAME_OLD, DUAL_NAME_NEW)
    changes += 1
    print(f"  Applied transformer_layer/mtp_model_layer dual-name fix")
elif DUAL_NAME_NEW not in src:
    print(f"FAIL: dual-name patch target not found in {TARGET}", file=sys.stderr)
    sys.exit(1)

with open(TARGET, "w") as f:
    f.write(src)

print(f"Patched {TARGET}: {changes} change(s) applied")
