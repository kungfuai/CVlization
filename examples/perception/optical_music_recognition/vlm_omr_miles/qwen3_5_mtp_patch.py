"""Patch for Miles + megatron-bridge Qwen3.5-VLM MTP layer name mismatch.

Bug (slime issue #1894): Megatron-LM produces MTP parameters named
`transformer_layer.*`, but megatron-bridge's qwen35_vl_bridge.py registers
mappings using `mtp_model_layer.*` (recent upstream rename).

The fix is in megatron-bridge PR #3769 but unmerged as of 2026-05-11.

Apply the established compat shim (already in glm45_bridge.py and mimo_bridge.py):
text-replace the new name with the old name throughout the VL bridge file.
Once megatron-bridge releases the proper fix, this patch can be removed.
"""

import sys
from pathlib import Path

VL_BRIDGE = Path("/usr/local/lib/python3.12/dist-packages/megatron/bridge/models/qwen_vl/qwen35_vl_bridge.py")

if not VL_BRIDGE.exists():
    print(f"FAIL: {VL_BRIDGE} not found", file=sys.stderr)
    sys.exit(1)

text = VL_BRIDGE.read_text()
if "mtp_model_layer" not in text:
    print(f"Patch already applied (no occurrences of mtp_model_layer in {VL_BRIDGE})")
    sys.exit(0)

count = text.count("mtp_model_layer")
text = text.replace("mtp_model_layer", "transformer_layer")
VL_BRIDGE.write_text(text)
print(f"Patched {VL_BRIDGE}: replaced {count} occurrences of mtp_model_layer -> transformer_layer")
