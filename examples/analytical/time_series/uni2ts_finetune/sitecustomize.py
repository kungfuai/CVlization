"""Project-specific Python startup customizations for Uni2TS fine-tuning."""

try:
    from uni2ts.transform import patch as patch_module
except ImportError:
    patch_module = None

if patch_module is not None:
    ranges = patch_module.DefaultPatchSizeConstraints.DEFAULT_RANGES
    if "h" not in ranges and "H" in ranges:
        ranges["h"] = ranges["H"]
