# CVlization Wan2GP Patches

These patches are applied to the vendored wan2gp code after syncing from upstream.

## Usage

Run `sync_vendor.sh` with `WAN2GP_SRC` set to your upstream Wan2GP clone:

```bash
git clone https://github.com/DeepBeepMeep/Wan2GP.git /tmp/Wan2GP
WAN2GP_SRC=/tmp/Wan2GP ./sync_vendor.sh
```

The script will:
1. Sync selected files from upstream via rsync
2. Apply all patches in this directory

## Patches

### 001-attention-safe-get.patch
**File:** `shared/attention.py`
**Change:** Use `.get("_attention", "flash")` instead of direct dict access
**Reason:** Provides a safe fallback when `_attention` is not set in `offload.shared_state`. The original code expects this to be set by the Gradio UI.

### 002-audio-video-aac-fix.patch
**File:** `shared/utils/audio_video.py`
**Change:** Use `-c:a aac -b:a 192k` instead of `-c:a copy`
**Reason:** MP4 containers don't support all audio codecs. The original `-c:a copy` fails when source audio is PCM or another incompatible codec.

### 003-base-encoder-device-fix.patch
**File:** `models/ltx2/ltx_core/text_encoders/gemma/encoders/base_encoder.py`
**Change:** Add explicit device movement for tensors in `_apply_feature_extractor`
**Reason:** Ensures tensors are on the correct device when mmgp moves models between CPU and GPU during inference.

## Adding New Patches

1. Make changes to vendor files
2. Generate patch:
   ```bash
   diff -u vendor/wan2gp/path/original.py vendor/wan2gp/path/modified.py > patches/00N-description.patch
   ```
3. Edit patch to use `a/` and `b/` prefixes for compatibility with `patch -p1`
4. Document in this README
5. Commit patch file
