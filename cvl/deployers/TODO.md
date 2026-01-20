# Deployers TODO

## Supporting More Examples for Cerebrium Deployment

Currently only `ltx2` is supported because `main.py` generation is hardcoded.

### Recommended Approach: Convention + Optional Override

**Discovery order:**
1. Check if `predict.py` has a `run()` function that returns `dict`
2. If not, check for `serve.py` with `run()` function
3. If not, example is unsupported for serverless deployment

**Convention for `run()` function:**
```python
def run(prompt: str, width: int = 512, ...) -> dict:
    """
    Serverless entry point.

    - All parameters must be JSON-serializable
    - Return dict with JSON-serializable values
    - For binary outputs (images, video, audio), use base64 encoding
    """
    # load model (lazy, cached globally)
    # run inference
    # return {"output_base64": "...", "format": "mp4", ...}
```

**Optional override in example.yaml:**
```yaml
serve:
  entry: "predict:run_pipeline"  # module:function
```

### Implementation Steps

1. Add `inspect_predict_py()` to check for `run()` function signature
2. If `run()` exists and returns dict, generate thin wrapper that:
   - Imports the function
   - Handles lazy model loading (if needed)
   - Calls function with request params
3. If `run()` doesn't exist, check for `serve.py`
4. Update `SUPPORTED_EXAMPLES` to be dynamic based on inspection

### Changes Required to Examples

For an example to support serverless deployment:

**Minimal change:** Add/modify `run()` in predict.py to return dict instead of writing to disk.

**Before:**
```python
def run_pipeline(prompt, output_path, ...):
    video = generate(prompt)
    save_video(video, output_path)
```

**After:**
```python
def run(prompt: str, ...) -> dict:
    video = generate(prompt)
    return {
        "video_base64": base64.b64encode(video).decode(),
        "format": "mp4"
    }
```

### Open Questions

- Should we support streaming responses?
- How to handle multi-modal inputs (image + text)?
- Should model loading be explicit (`init()` function) or implicit (lazy on first `run()`)?
- How to specify GPU/memory requirements if different from example.yaml defaults?
