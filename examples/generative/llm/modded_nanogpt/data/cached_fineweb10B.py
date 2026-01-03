import os
import sys
from huggingface_hub import hf_hub_download

# Download the GPT-2 tokens of Fineweb10B from huggingface. This
# saves about an hour of startup time compared to regenerating them.
# Files are cached in ~/.cache/huggingface and symlinked to local_dir.

REPO_ID = "kjj0/fineweb10B-gpt2"
local_dir = os.path.join(os.path.dirname(__file__), 'fineweb10B')
os.makedirs(local_dir, exist_ok=True)

def get(fname):
    local_path = os.path.join(local_dir, fname)
    if os.path.exists(local_path):
        return local_path
    # Download to HF cache and get the cached path
    cached_path = hf_hub_download(repo_id=REPO_ID, filename=fname, repo_type="dataset")
    # Create symlink from local_dir to cached file
    try:
        os.symlink(cached_path, local_path)
        print(f"  Linked {fname} -> {cached_path}")
    except OSError:
        # Symlink failed (cross-device), copy instead
        import shutil
        shutil.copy2(cached_path, local_path)
        print(f"  Copied {fname}")
    return local_path

get("fineweb_val_%06d.bin" % 0)
num_chunks = 103  # full fineweb10B. Each chunk is 100M tokens
if len(sys.argv) >= 2:  # we can pass an argument to download less
    num_chunks = int(sys.argv[1])
for i in range(1, num_chunks + 1):
    get("fineweb_train_%06d.bin" % i)
