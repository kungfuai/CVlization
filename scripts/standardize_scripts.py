#!/usr/bin/env python3
"""
Batch standardize build.sh and train.sh scripts across all examples.

This script updates shell scripts to follow the CVL pattern:
- Use SCRIPT_DIR for build context
- Use REPO_ROOT (go up 4 levels)
- Use --gpus=all instead of --runtime nvidia
- Use consistent mount patterns
- Remove CVL_WORK_DIR complexity
"""

import os
import re
from pathlib import Path

# Standard build.sh template
BUILD_TEMPLATE = '''#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

# Build from the script's directory, works from anywhere
docker build -t {image_name} "$SCRIPT_DIR"
'''

# Standard train.sh template (simple version)
TRAIN_TEMPLATE = '''#!/usr/bin/env bash
set -euo pipefail

# Always run from this folder
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

# Find repo root for cvlization package (go up 4 levels from example dir)
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Image name
IMG="${{CVL_IMAGE:-{image_name}}}"

# Mount workspace as writable (training writes outputs to /workspace)
docker run --rm --gpus=all --shm-size 16G \\
\t${{CVL_CONTAINER_NAME:+--name "$CVL_CONTAINER_NAME"}} \\
\t--workdir /workspace \\
\t--mount "type=bind,src=${{SCRIPT_DIR}},dst=/workspace" \\
\t--mount "type=bind,src=${{REPO_ROOT}},dst=/cvlization_repo,readonly" \\
\t--mount "type=bind,src=${{HOME}}/.cache/huggingface,dst=/root/.cache/huggingface" \\
\t--env "PYTHONPATH=/cvlization_repo" \\
\t--env "PYTHONUNBUFFERED=1" \\
\t${{WANDB_API_KEY:+-e WANDB_API_KEY=$WANDB_API_KEY}} \\
\t"$IMG" \\
\tpython train.py "$@"
'''

def extract_image_name(script_path):
    """Extract image name from existing build.sh or train.sh"""
    try:
        content = script_path.read_text()

        # Try various patterns
        patterns = [
            r'docker build -t (\S+)',
            r'IMAGE_NAME="(\S+)"',
            r'IMG="(\S+)"',
            r'docker run.*\s+(\S+)\s+python',
        ]

        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                name = match.group(1)
                # Clean up variable references
                name = name.replace('$IMAGE_NAME', '').replace('$IMG', '').replace('${IMG}', '')
                if name and not name.startswith('$'):
                    return name

        # Fallback: use directory name
        return script_path.parent.name
    except:
        return script_path.parent.name

def should_update_build(script_path):
    """Check if build.sh needs updating"""
    if not script_path.exists():
        return False

    content = script_path.read_text()

    # Check if it already follows the pattern
    if 'SCRIPT_DIR="$(cd' in content and 'docker build -t' in content and '"$SCRIPT_DIR"' in content:
        return False

    return True

def should_update_train(script_path):
    """Check if train.sh needs updating"""
    if not script_path.exists():
        return False

    content = script_path.read_text()

    # Skip if it's already updated or too complex
    if 'set -euo pipefail' in content and 'REPO_ROOT' in content and '--gpus=all' in content:
        return False

    # Skip complex scripts (more than 50 lines, or has functions)
    lines = content.split('\n')
    if len(lines) > 50 or 'function ' in content or '() {' in content:
        print(f"  Skipping complex script: {script_path}")
        return False

    return True

def update_build_sh(script_path):
    """Update a build.sh file"""
    image_name = extract_image_name(script_path)

    new_content = BUILD_TEMPLATE.format(image_name=image_name)

    print(f"Updating {script_path} (image: {image_name})")
    script_path.write_text(new_content)

def update_train_sh(script_path):
    """Update a train.sh file"""
    image_name = extract_image_name(script_path)

    new_content = TRAIN_TEMPLATE.format(image_name=image_name)

    print(f"Updating {script_path} (image: {image_name})")
    script_path.write_text(new_content)

def main():
    repo_root = Path(__file__).parent.parent
    examples_dir = repo_root / "examples"

    build_count = 0
    train_count = 0

    # Find all build.sh and train.sh files
    for script_path in examples_dir.rglob("build.sh"):
        if should_update_build(script_path):
            update_build_sh(script_path)
            build_count += 1

    for script_path in examples_dir.rglob("train.sh"):
        if should_update_train(script_path):
            update_train_sh(script_path)
            train_count += 1

    print(f"\nUpdated {build_count} build.sh files")
    print(f"Updated {train_count} train.sh files")

if __name__ == "__main__":
    main()
