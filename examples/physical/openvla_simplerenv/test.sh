#!/usr/bin/env bash
# Smoke test for OpenVLA SimplerEnv example
# Tests environment creation and random action rollout (no model loading)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${CVL_IMAGE:-cvlization/openvla-simplerenv:latest}"

echo "Running smoke test..."
echo "Image: ${IMAGE_NAME}"
echo ""

# Test 1: Check if simpler_env can be imported and environment created
echo "Test 1: Environment creation and random action rollout..."
docker run --rm \
    --gpus all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e DISPLAY="" \
    -e MUJOCO_GL=egl \
    -e ACCEPT_EULA=Y \
    --workdir /workspace \
    --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
    "${IMAGE_NAME}" \
    python -c "
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

print('Creating environment: widowx_spoon_on_towel')
env = simpler_env.make('widowx_spoon_on_towel')

print('Resetting environment...')
obs, info = env.reset()
instruction = env.get_language_instruction()
print(f'Instruction: {instruction}')

print('Getting image from observation...')
image = get_image_from_maniskill2_obs_dict(env, obs)
print(f'Image shape: {image.shape}, dtype: {image.dtype}')

print('Running 5 random action steps...')
for i in range(5):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f'  Step {i+1}: reward={reward:.4f}, done={done}, truncated={truncated}')

print('')
print('Test 1 PASSED: Environment works correctly!')
"

echo ""
echo "Test 2: Check server dependencies..."
docker run --rm \
    -e ACCEPT_EULA=Y \
    --workdir /workspace \
    --mount "type=bind,src=${SCRIPT_DIR},dst=/workspace" \
    "${IMAGE_NAME}" \
    python -c "
import fastapi
import uvicorn
import websockets
import cv2
import numpy as np
import yaml

print('All server dependencies available!')
print(f'  FastAPI: {fastapi.__version__}')
print(f'  OpenCV: {cv2.__version__}')
print(f'  NumPy: {np.__version__}')
print('')
print('Test 2 PASSED: Server dependencies OK!')
"

echo ""
echo "============================================"
echo "All smoke tests PASSED!"
echo "============================================"
echo ""
echo "To start the demo server, run:"
echo "  ./run.sh"
