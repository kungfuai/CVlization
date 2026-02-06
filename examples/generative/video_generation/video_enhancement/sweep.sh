#!/bin/bash
# Experiment sweep for video artifact removal
#
# Each experiment has a unique --run-name for easy TensorBoard organization.
# Ctrl+C will exit the entire sweep (not continue to next experiment).
#
# Run in parallel with different GPUs:
#   CUDA_VISIBLE_DEVICES=0 ./train.sh ... &
#   CUDA_VISIBLE_DEVICES=1 ./train.sh ... &
#
# Monitor with TensorBoard:
#   ./tensorboard.sh
#   Open http://localhost:6006

set -e  # Exit on error
trap "echo 'Interrupted. Exiting sweep.'; exit 1" INT TERM

# Baseline experiments (dummy dataset, 10k steps)
# ----------------------------------------------

# Exp 1: Baseline - no mask prediction
./train.sh --dummy --steps 10000 --val-every 500 \
    --run-name exp1_baseline \
    --checkpoint-dir ./checkpoints/exp1_baseline

# Exp 2: Multi-task - predict mask (no guidance)
./train.sh --dummy --steps 10000 --val-every 500 --predict-mask \
    --run-name exp2_multitask \
    --checkpoint-dir ./checkpoints/exp2_multitask

# Exp 3: Mask-guided modulation
./train.sh --dummy --steps 10000 --val-every 500 --mask-guidance modulation \
    --run-name exp3_modulation \
    --checkpoint-dir ./checkpoints/exp3_modulation

# Model size experiments
# ----------------------

# Exp 4: Smaller model (16 base channels)
./train.sh --dummy --steps 10000 --val-every 500 --channels 16 \
    --run-name exp4_small \
    --checkpoint-dir ./checkpoints/exp4_small

# Exp 5: Larger model (64 base channels)
./train.sh --dummy --steps 10000 --val-every 500 --channels 64 \
    --run-name exp5_large \
    --checkpoint-dir ./checkpoints/exp5_large

# Exp 6: Shallower model (3 stages)
./train.sh --dummy --steps 10000 --val-every 500 --depth 3 \
    --run-name exp6_shallow \
    --checkpoint-dir ./checkpoints/exp6_shallow

# Ablation experiments
# --------------------

# Exp 7: No temporal attention
./train.sh --dummy --steps 10000 --val-every 500 --no-temporal \
    --run-name exp7_no_temporal \
    --checkpoint-dir ./checkpoints/exp7_no_temporal

# Exp 8: Residual learning
./train.sh --dummy --steps 10000 --val-every 500 --residual \
    --run-name exp8_residual \
    --checkpoint-dir ./checkpoints/exp8_residual

# Exp 9: Residual + mask-guided
./train.sh --dummy --steps 10000 --val-every 500 --residual --mask-guidance modulation \
    --run-name exp9_residual_modulation \
    --checkpoint-dir ./checkpoints/exp9_residual_modulation

# Loss ablation
# -------------

# Exp 10: Pixel loss only (faster training)
./train.sh --dummy --steps 10000 --val-every 500 --pixel-only \
    --run-name exp10_pixel_only \
    --checkpoint-dir ./checkpoints/exp10_pixel_only
