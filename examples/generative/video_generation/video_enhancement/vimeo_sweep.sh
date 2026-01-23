#!/bin/bash
# Experiment sweep for video artifact removal on Vimeo Septuplet dataset
#
# Uses --vimeo with num_frames=7 (full septuplet sequence)
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

# Common settings
STEPS=50000
VAL_EVERY=1000
SAVE_EVERY=5000
NUM_FRAMES=7  # Vimeo Septuplet has 7 frames per sequence

COMMON="--vimeo --steps $STEPS --val-every $VAL_EVERY --save-every $SAVE_EVERY --num-frames $NUM_FRAMES"

# Baseline experiments
# --------------------

# Exp 1: Baseline - no mask prediction
./train.sh $COMMON \
    --run-name vimeo_exp1_baseline \
    --checkpoint-dir ./checkpoints/vimeo_exp1_baseline

# Exp 2: Multi-task - predict mask (no guidance)
./train.sh $COMMON --predict-mask \
    --run-name vimeo_exp2_multitask \
    --checkpoint-dir ./checkpoints/vimeo_exp2_multitask

# Exp 3: Mask-guided modulation
./train.sh $COMMON --mask-guidance modulation \
    --run-name vimeo_exp3_modulation \
    --checkpoint-dir ./checkpoints/vimeo_exp3_modulation

# Model size experiments
# ----------------------

# Exp 4: Larger model (64 base channels)
./train.sh $COMMON --channels 64 \
    --run-name vimeo_exp4_large \
    --checkpoint-dir ./checkpoints/vimeo_exp4_large

# Exp 5: Larger + mask-guided
./train.sh $COMMON --channels 64 --mask-guidance modulation \
    --run-name vimeo_exp5_large_modulation \
    --checkpoint-dir ./checkpoints/vimeo_exp5_large_modulation

# Ablation experiments
# --------------------

# Exp 6: No temporal attention
./train.sh $COMMON --no-temporal \
    --run-name vimeo_exp6_no_temporal \
    --checkpoint-dir ./checkpoints/vimeo_exp6_no_temporal

# Exp 7: Residual learning
./train.sh $COMMON --residual \
    --run-name vimeo_exp7_residual \
    --checkpoint-dir ./checkpoints/vimeo_exp7_residual

# Exp 8: Residual + mask-guided
./train.sh $COMMON --residual --mask-guidance modulation \
    --run-name vimeo_exp8_residual_modulation \
    --checkpoint-dir ./checkpoints/vimeo_exp8_residual_modulation
