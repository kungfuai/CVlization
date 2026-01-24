#!/bin/bash
# Experiment sweep for video artifact removal on Vimeo Septuplet dataset
#
# Uses --vimeo with num_frames=2 (faster iteration)
# Starts with pixel-only loss, then adds perceptual losses
# Prioritizes mask-guided modulation experiments
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
STEPS=10000
VAL_EVERY=500
SAVE_EVERY=5000
NUM_FRAMES=2  # Use 2 frames for faster iteration

COMMON="--vimeo --steps $STEPS --val-every $VAL_EVERY --save-every $SAVE_EVERY --num-frames $NUM_FRAMES"

# ===========================================
# Phase 1: Mask-guided modulation (priority)
# DONE - Results: exp3 > exp6, exp8 promising
# ===========================================

# Exp 1: Modulation + pixel-only (fastest baseline)
# ./train.sh $COMMON --mask-guidance modulation --pixel-only \
#     --run-name vimeo_exp1_modulation_pixel \
#     --checkpoint-dir ./checkpoints/vimeo_exp1_modulation_pixel

# Exp 2: Modulation + no-lpips (VGG perceptual)
# ./train.sh $COMMON --mask-guidance modulation --no-lpips \
#     --run-name vimeo_exp2_modulation_vgg \
#     --checkpoint-dir ./checkpoints/vimeo_exp2_modulation_vgg

# Exp 3: Modulation + full loss (LPIPS) *** BEST ***
# ./train.sh $COMMON --mask-guidance modulation \
#     --run-name vimeo_exp3_modulation_lpips \
#     --checkpoint-dir ./checkpoints/vimeo_exp3_modulation_lpips

# ===========================================
# Phase 2: Baselines for comparison
# DONE
# ===========================================

# Exp 4: No mask - pixel-only
# ./train.sh $COMMON --pixel-only \
#     --run-name vimeo_exp4_baseline_pixel \
#     --checkpoint-dir ./checkpoints/vimeo_exp4_baseline_pixel

# Exp 5: No mask - no-lpips
# ./train.sh $COMMON --no-lpips \
#     --run-name vimeo_exp5_baseline_vgg \
#     --checkpoint-dir ./checkpoints/vimeo_exp5_baseline_vgg

# Exp 6: No mask - full loss *** CLOSE TO EXP3 ***
# ./train.sh $COMMON \
#     --run-name vimeo_exp6_baseline_lpips \
#     --checkpoint-dir ./checkpoints/vimeo_exp6_baseline_lpips

# ===========================================
# Phase 3: Ablations
# DONE
# ===========================================

# Exp 7: Predict mask without guidance (multi-task only)
# ./train.sh $COMMON --predict-mask --no-lpips \
#     --run-name vimeo_exp7_multitask_vgg \
#     --checkpoint-dir ./checkpoints/vimeo_exp7_multitask_vgg

# Exp 8: Larger model + modulation *** PROMISING ***
# ./train.sh $COMMON --channels 64 --mask-guidance modulation --no-lpips \
#     --run-name vimeo_exp8_large_modulation \
#     --checkpoint-dir ./checkpoints/vimeo_exp8_large_modulation

# ===========================================
# Phase 4: Next experiments
# Based on learnings: LPIPS critical, modulation helps, larger model promising
# ===========================================

# Exp 9: Large model + modulation + LPIPS (best of exp3 + exp8)
./train.sh $COMMON --channels 64 --mask-guidance modulation \
    --run-name vimeo_exp9_large_modulation_lpips \
    --checkpoint-dir ./checkpoints/vimeo_exp9_large_modulation_lpips

# Exp 10: Modulation + LPIPS + mask-weight (focus loss on artifact regions)
./train.sh $COMMON --mask-guidance modulation --mask-weight 5.0 \
    --run-name vimeo_exp10_modulation_lpips_maskweight5 \
    --checkpoint-dir ./checkpoints/vimeo_exp10_modulation_lpips_maskweight5

# Exp 11: Large + modulation + LPIPS + mask-weight (all features combined)
./train.sh $COMMON --channels 64 --mask-guidance modulation --mask-weight 5.0 \
    --run-name vimeo_exp11_large_modulation_maskweight5 \
    --checkpoint-dir ./checkpoints/vimeo_exp11_large_modulation_maskweight5
