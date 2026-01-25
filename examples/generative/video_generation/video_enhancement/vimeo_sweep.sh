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
# DONE - Results: exp9 best, exp11/12 okay, exp13 broken (fixed now)
# ===========================================

# Exp 9: Large model + modulation + LPIPS (best of exp3 + exp8) *** BEST ***
# ./train.sh $COMMON --channels 64 --mask-guidance modulation \
#     --run-name vimeo_exp9_large_modulation_lpips \
#     --checkpoint-dir ./checkpoints/vimeo_exp9_large_modulation_lpips

# Exp 10: Modulation + LPIPS + mask-weight 3.0 (moderate focus on artifacts)
# ./train.sh $COMMON --mask-guidance modulation --mask-weight 3.0 \
#     --run-name vimeo_exp10_modulation_maskweight3 \
#     --checkpoint-dir ./checkpoints/vimeo_exp10_modulation_maskweight3

# ===========================================
# Phase 5: New mask guidance variants
# Test skip_gate, attn_gate, and ExplicitCompositeNet
# DONE - Results: exp11/12 not better than exp9, exp13 had mask collapse bug
# ===========================================

# Exp 11: skip_gate - suppress encoder features in artifact regions
# ./train.sh $COMMON --mask-guidance skip_gate \
#     --run-name vimeo_exp11_skipgate \
#     --checkpoint-dir ./checkpoints/vimeo_exp11_skipgate

# Exp 12: attn_gate - boost temporal attention to artifact regions
# ./train.sh $COMMON --mask-guidance attn_gate \
#     --run-name vimeo_exp12_attngate \
#     --checkpoint-dir ./checkpoints/vimeo_exp12_attngate

# Exp 13: ExplicitCompositeNet - had mask collapse bug, fixed in exp14
# ./train.sh $COMMON --model composite \
#     --run-name vimeo_exp13_composite \
#     --checkpoint-dir ./checkpoints/vimeo_exp13_composite

# ===========================================
# Phase 6: Verify fixes and extended training
# ===========================================

# Exp 14: Re-run composite with fix (auxiliary inpaint loss + auto mask_weight=5.0)
# Result: mask prediction still bad (small model, 2.3M params vs exp9's 8.9M)
# ./train.sh $COMMON --model composite \
#     --run-name vimeo_exp14_composite_fixed \
#     --checkpoint-dir ./checkpoints/vimeo_exp14_composite_fixed

# Exp 15: Composite with large model (same size as exp9) + detached mask fix
# Fix: mask_head learns only from mask_loss, recon_loss only trains inpaint_head
# Result: Better than exp14, higher mask loss than exp9
# ./train.sh $COMMON --model composite --channels 64 \
#     --run-name vimeo_exp15_composite_large \
#     --checkpoint-dir ./checkpoints/vimeo_exp15_composite_large

# ===========================================
# Phase 7: Larger artifacts (size_scale=1.5) + extended training (15k steps)
# Test if models handle larger text/logos
# ===========================================

STEPS_LONG=15000
COMMON_LONG="--vimeo --steps $STEPS_LONG --val-every $VAL_EVERY --save-every $SAVE_EVERY --num-frames $NUM_FRAMES"

# Exp 16: Best config (exp9) with larger artifacts
# ./train.sh $COMMON_LONG --channels 64 --mask-guidance modulation --size-scale 1.5 \
#     --run-name vimeo_exp16_large_modulation_sizescale \
#     --checkpoint-dir ./checkpoints/vimeo_exp16_large_modulation_sizescale

# Exp 17: Composite (exp15) with larger artifacts
./train.sh $COMMON_LONG --model composite --channels 64 --size-scale 1.5 \
    --run-name vimeo_exp17_composite_sizescale \
    --checkpoint-dir ./checkpoints/vimeo_exp17_composite_sizescale

# ===========================================
# Phase 8: LaMa architecture (FFC blocks + temporal attention)
# Test if global receptive field from Fourier convolutions helps
# ===========================================

# Exp 18: LaMa with default settings (64 base channels, 9 FFC blocks)
# From scratch, no pretrained weights
./train.sh $COMMON_LONG --model lama --channels 64 \
    --run-name vimeo_exp18_lama \
    --checkpoint-dir ./checkpoints/vimeo_exp18_lama

# Exp 19: LaMa with pretrained weights
# Download from: https://github.com/advimman/lama
# ./train.sh $COMMON_LONG --model lama --channels 64 \
#     --pretrained ./pretrained/big-lama.pt \
#     --run-name vimeo_exp19_lama_pretrained \
#     --checkpoint-dir ./checkpoints/vimeo_exp19_lama_pretrained

# Exp 20: ELIR with pretrained weights
# Download from: https://github.com/KAIST-VML/ELIR
# ./train.sh $COMMON_LONG --model elir --channels 64 \
#     --pretrained ./pretrained/elir.ckpt \
#     --run-name vimeo_exp20_elir_pretrained \
#     --checkpoint-dir ./checkpoints/vimeo_exp20_elir_pretrained

# ===========================================
# Phase 9: From-scratch experiments
# ===========================================

# Exp 21: ELIR from scratch (64 hidden channels, 128 flow channels)
# Training: flow matching loss + mask loss + composite loss
# Inference: 3-step Euler ODE in latent space
# ./train.sh $COMMON_LONG --model elir --channels 64 \
#     --run-name vimeo_exp21_elir \
#     --checkpoint-dir ./checkpoints/vimeo_exp21_elir

# Exp 22: LaMa with larger artifacts (size_scale=1.5)
# ./train.sh $COMMON_LONG --model lama --channels 64 --size-scale 1.5 \
#     --run-name vimeo_exp22_lama_sizescale \
#     --checkpoint-dir ./checkpoints/vimeo_exp22_lama_sizescale

# Exp 23: ELIR with larger artifacts (size_scale=1.5)
# ./train.sh $COMMON_LONG --model elir --channels 64 --size-scale 1.5 \
#     --run-name vimeo_exp23_elir_sizescale \
#     --checkpoint-dir ./checkpoints/vimeo_exp23_elir_sizescale
