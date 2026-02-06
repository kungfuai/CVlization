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
# Phase 1-9: Exploration experiments (DONE)
# All commented out - see results in notes.md
# ===========================================

# Exp 1-21: See git history for details
# Key findings:
#   - LPIPS is critical for perceptual quality
#   - 64 channels provides good capacity
#   - Modulation mask guidance helps
#   - LaMa (FFC blocks) promising for global context
#   - Best configs: exp9 (NAFUNet+modulation), exp16-18

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

# Exp 7: Predict mask without guidance (multi-task only)
# ./train.sh $COMMON --predict-mask --no-lpips \
#     --run-name vimeo_exp7_multitask_vgg \
#     --checkpoint-dir ./checkpoints/vimeo_exp7_multitask_vgg

# Exp 8: Larger model + modulation *** PROMISING ***
# ./train.sh $COMMON --channels 64 --mask-guidance modulation --no-lpips \
#     --run-name vimeo_exp8_large_modulation \
#     --checkpoint-dir ./checkpoints/vimeo_exp8_large_modulation

# Exp 9: Large model + modulation + LPIPS (best of exp3 + exp8) *** BEST ***
# ./train.sh $COMMON --channels 64 --mask-guidance modulation \
#     --run-name vimeo_exp9_large_modulation_lpips \
#     --checkpoint-dir ./checkpoints/vimeo_exp9_large_modulation_lpips

# Exp 10: Modulation + LPIPS + mask-weight 3.0 (moderate focus on artifacts)
# ./train.sh $COMMON --mask-guidance modulation --mask-weight 3.0 \
#     --run-name vimeo_exp10_modulation_maskweight3 \
#     --checkpoint-dir ./checkpoints/vimeo_exp10_modulation_maskweight3

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

# Exp 14: Re-run composite with fix (auxiliary inpaint loss + auto mask_weight=5.0)
# ./train.sh $COMMON --model composite \
#     --run-name vimeo_exp14_composite_fixed \
#     --checkpoint-dir ./checkpoints/vimeo_exp14_composite_fixed

# Exp 15: Composite with large model (same size as exp9) + detached mask fix
# ./train.sh $COMMON --model composite --channels 64 \
#     --run-name vimeo_exp15_composite_large \
#     --checkpoint-dir ./checkpoints/vimeo_exp15_composite_large

STEPS_LONG=15000
COMMON_LONG="--vimeo --steps $STEPS_LONG --val-every $VAL_EVERY --save-every $SAVE_EVERY --num-frames $NUM_FRAMES"

# Exp 16: Best config (exp9) with larger artifacts
# ./train.sh $COMMON_LONG --channels 64 --mask-guidance modulation --size-scale 1.5 \
#     --run-name vimeo_exp16_large_modulation_sizescale \
#     --checkpoint-dir ./checkpoints/vimeo_exp16_large_modulation_sizescale

# Exp 17: Composite (exp15) with larger artifacts
# ./train.sh $COMMON_LONG --model composite --channels 64 --size-scale 1.5 \
#     --run-name vimeo_exp17_composite_sizescale \
#     --checkpoint-dir ./checkpoints/vimeo_exp17_composite_sizescale

# Exp 18: LaMa with default settings (64 base channels, 9 FFC blocks)
# ./train.sh $COMMON_LONG --model lama --channels 64 \
#     --run-name vimeo_exp18_lama \
#     --checkpoint-dir ./checkpoints/vimeo_exp18_lama

# Exp 19: LaMa with pretrained weights
# ./train.sh $COMMON_LONG --model lama --channels 64 \
#     --pretrained auto \
#     --run-name vimeo_exp19_lama_pretrained \
#     --checkpoint-dir ./checkpoints/vimeo_exp19_lama_pretrained

# Exp 20: ELIR with pretrained weights
# ./train.sh $COMMON_LONG --model elir --channels 64 \
#     --pretrained auto \
#     --run-name vimeo_exp20_elir_pretrained \
#     --checkpoint-dir ./checkpoints/vimeo_exp20_elir_pretrained

# Exp 21: ELIR from scratch with MaskUNet
# ./train.sh $COMMON_LONG --model elir --channels 64 \
#     --mask-unet --focal-mask-loss \
#     --run-name vimeo_exp21_elir_maskunet \
#     --checkpoint-dir ./checkpoints/vimeo_exp21_elir_maskunet

# ===========================================
# Phase 10: Production-Ready Model Training
# ===========================================
#
# Goals:
# 1. Longer training (30k-50k steps)
# 2. More frames for temporal consistency (4-8 frames)
# 3. Robust to various artifact sizes
# 4. Best architecture from exploration phase
#
# Based on exp1-21 findings:
# - Best architecture: TemporalNAFUNet 64ch + modulation + LPIPS (exp9/16)
# - LaMa also promising for global context (exp18)
# - ELIR good for flow-based refinement (exp21)

STEPS_PROD=30000
VAL_EVERY_PROD=1000
SAVE_EVERY_PROD=10000

# ---------------------------------------------
# Exp 22: Production NAFUNet - 4 frames, 30k steps, multi-scale (DONE)
# ---------------------------------------------
# Best config (exp9/16) with more temporal context
# 4 frames provides better temporal consistency than 2
# Multi-scale: randomly samples 256/320/384, preserves aspect ratio
# Note: multi-scale forces batch_size=1

COMMON_PROD_4F="--vimeo --steps $STEPS_PROD --val-every $VAL_EVERY_PROD --save-every $SAVE_EVERY_PROD --num-frames 4"

# ./train.sh $COMMON_PROD_4F --channels 64 --mask-guidance modulation \
#     --multi-scale 256,320,384 \
#     --run-name vimeo_exp22_prod_nafunet_4f_multiscale \
#     --checkpoint-dir ./checkpoints/vimeo_exp22_prod_nafunet_4f_multiscale

# ---------------------------------------------
# Exp 23: LaMa with multi-scale - FFC blocks for global context (DONE)
# ---------------------------------------------
# LaMa's Fast Fourier Convolution provides global receptive field
# Good for large watermarks spanning the frame
# Multi-scale for resolution robustness

# ./train.sh $COMMON_PROD_4F --model lama --channels 64 \
#     --multi-scale 256,320,384 \
#     --run-name vimeo_exp23_lama_multiscale \
#     --checkpoint-dir ./checkpoints/vimeo_exp23_lama_multiscale

# ---------------------------------------------
# Exp 24: ExplicitComposite with multi-scale (DONE)
# ---------------------------------------------
# Explicit alpha blending guarantees clean region preservation
# Predicts both inpainted content and mask, composites them
# Multi-scale for resolution robustness

# ./train.sh $COMMON_PROD_4F --model composite --channels 64 \
#     --multi-scale 256,320,384 \
#     --run-name vimeo_exp24_composite_multiscale \
#     --checkpoint-dir ./checkpoints/vimeo_exp24_composite_multiscale

# ===========================================
# Phase 10b: Resume Training (warm start from exp22-24)
# ===========================================
# Continue training for another 30k steps (total 60k)

STEPS_RESUME=60000  # Total steps (not additional)

# ---------------------------------------------
# Exp 25: Resume NAFUNet from exp22 (best checkpoint)
# ---------------------------------------------
./train.sh --vimeo --steps $STEPS_RESUME --val-every $VAL_EVERY_PROD --save-every $SAVE_EVERY_PROD --num-frames 4 \
    --channels 64 --mask-guidance modulation \
    --multi-scale 256,320,384 \
    --resume ./checkpoints/vimeo_exp22_prod_nafunet_4f_multiscale/best.pt \
    --run-name vimeo_exp25_nafunet_resume \
    --checkpoint-dir ./checkpoints/vimeo_exp25_nafunet_resume

# ---------------------------------------------
# Exp 26: Resume LaMa from exp23 (best checkpoint)
# ---------------------------------------------
./train.sh --vimeo --steps $STEPS_RESUME --val-every $VAL_EVERY_PROD --save-every $SAVE_EVERY_PROD --num-frames 4 \
    --model lama --channels 64 \
    --multi-scale 256,320,384 \
    --resume ./checkpoints/vimeo_exp23_lama_multiscale/best.pt \
    --run-name vimeo_exp26_lama_resume \
    --checkpoint-dir ./checkpoints/vimeo_exp26_lama_resume

# ---------------------------------------------
# Exp 27: Resume ExplicitComposite from exp24 (best checkpoint)
# ---------------------------------------------
./train.sh --vimeo --steps $STEPS_RESUME --val-every $VAL_EVERY_PROD --save-every $SAVE_EVERY_PROD --num-frames 4 \
    --model composite --channels 64 \
    --multi-scale 256,320,384 \
    --resume ./checkpoints/vimeo_exp24_composite_multiscale/best.pt \
    --run-name vimeo_exp27_composite_resume \
    --checkpoint-dir ./checkpoints/vimeo_exp27_composite_resume

# ---------------------------------------------
# Exp 25: Production NAFUNet - Mixed artifact sizes
# ---------------------------------------------
# Train with varying size_scale for robustness
# size_scale=1.0 (default) + 1.5 (larger) + 0.7 (smaller)
# Note: Requires dataset augmentation support or multiple runs

# ./train.sh $COMMON_PROD_4F --channels 64 --mask-guidance modulation \
#     --size-scale 1.0 \
#     --run-name vimeo_exp25_prod_nafunet_size1.0 \
#     --checkpoint-dir ./checkpoints/vimeo_exp25_prod_nafunet_size1.0

# ./train.sh $COMMON_PROD_4F --channels 64 --mask-guidance modulation \
#     --size-scale 1.5 \
#     --run-name vimeo_exp25_prod_nafunet_size1.5 \
#     --checkpoint-dir ./checkpoints/vimeo_exp25_prod_nafunet_size1.5

# ---------------------------------------------
# Exp 26: Production ExplicitComposite - 4 frames
# ---------------------------------------------
# Explicit composite guarantees clean region preservation
# Good for production where artifacts have clear boundaries

# ./train.sh $COMMON_PROD_4F --model composite --channels 64 \
#     --run-name vimeo_exp26_prod_composite_4f \
#     --checkpoint-dir ./checkpoints/vimeo_exp26_prod_composite_4f

# ---------------------------------------------
# Exp 27: Final Production Model - 50k steps
# ---------------------------------------------
# Longest training with best architecture from exp22-26
# Run after evaluating exp22-26 results

# STEPS_FINAL=50000
# COMMON_FINAL="--vimeo --steps $STEPS_FINAL --val-every $VAL_EVERY_PROD --save-every $SAVE_EVERY_PROD --num-frames 4"
#
# ./train.sh $COMMON_FINAL --channels 64 --mask-guidance modulation \
#     --run-name vimeo_exp27_final_production \
#     --checkpoint-dir ./checkpoints/vimeo_exp27_final_production

# ===========================================
# Phase 11: Multi-Scale Training (Resolution Robustness)
# ===========================================
# Train with varying resolutions for robustness to different input sizes.
# Uses aspect-ratio-preserving resize (no cropping).
# batch_size=1 due to variable output sizes.

# ---------------------------------------------
# Exp 28: Multi-Scale NAFUNet - 4 frames, 30k steps
# ---------------------------------------------
# Scales: 256, 320, 384 (randomly sampled per example)
# Preserves aspect ratio, aligns to multiples of 8

# ./train.sh $COMMON_PROD_4F --channels 64 --mask-guidance modulation \
#     --multi-scale 256,320,384 \
#     --run-name vimeo_exp28_multiscale_nafunet \
#     --checkpoint-dir ./checkpoints/vimeo_exp28_multiscale_nafunet

# ---------------------------------------------
# Exp 29: Multi-Scale with wider range
# ---------------------------------------------
# Scales: 192, 256, 320, 384, 448

# ./train.sh $COMMON_PROD_4F --channels 64 --mask-guidance modulation \
#     --multi-scale 192,256,320,384,448 \
#     --run-name vimeo_exp29_multiscale_wide \
#     --checkpoint-dir ./checkpoints/vimeo_exp29_multiscale_wide

# ===========================================
# Phase 12: Long Video Optimizations (Future)
# ===========================================
# After best model is identified:
# - Add overlapping clip training with consistency loss
# - Progressive clip length curriculum
# - Hidden state for cross-clip memory

# Exp 30: Overlapping clip training (requires code changes)
# ./train.sh $COMMON_PROD_4F --channels 64 --mask-guidance modulation \
#     --overlap-training --overlap 2 --consistency-loss 0.1 \
#     --run-name vimeo_exp30_overlap_training \
#     --checkpoint-dir ./checkpoints/vimeo_exp30_overlap_training

echo "Sweep complete!"
