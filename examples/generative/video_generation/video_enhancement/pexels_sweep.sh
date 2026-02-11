#!/bin/bash
# Experiment sweep on Pexels Animals dataset
#
# 3 best architectures from vimeo_sweep.sh (exp25-27), retrained on Pexels.
# Pexels is the default dataset — no --vimeo flag needed.
#
# Run all 3 sequentially on one GPU:
#   CUDA_VISIBLE_DEVICES=0 ./pexels_sweep.sh
#
# Or run in parallel on separate GPUs:
#   CUDA_VISIBLE_DEVICES=0 ./pexels_sweep.sh 1
#   CUDA_VISIBLE_DEVICES=1 ./pexels_sweep.sh 2
#   CUDA_VISIBLE_DEVICES=0 ./pexels_sweep.sh 3   # after exp1 finishes
#
# Monitor with TensorBoard:
#   ./tensorboard.sh
#   Open http://localhost:6006

set -e
trap "echo 'Interrupted. Exiting sweep.'; exit 1" INT TERM

STEPS=30000
VAL_EVERY=1000
SAVE_EVERY=10000
NUM_FRAMES=4

COMMON="--steps $STEPS --val-every $VAL_EVERY --save-every $SAVE_EVERY --num-frames $NUM_FRAMES"

# Which experiment to run (empty = all)
EXP="${1:-}"

# ---------------------------------------------
# Exp 1: NAFUNet + modulation + LPIPS (best from Vimeo sweep)
# ---------------------------------------------
# Why: Clear winner on Vimeo (exp9/25). 64 channels, modulation mask
# guidance, multi-scale training. LPIPS perceptual loss is critical.
if [ -z "$EXP" ] || [ "$EXP" = "1" ]; then
echo "=== Exp 1: NAFUNet 64ch + modulation (best from Vimeo) ==="
./train.sh $COMMON --channels 64 --mask-guidance modulation \
    --multi-scale 256,320,384 \
    --run-name pexels_exp1_nafunet_modulation \
    --checkpoint-dir ./checkpoints/pexels_exp1_nafunet_modulation
fi

# ---------------------------------------------
# Exp 2: ExplicitComposite (runner-up from Vimeo sweep)
# ---------------------------------------------
# Why: Predicts mask + inpaint separately, composites via alpha blending.
# Guarantees clean regions are preserved (mask=0 -> output=input).
# Slightly behind NAFUNet on Vimeo but more interpretable.
if [ -z "$EXP" ] || [ "$EXP" = "2" ]; then
echo "=== Exp 2: ExplicitComposite 64ch ==="
./train.sh $COMMON --model composite --channels 64 \
    --multi-scale 256,320,384 \
    --run-name pexels_exp2_composite \
    --checkpoint-dir ./checkpoints/pexels_exp2_composite
fi

# ---------------------------------------------
# Exp 3: NAFUNet + modulation, wider multi-scale range
# ---------------------------------------------
# Why: Pexels videos have more resolution variety than Vimeo (SD/HD mixed,
# portrait + landscape). Wider scale range [192-448] should improve
# robustness on real-world inputs at various resolutions.
if [ -z "$EXP" ] || [ "$EXP" = "3" ]; then
echo "=== Exp 3: NAFUNet 64ch + modulation, wide multi-scale ==="
./train.sh $COMMON --channels 64 --mask-guidance modulation \
    --multi-scale 192,256,320,384,448 \
    --run-name pexels_exp3_nafunet_widescale \
    --checkpoint-dir ./checkpoints/pexels_exp3_nafunet_widescale
fi

echo "Sweep complete!"
