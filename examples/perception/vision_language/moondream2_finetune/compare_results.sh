#!/bin/bash

echo "================================"
echo "MOONDREAM2 TRAINING COMPARISON"
echo "================================"
echo ""

echo "=== Standard Causal Attention (baseline) ==="
echo "Settings: grad_accum=8, eval_samples=20"
if [ -f training_standard_causal.log ]; then
    echo "Bidirectional: $(grep 'Bidirectional image attention:' training_standard_causal.log | head -1)"
    echo ""
    echo "Validation checkpoints:"
    grep "Validation Accuracy:" training_standard_causal.log | tail -5
    echo ""
    echo "Final metrics:"
    tail -10 training_standard_causal.log | grep -E "(loss|token_acc|Final Validation)"
else
    echo "Log file not found or training still running..."
fi

echo ""
echo "=== 730-Token Bidirectional Attention ==="
echo "Settings: grad_accum=8, eval_samples=20"
if [ -f training_bidirectional_730.log ]; then
    echo "Bidirectional: $(grep 'Bidirectional image attention:' training_bidirectional_730.log | head -1)"
    echo ""
    echo "Validation checkpoints:"
    grep "Validation Accuracy:" training_bidirectional_730.log | tail -5
    echo ""
    echo "Final metrics:"
    tail -10 training_bidirectional_730.log | grep -E "(loss|token_acc|Final Validation)"
else
    echo "Log file not found or training still running..."
fi

echo ""
echo "================================"
