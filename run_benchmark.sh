#!/bin/bash
# =============================================================================
# BENCHMARK RUN - PURE CLASSIFIER (Auto-restart every 2k steps)
# =============================================================================

cd "/Users/akshgarg/Downloads/Physics Informed Neural Network/flare-pinn"

# Activate conda PROPERLY
eval "$(conda shell.bash hook)"
conda activate flare-pinn

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Set WANDB_DIR to save logs outside the repo
export WANDB_DIR=~/flare-pinn-personal-keep/wandb
mkdir -p "$WANDB_DIR"

CONFIG="src/configs/benchmark_classifier.yaml"
CKPT_DIR="outputs/checkpoints/benchmark_classifier"

# Ensure directory exists
mkdir -p "$CKPT_DIR"

# Auto-restart loop
while true; do
    echo "============================================================================="
    echo "🚀 BENCHMARK RUN at $(date)"
    echo "   Config: $CONFIG"
    echo "   Goal: Establish pure classifier ceiling (No Physics)"
    echo "============================================================================="
    
    # Find latest checkpoint in BENCHMARK folder
    LATEST=$(ls -t "$CKPT_DIR"/checkpoint_step_*.pt 2>/dev/null | head -1)
    
    if [ -n "$LATEST" ]; then
        echo "   → Continuing from $LATEST"
        # Set fresh_lr_schedule to false when resuming
        sed -i '' 's/fresh_lr_schedule: true/fresh_lr_schedule: false/' "$CONFIG"
        python3 src/train.py --config "$CONFIG" --resume "$LATEST"
    else
        echo "   → Starting FRESH benchmark run"
        # Ensure fresh_lr_schedule is true for first run
        sed -i '' 's/fresh_lr_schedule: false/fresh_lr_schedule: true/' "$CONFIG"
        python3 src/train.py --config "$CONFIG"
    fi
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Benchmark completed!"
        break
    else
        echo "🔄 AUTO-RESTART (Memory Cleanup) - sleeping 5s..."
        sleep 5
    fi
done

