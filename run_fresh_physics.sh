#!/bin/bash
# Auto-restart training script - FRESH START with High-Freq Fourier Features
# Architecture change: max_log2_freq 4 -> 6 requires fresh start

cd "$(dirname "$0")"

# Clear old checkpoints (incompatible architecture)
rm -f outputs/checkpoints/train_balanced/*.pt

echo "🚀 Starting FRESH training with HIGH-FREQ PHYSICS (max_log2_freq=6)"
echo "   ✅ Dropout: 0.40"
echo "   ✅ Physics scale: 0.20 (stronger)"
echo "   ✅ Max lambda: 0.7"
echo "   ✅ LR: 2e-4 (fresh start)"
echo ""

# Loop forever - script exits with code 42 for auto-restart
while true; do
    # Check if there's a checkpoint to resume from
    LATEST_CKPT=$(ls -t outputs/checkpoints/train_balanced/checkpoint_step_*.pt 2>/dev/null | head -1)
    
    if [ -n "$LATEST_CKPT" ]; then
        echo "🔄 Resuming from: $LATEST_CKPT"
        PYTHONPATH="$PWD:$PYTHONPATH" python src/train.py \
            --config src/configs/train_balanced.yaml \
            --wandb \
            --resume "$LATEST_CKPT"
    else
        echo "🆕 Starting FRESH run (architecture changed)..."
        PYTHONPATH="$PWD:$PYTHONPATH" python src/train.py \
            --config src/configs/train_balanced.yaml \
            --wandb
    fi
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 42 ]; then
        echo "🔄 Auto-restart triggered, continuing..."
        sleep 2
    elif [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Training completed successfully!"
        break
    else
        echo "❌ Training failed with exit code $EXIT_CODE"
        echo "   Check logs and restart manually if needed"
        break
    fi
done
