#!/bin/bash
# Train Nirasa-0.5B on Apple Silicon using MLX
# Usage: bash scripts/train_mlx.sh
set -e

echo "=== Nirasa-0.5B MLX Training ==="

# Install mlx-lm if needed
if ! .venv/bin/python -c "import mlx_lm" 2>/dev/null; then
    echo "Installing mlx-lm..."
    .venv/bin/pip install -q "mlx-lm[train]"
fi

# Step 1: Prepare data from dedup output
echo ""
echo "=== Preparing training data ==="
PYTHONPATH="$(pwd)" .venv/bin/python scripts/prepare_mlx_data.py

# Step 2: Train
echo ""
echo "=== Starting training ==="
.venv/bin/python -m mlx_lm.lora --config configs/training/mlx_0.5b_full.yaml

echo ""
echo "=== Training complete ==="
echo "Checkpoint saved to: ./checkpoints/nirasa-0.5b-th"
echo ""
echo "To test generation:"
echo "  .venv/bin/python -m mlx_lm.generate --model Qwen/Qwen2.5-0.5B --adapter-path ./checkpoints/nirasa-0.5b-th --prompt 'ประเทศไทยมี'"
