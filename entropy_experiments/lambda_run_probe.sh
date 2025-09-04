#!/bin/bash
# Lambda A100 Entropy Probe Launcher
# For Lord Krang's entropy experiments

set -e

echo "🚀 Lambda A100 Entropy Probe Launcher"
echo "======================================"

# Configuration
CONFIG="${1:-entropy_experiments/configs/A100_config.yaml}"
ITERATIONS="${2:-3}"
TEMPS="${3:-1.0}"

echo "📋 Config: $CONFIG"
echo "🔄 Iterations: $ITERATIONS"
echo "🌡️ Temperatures: $TEMPS"
echo ""

# Check GPU
echo "🔍 Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits || true
echo ""

# Run the probe
echo "⚡ Starting entropy probe..."
echo "======================================"

PYTHONPATH=. python entropy_experiments/run_a100_entropy_probe.py \
    --config "$CONFIG" \
    --iterations "$ITERATIONS" \
    --temps "$TEMPS" \
    --gpu-check

echo ""
echo "✅ Probe execution completed!"
echo ""

# Show results location
echo "📁 Results saved in:"
ls -la entropy_experiments/results/a100_run_* 2>/dev/null | tail -1 || echo "No results found"