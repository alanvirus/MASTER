#!/usr/bin/env bash
# Run all 6 baselines sequentially across 5 seeds each.
# Usage: ./run_all.sh

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
for m in lstm gru transformer gats tcn xgboost; do
    echo "=================================================="
    echo "  $m"
    echo "=================================================="
    "$ROOT/run_seeds.sh" "$m"
done
