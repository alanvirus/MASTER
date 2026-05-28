#!/usr/bin/env bash
# Run a single baseline across 5 seeds.
# Usage: ./run_seeds.sh <model_dir>      e.g. ./run_seeds.sh lstm
# Reads <model_dir>/<model_dir>.yaml as the seed-0 template, generates seed-1..4 copies via sed,
# then qrun each. Logs to <model_dir>/<model_dir>_s${seed}.log.

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
MODEL="${1:?usage: $0 <model_dir>}"
DIR="$ROOT/$MODEL"
TPL="$DIR/$MODEL.yaml"

if [ ! -f "$TPL" ]; then
    echo "template not found: $TPL" >&2
    exit 1
fi

cd "$DIR"
for s in 0 1 2 3 4; do
    out="${MODEL}_s${s}.yaml"
    if [ "$s" -eq 0 ]; then
        cp "$MODEL.yaml" "$out"
    else
        sed "s/seed: 0/seed: $s/" "$MODEL.yaml" > "$out"
    fi
    echo ">>> qrun $out (seed=$s)"
    qrun "$out" 2>&1 | tee "${MODEL}_s${s}.log"
done
