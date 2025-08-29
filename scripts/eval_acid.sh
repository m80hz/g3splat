#!/usr/bin/env bash
set -euo pipefail

# Wrapper to evaluate a checkpoint on ACID across tasks.
# Usage:
#   scripts/eval_acid.sh -c <ckpt> -e <nvs_experiment> [-g 0] [-o results]

CHECKPOINT=""
EXPERIMENT="acid_grid_normal_1x8"   # NVS experiment for ACID
GPU=0
OUT_DIR="results"
WANDB_NAME=""
INDEX="assets/evaluation_index_acid.json"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--checkpoint) CHECKPOINT="$2"; shift 2;;
    -e|--experiment) EXPERIMENT="$2"; shift 2;;
    -g|--gpu) GPU="$2"; shift 2;;
    -o|--out) OUT_DIR="$2"; shift 2;;
    --wandb-name) WANDB_NAME="$2"; shift 2;;
    -h|--help)
      echo "Usage: $0 -c <ckpt> [-e <exp>] [-g <gpu>] [-o <out>] [--wandb-name <name>]"; exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$CHECKPOINT" ]]; then
  echo "--checkpoint is required"; exit 2
fi

mkdir -p "$OUT_DIR"

# NVS
./scripts/eval_checkpoint.sh -c "$CHECKPOINT" -e "$EXPERIMENT" \
  --only nvs --gpu "$GPU" --out "$OUT_DIR" \
  --index "$INDEX" --view-ns dataset.re10k \
  ${WANDB_NAME:+--wandb-name "$WANDB_NAME"}

# Pose + Depth experiments for ACID configs (pose uses acid_grid_normal)
./scripts/eval_checkpoint.sh -c "$CHECKPOINT" -e acid_grid_normal \
  --only pose --gpu "$GPU" --out "$OUT_DIR" \
  --index "$INDEX" --view-ns dataset.re10k
