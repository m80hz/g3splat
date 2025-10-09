#!/usr/bin/env bash
set -euo pipefail

# Wrapper to evaluate a checkpoint on DTU across tasks (Depth, Pose, NVS).
# Usage:
#   scripts/eval_dtu.sh -c <ckpt> [-g 0] [-o results] [-e <exp>] [--wandb-name <name>]

CHECKPOINT=""
GPU=0
OUT_DIR="results"
EXP="dtu_grid_normal_1x8"  # default DTU experiment
WANDB_NAME=""
INDEX="assets/evaluation_index_dtu.json"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--checkpoint) CHECKPOINT="$2"; shift 2;;
    -g|--gpu) GPU="$2"; shift 2;;
    -o|--out) OUT_DIR="$2"; shift 2;;
    -e|--experiment) EXP="$2"; shift 2;;
    --wandb-name) WANDB_NAME="$2"; shift 2;;
    -h|--help)
      echo "Usage: $0 -c <ckpt> [-g <gpu>] [-o <out>] [-e <exp>] [--wandb-name <name>]"; exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$CHECKPOINT" ]]; then
  echo "--checkpoint is required"; exit 2
fi

mkdir -p "$OUT_DIR"

# Depth on DTU (GT depth available)
./scripts/eval_checkpoint.sh -c "$CHECKPOINT" -e "$EXP" \
  --only depth --gpu "$GPU" --out "$OUT_DIR"

# Pose on DTU
./scripts/eval_checkpoint.sh -c "$CHECKPOINT" -e "$EXP" \
  --only pose --gpu "$GPU" --out "$OUT_DIR" \
  --index "$INDEX" --view-ns dataset.re10k

# NVS on DTU (with the same evaluation index)
./scripts/eval_checkpoint.sh -c "$CHECKPOINT" -e "$EXP" \
  --only nvs --gpu "$GPU" --out "$OUT_DIR" \
  --index "$INDEX" --view-ns dataset.re10k \
  ${WANDB_NAME:+--wandb-name "$WANDB_NAME"}
