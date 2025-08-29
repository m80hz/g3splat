#!/usr/bin/env bash
set -euo pipefail

# Wrapper to evaluate a checkpoint on ScanNetV1 across tasks.
# Usage:
#   scripts/eval_scannet.sh -c <ckpt> [-g 0] [-o results]

CHECKPOINT=""
GPU=0
OUT_DIR="results"
DEPTH_EXP="scannet_depth_grid_normal"  # default depth and nvs experiments for ScanNet
POSE_EXP="scannet_pose_grid_normal"    # default pose experiment for ScanNet
RUN_POSE=true
RUN_DEPTH=true
RUN_NVS=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--checkpoint) CHECKPOINT="$2"; shift 2;;
    -g|--gpu) GPU="$2"; shift 2;;
    -o|--out) OUT_DIR="$2"; shift 2;;
    --pose-exp) POSE_EXP="$2"; shift 2;;
    --no-pose) RUN_POSE=false; shift 1;;
    --pose-only) RUN_DEPTH=false; RUN_POSE=true; shift 1;;
    -h|--help)
      echo "Usage: $0 -c <ckpt> [-g <gpu>] [-o <out>] [--pose-exp <exp>] [--no-pose|--pose-only]"; exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$CHECKPOINT" ]]; then
  echo "--checkpoint is required"; exit 2
fi

mkdir -p "$OUT_DIR"

# Depth on ScanNet (default experiment: $DEPTH_EXP)
if [[ "$RUN_DEPTH" == true ]]; then
  ./scripts/eval_checkpoint.sh -c "$CHECKPOINT" -e "$DEPTH_EXP" \
    --only depth --gpu "$GPU" --out "$OUT_DIR"
fi

# Pose on ScanNet (default experiment: $POSE_EXP)
if [[ "$RUN_POSE" == true ]]; then
  ./scripts/eval_checkpoint.sh -c "$CHECKPOINT" -e "$POSE_EXP" \
    --only pose --gpu "$GPU" --out "$OUT_DIR"
fi

## NVS on ScanNet (run by default after depth/pose). This will run both with- and without-pose-refinement
if [[ "$RUN_NVS" == true ]]; then
  ./scripts/eval_checkpoint.sh -c "$CHECKPOINT" -e "$DEPTH_EXP" --only nvs --gpu "$GPU" --out "$OUT_DIR"
fi
