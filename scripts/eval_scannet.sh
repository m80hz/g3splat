#!/usr/bin/env bash
set -euo pipefail

# Wrapper to evaluate a checkpoint on ScanNetV1 across tasks.
# Usage:
#   scripts/eval_scannet.sh -c <ckpt> [-g 0] [-o results] [-e <depth_exp>] [--pose-exp <pose_exp>] [--nvs-exp <nvs_exp>] [--wandb-name <name>]

CHECKPOINT=""
GPU=0
OUT_DIR="results"
DEPTH_EXP="scannet_depth_align_orient"  # default depth and nvs experiments for ScanNet
POSE_EXP="scannet_pose_align_orient"    # default pose experiment for ScanNet
NVS_EXP="$DEPTH_EXP"                    # default NVS experiment for ScanNet (default experiment: depth (same virtual views as depth evaluation))
RUN_POSE=true
RUN_DEPTH=true
RUN_NVS=true
NVS_SET=false
WANDB_NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--checkpoint) CHECKPOINT="$2"; shift 2;;
    -g|--gpu) GPU="$2"; shift 2;;
    -o|--out) OUT_DIR="$2"; shift 2;;
    -e|--experiment) DEPTH_EXP="$2"; shift 2;;
    --pose-exp) POSE_EXP="$2"; shift 2;;
    --depth-exp) DEPTH_EXP="$2"; shift 2;;
  --nvs-exp) NVS_EXP="$2"; NVS_SET=true; shift 2;;
  --wandb-name) WANDB_NAME="$2"; shift 2;;
    --no-pose) RUN_POSE=false; shift 1;;
    --pose-only) RUN_DEPTH=false; RUN_POSE=true; shift 1;;
    --no-nvs) RUN_NVS=false; shift 1;;
    --nvs-only) RUN_NVS=true; RUN_DEPTH=false; RUN_POSE=false; shift 1;;
    -h|--help)
  echo "Usage: $0 -c <ckpt> [-g <gpu>] [-o <out>] [-e <depth_exp>] [--pose-exp <pose_exp>] [--depth-exp <exp>] [--nvs-exp <exp>] [--wandb-name <name>] [--no-pose|--pose-only] [--no-nvs|--nvs-only]"; exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$CHECKPOINT" ]]; then
  echo "--checkpoint is required"; exit 2
fi

mkdir -p "$OUT_DIR"

# If user provided a depth experiment via -e/--depth-exp and did not explicitly set NVS exp,
# default NVS to the same as depth.
if [[ "$NVS_SET" == false ]]; then
  NVS_EXP="$DEPTH_EXP"
fi

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

## NVS on ScanNet. This will run both with- and without-pose-refinement (experiment: $NVS_EXP)
if [[ "$RUN_NVS" == true ]]; then
  ./scripts/eval_checkpoint.sh -c "$CHECKPOINT" -e "$NVS_EXP" --only nvs --gpu "$GPU" --out "$OUT_DIR" \
    ${WANDB_NAME:+--wandb-name "$WANDB_NAME"}
fi
