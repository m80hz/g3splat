#!/usr/bin/env bash
set -euo pipefail

print_usage() {
  cat <<EOF
Usage: $0 --checkpoint PATH --experiment NAME [options]

Required:
  -c, --checkpoint PATH         Path to the checkpoint file to evaluate
  -e, --experiment NAME         Experiment name (value for +experiment)

Options:
  -g, --gpu ID                  CUDA device id (default: 0)
  -o, --out DIR                 Output directory for log files (default: .)
  --only TASKS                  Which evaluations to run. Accepts:
                                - both (pose+depth), pose, depth
                                - nvs
                                - all (pose+depth+nvs)
                                - comma-separated list: e.g. pose,nvs
  --no-refinement               Skip the "with pose refinement" runs (only run without refinement)
  --index PATH                  Evaluation index file (adds dataset view_sampler overrides)
  --view-ns NAME                Dataset view namespace (default: dataset.re10k)
  --wandb-name NAME             WandB run name (NVS only, optional)
  --nvs-save-with BOOL          test.save_image in NVS with-refinement run (default: false)
  --nvs-save-without BOOL       test.save_image in NVS without-refinement run (default: false)
  --dry-run                     Print commands without executing
  --fail-fast                   Exit immediately on first failure (default: continue and report at end)
  --extra "ARGS"                Extra Hydra overrides to append to ALL runs
  --pose-extra "ARGS"           Extra Hydra overrides to append to pose runs
  --depth-extra "ARGS"          Extra Hydra overrides to append to depth runs
  --nvs-extra "ARGS"            Extra Hydra overrides to append to NVS runs
  -h, --help                    Show this help and exit

Examples:
  - Pose+Depth (default):
    $0 -c ./pretrained_weights/model.ckpt -e scannet_pose_align_orient

  - Depth only, GPU 1, results dir:
    $0 --checkpoint ./pretrained_weights/m.ckpt --experiment scannet_depth_align_orient --only depth --gpu 1 --out results

  - NVS on RE10K with evaluation index and WandB name:
    $0 -c ./pretrained_weights/model.ckpt -e re10k_align_orient_1x8 --only nvs \
       --index assets/evaluation_index_re10k.json --wandb-name test_re10k_align_orient

  - Pose on ACID with index overrides:
    $0 -c ./pretrained_weights/model.ckpt -e acid_align_orient --only pose \
       --index assets/evaluation_index_acid.json

Notes:
  See dataset wrappers in scripts/eval_*.sh to orchestrate multiple tasks for a dataset.
EOF
}

CHECKPOINT=""
EXPERIMENT=""
GPU=0
OUT_DIR="."
ONLY="both"
RUN_WITH_REFINEMENT=true
INDEX=""
VIEW_NS="dataset.re10k"
WANDB_NAME=""
NVS_SAVE_WITH=false
NVS_SAVE_WITHOUT=false
EXTRA=""
POSE_EXTRA=""
DEPTH_EXTRA=""
NVS_EXTRA=""
DRY_RUN=false
FAIL_FAST=false
# Support repeatable extras via arrays (comma-separated supported)
EXTRA_LIST=()
POSE_EXTRA_LIST=()
DEPTH_EXTRA_LIST=()
NVS_EXTRA_LIST=()

if [[ $# -eq 0 ]]; then
  print_usage
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--checkpoint)
      CHECKPOINT="$2"; shift 2;;
    -e|--experiment)
      EXPERIMENT="$2"; shift 2;;
    -g|--gpu)
      GPU="$2"; shift 2;;
    -o|--out)
      OUT_DIR="$2"; shift 2;;
    --only)
      ONLY="$2"; shift 2;;
    --no-refinement)
      RUN_WITH_REFINEMENT=false; shift 1;;
    --index)
      INDEX="$2"; shift 2;;
    --view-ns)
      VIEW_NS="$2"; shift 2;;
    --wandb-name)
      WANDB_NAME="$2"; shift 2;;
    --nvs-save-with)
      NVS_SAVE_WITH="$2"; shift 2;;
    --nvs-save-without)
      NVS_SAVE_WITHOUT="$2"; shift 2;;
    --dry-run)
      DRY_RUN=true; shift 1;;
    --fail-fast)
      FAIL_FAST=true; shift 1;;
    --extra)
      VAL="$2"; shift 2;
      if [[ "$VAL" == *,* ]]; then IFS=',' read -r -a TMP <<< "$VAL"; EXTRA_LIST+=("${TMP[@]}"); else EXTRA_LIST+=("$VAL"); fi
      ;;
    --pose-extra)
      VAL="$2"; shift 2;
      if [[ "$VAL" == *,* ]]; then IFS=',' read -r -a TMP <<< "$VAL"; POSE_EXTRA_LIST+=("${TMP[@]}"); else POSE_EXTRA_LIST+=("$VAL"); fi
      ;;
    --depth-extra)
      VAL="$2"; shift 2;
      if [[ "$VAL" == *,* ]]; then IFS=',' read -r -a TMP <<< "$VAL"; DEPTH_EXTRA_LIST+=("${TMP[@]}"); else DEPTH_EXTRA_LIST+=("$VAL"); fi
      ;;
    --nvs-extra)
      VAL="$2"; shift 2;
      if [[ "$VAL" == *,* ]]; then IFS=',' read -r -a TMP <<< "$VAL"; NVS_EXTRA_LIST+=("${TMP[@]}"); else NVS_EXTRA_LIST+=("$VAL"); fi
      ;;
    -h|--help)
      print_usage; exit 0;;
    *)
      echo "Unknown option: $1"; print_usage; exit 1;;
  esac
done

if [[ -z "$CHECKPOINT" || -z "$EXPERIMENT" ]]; then
  echo "Error: --checkpoint and --experiment are required."
  print_usage
  exit 2
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Error: checkpoint not found: $CHECKPOINT"
  exit 3
fi

mkdir -p "$OUT_DIR"

BASE_NAME=$(basename "$CHECKPOINT")
SAFE_NAME=$(echo "$BASE_NAME" | sed 's/[^a-zA-Z0-9._-]/_/g')

# Build task list from --only
declare -a EVALS
if [[ "$ONLY" == *","* ]]; then
  IFS=',' read -r -a EVALS <<< "$ONLY"
else
  case "$ONLY" in
    both) EVALS=(eval_pose eval_depth) ;;
    pose) EVALS=(eval_pose) ;;
    depth) EVALS=(eval_depth) ;;
    nvs) EVALS=(nvs) ;;
    all) EVALS=(eval_pose eval_depth nvs) ;;
    *) echo "Unknown value for --only: $ONLY"; exit 4 ;;
  esac
fi

# Common view overrides if index provided
build_view_overrides() {
  local -n _arr=$1
  if [[ -n "$INDEX" ]]; then
    _arr+=("dataset/view_sampler@${VIEW_NS}.view_sampler=evaluation")
    _arr+=("${VIEW_NS}.view_sampler.index_path=$INDEX")
  fi
}

# Track failures and provide a safe runner
FAILED=0
run_cmd() {
  local desc="$1"; shift
  local out_file="$1"; shift
  echo "Running: $desc -> $out_file"
  if [[ "$DRY_RUN" == true ]]; then
    echo "DRY-RUN: $*" | tee "$out_file" >/dev/null
    return 0
  fi
  set +e
  CUDA_VISIBLE_DEVICES=$GPU "$@" > "$out_file" 2>&1
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "ERROR: $desc failed (exit $rc). See $out_file"
    FAILED=1
    if [[ "$FAIL_FAST" == true ]]; then
      exit $rc
    fi
  else
    echo "SUCCESS: $desc"
  fi
  return $rc
}

echo "Running evaluations for checkpoint=$CHECKPOINT experiment=$EXPERIMENT on GPU=$GPU"

for E in "${EVALS[@]}"; do
  case "$E" in
    eval_pose)
      MODULE="src.eval_pose"; PREFIX="pose";
      BASE_ARGS=(+experiment="$EXPERIMENT" +evaluation=eval_pose checkpointing.load="$CHECKPOINT")
      build_view_overrides BASE_ARGS
      EXTRA_ARGS=()
      # combine legacy single-string flags and repeatable lists
      [[ -n "$EXTRA" ]] && EXTRA_ARGS+=("$EXTRA")
      [[ -n "$POSE_EXTRA" ]] && EXTRA_ARGS+=("$POSE_EXTRA")
      [[ ${#EXTRA_LIST[@]} -gt 0 ]] && EXTRA_ARGS+=("${EXTRA_LIST[@]}")
      [[ ${#POSE_EXTRA_LIST[@]} -gt 0 ]] && EXTRA_ARGS+=("${POSE_EXTRA_LIST[@]}")
      # without refinement
      OUT_FILE="$OUT_DIR/${PREFIX}_${EXPERIMENT}-without_pose_refinement-${SAFE_NAME}.txt"
      echo "-> $MODULE ${BASE_ARGS[*]} evaluation.use_pose_refinement=false -> $OUT_FILE"
      run_cmd "$MODULE ${BASE_ARGS[*]} evaluation.use_pose_refinement=false" "$OUT_FILE" \
        python -m $MODULE "${BASE_ARGS[@]}" evaluation.use_pose_refinement=false "${EXTRA_ARGS[@]}"
      # with refinement
      if [[ "$RUN_WITH_REFINEMENT" == true ]]; then
        OUT_FILE="$OUT_DIR/${PREFIX}_${EXPERIMENT}-with_pose_refinement-${SAFE_NAME}.txt"
        echo "-> $MODULE ${BASE_ARGS[*]} (with refinement) -> $OUT_FILE"
        run_cmd "$MODULE ${BASE_ARGS[*]} (with refinement)" "$OUT_FILE" \
          python -m $MODULE "${BASE_ARGS[@]}" "${EXTRA_ARGS[@]}"
      fi
      ;;
    eval_depth)
      MODULE="src.eval_depth"; PREFIX="depth";
      BASE_ARGS=(+experiment="$EXPERIMENT" +evaluation=eval_depth checkpointing.load="$CHECKPOINT")
      build_view_overrides BASE_ARGS
      EXTRA_ARGS=()
      [[ -n "$EXTRA" ]] && EXTRA_ARGS+=("$EXTRA")
      [[ -n "$DEPTH_EXTRA" ]] && EXTRA_ARGS+=("$DEPTH_EXTRA")
      [[ ${#EXTRA_LIST[@]} -gt 0 ]] && EXTRA_ARGS+=("${EXTRA_LIST[@]}")
      [[ ${#DEPTH_EXTRA_LIST[@]} -gt 0 ]] && EXTRA_ARGS+=("${DEPTH_EXTRA_LIST[@]}")
      # without refinement
      OUT_FILE="$OUT_DIR/${PREFIX}_${EXPERIMENT}-without_pose_refinement-${SAFE_NAME}.txt"
      echo "-> $MODULE ${BASE_ARGS[*]} evaluation.use_pose_refinement=false -> $OUT_FILE"
      run_cmd "$MODULE ${BASE_ARGS[*]} evaluation.use_pose_refinement=false" "$OUT_FILE" \
        python -m $MODULE "${BASE_ARGS[@]}" evaluation.use_pose_refinement=false "${EXTRA_ARGS[@]}"
      # with refinement
      if [[ "$RUN_WITH_REFINEMENT" == true ]]; then
        OUT_FILE="$OUT_DIR/${PREFIX}_${EXPERIMENT}-with_pose_refinement-${SAFE_NAME}.txt"
        echo "-> $MODULE ${BASE_ARGS[*]} (with refinement) -> $OUT_FILE"
        run_cmd "$MODULE ${BASE_ARGS[*]} (with refinement)" "$OUT_FILE" \
          python -m $MODULE "${BASE_ARGS[@]}" "${EXTRA_ARGS[@]}"
      fi
      ;;
    nvs)
      MODULE="src.main"; PREFIX="nvs";
      BASE_ARGS=(+experiment="$EXPERIMENT" mode=test checkpointing.load="$CHECKPOINT")
      build_view_overrides BASE_ARGS
      # Ensure default WandB name if not provided
      if [[ -z "$WANDB_NAME" ]]; then WANDB_NAME="test_${EXPERIMENT}"; fi
      BASE_ARGS+=("wandb.name=$WANDB_NAME")
      EXTRA_ARGS=()
      [[ -n "$EXTRA" ]] && EXTRA_ARGS+=("$EXTRA")
      [[ -n "$NVS_EXTRA" ]] && EXTRA_ARGS+=("$NVS_EXTRA")
      [[ ${#EXTRA_LIST[@]} -gt 0 ]] && EXTRA_ARGS+=("${EXTRA_LIST[@]}")
      [[ ${#NVS_EXTRA_LIST[@]} -gt 0 ]] && EXTRA_ARGS+=("${NVS_EXTRA_LIST[@]}")
      # with refinement (align_pose must be explicitly enabled)
      if [[ "$RUN_WITH_REFINEMENT" == true ]]; then
        OUT_FILE="$OUT_DIR/${PREFIX}_${EXPERIMENT}-with_pose_refinement-${SAFE_NAME}.txt"
        echo "-> $MODULE ${BASE_ARGS[*]} test.align_pose=true test.save_image=$NVS_SAVE_WITH -> $OUT_FILE"
        run_cmd "$MODULE ${BASE_ARGS[*]} test.align_pose=true test.save_image=$NVS_SAVE_WITH" "$OUT_FILE" \
          python -m $MODULE "${BASE_ARGS[@]}" test.align_pose=true "test.save_image=$NVS_SAVE_WITH" "${EXTRA_ARGS[@]}"
      fi
      # without refinement (align_pose=false is default)
      OUT_FILE="$OUT_DIR/${PREFIX}_${EXPERIMENT}-without_pose_refinement-${SAFE_NAME}.txt"
      echo "-> $MODULE ${BASE_ARGS[*]} test.align_pose=false test.save_image=$NVS_SAVE_WITHOUT -> $OUT_FILE"
      run_cmd "$MODULE ${BASE_ARGS[*]} test.align_pose=false test.save_image=$NVS_SAVE_WITHOUT" "$OUT_FILE" \
        python -m $MODULE "${BASE_ARGS[@]}" test.align_pose=false "test.save_image=$NVS_SAVE_WITHOUT" "${EXTRA_ARGS[@]}"
      ;;
    *) echo "Unknown task: $E"; exit 5;;
  esac
done

echo "All requested evaluations executed. Results are in: $OUT_DIR"
if [[ $FAILED -ne 0 ]]; then
  echo "One or more evaluations failed. Inspect logs under $OUT_DIR"
  exit 1
fi
