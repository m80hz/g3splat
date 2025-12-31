#!/bin/bash

#SBATCH -p a100 -A strategic               # Partition/queue with A100 GPUs
#SBATCH --nodes=1             # 6 nodes
#SBATCH --ntasks-per-node=1   # 4 tasks per node (total 24 tasks for 6 nodes)
#SBATCH --gres=gpu:1          # 4 GPUs per node (A100)
#SBATCH --cpus-per-task=4     # Number of CPU cores per GPU process
#SBATCH --time=24:00:00       # Time limit (D-HH:MM)
#SBATCH --mem=64GB            # Memory per node

# Notification configuration 
#SBATCH --mail-type=END       # Send a notification email when the job is done
#SBATCH --mail-type=FAIL      # Send a notification email when the job fails
#SBATCH --mail-user=mehdi.hosseinzadeh@adelaide.edu.au  # Email to receive notifications

source ~/.bashrc
module load CUDA/11.8.0
module load GCC/11.2.0
module use /apps/icl/modules/all

source ~/.bashrc
conda activate /hpcfs/users/$USER/projects/g3splat/.conda

# --- redirect temp & cache dirs to a directory with sufficient space ---
export SCRATCH_BASE="${SLURM_TMPDIR:-/scratchdata1/users/$USER}"
export TMPDIR="$SCRATCH_BASE/tmp-$SLURM_JOB_ID"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export WANDB_CACHE_DIR="$TMPDIR/wandb_cache"  # artifacts/cache
export CUDA_CACHE_PATH="$TMPDIR/cuda"         # CUDA kernel cache

mkdir -p "$TMPDIR" "$WANDB_CACHE_DIR" "$CUDA_CACHE_PATH"

echo "Starting job: $SLURM_JOB_NAME with ID $SLURM_JOB_ID"
echo "Nodes allocated: $SLURM_NODELIST"
echo "[temps] TMPDIR=$TMPDIR"
echo "[temps] WANDB_CACHE_DIR=$WANDB_CACHE_DIR"
echo "[temps] CUDA_CACHE_PATH=$CUDA_CACHE_PATH"

echo "PYTHON=$(which python)"

srun --export=ALL python -m src.main \
                            +experiment=re10k_align_orient \
                            mode=test \
                            wandb.name=test_re10k_align_orient \
                            dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                            dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
                            checkpointing.load=./pretrained_weights/g3splat_mast3r_3dgs_align_orient_re10k.ckpt \
                            > "nvs_re10k_align_orient_without-pose-refinement_g3splat_mast3r_3dgs_align_orient_re10k.ckpt.txt" 2>&1
