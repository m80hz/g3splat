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
conda activate /hpcfs/users/$USER/project/Gen2DGS/.conda

# Redirect temporary files to a directory with sufficient space
# export TMPDIR=/hpcfs/users/$USER/tmp
# export TMP=/hpcfs/users/$USER/tmp
# export TEMP=/hpcfs/users/$USER/tmp
# export WANDB_CACHE_DIR=/hpcfs/users/$USER/tmp/wandb_cache
# mkdir -p $TMPDIR
# mkdir -p $WANDB_CACHE_DIR

echo "Starting job: $SLURM_JOB_NAME with ID $SLURM_JOB_ID"
echo "Using TMPDIR: $TMPDIR"
echo "Nodes allocated: $SLURM_NODELIST"

srun --export=ALL python -m src.main \
                            +experiment=re10k_grid_normal \
                            mode=test \
                            wandb.name=test_re10k_grid_normal_v2 \
                            dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                            dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
                            test.save_image=false \
                            checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.ckpt \
                            > "nvs_re10k_grid_normal_with-pose-refinement_ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.txt" 2>&1
    