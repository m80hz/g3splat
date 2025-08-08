#!/bin/bash

#SBATCH -p a100 -A strategic               # Partition/queue with A100 GPUs
#SBATCH --nodes=6             # 6 nodes
#SBATCH --ntasks-per-node=4   # 4 tasks per node (total 24 tasks for 6 nodes)
#SBATCH --gres=gpu:4          # 4 GPUs per node (A100)
#SBATCH --cpus-per-task=4     # Number of CPU cores per GPU process
#SBATCH --time=24:00:00       # Time limit (D-HH:MM)
#SBATCH --mem=250GB            # Memory per node

# Notification configuration 
#SBATCH --mail-type=END       # Send a notification email when the job is done
#SBATCH --mail-type=FAIL      # Send a notification email when the job fails
#SBATCH --mail-user=mehdi.hosseinzadeh@adelaide.edu.au  # Email to receive notifications

source ~/.bashrc
module load CUDA/11.8.0
module load GCC/11.2.0
module use /apps/icl/modules/all

source ~/.bashrc
conda activate /hpcfs/users/$USER/projects/Gen2DGS/.conda

echo "Starting job: $SLURM_JOB_NAME with ID $SLURM_JOB_ID"
echo "Using TMPDIR: $TMPDIR"
echo "Nodes allocated: $SLURM_NODELIST"

srun --export=ALL python -m src.main +experiment=re10k_grid_normal wandb.mode=offline
