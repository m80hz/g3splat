# Train on RE10k on the cluster (6 nodes, 4 GPUs per node)
# Change the experiment config to train with different parameters, for example: re10k_grid or re10k_normal
python -m src.main +experiment=re10k_grid_normal wandb.mode=online


# Train on RE10k on a single GPU
# Change the experiment config to train with different parameters, for example: re10k_grid_1x8 or re10k_normal_1x8
# python -m src.main +experiment=re10k_grid_normal_1x8 wandb.mode=online
