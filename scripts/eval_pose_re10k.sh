# In-Domain Pose Evaluation on RE10k
CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=re10k_grid_1x8 \
                        +evaluation=eval_pose \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_hpc_2025-03-01_22-56-21_step_18748.ckpt
