# Cross-Domain Zero-shot Pose Evaluation on ACID
CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=acid_grid \
                        +evaluation=eval_pose \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_hpc_2025-03-01_22-56-21_step_18748.ckpt