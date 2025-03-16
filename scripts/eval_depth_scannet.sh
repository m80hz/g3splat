# Cross-Domain Zero-shot Depth Evaluation on ScanNetV1
CUDA_VISIBLE_DEVICES=0 python -m src.eval_depth \
                        +experiment=scannet_depth_grid \
                        +evaluation=eval_depth \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_hpc_2025-03-01_22-56-21_step_18748.ckpt
