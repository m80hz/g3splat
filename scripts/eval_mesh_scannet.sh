# Cross-Domain Zero-shot Mesh Evaluation on ScanNetV1

CUDA_VISIBLE_DEVICES=1 python -m src.eval_mesh \
                        +experiment=scannet_depth_grid_normal \
                        +evaluation=eval_mesh \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v5_edge_aware_hpc_2025-08-13_21-02-57_18750.ckpt 
