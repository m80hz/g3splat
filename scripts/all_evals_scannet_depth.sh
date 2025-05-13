# This script runs all depth evaluations on ScanNetV1.

echo "Running all depth evaluations ..."

# Cross-Domain Zero-shot Depth Evaluation on ScanNetV1 (the current models in the paper)


CUDA_VISIBLE_DEVICES=1 python -m src.eval_depth \
                        +experiment=scannet_depth_grid_normal \
                        +evaluation=eval_depth \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.ckpt \
                        > "depth_scannet_depth_grid_normal-with_pose_refinement-ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.txt" 2>&1

CUDA_VISIBLE_DEVICES=1 python -m src.eval_depth \
                        +experiment=scannet_depth_grid_normal \
                        +evaluation=eval_depth \
                        evaluation.use_pose_refinement=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.ckpt \
                        > "depth_scannet_depth_grid_normal-without_pose_refinement-ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.txt" 2>&1




CUDA_VISIBLE_DEVICES=1 python -m src.eval_depth \
                        +experiment=scannet_depth_normal \
                        +evaluation=eval_depth \
                        checkpointing.load=./pretrained_weights/ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_18748.ckpt \
                        > "depth_scannet_depth_normal-with_pose_refinement-ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_18748.txt" 2>&1

CUDA_VISIBLE_DEVICES=1 python -m src.eval_depth \
                        +experiment=scannet_depth_normal \
                        +evaluation=eval_depth \
                        evaluation.use_pose_refinement=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_18748.ckpt \
                        > "depth_scannet_depth_normal-without_pose_refinement-ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_18748.txt" 2>&1




CUDA_VISIBLE_DEVICES=1 python -m src.eval_depth \
                        +experiment=scannet_depth_grid \
                        +evaluation=eval_depth \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_hpc_2025-03-01_22-56-21_step_18748.ckpt \
                        > "depth_scannet_depth_grid-with_pose_refinement-ours_re10k_grid_hpc_2025-03-01_22-56-21_step_18748.txt" 2>&1

CUDA_VISIBLE_DEVICES=1 python -m src.eval_depth \
                        +experiment=scannet_depth_grid \
                        +evaluation=eval_depth \
                        evaluation.use_pose_refinement=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_hpc_2025-03-01_22-56-21_step_18748.ckpt \
                        > "depth_scannet_depth_grid-without_pose_refinement-ours_re10k_grid_hpc_2025-03-01_22-56-21_step_18748.txt" 2>&1





CUDA_VISIBLE_DEVICES=1 python -m src.eval_depth \
                        +experiment=scannet_depth \
                        +evaluation=eval_depth \
                        checkpointing.load=./pretrained_weights/ours_re10k_hpc_2025-03-04_02-02-31_step_18748.ckpt \
                        > "depth_scannet_depth-with_pose_refinement-ours_re10k_hpc_2025-03-04_02-02-31_step_18748.txt" 2>&1

CUDA_VISIBLE_DEVICES=1 python -m src.eval_depth \
                        +experiment=scannet_depth \
                        +evaluation=eval_depth \
                        evaluation.use_pose_refinement=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_hpc_2025-03-04_02-02-31_step_18748.ckpt \
                        > "depth_scannet_depth-without_pose_refinement-ours_re10k_hpc_2025-03-04_02-02-31_step_18748.txt" 2>&1





CUDA_VISIBLE_DEVICES=1 python -m src.eval_depth \
                        +experiment=scannet_depth_grid_normal \
                        +evaluation=eval_depth \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_18748.ckpt \
                        > "depth_scannet_depth_grid_normal-with_pose_refinement-ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_18748.txt" 2>&1

CUDA_VISIBLE_DEVICES=1 python -m src.eval_depth \
                        +experiment=scannet_depth_grid_normal \
                        +evaluation=eval_depth \
                        evaluation.use_pose_refinement=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_18748.ckpt \
                        > "depth_scannet_depth_grid_normal-without_pose_refinement-ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_18748.txt" 2>&1




CUDA_VISIBLE_DEVICES=1 python -m src.eval_depth \
                        +experiment=scannet_depth_grid_normal \
                        +evaluation=eval_depth \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_hpc_2025-03-05_16-31-33_step_18748.ckpt \
                        > "depth_scannet_depth_grid_normal-with_pose_refinement-ours_re10k_grid_normal_hpc_2025-03-05_16-31-33_step_18748.txt" 2>&1

CUDA_VISIBLE_DEVICES=1 python -m src.eval_depth \
                        +experiment=scannet_depth_grid_normal \
                        +evaluation=eval_depth \
                        evaluation.use_pose_refinement=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_hpc_2025-03-05_16-31-33_step_18748.ckpt \
                        > "depth_scannet_depth_grid_normal-without_pose_refinement-ours_re10k_grid_normal_hpc_2025-03-05_16-31-33_step_18748.txt" 2>&1



# the 20001 models 


CUDA_VISIBLE_DEVICES=1 python -m src.eval_depth \
                        +experiment=scannet_depth_grid_normal \
                        +evaluation=eval_depth \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_20001.ckpt \
                        > "depth_scannet_depth_grid_normal-with_pose_refinement-ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_20001.txt" 2>&1

CUDA_VISIBLE_DEVICES=1 python -m src.eval_depth \
                        +experiment=scannet_depth_grid_normal \
                        +evaluation=eval_depth \
                        evaluation.use_pose_refinement=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_20001.ckpt \
                        > "depth_scannet_depth_grid_normal-without_pose_refinement-ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_20001.txt" 2>&1




CUDA_VISIBLE_DEVICES=1 python -m src.eval_depth \
                        +experiment=scannet_depth_normal \
                        +evaluation=eval_depth \
                        checkpointing.load=./pretrained_weights/ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_20001.ckpt \
                        > "depth_scannet_depth_normal-with_pose_refinement-ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_20001.txt" 2>&1

CUDA_VISIBLE_DEVICES=1 python -m src.eval_depth \
                        +experiment=scannet_depth_normal \
                        +evaluation=eval_depth \
                        evaluation.use_pose_refinement=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_20001.ckpt \
                        > "depth_scannet_depth_normal-without_pose_refinement-ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_20001.txt" 2>&1





CUDA_VISIBLE_DEVICES=1 python -m src.eval_depth \
                        +experiment=scannet_depth_grid_normal \
                        +evaluation=eval_depth \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_20001.ckpt \
                        > "depth_scannet_depth_grid_normal-with_pose_refinement-ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_20001.txt" 2>&1

CUDA_VISIBLE_DEVICES=1 python -m src.eval_depth \
                        +experiment=scannet_depth_grid_normal \
                        +evaluation=eval_depth \
                        evaluation.use_pose_refinement=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_20001.ckpt \
                        > "depth_scannet_depth_grid_normal-without_pose_refinement-ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_20001.txt" 2>&1






echo "All depth scripts executed." 

