echo "Running all nvs evaluations for scannet..."

# Cross-Domain Zero-shot NVS Evaluation on ScanNetV1 

CUDA_VISIBLE_DEVICES=0 python -m src.main \
                        +experiment=scannet_depth \
                        mode=test \
                        wandb.name=test_scannet_grid_normal_v2 \
                        test.save_image=true \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.ckpt \
                        > "nvs_scannetv1_grid_normal-with_pose_refinement-ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.txt" 2>&1

CUDA_VISIBLE_DEVICES=0 python -m src.main \
                        +experiment=scannet_depth \
                        mode=test \
                        test.align_pose=false \
                        wandb.name=test_scannet_grid_normal_v2 \
                        test.save_image=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.ckpt \
                        > "nvs_scannetv1_grid_normal-without_pose_refinement-ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.txt" 2>&1



CUDA_VISIBLE_DEVICES=0 python -m src.main \
                        +experiment=scannet_depth \
                        mode=test \
                        wandb.name=test_scannet_grid \
                        test.save_image=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_hpc_2025-03-01_22-56-21_step_18748.ckpt \
                        > "nvs_scannetv1_grid-with_pose_refinement-ours_re10k_grid_hpc_2025-03-01_22-56-21_step_18748.txt" 2>&1

CUDA_VISIBLE_DEVICES=0 python -m src.main \
                        +experiment=scannet_depth \
                        mode=test \
                        test.align_pose=false \
                        wandb.name=test_scannet_grid \
                        test.save_image=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_hpc_2025-03-01_22-56-21_step_18748.ckpt \
                        > "nvs_scannetv1_grid-without_pose_refinement-ours_re10k_grid_hpc_2025-03-01_22-56-21_step_18748.txt" 2>&1




CUDA_VISIBLE_DEVICES=0 python -m src.main \
                        +experiment=scannet_depth \
                        mode=test \
                        wandb.name=test_scannet_normal_v2 \
                        test.save_image=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_18748.ckpt \
                        > "nvs_scannetv1_normal-with_pose_refinement-ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_18748.txt" 2>&1


CUDA_VISIBLE_DEVICES=0 python -m src.main \
                        +experiment=scannet_depth \
                        mode=test \
                        test.align_pose=false \
                        wandb.name=test_scannet_normal_v2 \
                        test.save_image=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_18748.ckpt \
                        > "nvs_scannetv1_normal-without_pose_refinement-ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_18748.txt" 2>&1




CUDA_VISIBLE_DEVICES=0 python -m src.main \
                        +experiment=scannet_depth \
                        mode=test \
                        wandb.name=test_scannet_2dgs \
                        test.save_image=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_hpc_2025-03-04_02-02-31_step_18748.ckpt \
                        > "nvs_scannetv1_2dgs-with_pose_refinement-ours_re10k_hpc_2025-03-04_02-02-31_step_18748.txt" 2>&1

CUDA_VISIBLE_DEVICES=0 python -m src.main \
                        +experiment=scannet_depth \
                        mode=test \
                        test.align_pose=false \
                        wandb.name=test_scannet_2dgs \
                        test.save_image=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_hpc_2025-03-04_02-02-31_step_18748.ckpt \
                        > "nvs_scannetv1_2dgs-without_pose_refinement-ours_re10k_hpc_2025-03-04_02-02-31_step_18748.txt" 2>&1



echo "All scannet nvs scripts executed." 
