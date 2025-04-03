echo "Running all nvs evaluations for ACID..."

# Cross-Domain Zero-shot NVS Evaluation on ACID 

CUDA_VISIBLE_DEVICES=0 python -m src.main \
                        +experiment=acid_1x8 \
                        mode=test \
                        wandb.name=test_acid_grid_normal_v2 \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
                        test.save_image=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.ckpt \
                        > "nvs_acid_grid_normal-with_pose_refinement-ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.txt" 2>&1

CUDA_VISIBLE_DEVICES=0 python -m src.main \
                        +experiment=acid_1x8 \
                        mode=test \
                        test.align_pose=false \
                        wandb.name=test_acid_grid_normal_v2 \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
                        test.save_image=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.ckpt \
                        > "nvs_acid_grid_normal-without_pose_refinement-ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.txt" 2>&1



CUDA_VISIBLE_DEVICES=0 python -m src.main \
                        +experiment=acid_1x8 \
                        mode=test \
                        wandb.name=test_acid_grid \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
                        test.save_image=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_hpc_2025-03-01_22-56-21_step_18748.ckpt \
                        > "nvs_acid_grid-with_pose_refinement-ours_re10k_grid_hpc_2025-03-01_22-56-21_step_18748.txt" 2>&1

CUDA_VISIBLE_DEVICES=0 python -m src.main \
                        +experiment=acid_1x8 \
                        mode=test \
                        test.align_pose=false \
                        wandb.name=test_acid_grid \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
                        test.save_image=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_hpc_2025-03-01_22-56-21_step_18748.ckpt \
                        > "nvs_acid_grid-without_pose_refinement-ours_re10k_grid_hpc_2025-03-01_22-56-21_step_18748.txt" 2>&1




CUDA_VISIBLE_DEVICES=0 python -m src.main \
                        +experiment=acid_1x8 \
                        mode=test \
                        wandb.name=test_acid_normal_v2 \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
                        test.save_image=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_18748.ckpt \
                        > "nvs_acid_normal-with_pose_refinement-ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_18748.txt" 2>&1


CUDA_VISIBLE_DEVICES=0 python -m src.main \
                        +experiment=acid_1x8 \
                        mode=test \
                        test.align_pose=false \
                        wandb.name=test_acid_normal_v2 \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
                        test.save_image=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_18748.ckpt \
                        > "nvs_acid_normal-without_pose_refinement-ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_18748.txt" 2>&1




CUDA_VISIBLE_DEVICES=0 python -m src.main \
                        +experiment=acid_1x8 \
                        mode=test \
                        wandb.name=test_acid_2dgs \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
                        test.save_image=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_hpc_2025-03-04_02-02-31_step_18748.ckpt \
                        > "nvs_acid_2dgs-with_pose_refinement-ours_re10k_hpc_2025-03-04_02-02-31_step_18748.txt" 2>&1

CUDA_VISIBLE_DEVICES=0 python -m src.main \
                        +experiment=acid_1x8 \
                        mode=test \
                        test.align_pose=false \
                        wandb.name=test_acid_2dgs \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
                        test.save_image=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_hpc_2025-03-04_02-02-31_step_18748.ckpt \
                        > "nvs_acid_2dgs-without_pose_refinement-ours_re10k_hpc_2025-03-04_02-02-31_step_18748.txt" 2>&1



echo "All ACID nvs scripts executed." 
