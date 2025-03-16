echo "Running all nvs evaluations ..."


# In-Domain NVS Evaluation on RE10k (the current models in the paper)
CUDA_VISIBLE_DEVICES=1 python -m src.main \
                        +experiment=re10k_grid_1x8 \
                        mode=test \
                        wandb.name=test_re10k_grid \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
                        test.save_image=true \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_hpc_2025-03-01_22-56-21_step_18748.ckpt \
                        > "nvs_re10k_grid_with-pose-refinement_ours_re10k_grid_hpc_2025-03-01_22-56-21_step_18748.txt" 2>&1

CUDA_VISIBLE_DEVICES=1 python -m src.main \
                        +experiment=re10k_grid_1x8 \
                        mode=test \
                        test.align_pose=false \
                        wandb.name=test_re10k_grid \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
                        test.save_image=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_hpc_2025-03-01_22-56-21_step_18748.ckpt \
                        > "nvs_re10k_grid_without-pose-refinement_ours_re10k_grid_hpc_2025-03-01_22-56-21_step_18748.txt" 2>&1




# In-Domain NVS Evaluation on RE10k (the newly trained models)
CUDA_VISIBLE_DEVICES=1 python -m src.main \
                        +experiment=re10k_grid_normal_1x8 \
                        mode=test \
                        wandb.name=test_re10k_grid_normal_v3 \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
                        test.save_image=true \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_18748.ckpt \
                        > "nvs_re10k_grid_normal_with-pose-refinement_ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_18748.txt" 2>&1

CUDA_VISIBLE_DEVICES=1 python -m src.main \
                        +experiment=re10k_grid_normal_1x8 \
                        mode=test \
                        test.align_pose=false \
                        wandb.name=test_re10k_grid_normal_v3 \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
                        test.save_image=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_18748.ckpt \
                        > "nvs_re10k_grid_normal_without-pose-refinement_ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_18748.txt" 2>&1




CUDA_VISIBLE_DEVICES=1 python -m src.main \
                        +experiment=re10k_normal_1x8 \
                        mode=test \
                        wandb.name=test_re10k_normal_v2 \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
                        test.save_image=true \
                        checkpointing.load=./pretrained_weights/ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_18748.ckpt \
                        > "nvs_re10k_normal_with-pose-refinement_ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_18748.txt" 2>&1

CUDA_VISIBLE_DEVICES=1 python -m src.main \
                        +experiment=re10k_normal_1x8 \
                        mode=test \
                        test.align_pose=false \
                        wandb.name=test_re10k_normal_v2 \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
                        test.save_image=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_18748.ckpt \
                        > "nvs_re10k_normal_without-pose-refinement_ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_18748.txt" 2>&1




CUDA_VISIBLE_DEVICES=1 python -m src.main \
                        +experiment=re10k_grid_normal_1x8 \
                        mode=test \
                        wandb.name=test_re10k_grid_normal_v2 \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
                        test.save_image=true \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.ckpt \
                        > "nvs_re10k_grid_normal_with-pose-refinement_ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.txt" 2>&1


CUDA_VISIBLE_DEVICES=1 python -m src.main \
                        +experiment=re10k_grid_normal_1x8 \
                        mode=test \
                        test.align_pose=false \
                        wandb.name=test_re10k_grid_normal_v2 \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
                        test.save_image=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.ckpt \
                        > "nvs_re10k_grid_normal_without-pose-refinement_ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.txt" 2>&1



echo "All scripts executed." 
