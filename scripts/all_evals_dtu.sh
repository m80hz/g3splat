echo "Running all dtu evaluations ..."


# Pose
CUDA_VISIBLE_DEVICES=1 python -m src.eval_pose \
                        +experiment=dtu_1x8 \
                        +evaluation=eval_pose \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_dtu.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.ckpt \
                        > "pose_dtu_grid_normal_with-refinement_ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.txt" 2>&1

CUDA_VISIBLE_DEVICES=1 python -m src.eval_pose \
                        +experiment=dtu_1x8 \
                        +evaluation=eval_pose \
                        evaluation.use_pose_refinement=false \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_dtu.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.ckpt \
                        > "pose_dtu_grid_normal_without-refinement_ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.txt" 2>&1




CUDA_VISIBLE_DEVICES=1 python -m src.eval_pose \
                        +experiment=dtu_1x8 \
                        +evaluation=eval_pose \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_dtu.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_18748.ckpt \
                        > "pose_dtu_normal_with-refinement_ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_18748.txt" 2>&1

CUDA_VISIBLE_DEVICES=1 python -m src.eval_pose \
                        +experiment=dtu_1x8 \
                        +evaluation=eval_pose \
                        evaluation.use_pose_refinement=false \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_dtu.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_18748.ckpt \
                        > "pose_dtu_normal_without-refinement_ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_18748.txt" 2>&1





CUDA_VISIBLE_DEVICES=1 python -m src.eval_pose \
                        +experiment=dtu_1x8 \
                        +evaluation=eval_pose \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_dtu.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_hpc_2025-03-01_22-56-21_step_18748.ckpt \
                        > "pose_dtu_pose_grid-with_pose_refinement-ours_re10k_grid_hpc_2025-03-01_22-56-21_step_18748.txt" 2>&1

CUDA_VISIBLE_DEVICES=1 python -m src.eval_pose \
                        +experiment=dtu_1x8 \
                        +evaluation=eval_pose \
                        evaluation.use_pose_refinement=false \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_dtu.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_hpc_2025-03-01_22-56-21_step_18748.ckpt \
                        > "pose_dtu_pose_grid-without_pose_refinement-ours_re10k_grid_hpc_2025-03-01_22-56-21_step_18748.txt" 2>&1




CUDA_VISIBLE_DEVICES=1 python -m src.eval_pose \
                        +experiment=dtu_1x8 \
                        +evaluation=eval_pose \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_dtu.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_hpc_2025-03-04_02-02-31_step_18748.ckpt \
                        > "pose_dtu_pose-with_pose_refinement-ours_re10k_hpc_2025-03-04_02-02-31_step_18748.txt" 2>&1

CUDA_VISIBLE_DEVICES=1 python -m src.eval_pose \
                        +experiment=dtu_1x8 \
                        +evaluation=eval_pose \
                        evaluation.use_pose_refinement=false \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_dtu.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_hpc_2025-03-04_02-02-31_step_18748.ckpt \
                        > "pose_dtu_pose-without_pose_refinement-ours_re10k_hpc_2025-03-04_02-02-31_step_18748.txt" 2>&1





CUDA_VISIBLE_DEVICES=1 python -m src.eval_pose \
                        +experiment=dtu_1x8 \
                        +evaluation=eval_pose \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_dtu.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_hpc_2025-03-05_16-31-33_step_18748.ckpt \
                        > "pose_dtu_pose_grid_normal-with_pose_refinement-ours_re10k_grid_normal_hpc_2025-03-05_16-31-33_step_18748.txt" 2>&1

CUDA_VISIBLE_DEVICES=1 python -m src.eval_pose \
                        +experiment=dtu_1x8 \
                        +evaluation=eval_pose \
                        evaluation.use_pose_refinement=false \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_dtu.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_hpc_2025-03-05_16-31-33_step_18748.ckpt \
                        > "pose_dtu_pose_grid_normal-without_pose_refinement-ours_re10k_grid_normal_hpc_2025-03-05_16-31-33_step_18748.txt" 2>&1


CUDA_VISIBLE_DEVICES=1 python -m src.eval_pose \
                        +experiment=dtu_1x8 \
                        +evaluation=eval_pose \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_dtu.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_18748.ckpt \
                        > "pose_dtu_grid_normal_with-refinement_ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_18748.txt" 2>&1

CUDA_VISIBLE_DEVICES=1 python -m src.eval_pose \
                        +experiment=dtu_1x8 \
                        +evaluation=eval_pose \
                        evaluation.use_pose_refinement=false \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_dtu.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_18748.ckpt \
                        > "pose_dtu_grid_normal_without-refinement_ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_18748.txt" 2>&1



CUDA_VISIBLE_DEVICES=1 python -m src.eval_pose \
                        +experiment=dtu_1x8 \
                        +evaluation=eval_pose \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_dtu.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_20001.ckpt \
                        > "pose_dtu_grid_normal_with-refinement_ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_20001.txt" 2>&1

CUDA_VISIBLE_DEVICES=1 python -m src.eval_pose \
                        +experiment=dtu_1x8 \
                        +evaluation=eval_pose \
                        evaluation.use_pose_refinement=false \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_dtu.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_20001.ckpt \
                        > "pose_dtu_grid_normal_without-refinement_ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_20001.txt" 2>&1



CUDA_VISIBLE_DEVICES=1 python -m src.eval_pose \
                        +experiment=dtu_1x8 \
                        +evaluation=eval_pose \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_dtu.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_20001.ckpt \
                        > "pose_dtu_normal_with-refinement_ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_20001.txt" 2>&1

CUDA_VISIBLE_DEVICES=1 python -m src.eval_pose \
                        +experiment=dtu_1x8 \
                        +evaluation=eval_pose \
                        evaluation.use_pose_refinement=false \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_dtu.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_20001.ckpt \
                        > "pose_dtu_normal_without-refinement_ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_20001.txt" 2>&1




CUDA_VISIBLE_DEVICES=1 python -m src.eval_pose \
                        +experiment=dtu_1x8 \
                        +evaluation=eval_pose \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_dtu.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_20001.ckpt \
                        > "pose_dtu_grid_normal_with-refinement_ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_20001.txt" 2>&1

CUDA_VISIBLE_DEVICES=1 python -m src.eval_pose \
                        +experiment=dtu_1x8 \
                        +evaluation=eval_pose \
                        evaluation.use_pose_refinement=false \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_dtu.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_20001.ckpt \
                        > "pose_dtu_grid_normal_without-refinement_ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_20001.txt" 2>&1



# NVS
# Cross-Domain Zero-Shot NVS Evaluation on DTU
CUDA_VISIBLE_DEVICES=1 python -m src.main \
                        +experiment=dtu_1x8 \
                        mode=test \
                        wandb.name=test_dtu_grid_normal_v2 \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_dtu.json \
                        test.save_image=true \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.ckpt \
                        > "nvs_dtu_grid_normal_with-pose-refinement_ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.txt" 2>&1


CUDA_VISIBLE_DEVICES=1 python -m src.main \
                        +experiment=dtu_1x8 \
                        mode=test \
                        test.align_pose=false \
                        wandb.name=test_dtu_grid_normal_v2 \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_dtu.json \
                        test.save_image=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.ckpt \
                        > "nvs_dtu_grid_normal_without-pose-refinement_ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.txt" 2>&1





CUDA_VISIBLE_DEVICES=1 python -m src.main \
                        +experiment=dtu_1x8 \
                        mode=test \
                        wandb.name=test_dtu_normal_v2 \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_dtu.json \
                        test.save_image=true \
                        checkpointing.load=./pretrained_weights/ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_18748.ckpt \
                        > "nvs_dtu_normal_with-refinement_ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_18748.txt" 2>&1


CUDA_VISIBLE_DEVICES=1 python -m src.main \
                        +experiment=dtu_1x8 \
                        mode=test \
                        test.align_pose=false \
                        wandb.name=test_dtu_normal_v2 \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_dtu.json \
                        test.save_image=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_18748.ckpt \
                        > "nvs_dtu_normal_without-refinement_ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_18748.txt" 2>&1



CUDA_VISIBLE_DEVICES=1 python -m src.main \
                        +experiment=dtu_1x8 \
                        mode=test \
                        wandb.name=test_dtu_grid \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_dtu.json \
                        test.save_image=true \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_hpc_2025-03-01_22-56-21_step_18748.ckpt \
                        > "nvs_dtu_grid_with-refinement_ours_re10k_grid_hpc_2025-03-01_22-56-21_step_18748.txt" 2>&1


CUDA_VISIBLE_DEVICES=1 python -m src.main \
                        +experiment=dtu_1x8 \
                        mode=test \
                        test.align_pose=false \
                        wandb.name=test_dtu_grid \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_dtu.json \
                        test.save_image=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_hpc_2025-03-01_22-56-21_step_18748.ckpt \
                        > "nvs_dtu_grid_without-refinement_ours_re10k_grid_hpc_2025-03-01_22-56-21_step_18748.txt" 2>&1



CUDA_VISIBLE_DEVICES=1 python -m src.main \
                        +experiment=dtu_1x8 \
                        mode=test \
                        wandb.name=test_dtu \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_dtu.json \
                        test.save_image=true \
                        checkpointing.load=./pretrained_weights/ours_re10k_hpc_2025-03-04_02-02-31_step_18748.ckpt \
                        > "nvs_dtu_with-refinement_ours_re10k_hpc_2025-03-04_02-02-31_step_18748.txt" 2>&1


CUDA_VISIBLE_DEVICES=1 python -m src.main \
                        +experiment=dtu_1x8 \
                        mode=test \
                        test.align_pose=false \
                        wandb.name=test_dtu \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_dtu.json \
                        test.save_image=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_hpc_2025-03-04_02-02-31_step_18748.ckpt \
                        > "nvs_dtu_without-refinement_ours_re10k_hpc_2025-03-04_02-02-31_step_18748.txt" 2>&1



CUDA_VISIBLE_DEVICES=1 python -m src.main \
                        +experiment=dtu_1x8 \
                        mode=test \
                        wandb.name=test_dtu_grid_normal_v3 \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_dtu.json \
                        test.save_image=true \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_18748.ckpt \
                        > "nvs_dtu_grid_normal_with-pose-refinement_ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_18748.txt" 2>&1


CUDA_VISIBLE_DEVICES=1 python -m src.main \
                        +experiment=dtu_1x8 \
                        mode=test \
                        test.align_pose=false \
                        wandb.name=test_dtu_grid_normal_v3 \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_dtu.json \
                        test.save_image=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_18748.ckpt \
                        > "nvs_dtu_grid_normal_without-pose-refinement_ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_18748.txt" 2>&1



CUDA_VISIBLE_DEVICES=1 python -m src.main \
                        +experiment=dtu_1x8 \
                        mode=test \
                        wandb.name=test_dtu_grid_normal_v2 \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_dtu.json \
                        test.save_image=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_20001.ckpt \
                        > "nvs_dtu_grid_normal_with-pose-refinement_ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_20001.txt" 2>&1


CUDA_VISIBLE_DEVICES=1 python -m src.main \
                        +experiment=dtu_1x8 \
                        mode=test \
                        test.align_pose=false \
                        wandb.name=test_dtu_grid_normal_v2 \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_dtu.json \
                        test.save_image=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_20001.ckpt \
                        > "nvs_dtu_grid_normal_without-pose-refinement_ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_20001.txt" 2>&1




CUDA_VISIBLE_DEVICES=1 python -m src.main \
                        +experiment=dtu_1x8 \
                        mode=test \
                        wandb.name=test_dtu_normal_v2 \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_dtu.json \
                        test.save_image=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_20001.ckpt \
                        > "nvs_dtu_normal_with-refinement_ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_20001.txt" 2>&1


CUDA_VISIBLE_DEVICES=1 python -m src.main \
                        +experiment=dtu_1x8 \
                        mode=test \
                        test.align_pose=false \
                        wandb.name=test_dtu_normal_v2 \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_dtu.json \
                        test.save_image=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_20001.ckpt \
                        > "nvs_dtu_normal_without-refinement_ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_20001.txt" 2>&1




CUDA_VISIBLE_DEVICES=1 python -m src.main \
                        +experiment=dtu_1x8 \
                        mode=test \
                        wandb.name=test_dtu_grid_normal_v3 \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_dtu.json \
                        test.save_image=true \
                        test.save_mesh=true \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_20001.ckpt \
                        > "nvs_dtu_grid_normal_with-pose-refinement_ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_20001.txt" 2>&1


CUDA_VISIBLE_DEVICES=1 python -m src.main \
                        +experiment=dtu_1x8 \
                        mode=test \
                        test.align_pose=false \
                        wandb.name=test_dtu_grid_normal_v3 \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_dtu.json \
                        test.save_image=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_20001.ckpt \
                        > "nvs_dtu_grid_normal_without-pose-refinement_ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_20001.txt" 2>&1




# #######################################################################################################

# # Depth
CUDA_VISIBLE_DEVICES=1 python -m src.eval_depth \
                        +experiment=dtu_1x8 \
                        +evaluation=eval_depth \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.ckpt \
                        > "depth_dtu_grid_normal_with-refinement_ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.txt" 2>&1

CUDA_VISIBLE_DEVICES=1 python -m src.eval_depth \
                        +experiment=dtu_1x8 \
                        +evaluation=eval_depth \
                        evaluation.use_pose_refinement=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.ckpt \
                        > "depth_dtu_grid_normal_without-refinement_ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.txt" 2>&1



CUDA_VISIBLE_DEVICES=1 python -m src.eval_depth \
                        +experiment=dtu_1x8 \
                        +evaluation=eval_depth \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_20001.ckpt \
                        > "depth_dtu_grid_normal_with-refinement_ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_20001.txt" 2>&1

CUDA_VISIBLE_DEVICES=1 python -m src.eval_depth \
                        +experiment=dtu_1x8 \
                        +evaluation=eval_depth \
                        evaluation.use_pose_refinement=false \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_20001.ckpt \
                        > "depth_dtu_grid_normal_without-refinement_ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_20001.txt" 2>&1




echo "All dtu evaluations executed." 
