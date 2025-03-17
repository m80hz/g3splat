# In-Domain Pose Evaluation on RE10k


CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=re10k_grid_normal \
                        +evaluation=eval_pose \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.ckpt \
                        > "pose_re10k_grid_normal_with-refinement_ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.txt" 2>&1

CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=re10k_grid_normal \
                        +evaluation=eval_pose \
                        evaluation.use_pose_refinement=false \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.ckpt \
                        > "pose_re10k_grid_normal_without-refinement_ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.txt" 2>&1




CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=re10k_normal \
                        +evaluation=eval_pose \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_18748.ckpt \
                        > "pose_re10k_normal_with-refinement_ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_18748.txt" 2>&1

CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=re10k_normal \
                        +evaluation=eval_pose \
                        evaluation.use_pose_refinement=false \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_18748.ckpt \
                        > "pose_re10k_normal_without-refinement_ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_18748.txt" 2>&1





CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=re10k_grid_normal \
                        +evaluation=eval_pose \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_18748.ckpt \
                        > "pose_re10k_grid_normal_with-refinement_ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_18748.txt" 2>&1

CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=re10k_grid_normal \
                        +evaluation=eval_pose \
                        evaluation.use_pose_refinement=false \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_18748.ckpt \
                        > "pose_re10k_grid_normal_without-refinement_ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_18748.txt" 2>&1





CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=re10k_grid \
                        +evaluation=eval_pose \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_hpc_2025-03-01_22-56-21_step_18748.ckpt \
                        > "pose_re10k_grid_with-refinement_ours_re10k_grid_hpc_2025-03-01_22-56-21_step_18748.txt" 2>&1

CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=re10k_grid \
                        +evaluation=eval_pose \
                        evaluation.use_pose_refinement=false \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_hpc_2025-03-01_22-56-21_step_18748.ckpt \
                        > "pose_re10k_grid_without-refinement_ours_re10k_grid_hpc_2025-03-01_22-56-21_step_18748.txt" 2>&1



CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=re10k \
                        +evaluation=eval_pose \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_hpc_2025-03-04_02-02-31_step_18748.ckpt \
                        > "pose_re10k_with-refinement_ours_re10k_hpc_2025-03-04_02-02-31_step_18748.txt" 2>&1

CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=re10k \
                        +evaluation=eval_pose \
                        evaluation.use_pose_refinement=false \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_hpc_2025-03-04_02-02-31_step_18748.ckpt \
                        > "pose_re10k_without-refinement_ours_re10k_hpc_2025-03-04_02-02-31_step_18748.txt" 2>&1


CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=re10k_grid_normal \
                        +evaluation=eval_pose \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_hpc_2025-03-05_16-31-33_step_18748.ckpt \
                        > "pose_re10k_grid_normal_with-refinement_ours_re10k_grid_normal_hpc_2025-03-05_16-31-33_step_18748.txt" 2>&1

CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=re10k_grid_normal \
                        +evaluation=eval_pose \
                        evaluation.use_pose_refinement=false \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_hpc_2025-03-05_16-31-33_step_18748.ckpt \
                        > "pose_re10k_grid_normal_without-refinement_ours_re10k_grid_normal_hpc_2025-03-05_16-31-33_step_18748.txt" 2>&1


##########################
##########################


CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=re10k_grid_normal \
                        +evaluation=eval_pose \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_20001.ckpt \
                        > "pose_re10k_grid_normal_with-refinement_ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_20001.txt" 2>&1

CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=re10k_grid_normal \
                        +evaluation=eval_pose \
                        evaluation.use_pose_refinement=false \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_20001.ckpt \
                        > "pose_re10k_grid_normal_without-refinement_ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_20001.txt" 2>&1



CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=re10k_normal \
                        +evaluation=eval_pose \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_20001.ckpt \
                        > "pose_re10k_normal_with-refinement_ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_20001.txt" 2>&1

CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=re10k_normal \
                        +evaluation=eval_pose \
                        evaluation.use_pose_refinement=false \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_20001.ckpt \
                        > "pose_re10k_normal_without-refinement_ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_20001.txt" 2>&1





CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=re10k_grid_normal \
                        +evaluation=eval_pose \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_20001.ckpt \
                        > "pose_re10k_grid_normal_with-refinement_ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_20001.txt" 2>&1

CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=re10k_grid_normal \
                        +evaluation=eval_pose \
                        evaluation.use_pose_refinement=false \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_20001.ckpt \
                        > "pose_re10k_grid_normal_without-refinement_ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_20001.txt" 2>&1


#######################################################################
#######################################################################

# Cross-Domain Zero-shot Pose Evaluation on ACID

CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=acid_grid_normal \
                        +evaluation=eval_pose \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.ckpt \
                        > "pose_acid_grid_normal_with-refinement_ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.txt" 2>&1

CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=acid_grid_normal \
                        +evaluation=eval_pose \
                        evaluation.use_pose_refinement=false \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.ckpt \
                        > "pose_acid_grid_normal_without-refinement_ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.txt" 2>&1




CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=acid_normal \
                        +evaluation=eval_pose \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_18748.ckpt \
                        > "pose_acid_normal_with-refinement_ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_18748.txt" 2>&1

CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=acid_normal \
                        +evaluation=eval_pose \
                        evaluation.use_pose_refinement=false \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_18748.ckpt \
                        > "pose_acid_normal_without-refinement_ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_18748.txt" 2>&1





CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=acid_grid_normal \
                        +evaluation=eval_pose \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_18748.ckpt \
                        > "pose_acid_grid_normal_with-refinement_ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_18748.txt" 2>&1

CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=acid_grid_normal \
                        +evaluation=eval_pose \
                        evaluation.use_pose_refinement=false \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_18748.ckpt \
                        > "pose_acid_grid_normal_without-refinement_ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_18748.txt" 2>&1




CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=acid_grid \
                        +evaluation=eval_pose \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_hpc_2025-03-01_22-56-21_step_18748.ckpt \
                        > "pose_acid_grid_with-refinement_ours_re10k_grid_hpc_2025-03-01_22-56-21_step_18748.txt" 2>&1

CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=acid_grid \
                        +evaluation=eval_pose \
                        evaluation.use_pose_refinement=false \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_hpc_2025-03-01_22-56-21_step_18748.ckpt \
                        > "pose_acid_grid_without-refinement_ours_re10k_grid_hpc_2025-03-01_22-56-21_step_18748.txt" 2>&1




CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=acid \
                        +evaluation=eval_pose \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_hpc_2025-03-04_02-02-31_step_18748.ckpt \
                        > "pose_acid_with-refinement_ours_re10k_hpc_2025-03-04_02-02-31_step_18748.txt" 2>&1

CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=acid \
                        +evaluation=eval_pose \
                        evaluation.use_pose_refinement=false \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_hpc_2025-03-04_02-02-31_step_18748.ckpt \
                        > "pose_acid_without-refinement_ours_re10k_hpc_2025-03-04_02-02-31_step_18748.txt" 2>&1


CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=acid_grid_normal \
                        +evaluation=eval_pose \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_hpc_2025-03-05_16-31-33_step_18748.ckpt \
                        > "pose_acid_grid_normal_with-refinement_ours_re10k_grid_normal_hpc_2025-03-05_16-31-33_step_18748.txt" 2>&1

CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=acid_grid_normal \
                        +evaluation=eval_pose \
                        evaluation.use_pose_refinement=false \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_hpc_2025-03-05_16-31-33_step_18748.ckpt \
                        > "pose_acid_grid_normal_without-refinement_ours_re10k_grid_normal_hpc_2025-03-05_16-31-33_step_18748.txt" 2>&1


# ######################################################


CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=acid_grid_normal \
                        +evaluation=eval_pose \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_20001.ckpt \
                        > "pose_acid_grid_normal_with-refinement_ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_20001.txt" 2>&1

CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=acid_grid_normal \
                        +evaluation=eval_pose \
                        evaluation.use_pose_refinement=false \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_20001.ckpt \
                        > "pose_acid_grid_normal_without-refinement_ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_20001.txt" 2>&1




CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=acid_normal \
                        +evaluation=eval_pose \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_20001.ckpt \
                        > "pose_acid_normal_with-refinement_ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_20001.txt" 2>&1

CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=acid_normal \
                        +evaluation=eval_pose \
                        evaluation.use_pose_refinement=false \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_20001.ckpt \
                        > "pose_acid_normal_without-refinement_ours_re10k_normal_v2_gaussians-detached-grid_hpc_2025-03-16_02-33-05_step_20001.txt" 2>&1





CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=acid_grid_normal \
                        +evaluation=eval_pose \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_20001.ckpt \
                        > "pose_acid_grid_normal_with-refinement_ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_20001.txt" 2>&1

CUDA_VISIBLE_DEVICES=0 python -m src.eval_pose \
                        +experiment=acid_grid_normal \
                        +evaluation=eval_pose \
                        evaluation.use_pose_refinement=false \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_20001.ckpt \
                        > "pose_acid_grid_normal_without-refinement_ours_re10k_grid_normal_v3_gaussians_hpc_2025-03-16_02-51-15_step_20001.txt" 2>&1



