# In-Domain NVS Evaluation on RE10k
CUDA_VISIBLE_DEVICES=0 python -m src.main \
                        +experiment=re10k_grid_normal \
                        mode=test \
                        wandb.name=test_re10k_grid_normal_v2 \
                        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
                        dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
                        test.save_image=true \
                        checkpointing.load=./pretrained_weights/ours_re10k_grid_normal_v2_gaussians-detached-grid_hpc_2025-03-15_20-47-46_step_18748.ckpt 
