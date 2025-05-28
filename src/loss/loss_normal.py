from dataclasses import dataclass

import torch
from einops import reduce, rearrange
from jaxtyping import Float
from torch import Tensor
import torch.nn.functional as F

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss

from ..geometry.projection import points_to_normal
from ..model.encoder.common.gaussians import quaternion_to_matrix


@dataclass
class LossNormalCfg:
    lambda_context_views_normal: float
    on_novel_views: bool
    lambda_novel_views_normal: float
    lambda_novel_views_distortion: float
    apply_normal_after_step: int
    apply_distortion_after_step: int
    valid_threshold: float = 1e-1  
    depth_valid_threshold: float = 1e-3
    depth_disc_multiplier: float = 1.5
    depth_disc_slope: float = 0.1
    huber_delta: float = 0.1


@dataclass
class LossNormalCfgWrapper:
    normal: LossNormalCfg


class LossNormal(Loss[LossNormalCfg, LossNormalCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        
        if self.cfg.lambda_context_views_normal == 0.0:
            context_view_loss = torch.tensor(0.0, device=prediction.depth.device)
        else:
            # Extract image dimensions; batch["context"]["image"] shape: (B, V, C, H, W), V == 2.
            B, V, C, H, W = batch["context"]["image"].shape

            # # Reshape gaussians.means into (B, V, H, W, 3)
            all_pts3d = rearrange(gaussians.means, "b (v h w) d -> (b v) h w d", h=H, w=W)    # (B*V, H, W, 3)
            # pts3d1 = all_pts3d[:, 0, ...]  # (B, H, W, 3)
            # pts3d2 = all_pts3d[:, 1, ...]  # (B, H, W, 3)
            surf_normals_ptc = points_to_normal(all_pts3d)      # (B*V, H, W, 3)

            # extract gaussian normals from their rotation quaternion
            gaussian_rotations = rearrange(gaussians.rotations, "b (v h w) d -> (b v) h w d", v=V, h=H, w=W)     # shape (B*V, H, W, 4)
            # Normalize the quaternions to ensure they are unit quaternions.
            gaussian_rotations_normalised = gaussian_rotations / gaussian_rotations.norm(dim=-1, keepdim=True)
            # Convert quaternions to rotation matrices. The resulting shape is (B*V, H, W, 3, 3).
            gaussian_rot_matrices = quaternion_to_matrix(gaussian_rotations_normalised)
            # Extract the third column from each rotation matrix, which represents the surfel normal.
            gs_surfel_normals = gaussian_rot_matrices[..., :, 2]  # shape: (B*V, H, W, 3)

            eps = 1e-6  # small constant for numerical stability
            norm_surf_ptc = torch.norm(surf_normals_ptc, dim=-1, keepdim=True)  # (B*V, H, W, 1)
            norm_gs_surfels = torch.norm(gs_surfel_normals, dim=-1, keepdim=True)  # (B*V, H, W, 1)

            # Normalize the normals (avoid division-by-zero using eps).
            surf_normals_ptc_normed = surf_normals_ptc / (norm_surf_ptc + eps)
            gs_surfel_normals_normed = gs_surfel_normals / (norm_gs_surfels + eps)

            surf_normals_ptc_normed = rearrange(surf_normals_ptc_normed, "(b v) h w d -> b v h w d", b=B, v=V)  # (B, V, H, W, 3)
            gs_surfel_normals_normed = rearrange(gs_surfel_normals_normed, "(b v) h w d -> b v h w d", b=B, v=V)  # (B, V, H, W, 3)


            # Gradient-Based Soft Mask for Depth Discontinuities In Context Views
            # -------------------------------------------------------------
            extrinsics = batch["context"]["extrinsics"]  # (B, V, 4, 4)
            all_pts3d = rearrange(all_pts3d, "(b v) h w d -> b v h w d", b=B, v=V)    # (B, V, H, W, 3)
            T0 = extrinsics[:, 0]                       # cam0 to world
            T0_inv  = torch.inverse(T0)                 # world to cam0
            T_inv_v = torch.inverse(extrinsics)         # world to cam_v for each v
            
            depth_maps = []
            for v in range(V):
                # take the v-th pointcloud (in cam0 frame)
                pts_cam0 = all_pts3d[:, v]            # (B, H, W, 3)
                # to homogeneous: (B,4,H,W)
                ones = torch.ones(B, H, W, 1, device=pts_cam0.device, dtype=pts_cam0.dtype)
                pts_h = torch.cat([pts_cam0, ones], dim=-1).permute(0, 3, 1, 2)  # (B,4,H,W)
                pts_flat = pts_h.view(B, 4, -1)                              # (B,4,N)

                # cam0 to world to cam_v
                world = torch.bmm(T0, pts_flat)                         # (B,4,N)
                cam_v = torch.bmm(T_inv_v[:, v], world)                 # (B,4,N)
                cam_v = cam_v.view(B, 4, H, W)
                depth_maps.append(cam_v[:, 2, :, :])                    # (B,H,W)
            depth_map = torch.stack(depth_maps, dim=1)      # (B, V, H, W)
            
            # Compute per-view soft masks exactly like GridLoss
            soft_masks = []
            for v in range(V):
                d = depth_map[:, v].float()                  # (B,H,W)
                flat = d.view(B, -1)
                med = flat.median(dim=1)[0].view(B,1,1)
                std = flat.std(dim=1).view(B,1,1)
                dn = (d - med) / (std + eps)              # normalize
                gx = torch.abs(dn[:, :, 1:] - dn[:, :, :-1])
                gx = F.pad(gx, (0,1), mode="replicate")
                gy = torch.abs(dn[:, 1:, :] - dn[:, :-1, :])
                gy = F.pad(gy, (0,0,0,1), mode="replicate")
                gm = torch.sqrt(gx**2 + gy**2)               # gradient magnitude

                thr = torch.median(gm.view(B,-1), dim=1)[0].view(B,1,1)
                thr = thr * self.cfg.depth_disc_multiplier
                sm  = 1.0 / (1.0 + torch.exp((gm - thr) / self.cfg.depth_disc_slope))
                soft_masks.append(sm)                        # (B,H,W)
            soft_mask = torch.stack(soft_masks, dim=1)       # (B,V,H,W)

            norm_s = surf_normals_ptc_normed.norm(dim=-1)                             # (B,V,H,W)
            norm_g = gs_surfel_normals_normed.norm(dim=-1)                               # (B,V,H,W)
            valid_normal_mask = ((norm_s > self.cfg.valid_threshold) & (norm_g > self.cfg.valid_threshold)).float()

            # Compute the dot product per pixel
            dot_product = torch.sum(surf_normals_ptc_normed * gs_surfel_normals_normed, dim=-1)  # (B, V, H, W)
            angular_error = 1.0 - dot_product  # Zero error when perfectly aligned

            # Apply a robust Huber loss (Smooth L1) to the angular error.
            loss_per_pixel = F.smooth_l1_loss(
                angular_error, 
                torch.zeros_like(angular_error), 
                reduction='none', 
                beta=self.cfg.huber_delta
            )
            combined_mask = soft_mask * valid_normal_mask                  # (B,V,H,W)
            masked_loss = loss_per_pixel * combined_mask
            context_view_loss = self.cfg.lambda_context_views_normal * masked_loss.mean()
        

        if not self.cfg.on_novel_views:
            novel_view_loss = torch.tensor(0.0, device=prediction.depth.device)
        else:
            
            # Only apply the normal consistency loss after a certain training step.
            lambda_novel_views_normal = self.cfg.lambda_novel_views_normal if global_step > self.cfg.apply_normal_after_step else 0.0
            lambda_novel_views_dist = self.cfg.lambda_novel_views_distortion if global_step > self.cfg.apply_distortion_after_step else 0.0
            if lambda_novel_views_normal == 0.0 and lambda_novel_views_dist == 0.0:
                return context_view_loss + torch.tensor(0.0, device=prediction.depth.device)

            eps = 1e-6  # small constant for numerical stability

            depth = rearrange(prediction.depth, "b v h w -> (b v) h w").detach()                        # (B, H, W)
            surf_normal = rearrange(prediction.surf_normal, "b v c h w -> (b v) c h w").detach()        # (B, 3, H, W)   (depth-derived normals)
            rend_normal = rearrange(prediction.rend_normal, "b v c h w -> (b v) c h w")        # (B, 3, H, W)   (rendered normals)
            # rend_dist = rearrange(prediction.dist, "b v h w -> (b v) h w")                     # (B, H, W)   (depth distortion)        
            

            # -------------------------------
            # 1. Normal Validity Masking
            # -------------------------------
            norm_surf = torch.norm(surf_normal, dim=1, keepdim=True)  # (B, 1, H, W)
            norm_rend = torch.norm(rend_normal, dim=1, keepdim=True)  # (B, 1, H, W)
            valid_normal_mask = ((norm_surf > self.cfg.valid_threshold) & (norm_rend > self.cfg.valid_threshold)).float()
            
            
            # -------------------------------
            # 2. Depth Validity Masking
            # -------------------------------
            valid_depth_mask = (depth > self.cfg.depth_valid_threshold).float().unsqueeze(1)  # (B, 1, H, W)
            
            # Combined mask from normals and depth validity.
            valid_mask = valid_normal_mask * valid_depth_mask  # (B, 1, H, W)
            
            
            # -------------------------------
            # 3. Gradient-Based Soft Mask for Depth Discontinuities
            # -------------------------------
            # Robustly normalize depth per image using median and standard deviation to mitigate outliers.
            B, H, W = depth.shape
            depth = depth.float()  # ensure floating point for computations
            
            # Compute median and standard deviation per image (flattening spatial dimensions)
            depth_flat = depth.view(B, -1)
            depth_median = depth_flat.median(dim=1)[0].view(B, 1, 1)
            depth_std = depth_flat.std(dim=1).view(B, 1, 1)
            depth_norm = (depth - depth_median) / (depth_std + eps)  # (B, H, W)

            # Compute spatial gradients of the normalized depth using finite differences.
            grad_x = torch.abs(depth_norm[:, :, 1:] - depth_norm[:, :, :-1])
            grad_x = F.pad(grad_x, (0, 1), mode='replicate')  
            grad_y = torch.abs(depth_norm[:, 1:, :] - depth_norm[:, :-1, :])
            grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')

            # Compute gradient magnitude.
            grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)  # (B, H, W)
            
            # Compute an adaptive threshold per image based on the median gradient magnitude.
            adaptive_threshold = torch.median(grad_mag.view(B, -1), dim=1)[0].view(B, 1, 1)
            adaptive_threshold = adaptive_threshold * self.cfg.depth_disc_multiplier

            # Create a soft mask: pixels with gradient magnitude much lower than the adaptive threshold have weight ~1,
            # while those with high gradient magnitude are downweighted.
            # Using a sigmoid function for smooth transition.
            soft_mask = 1.0 / (1.0 + torch.exp((grad_mag - adaptive_threshold) / self.cfg.depth_disc_slope))    # soft_mask: shape (B, H, W)
            
            # Incorporate the soft mask into the overall validity mask.
            final_valid_mask = valid_mask * soft_mask.unsqueeze(1)  # (B, 1, H, W)
            # Check if the number of valid pixels is below a threshold (e.g., 100 pixels).
            if final_valid_mask.sum().item() < 100:
                return torch.tensor(0.0, device=prediction.depth.device)

        
            # -------------------------------
            # 4. Compute Robust Angular Loss
            # -------------------------------
            # Normalize the normals (avoid division-by-zero using eps).
            surf_normal_normed = surf_normal / (norm_surf + eps)
            rend_normal_normed = rend_normal / (norm_rend + eps)

            # Compute the dot product per pixel; using absolute value to account for direction ambiguity (n and -n are equivalent).
            dot_product = torch.sum(surf_normal_normed * rend_normal_normed, dim=1)  # (B, H, W)
            abs_dot = torch.abs(dot_product)
            angular_error = 1.0 - abs_dot  # Zero error when perfectly aligned (or anti-aligned).

            # Apply a robust Huber loss (Smooth L1) to the angular error.
            loss_per_pixel = F.smooth_l1_loss(
                angular_error, 
                torch.zeros_like(angular_error), 
                reduction='none', 
                beta=self.cfg.huber_delta
            )

            # Mask out invalid pixels: final_valid_mask is (B, 1, H, W)
            masked_loss = loss_per_pixel * final_valid_mask.squeeze(1)

            # Average the loss over the valid pixels (avoid division by zero).
            loss_sum = masked_loss.sum()
            valid_count = final_valid_mask.sum() + eps
            normal_consistency_loss = loss_sum / valid_count
            total_normal_loss = lambda_novel_views_normal * normal_consistency_loss

            # dist_loss = lambda_novel_views_dist * (rend_dist).mean()

            novel_view_loss = total_normal_loss
        
       
        total_loss = context_view_loss + novel_view_loss
        
        return total_loss
