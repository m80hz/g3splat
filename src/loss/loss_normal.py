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

@dataclass
class LossNormalCfg:
    lambda_normal: float
    lambda_distortion: float
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
        
        # Only apply the normal consistency loss after a certain training step.
        lambda_normal = self.cfg.lambda_normal if global_step > self.cfg.apply_normal_after_step else 0.0
        lambda_dist = self.cfg.lambda_distortion if global_step > self.cfg.apply_distortion_after_step else 0.0
        if lambda_normal == 0.0 and lambda_dist == 0.0:
            return torch.tensor(0.0, device=prediction.depth.device)

        eps = 1e-6  # small constant for numerical stability

        depth = rearrange(prediction.depth, "b v h w -> (b v) h w").detach()                        # (B, H, W)
        surf_normal = rearrange(prediction.surf_normal, "b v c h w -> (b v) c h w").detach()        # (B, 3, H, W)   (depth-derived normals)
        rend_normal = rearrange(prediction.rend_normal, "b v c h w -> (b v) c h w")        # (B, 3, H, W)   (rendered normals)
        rend_dist = rearrange(prediction.dist, "b v h w -> (b v) h w")                     # (B, H, W)   (depth distortion)        
        

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
        total_normal_loss = lambda_normal * normal_consistency_loss

        dist_loss = lambda_dist * (rend_dist).mean()

        total_loss = total_normal_loss + dist_loss
        
        return total_loss
