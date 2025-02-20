from dataclasses import dataclass

import torch
from einops import reduce
from jaxtyping import Float
from torch import Tensor
import torch.nn.functional as F

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss

from ..misc.utils import inspect_depth_tensor


@dataclass
class LossNormalCfg:
    lambda_normal: float
    lambda_distortion: float
    apply_normal_after_step: int
    apply_distortion_after_step: int


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
        
        # regularization
        lambda_normal = self.cfg.lambda_normal if global_step > self.cfg.apply_normal_after_step else 0.0
        lambda_dist = self.cfg.lambda_distortion if global_step > self.cfg.apply_distortion_after_step else 0.0

        depth = prediction.depth  # shape: (B, V, H, W)
        
        # Compute a robust statistic – here we use the median depth across all elements.
        median_depth = depth.median()
        # Adaptive threshold: 10% of the median depth
        adaptive_threshold = 0.1 * median_depth  # adjust the factor as needed

        # Compute depth differences along x and y
        d_dx = torch.abs(depth[:, :, :, 1:] - depth[:, :, :, :-1])
        d_dy = torch.abs(depth[:, :, 1:, :] - depth[:, :, :-1, :])

        # Pad to get back to (B, V, H, W)
        d_dx = F.pad(d_dx, (0, 1, 0, 0), mode='replicate')
        d_dy = F.pad(d_dy, (0, 0, 0, 1), mode='replicate')

        # One option: use the maximum difference per pixel (or you could use a norm)
        grad_mag = torch.max(d_dx, d_dy)

        # Create a mask that selects only the pixels with small depth differences
        mask = grad_mag < adaptive_threshold  # shape: (B, V, H, W)

        # Compute the dot product between rendered and surface normals; both are (B, V, 3, H, W)
        dot = (prediction.rend_normal * prediction.surf_normal).sum(dim=2)  # now (B, V, H, W)
        normal_error = 1 - dot

        # Apply the mask (only where depth is continuous) to compute the loss
        if mask.sum() > 0:
            normal_consistency_loss = lambda_normal * normal_error[mask].mean()
        else:
            normal_consistency_loss = torch.tensor(0.0, dtype=normal_error.dtype, device=normal_error.device)

        # dist_loss = lambda_dist * prediction.dist.mean()
        
        # total_normal_loss = normal_consistency_loss + dist_loss
        total_normal_loss = normal_consistency_loss
        
        return total_normal_loss
            
