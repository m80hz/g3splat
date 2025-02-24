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

# from ..misc.utils import inspect_depth_tensor, vis_depth_map
# import matplotlib.pyplot as plt

@dataclass
class LossNormalCfg:
    lambda_normal: float
    lambda_distortion: float
    apply_normal_after_step: int
    apply_distortion_after_step: int
    normal_valid_threshold: float = 1e-6  # threshold to consider a normal valid


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
        adaptive_threshold = 0.05 * median_depth  # adjust the factor as needed

        # Compute depth differences along x and y
        d_dx = torch.abs(depth[:, :, :, 1:] - depth[:, :, :, :-1])
        d_dy = torch.abs(depth[:, :, 1:, :] - depth[:, :, :-1, :])

        # Pad to get back to (B, V, H, W)
        d_dx = F.pad(d_dx, (0, 1, 0, 0), mode='replicate')
        d_dy = F.pad(d_dy, (0, 0, 0, 1), mode='replicate')

        # One option: use the maximum difference per pixel (or you could use a norm)
        grad_mag = torch.max(d_dx, d_dy)

        # Create a mask that selects only the pixels with small depth differences
        depth_mask = grad_mag < adaptive_threshold  # shape: (B, V, H, W)

        # Validate normals: check that the magnitude of both normals is above a small threshold.
        valid_rend = torch.norm(prediction.rend_normal, dim=2) > self.cfg.normal_valid_threshold
        valid_surf = torch.norm(prediction.surf_normal, dim=2) > self.cfg.normal_valid_threshold
        normals_valid_mask = valid_rend & valid_surf

        # Create a near-far mask based on raw depth:
        nonzero_depth = depth[depth > 0]
        if nonzero_depth.numel() > 0:
            near_depth = nonzero_depth.quantile(0.10)
            far_depth = nonzero_depth.quantile(0.90)
        else:
            near_depth = depth.min()
            far_depth = depth.max()
        near_far_mask = (depth >= near_depth) & (depth <= far_depth)

        # Combine masks: continuous depth, valid normals, and in the near-far range contribute
        valid_mask = depth_mask & normals_valid_mask & near_far_mask

        # Compute dot product between rendered and surface normals; both are (B, V, 3, H, W)
        dot = (prediction.rend_normal * prediction.surf_normal).sum(dim=2)  # now (B, V, H, W)
        normal_error = 1 - torch.abs(dot)

        # Apply the combined valid mask to compute the loss
        if valid_mask.sum() > 0:
            normal_consistency_loss = lambda_normal * normal_error[valid_mask].mean()
        else:
            normal_consistency_loss = torch.tensor(0.0, dtype=normal_error.dtype, device=normal_error.device)

        # fig, ax = plt.subplots(2, 6, figsize=(21, 14))
        # # Top row titles
        # ax[0, 0].set_title("Context Image")
        # ax[0, 1].set_title("Target")
        # ax[0, 2].set_title("Rendered Depth")
        # ax[0, 3].set_title("Rendered Normal")
        # ax[0, 4].set_title("Surface Normal")
        # ax[0, 5].set_title("Valid Mask")

        # ax[0, 0].imshow(batch["context"]["image"][0, 0].detach().cpu().permute(1, 2, 0) * 0.5 + 0.5)
        # ax[1, 0].imshow(batch["context"]["image"][0, 1].detach().cpu().permute(1, 2, 0) * 0.5 + 0.5)
        # ax[0, 1].imshow(batch["target"]["image"][0, 0].detach().cpu().permute(1, 2, 0))
        # ax[1, 1].imshow(batch["target"]["image"][0, 1].detach().cpu().permute(1, 2, 0))
        # ax[0, 2].imshow(vis_depth_map(depth[0, 0]).detach().cpu().permute(1, 2, 0), cmap='grey')
        # ax[1, 2].imshow(vis_depth_map(depth[0, 1]).detach().cpu().permute(1, 2, 0), cmap='grey')
        # ax[0, 3].imshow(prediction.rend_normal[0, 0].detach().cpu().permute(1, 2, 0))
        # ax[1, 3].imshow(prediction.rend_normal[0, 1].detach().cpu().permute(1, 2, 0))
        # ax[0, 4].imshow(prediction.surf_normal[0, 0].detach().cpu().permute(1, 2, 0))
        # ax[1, 4].imshow(prediction.surf_normal[0, 1].detach().cpu().permute(1, 2, 0))
        # ax[0, 5].imshow(valid_mask[0, 0].detach().cpu(), cmap='grey')
        # ax[1, 5].imshow(valid_mask[0, 1].detach().cpu(), cmap='grey')
        
        # plt.tight_layout()
        # # plt.show()
        # plt.savefig(f"normal_eval_{global_step}.png")


        # dist_loss = lambda_dist * prediction.dist.mean()
        # total_normal_loss = normal_consistency_loss + dist_loss

        total_normal_loss = normal_consistency_loss
        
        return total_normal_loss
            
