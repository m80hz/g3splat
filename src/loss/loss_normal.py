from dataclasses import dataclass

import torch
from einops import reduce
from jaxtyping import Float
from torch import Tensor

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

        # compute the dot product over the normal channels  shape: [batch, view, height, width]
        dot = (prediction.rend_normal * prediction.surf_normal).sum(dim=2)
        normal_error = 1 - dot
        normal_consistency_loss = lambda_normal * normal_error.mean()

        # dist_loss = lambda_dist * prediction.dist.mean()
        
        # total_normal_loss = normal_consistency_loss + dist_loss
        total_normal_loss = normal_consistency_loss
        
        return total_normal_loss
            
