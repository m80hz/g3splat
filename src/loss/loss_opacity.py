from dataclasses import dataclass
import torch
from torch import Tensor
import torch.nn.functional as F

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss

@dataclass
class LossOpacityCfg:
    lambda_opacity: float           # scaling factor for this loss
    apply_opacity_after_step: int   # only apply after this many iterations
    threshold: float = 0.7          # desired minimum opacity; no penalty for opacities >= threshold
    gamma: float = 5.0              # steepness factor; adjust based on typical opacity gap
    clamp_min: float = -10.0        
    clamp_max: float = 10.0         

@dataclass
class LossOpacityCfgWrapper:
    opacity: LossOpacityCfg

class LossOpacity(Loss[LossOpacityCfg, LossOpacityCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> torch.Tensor:
        # Only apply the opacity loss after the specified training step.
        lambda_opacity = self.cfg.lambda_opacity if global_step > self.cfg.apply_opacity_after_step else 0.0
        if lambda_opacity == 0.0:
            return torch.tensor(0.0, device=gaussians.opacities.device)
       
        opacities: Tensor = gaussians.opacities           # (B, N) 
        
        x = self.cfg.gamma * (self.cfg.threshold - opacities)
        x = torch.clamp(x, min=self.cfg.clamp_min, max=self.cfg.clamp_max)
        
        epsilon = 1e-6
        raw_loss = F.softplus(x + epsilon) - F.softplus(torch.tensor(0.0, device=opacities.device))
        
        loss_per_gaussian = torch.clamp(raw_loss, min=0.0)
        
        # Dynamic normalization of the loss by the standard deviation of x, 
        # to maintain a consistent scale across batches and gradient magnitude stable.
        norm_factor = torch.std(x) + epsilon
        normalized_loss = loss_per_gaussian / norm_factor
        
        # Average the loss over all Gaussians and scale by lambda_opacity.
        total_loss = lambda_opacity * normalized_loss.mean()

        return total_loss

