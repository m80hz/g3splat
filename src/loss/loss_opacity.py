from dataclasses import dataclass
import torch
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossOpacityCfg:
    lambda_opacity: float
    apply_opacity_after_step: int
    threshold: float = 0.5  # threshold below which we start penalizing opacities


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
        
        all_opacities: Tensor = gaussians.opacities  # (B, N)
        
        # This loss is nearly zero when opacity > threshold and increases smoothly when opacity falls below threshold.
        penalty = F.softplus(self.cfg.threshold - all_opacities)
        
        # Compute the mean loss over all gaussians in the batch
        loss_value = lambda_opacity * penalty.mean()
        
        return loss_value


# --- Dummy Test ---
if __name__ == "__main__":
    import sys
    B = 8
    V = 2
    C = 3
    H, W = 256, 256
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    dummy_image = torch.ones(B, V, C, H, W, device=device)
    intr_mat = torch.tensor([[0.8969, 0.0, 0.5],
                              [0.0, 0.8971, 0.5],
                              [0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    intrinsics = intr_mat.unsqueeze(0).unsqueeze(0).repeat(B, V, 1, 1)  # (B,V,3,3)
    
    # Set view1 extrinsics to identity; set view2 extrinsics to a translation along x by 0.1.
    T1 = torch.eye(4, device=device, dtype=torch.float32)
    T2 = torch.eye(4, device=device, dtype=torch.float32)
    T2[0, 3] = 0.1  # translation along x
    extrinsics = torch.stack([T1, T2], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # (B,V,4,4)
    
    batch = {
        "context": {
            "image": dummy_image,       # (B,V,C,H,W)
            "intrinsics": intrinsics,     # (B,V,3,3)
            "extrinsics": extrinsics,     # (B,V,4,4)
            "img_shape": (H, W),
        }
    }
    
    # For testing, we create a dummy gaussians object with opacities.
    # Let's assume there are N gaussians per batch.
    N = 1000
    class DummyGaussians:
        def __init__(self, B, N, device):
            # Random opacities in [0, 1] (simulate some low values as well)
            self.opacities = torch.rand(B, N, device=device)
    
    gaussians = DummyGaussians(B, N, device)
    
    # Set up configuration for the loss.
    cfg = LossOpacityCfg(lambda_opacity=1.0, apply_opacity_after_step=1000, threshold=0.1)
    cfg_wrapper = LossOpacityCfgWrapper(opacity=cfg)
    
    # Create the loss module.
    loss_module = LossOpacity(cfg, cfg_wrapper)
    
    # Simulate a global step beyond the threshold.
    global_step = 1500
    
    loss_value = loss_module.forward(prediction=None, batch=batch, gaussians=gaussians, global_step=global_step)
    print("Opacity Loss:", loss_value.item())
