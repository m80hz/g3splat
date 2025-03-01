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
class LossGridCfg:
    lambda_grid: float
    apply_grid_after_step: int


@dataclass
class LossGridCfgWrapper:
    grid: LossGridCfg
    

class LossGrid(Loss[LossGridCfg, LossGridCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> torch.Tensor:
        # Use grid loss only after the specified step.
        lambda_grid = self.cfg.lambda_grid if global_step > self.cfg.apply_grid_after_step else 0.0
        if lambda_grid == 0.0:
            return torch.tensor(0.0, device=gaussians.means.device)

        # Extract image dimensions; batch["context"]["image"] is (B, V, C, H, W), V==2.
        B, V, C, H, W = batch["context"]["image"].shape
        
        all_pts3d = rearrange(gaussians.means, "b (v h w) d -> b v h w d", h=H, w=W)     # (B, V, H, W, 3)
        pts3d1 = all_pts3d[:, 0, ...]  # (B, H, W, 3)
        pts3d2 = all_pts3d[:, 1, ...]  # (B, H, W, 3)
      
        # Get intrinsics and extrinsics.
        intrinsics = batch["context"]["intrinsics"]  # (B, V, 3, 3)
        extrinsics = batch["context"]["extrinsics"]  # (B, V, 4, 4)
        intr0 = intrinsics[:, 0, :, :]  # for view1
        intr1 = intrinsics[:, 1, :, :]  # for view2
        T1 = extrinsics[:, 0, :, :]     # view1 extrinsics (camera-to-world)
        T2 = extrinsics[:, 1, :, :]     # view2 extrinsics (camera-to-world)
        
        device = pts3d1.device

        # Helper: convert (B, H, W, 3) to homogeneous coordinates (B, 4, H, W)
        def pts3d_to_hom(pts):
            B, H, W, _ = pts.shape
            ones = torch.ones(B, H, W, 1, device=pts.device, dtype=pts.dtype)
            pts_h = torch.cat([pts, ones], dim=-1)  # (B, H, W, 4)
            return pts_h.permute(0, 3, 1, 2)         # (B, 4, H, W)
        
        # --- For view1 ---
        pts3d1_h = pts3d_to_hom(pts3d1)  # (B, 4, H, W)
        # Even if pts3d1 are in view1’s frame, to be cautious we use T1:
        world_pts1 = torch.bmm(T1, pts3d1_h.view(B, 4, -1)).view(B, 4, H, W)
        T1_inv = torch.inverse(T1)
        pts1_cam = torch.bmm(T1_inv, world_pts1.view(B, 4, -1)).view(B, 4, H, W)
        proj_view1 = self.project_pts3d_to_px2d(pts1_cam, intr0, eps=1e-4, normalize=True)
        proj_view1_flat = proj_view1.view(B, -1, 2)  # (B, N, 2)
        
        # --- For view2 ---
        # Compute relative transformation from view1 to view2:
        # T_rel = T2^{-1} * T1 maps points from view1’s frame to view2’s camera coordinates.
        T2_inv = torch.inverse(T2)
        T_rel = torch.bmm(T2_inv, T1)  # (B, 4, 4)
        pts3d2_h = pts3d_to_hom(pts3d2)  # (B, 4, H, W)
        pts2_cam = torch.bmm(T_rel, pts3d2_h.view(B, 4, -1)).view(B, 4, H, W)
        proj_view2 = self.project_pts3d_to_px2d(pts2_cam, intr1, eps=1e-4, normalize=True)
        proj_view2_flat = proj_view2.view(B, -1, 2)  # (B, N, 2)
        
        # Option 1: Create target grid using xy_grid and then normalize it.
        grid_int = self.xy_grid(W, H, device=device, cat_dim=-1, homogeneous=False)  # (H, W, 2) in [0, W-1] and [0, H-1]
        grid_norm = grid_int.float() / torch.tensor([W - 1, H - 1], device=device).reshape(1, 1, 2)
        grid_norm = (grid_norm - 0.5) * 2  # now in [-1, 1]
        grid_flat = grid_norm.view(-1, 2)   # (N, 2)
        
        def compute_view_loss(proj, grid):
            valid_mask = ((proj[..., 0] >= -1) & (proj[..., 0] <= 1) &
                          (proj[..., 1] >= -1) & (proj[..., 1] <= 1)).unsqueeze(-1)
            diff = proj - grid.unsqueeze(0)
            sq_err = diff.pow(2).sum(dim=-1, keepdim=True)
            sq_err = sq_err * valid_mask.float()
            eps = 1e-6
            loss = sq_err.sum() / (valid_mask.float().sum() + eps)
            return loss
        
        loss_view1 = compute_view_loss(proj_view1_flat, grid_flat)
        loss_view2 = compute_view_loss(proj_view2_flat, grid_flat)
        total_loss = lambda_grid * (loss_view1 + loss_view2)
        
        return total_loss

    def project_pts3d_to_px2d(self, points3d, K, eps=1.e-4, normalize=True):
        B, C, H, W = points3d.shape
        if K.shape[-1] == 3:
            K_4x4 = torch.eye(4, dtype=K.dtype, device=K.device).unsqueeze(0).repeat(B, 1, 1)
            K_4x4[:, :3, :3] = K.clone()
        else:
            K_4x4 = K.clone()
        points3d_flat = points3d.view(B, C, -1)
        points2d = torch.bmm(K_4x4[:, :3, :], points3d_flat)
        xy = points2d[:, :2, :] / (points2d[:, 2:3, :] + eps)
        xy = xy.view(B, 2, H, W).permute(0, 2, 3, 1)
        if normalize:
            xy = (xy - 0.5) * 2  # intrinsics are assumed to be already normalized, we map [0,1] to [-1,1]
        return xy
    
    @staticmethod
    # xy_grid returns integer coordinates in [0, W-1] and [0, H-1]
    def xy_grid(W, H, device=None, origin=(0, 0), unsqueeze=None, cat_dim=-1, homogeneous=False, **arange_kw):
        if device is None:
            import numpy as np
            arange, meshgrid, stack, ones = np.arange, np.meshgrid, np.stack, np.ones
        else:
            arange = lambda *a, **kw: torch.arange(*a, device=device, **kw)
            meshgrid, stack = torch.meshgrid, torch.stack
            ones = lambda *a: torch.ones(*a, device=device)
        tw, th = [arange(o, o+s, **arange_kw) for s, o in zip((W, H), origin)]
        grid = meshgrid(tw, th, indexing='xy')
        if homogeneous:
            grid = grid + (ones((H, W)),)
        if unsqueeze is not None:
            grid = (grid[0].unsqueeze(unsqueeze), grid[1].unsqueeze(unsqueeze))
        if cat_dim is not None:
            grid = stack(grid, cat_dim)
        return grid  # shape (H, W, 2)




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
    
    # Create ideal grid-aligned 3D points for view1.
    grid = LossGrid.xy_grid(W, H, device=device, cat_dim=-1, homogeneous=False)  # (H,W,2) with values in [0,W-1] and [0,H-1]
    grid_norm = grid.float() / torch.tensor([W-1, H-1], device=device).reshape(1, 1, 2)  # normalized to [0,1]
    X = (grid_norm[..., 0] - 0.5) / 0.8969
    Y = (grid_norm[..., 1] - 0.5) / 0.8971
    Z = torch.ones_like(X)
    pts3d_view = torch.stack([X, Y, Z], dim=-1)  # (H,W,3)
    
    # Expand to batch and two views.
    pts3d_view = pts3d_view.unsqueeze(0).unsqueeze(0).repeat(B, V, 1, 1, 1)  # (B,V,H,W,3)
    gaussians = rearrange(pts3d_view, "b v h w d -> b (v h w) d")
    
    loss_module = LossGrid()
    loss_module.cfg = LossGridCfg(lambda_grid=1.0, apply_grid_after_step=0)
    
    loss = loss_module.forward(None, batch, gaussians, global_step=1)
    print("Total grid loss:", loss.item())
