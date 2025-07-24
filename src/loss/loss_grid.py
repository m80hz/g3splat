from dataclasses import dataclass
import torch
from einops import rearrange
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
    huber_delta: float = 0.0         # optional huber on grid error (pixel_delta / (W - 1))
    # hinge_coef: float = 0.1         # weight for out-of-bounds hinge
    # neg_depth_coef: float = 0.1     # weight for negative-depth penalty

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
        # only apply after threshold step
        lambda_grid = self.cfg.lambda_grid if global_step > self.cfg.apply_grid_after_step else 0.0
        if lambda_grid == 0.0:
            return torch.tensor(0.0, device=gaussians.means.device)

        B, V, C, H, W = batch["context"]["image"].shape
        device = gaussians.means.device

        # reshape points
        all_pts3d = rearrange(gaussians.means, "b (v h w) d -> b v h w d", h=H, w=W)
        
        # retrieve camera intrinsics & extrinsics
        intrinsics = batch["context"]["intrinsics"]  # (B, V, 3, 3)
        extrinsics = batch["context"]["extrinsics"]  # (B, V, 4, 4): cam->world transforms

        # precompute transforms
        T0_cam0_to_world = extrinsics[:, 0]               # (B,4,4)
        T_world_to_cam_v = torch.inverse(extrinsics)      # (B,V,4,4)

        # prepare grid targets
        grid_int = self.xy_grid(W, H, device=device, cat_dim=-1, homogeneous=False)
        grid_norm = (grid_int.float() / torch.tensor([W-1, H-1], device=device).view(1,1,2) - 0.5) * 2    # map to [-1, 1]
        grid_flat = grid_norm.view(-1, 2)  # (N,2)

        total_align = torch.tensor(0.0, device=device)
        # total_hinge = torch.tensor(0.0, device=device)
        # total_negz  = torch.tensor(0.0, device=device)
        eps = 1e-6
        
        # helper: project from cam0 frame to cam-v
        def project_cam0_to_camv(pts_cam0: Tensor, v: int, K: Tensor):
            # pts_cam0: (B, H, W, 3) in cam0 frame
            # K: (B,3,3) intrinsics for view v
            ones = pts_cam0.new_ones(B, H, W, 1)
            pts_h = torch.cat([pts_cam0, ones], dim=-1)                  # (B,H,W,4)
            pts_flat = pts_h.permute(0,3,1,2).reshape(B,4,-1)            # (B,4,N)

            # cam0 -> world -> cam_v
            world_flat = T0_cam0_to_world.bmm(pts_flat)                  # (B,4,N)
            cam_flat = T_world_to_cam_v[:, v].bmm(world_flat)            # (B,4,N)
            cam = cam_flat.view(B,4,H,W)

            depth = cam[:,2].reshape(B,-1,1)                             # (B,N,1)
            proj = self.project_pts3d_to_px2d(cam, K, eps=1e-6, normalize=True)
            proj = proj.view(B,-1,2)                                    # (B,N,2)
            return proj, depth

        # loop over views
        used_view_counter = 0
        for v in range(V):
            K = intrinsics[:,v]  # (B,3,3)
            pts_cam0 = all_pts3d[:, v]    # (B,H,W,3)
            proj, depth = project_cam0_to_camv(pts_cam0, v, K)
            proj = torch.nan_to_num(proj) 

            # alignment L2 (optionally with Huber)
            diff = proj - grid_flat.unsqueeze(0)
            if self.cfg.huber_delta > 0:
                err = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction='none', beta=self.cfg.huber_delta).sum(-1, keepdim=True)
            else:
                err = diff.pow(2).sum(-1, keepdim=True)

            # # hinge for out-of-bounds (u,v >1)
            # over_u = torch.clamp(proj[...,0].abs() - 1, min=0, max=2)   # clamp to [0,2] in normalized coordinates
            # over_v = torch.clamp(proj[...,1].abs() - 1, min=0, max=2)   # clamp to [0,2] in normalized coordinates
            # L_hinge = (over_u.pow(2) + over_v.pow(2)).mean()
            # total_hinge += L_hinge
            
            # # negative-depth penalty
            # neg_mask = (depth.squeeze(-1) < 0)
            # if neg_mask.any():
            #     L_negz = depth[neg_mask].abs().mean()
            # else:
            #     L_negz = depth.new_tensor(0.0)

            # total_negz  += L_negz
            
            # valid mask: in [-1,1] and z>0
            mask = ((proj[...,0].abs() <= 1) & (proj[...,1].abs() <= 1) & (depth.squeeze(-1) > 0))
            mask = mask.float().unsqueeze(-1)  # (B, N, 1)
            Nvalid = mask.sum()
            if Nvalid < 100:
                continue

            # average alignment over all valid depth pixels
            L_align = (err * mask).sum() / (Nvalid + eps)

            total_align += L_align

            used_view_counter += 1
        
        # average across used views
        views_used = max(1, used_view_counter)
        mean_align = total_align / views_used
        # mean_hinge = total_hinge / views_used
        # mean_negz  = total_negz  / views_used

        # total_loss = lambda_grid * (mean_align + self.cfg.hinge_coef * mean_hinge + self.cfg.neg_depth_coef * mean_negz)        
        total_loss = lambda_grid * mean_align        
                
        return total_loss

    def project_pts3d_to_px2d(self, points3d: Tensor, K: Tensor, eps: float = 1.e-4, normalize: bool = True) -> Tensor:
        """
        Project 3D homogeneous points (B,4,H,W) to 2D pixel coordinates.
        K is assumed to be (B,3,3). With normalize=True, maps [0,1] to [-1,1].
        """
        B, C, H, W = points3d.shape
        if K.shape[-1] == 3:
            K_4x4 = torch.eye(4, dtype=K.dtype, device=K.device).unsqueeze(0).repeat(B, 1, 1)
            K_4x4[:, :3, :3] = K.clone()
        else:
            K_4x4 = K.clone()
        points3d_flat = points3d.view(B, C, -1)  # (B, 4, H*W)
        points2d = torch.bmm(K_4x4[:, :3, :], points3d_flat)  # (B, 3, H*W)
        xy = points2d[:, :2, :] / torch.clamp(points2d[:, 2:3, :], eps)
        xy = xy.view(B, 2, H, W).permute(0, 2, 3, 1)  # (B, H, W, 2)
        if normalize:
            # already normalised intrinsics
            xy = (xy - 0.5) * 2  # maps [0,1] to [-1,1]
        return xy
    
    @staticmethod
    def xy_grid(W: int, H: int, device=None, origin=(0, 0), unsqueeze=None, cat_dim=-1, homogeneous: bool = False, **arange_kw) -> Tensor:
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
    grid = LossGrid.xy_grid(W, H, device=device, cat_dim=-1, homogeneous=False)  # (H,W,2) in [0,W-1] and [0,H-1]
    grid_norm = grid.float() / torch.tensor([W-1, H-1], device=device).reshape(1, 1, 2)  # normalize to [0,1]
    X = (grid_norm[..., 0] - 0.5) / 0.8969
    Y = (grid_norm[..., 1] - 0.5) / 0.8971
    Z = torch.ones_like(X)
    pts3d_view = torch.stack([X, Y, Z], dim=-1)  # (H,W,3)
    
    # Expand to batch and two views.
    pts3d_view = pts3d_view.unsqueeze(0).unsqueeze(0).repeat(B, V, 1, 1, 1)  # (B,V,H,W,3)
    gaussians = rearrange(pts3d_view, "b v h w d -> b (v h w) d")
    
    cfg = LossGridCfgWrapper(grid=LossGridCfg(lambda_grid=0.1, apply_grid_after_step=0, huber_delta=0.0))
    loss_module = LossGrid(cfg=cfg)
    
    # Forward pass to compute grid loss.
    gaussians = Gaussians(means=gaussians, covariances=None, rotations=None, scales=None, harmonics=None, opacities=None)
    
    loss = loss_module.forward(None, batch, gaussians, global_step=1)
    print("Total grid loss:", loss.item())
