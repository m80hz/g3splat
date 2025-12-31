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
from ..model.encoder.common.gaussians import gaussian_orientation_from_scales


@dataclass
class LossOrientCfg:
    lambda_context_views_orient: float
    lambda_context_orient_smoothness: float
    on_novel_views: bool
    lambda_novel_views_orient: float
    lambda_novel_views_distortion: float
    apply_orient_after_step: int
    apply_distortion_after_step: int
    valid_threshold: float = 1e-1  
    depth_valid_threshold: float = 1e-3
    depth_disc_multiplier: float = 3.0
    depth_disc_slope: float = 0.1
    huber_delta: float = 0.1        # in cosine space
    lambda_context_scale_flatten: float = 0.0


@dataclass
class LossOrientCfgWrapper:
    orient: LossOrientCfg


class LossOrient(Loss[LossOrientCfg, LossOrientCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        
        # --------------------
        # Context View Loss
        # --------------------
        # Extract image dimensions; batch["context"]["image"] shape: (B, V, C, H, W), V == 2.
        B, V, C, H, W = batch["context"]["image"].shape
        eps = 1e-8  # small constant for numerical stability

        context_view_loss = torch.tensor(0.0, device=prediction.depth.device)
        lambda_ctx_normal = self.cfg.lambda_context_views_orient
        lambda_ctx_smooth = self.cfg.lambda_context_orient_smoothness
        need_normals = (lambda_ctx_normal != 0.0) or (lambda_ctx_smooth != 0.0)

        gaussian_scales = rearrange(gaussians.scales, "b (v h w) d -> (b v) h w d", v=V, h=H, w=W)

        if need_normals:
            # -- compute point-cloud normals --
            all_pts3d = rearrange(gaussians.means, "b (v h w) d -> (b v) h w d", v=V, h=H, w=W)
            surf_normals_ptc, weights = points_to_normal(all_pts3d)  # (B*V, H, W, 3)

            # -- compute gaussian surfel normals --
            gaussian_rot = rearrange(gaussians.rotations, "b (v h w) d -> (b v) h w d", v=V, h=H, w=W)
            gs_surfel_normals = gaussian_orientation_from_scales(gaussian_rot, gaussian_scales)  # (B*V, H, W, 3)

            # normalize both sets of normals
            norm_ptc = surf_normals_ptc.norm(dim=-1, keepdim=True)
            norm_gs = gs_surfel_normals.norm(dim=-1, keepdim=True)
            surf_normals = surf_normals_ptc / (norm_ptc + eps)
            gs_normals = gs_surfel_normals / (norm_gs + eps)

            if self.cfg.lambda_context_orient_smoothness == 0.0:
                # -- only normal consistency loss --
                dot = (surf_normals * gs_normals).sum(-1)
                ang_err = 1.0 - dot
                loss_ang_per_pixel = F.smooth_l1_loss(ang_err, torch.zeros_like(ang_err), reduction='none', beta=self.cfg.huber_delta)
                loss_ang_per_pixel = weights * loss_ang_per_pixel
                loss_ang_mean = loss_ang_per_pixel.mean()
                context_view_loss = self.cfg.lambda_context_views_orient * loss_ang_mean

            else:
                # -- both normal consistency and smoothness loss --
                # reshape to (B, V, H, W, 3)
                surf_normals = rearrange(surf_normals, "(b v) h w d -> b v h w d", b=B, v=V)
                gs_normals = rearrange(gs_normals, "(b v) h w d -> b v h w d", b=B, v=V)

                # -- build depth-edge-based soft mask --
                extrinsics = batch["context"]["extrinsics"]  # (B, V, 4, 4)
                T0 = extrinsics[:, 0]                       # cam0 to world
                T_inv_all = torch.inverse(extrinsics)       # world to cam_v for each v
                all_pts3d = rearrange(all_pts3d, "(b v) h w d -> b v h w d", b=B, v=V)

                soft_masks, w_x_list, w_y_list, depth_maps = [], [], [], []
                for v in range(V):
                    pts0 = all_pts3d[:, v]
                    ones = torch.ones(B, H, W, 1, device=pts0.device, dtype=pts0.dtype)
                    pts_h = torch.cat([pts0, ones], dim=-1).permute(0, 3, 1, 2)
                    flat = pts_h.view(B, 4, -1)
                    world = torch.bmm(T0, flat)
                    cam_v = torch.bmm(T_inv_all[:, v], world).view(B, 4, H, W)
                    d = cam_v[:,2]  # (B,H,W)
                    depth_maps.append(d)
                
                    # normalize depth per image
                    flat_d = d.view(B, -1)
                    med = flat_d.median(dim=1)[0].view(B, 1, 1)
                    std = flat_d.std(dim=1).view(B, 1, 1).clamp(min=1e-3)
                    dn = (d - med) / (std + eps)
                    

                    # spatial depth gradients
                    gx = F.pad(torch.abs(dn[:, :, 1:] - dn[:, :, :-1]), (0, 1), 'replicate')
                    gy = F.pad(torch.abs(dn[:, 1:, :] - dn[:, :-1, :]), (0, 0, 0, 1), 'replicate')

                    # combined magnitude for unified soft mask
                    gm = torch.sqrt(gx ** 2 + gy ** 2 + eps)
                    thr_g = torch.median(gm.view(B, -1), dim=1)[0].view(B, 1, 1) * self.cfg.depth_disc_multiplier
                    sm = torch.sigmoid(-((gm - thr_g) / self.cfg.depth_disc_slope).clamp(-40, 40))
                    soft_masks.append(sm)

                    # **separate thresholds per axis**
                    thr_x = torch.median(gx.view(B, -1), dim=1)[0].view(B, 1, 1) * self.cfg.depth_disc_multiplier * 2.0  # to put less weight near weaker depth edges
                    thr_y = torch.median(gy.view(B, -1), dim=1)[0].view(B, 1, 1) * self.cfg.depth_disc_multiplier * 2.0

                    # directional weights (detached)
                    wx = torch.sigmoid(-((gx - thr_x) / self.cfg.depth_disc_slope).clamp(-40, 40)).detach()
                    wy = torch.sigmoid(-((gy - thr_y) / self.cfg.depth_disc_slope).clamp(-40, 40)).detach()
                    w_x_list.append(wx)
                    w_y_list.append(wy)

                depth_map = torch.stack(depth_maps, dim=1)  # (B, V, H, W)
                soft_mask = torch.stack(soft_masks, dim=1)  # (B, V, H, W)
                w_x = torch.stack(w_x_list, dim=1)         # (B, V, H, W)
                w_y = torch.stack(w_y_list, dim=1)         # (B, V, H, W)
                
                # validity mask detached
                vn = surf_normals.norm(dim=-1) > self.cfg.valid_threshold
                vg = gs_normals.norm(dim=-1)   > self.cfg.valid_threshold
                vd = depth_map > self.cfg.depth_valid_threshold
                valid = (vn & vg & vd).float().detach()  
                valid_mask = (valid * soft_mask).detach()
                
                # -- normal consistency loss --
                dot = (surf_normals * gs_normals).sum(-1)
                ang_err = 1.0 - dot
                loss_ang = F.smooth_l1_loss(ang_err, torch.zeros_like(ang_err), reduction='none', beta=self.cfg.huber_delta)
                loss_ang = (loss_ang * valid_mask).sum() / (valid_mask.sum() + eps)
                consistency_loss = self.cfg.lambda_context_views_orient * loss_ang

                # -- normal smoothness prior (edge-aware) --
                gs_c = gs_normals.permute(0, 1, 4, 2, 3)  # (B, V, 3, H, W)
                # compute diffs manually to avoid unsupported F.pad on 5D
                # X-direction diffs (width axis dim=4)
                diff_x = gs_c[..., :, :, 1:] - gs_c[..., :, :, :-1]  # (B,V,3,H,W-1)
                last_col = diff_x[..., :, :, -1:].clone()            # replicate last column
                dx_n = torch.cat([diff_x, last_col], dim=4)          # (B,V,3,H,W)
                # Y-direction diffs (height axis dim=3)
                diff_y = gs_c[..., :, 1:, :] - gs_c[..., :, :-1, :]  # (B,V,3,H-1,W)
                last_row = diff_y[..., :, -1:, :].clone()            # replicate last row
                dy_n = torch.cat([diff_y, last_row], dim=3)          # (B,V,3,H,W)
                
                # aggregate magnitude per pixel
                dx_abs = dx_n.abs().sum(dim=2)  # (B, V, H, W)
                dy_abs = dy_n.abs().sum(dim=2)

                smooth_x = (dx_abs * w_x * valid_mask).sum() / ((w_x * valid_mask).sum() + eps)
                smooth_y = (dy_abs * w_y * valid_mask).sum() / ((w_y * valid_mask).sum() + eps)
                smoothness_loss = self.cfg.lambda_context_orient_smoothness * (smooth_x + smooth_y)

                # Combine both context terms
                context_view_loss = consistency_loss + smoothness_loss
                if valid_mask.sum().item() < 100:
                    context_view_loss = torch.tensor(0.0, device=prediction.depth.device)

        # -- scale flattening regularization --
        if self.cfg.lambda_context_scale_flatten != 0.0:
            min_scales = gaussian_scales.min(dim=-1).values  # (B*V, H, W)
            scale_reg_loss = self.cfg.lambda_context_scale_flatten * min_scales.mean()
            context_view_loss = context_view_loss + scale_reg_loss


        # -----------------------------------
        # Novel View Loss
        # -----------------------------------
        if not self.cfg.on_novel_views:
            novel_view_loss = torch.tensor(0.0, device=prediction.depth.device)
        else:
            
            # Only apply the normal consistency loss after a certain training step.
            lambda_novel_views_orient = self.cfg.lambda_novel_views_orient if global_step > self.cfg.apply_orient_after_step else 0.0
            lambda_novel_views_dist = self.cfg.lambda_novel_views_distortion if global_step > self.cfg.apply_distortion_after_step else 0.0
            if lambda_novel_views_orient == 0.0 and lambda_novel_views_dist == 0.0:
                return context_view_loss + torch.tensor(0.0, device=prediction.depth.device)

            eps = 1e-6  # small constant for numerical stability

            depth = rearrange(prediction.depth, "b v h w -> (b v) h w").detach()                        # (B, H, W)
            surf_normal = rearrange(prediction.surf_normal, "b v c h w -> (b v) c h w").detach()        # (B, 3, H, W)   (depth-derived normals)
            rend_normal = rearrange(prediction.rend_normal, "b v c h w -> (b v) c h w")        # (B, 3, H, W)   (rendered normals)
            # rend_dist = rearrange(prediction.dist, "b v h w -> (b v) h w")                     # (B, H, W)   (depth distortion)        
            
            norm_surf = torch.norm(surf_normal, dim=1, keepdim=True)  # (B, 1, H, W)
            norm_rend = torch.norm(rend_normal, dim=1, keepdim=True)  # (B, 1, H, W)
            valid_normal_mask = ((norm_surf > self.cfg.valid_threshold) & (norm_rend > self.cfg.valid_threshold)).float()
            
            valid_depth_mask = (depth > self.cfg.depth_valid_threshold).float().unsqueeze(1)  # (B, 1, H, W)
            
            # combined mask from normals and depth validity.
            valid_mask = valid_normal_mask * valid_depth_mask  # (B, 1, H, W)
            
            
            # gradient-based soft mask for depth discontinuities
            # -------------------------------
            B, H, W = depth.shape
            depth = depth.float()  # ensure floating point for computations
            
            depth_flat = depth.view(B, -1)
            depth_median = depth_flat.median(dim=1)[0].view(B, 1, 1)
            depth_std = depth_flat.std(dim=1).view(B, 1, 1).clamp(min=1e-3)
            depth_norm = (depth - depth_median) / (depth_std + eps)  # (B, H, W)

            # spatial gradients of the normalized depth using finite difference
            grad_x = torch.abs(depth_norm[:, :, 1:] - depth_norm[:, :, :-1])
            grad_x = F.pad(grad_x, (0, 1), mode='replicate')  
            grad_y = torch.abs(depth_norm[:, 1:, :] - depth_norm[:, :-1, :])
            grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')

            grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + eps)  # (B, H, W)
            
            # compute an adaptive threshold per image based on the median gradient magnitude.
            adaptive_threshold = torch.median(grad_mag.view(B, -1), dim=1)[0].view(B, 1, 1)
            adaptive_threshold = adaptive_threshold * self.cfg.depth_disc_multiplier

            # create a soft mask: pixels with gradient magnitude much lower than the adaptive threshold have weight ~1,
            # while those with high gradient magnitude are downweighted.
            # Using a sigmoid function for smooth transition.
            # soft_mask = 1.0 / (1.0 + torch.exp((grad_mag - adaptive_threshold) / self.cfg.depth_disc_slope))    # soft_mask: shape (B, H, W)
            x = ((grad_mag - adaptive_threshold) / self.cfg.depth_disc_slope).clamp(-40, 40)
            soft_mask = torch.sigmoid(-x)    # numerically equivalent to 1/(1+exp(x)), but more stable

            # the overall validity mask.
            final_valid_mask = valid_mask * soft_mask.unsqueeze(1)  # (B, 1, H, W)
            final_valid_mask = final_valid_mask.detach()  # Detach to avoid gradients through the mask

            surf_normal_normed = surf_normal / (norm_surf + eps)
            rend_normal_normed = rend_normal / (norm_rend + eps)

            # using absolute value to account for direction ambiguity (n and -n are equivalent).
            dot_product = torch.sum(surf_normal_normed * rend_normal_normed, dim=1)  # (B, H, W)
            abs_dot = torch.abs(dot_product)
            angular_error = 1.0 - abs_dot  # Zero error when perfectly aligned (or anti-aligned).

            loss_per_pixel = F.smooth_l1_loss(
                angular_error, 
                torch.zeros_like(angular_error), 
                reduction='none', 
                beta=self.cfg.huber_delta
            )

            masked_loss = loss_per_pixel * final_valid_mask.squeeze(1)

            loss_sum = masked_loss.sum()
            valid_count = final_valid_mask.sum() + eps
            normal_consistency_loss = loss_sum / valid_count
            total_normal_loss = lambda_novel_views_orient * normal_consistency_loss
            novel_view_loss = total_normal_loss
            # -------------------------------
            # dist_loss = lambda_novel_views_dist * (rend_dist).mean()
            # novel_view_loss += dist_loss
            # -------------------------------
            # If the number of valid pixels is too low, set the loss to zero (safeguard)
            if valid_count.item() < 100:
                novel_view_loss = torch.tensor(0.0, device=prediction.depth.device)
        
        
        # -----------------------------------
        # Combine context view loss and novel view loss.
        total_loss = context_view_loss + novel_view_loss
        
        return total_loss
