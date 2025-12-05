from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor, nn

from ....geometry.projection import get_world_rays
from ....misc.sh_rotation import rotate_sh
from .gaussians import build_covariance


@dataclass
class Gaussians:
    means: Float[Tensor, "*batch 3"]
    covariances: Float[Tensor, "*batch 3 3"]
    scales: Float[Tensor, "*batch 3"]
    rotations: Float[Tensor, "*batch 4"]
    harmonics: Float[Tensor, "*batch 3 _"]
    opacities: Float[Tensor, " *batch"]


@dataclass
class GaussianAdapterCfg:
    gaussian_scale_min: float
    gaussian_scale_max: float
    sh_degree: int
    gaussian_type: Literal["2d", "3d"] = "3d"


class GaussianAdapter(nn.Module):
    cfg: GaussianAdapterCfg

    def __init__(self, cfg: GaussianAdapterCfg):
        super().__init__()
        self.cfg = cfg

        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.cfg.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    # --- helpers shared between 2d/3d variants ---

    def _normalize_quaternion(self, rotations: Tensor, eps: float) -> Tensor:
        # Quaternions are assumed to be in (w, x, y, z) order
        return rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

    def _process_sh(
        self,
        sh: Tensor,
        opacities: Tensor,
    ) -> Tensor:
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        return sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask

    def _map_raw_scales(self, raw_scales: Tensor) -> Tensor:
        # Shared nonlinearity used by UnifiedGaussianAdapter (2D and 3D)
        scales = 0.001 * F.softplus(raw_scales)
        return scales.clamp_max(0.3)

    def _compute_depth_pixel_scaling(
        self,
        depths: Tensor,
        intrinsics: Tensor,
        image_shape: tuple[int, int],
    ) -> Tensor:
        device = intrinsics.device
        h, w = image_shape
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        multiplier = self.get_scale_multiplier(intrinsics, pixel_size)

        factor = depths * multiplier  # shape (...,)
        return factor

    def forward(
        self,
        extrinsics: Float[Tensor, "*#batch 4 4"],
        intrinsics: Float[Tensor, "*#batch 3 3"],
        coordinates: Float[Tensor, "*#batch 2"],
        depths: Float[Tensor, "*#batch"],
        opacities: Float[Tensor, "*#batch"],
        raw_gaussians: Float[Tensor, "*#batch _"],
        image_shape: tuple[int, int],
        eps: float = 1e-8,
    ) -> Gaussians:
        if self.cfg.gaussian_type == "3d":
            num_scales = 3
        else:
            num_scales = 2

        raw_scales, rotations, sh = raw_gaussians.split(
            (num_scales, 4, 3 * self.d_sh), dim=-1
        )

        # map raw scales
        scale_min = self.cfg.gaussian_scale_min
        scale_max = self.cfg.gaussian_scale_max
        scales = scale_min + (scale_max - scale_min) * raw_scales.sigmoid()  # (..., 2|3)

        rotations = self._normalize_quaternion(rotations, eps)
        
        factor = self._compute_depth_pixel_scaling(depths, intrinsics, image_shape)
        scales = scales * factor[..., None]

        if self.cfg.gaussian_type == "3d":
            scales_3d = scales  # (..., 3)
        else:
            # Use ratio-based third scale as in UnifiedGaussianAdapter
            ratio = 0.01
            min_scale_2d, _ = scales.min(dim=-1, keepdim=True)
            min_third_scale = 1.0e-6
            third_scale = torch.clamp(min_scale_2d * ratio, min=min_third_scale)
            scales_3d = torch.cat([scales, third_scale], dim=-1)  # (..., 3)

        sh = self._process_sh(sh, opacities)
        covariances = build_covariance(scales_3d, rotations)
        c2w_rotations = extrinsics[..., :3, :3]
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)

        origins, directions = get_world_rays(coordinates, extrinsics, intrinsics)
        means = origins + directions * depths[..., None]

        return Gaussians(
            means=means,
            covariances=covariances,
            # harmonics=rotate_sh(sh, c2w_rotations[..., None, :, :]),
            harmonics=sh,
            opacities=opacities,
            scales=scales_3d,  # always 3D scales
            # Note: rotations of the Gaussians built with this variant of the adapter are not 
            # in the world frame and are left as-is (unlike covariances which are rotated to world frame)
            rotations=rotations.broadcast_to((*scales_3d.shape[:-1], 4)),
        )

    def get_scale_multiplier(
        self,
        intrinsics: Float[Tensor, "*#batch 3 3"],
        pixel_size: Float[Tensor, "*#batch 2"],
        multiplier: float = 0.1,
    ) -> Float[Tensor, " *batch"]:
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    @property
    def d_sh(self) -> int:
        return (self.cfg.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        if self.cfg.gaussian_type == "3d":
            # 3 for scale + 4 for rotation + 3*d_sh for harmonics
            return 7 + 3 * self.d_sh
        else:
            # 2 for scale + 4 for rotation + 3*d_sh for harmonics
            return 6 + 3 * self.d_sh


class UnifiedGaussianAdapter(GaussianAdapter):
    def forward(
        self,
        means: Float[Tensor, "*#batch 3"],
        depths: Float[Tensor, "*#batch"],
        opacities: Float[Tensor, "*#batch"],
        raw_gaussians: Float[Tensor, "*#batch _"],
        eps: float = 1e-8,
        intrinsics: Optional[Float[Tensor, "*#batch 3 3"]] = None,
        coordinates: Optional[Float[Tensor, "*#batch 2"]] = None,
    ) -> Gaussians:
        if self.cfg.gaussian_type == "2d":
            raw_scales_2d, rotations, sh = raw_gaussians.split((2, 4, 3 * self.d_sh), dim=-1)
            scales_2d = self._map_raw_scales(raw_scales_2d)
            ratio = 0.01
            min_scale_2d, _ = scales_2d.min(dim=-1, keepdim=True)
            min_third_scale = 1.0e-6
            third_scale = torch.clamp(min_scale_2d * ratio, min=min_third_scale)
            scales_3d = torch.cat([scales_2d, third_scale], dim=-1)
            rotations = self._normalize_quaternion(rotations, eps)
            sh = self._process_sh(sh, opacities)
            covariances = build_covariance(scales_3d, rotations)
            return Gaussians(
                means=means,
                covariances=covariances,
                harmonics=sh,
                opacities=opacities,
                scales=scales_3d,
                rotations=rotations.broadcast_to((*scales_3d.shape[:-1], 4)),
            )

        raw_scales_3d, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)
        scales_3d = self._map_raw_scales(raw_scales_3d)
        rotations = self._normalize_quaternion(rotations, eps)
        sh = self._process_sh(sh, opacities)
        covariances = build_covariance(scales_3d, rotations)
        return Gaussians(
            means=means,
            covariances=covariances,
            harmonics=sh,
            opacities=opacities,
            scales=scales_3d,
            rotations=rotations.broadcast_to((*scales_3d.shape[:-1], 4)),
        )

