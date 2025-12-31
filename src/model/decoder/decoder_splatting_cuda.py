from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor, exp

from ...dataset import DatasetCfg
from ..types import Gaussians
from .cuda_splatting import DepthRenderingMode, render_cuda, render_cuda_3d
from .decoder import Decoder, DecoderOutput

DecoderType = Literal["2D", "3D"]


@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]
    background_color: list[float]
    make_scale_invariant: bool
    depth_ratio: int
    expected_depth: bool


class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: DecoderSplattingCUDACfg,
    ) -> None:
        super().__init__(cfg)
        self.make_scale_invariant = cfg.make_scale_invariant
        self.expected_depth = cfg.expected_depth
        self.depth_ratio = cfg.depth_ratio
        self.register_buffer(
            "background_color",
            torch.tensor(cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        cam_rot_delta: Float[Tensor, "batch view 3"] | None = None,
        cam_trans_delta: Float[Tensor, "batch view 3"] | None = None,
        decoder_type: DecoderType = "3D",
    ) -> DecoderOutput:
        b, v, _, _ = extrinsics.shape
        decoder_type = decoder_type.upper()
        if decoder_type == "2D":
            # 2D splatting expects only 2 scale values; take the first 2 if 3 are provided
            scales_2d = gaussians.scales[..., :2] if gaussians.scales.shape[-1] == 3 else gaussians.scales
            color, alpha, rend_normal, dist, depth, surf_normal = render_cuda(
                rearrange(extrinsics, "b v i j -> (b v) i j"),
                rearrange(intrinsics, "b v i j -> (b v) i j"),
                rearrange(near, "b v -> (b v)"),
                rearrange(far, "b v -> (b v)"),
                image_shape,
                repeat(self.background_color, "c -> (b v) c", b=b, v=v),
                repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
                repeat(scales_2d, "b g ss -> (b v) g ss", v=v),
                repeat(gaussians.rotations, "b g wxyz -> (b v) g wxyz", v=v),
                repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
                repeat(gaussians.opacities, "b g -> (b v) g", v=v),
                depth_ratio=self.depth_ratio,
                # repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
                gaussian_covariances=None,
                scale_invariant=self.make_scale_invariant,
                cam_rot_delta=rearrange(cam_rot_delta, "b v i -> (b v) i") if cam_rot_delta is not None else None,
                cam_trans_delta=rearrange(cam_trans_delta, "b v i -> (b v) i") if cam_trans_delta is not None else None,
                expected_depth=self.expected_depth,
            )
        elif decoder_type == "3D":
            color, alpha, rend_normal, dist, depth, surf_normal = render_cuda_3d(
                rearrange(extrinsics, "b v i j -> (b v) i j"),
                rearrange(intrinsics, "b v i j -> (b v) i j"),
                rearrange(near, "b v -> (b v)"),
                rearrange(far, "b v -> (b v)"),
                image_shape,
                repeat(self.background_color, "c -> (b v) c", b=b, v=v),
                repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
                repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
                repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
                repeat(gaussians.opacities, "b g -> (b v) g", v=v),
                scale_invariant=self.make_scale_invariant,
                cam_rot_delta=rearrange(cam_rot_delta, "b v i -> (b v) i") if cam_rot_delta is not None else None,
                cam_trans_delta=rearrange(cam_trans_delta, "b v i -> (b v) i") if cam_trans_delta is not None else None,
                expected_depth=self.expected_depth,
            )
        else:
            raise ValueError("Decoder type should be either 2D or 3D.")
        
        
        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)
        alpha = rearrange(alpha, "(b v) h w -> b v h w", b=b, v=v) if alpha is not None else None
        rend_normal = rearrange(rend_normal, "(b v) xyz h w -> b v xyz h w", b=b, v=v) if rend_normal is not None else None
        dist = rearrange(dist, "(b v) h w -> b v h w", b=b, v=v) if dist is not None else None
        depth = rearrange(depth, "(b v) h w -> b v h w", b=b, v=v) if depth is not None else None
        surf_normal = rearrange(surf_normal, "(b v) xyz h w -> b v xyz h w", b=b, v=v) if surf_normal is not None else None
        
        return DecoderOutput(color, alpha, rend_normal, dist, depth, surf_normal)
