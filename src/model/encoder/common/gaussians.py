import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor


# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
def quaternion_to_matrix(
    quaternions: Float[Tensor, "*batch 4"],
    eps: float = 1e-8,
) -> Float[Tensor, "*batch 3 3"]:
    # Order changed to match 2D GS rasterizer
    r, i, j, k = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return rearrange(o, "... (i j) -> ... i j", i=3, j=3)


def build_covariance(
    scale: Float[Tensor, "*#batch 3"],
    rotation_wxyz: Float[Tensor, "*#batch 4"],
) -> Float[Tensor, "*batch 3 3"]:
    scale = scale.diag_embed()
    rotation = quaternion_to_matrix(rotation_wxyz)
    return (
        rotation
        @ scale
        @ rearrange(scale, "... i j -> ... j i")
        @ rearrange(rotation, "... i j -> ... j i")
    )


def gaussian_orientation_from_scales(
    rotation_wxyz: Float[Tensor, "*batch 4"],
    scale: Float[Tensor, "*batch 3"],
    column: int | None = None,
    eps: float = 1e-8,
) -> Float[Tensor, "*batch 3"]:
    """Return the rotation axis aligned with the chosen scale component."""
    rotation_wxyz = rotation_wxyz / (rotation_wxyz.norm(dim=-1, keepdim=True) + eps)
    rotation = quaternion_to_matrix(rotation_wxyz, eps=eps)

    if column is None:
        column_idx = scale.argmin(dim=-1)
    else:
        if column < 0 or column > 2:
            raise ValueError("column must be in [0, 2] when provided")
        column_idx = torch.full_like(scale[..., 0], column, dtype=torch.long)

    column_idx = column_idx.to(torch.long)
    # rotation shape: (..., 3, 3); build an index of shape (..., 3, 1)
    gather_idx = column_idx.unsqueeze(-1).unsqueeze(-2).expand(*rotation.shape[:-1], 1)
    direction = torch.gather(rotation, dim=-1, index=gather_idx).squeeze(-1)
    return direction / (direction.norm(dim=-1, keepdim=True) + eps)
