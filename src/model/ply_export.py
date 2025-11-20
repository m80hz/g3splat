from pathlib import Path

import numpy as np
import torch
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from torch import Tensor


def construct_list_of_attributes(num_rest: int) -> list[str]:
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    for i in range(num_rest):
        attributes.append(f"f_rest_{i}")
    attributes.append("opacity")
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")
    return attributes


def export_ply(
    means: Float[Tensor, "gaussian 3"],
    scales: Float[Tensor, "gaussian 3"],
    rotations: Float[Tensor, "gaussian 4"],   # expected in wxyz order
    harmonics: Float[Tensor, "gaussian 3 d_sh"],
    opacities: Float[Tensor, "gaussian"],
    path: Path,
    shift_and_scale: bool = False,
    save_sh_dc_only: bool = True,
):
    if shift_and_scale:
        # Shift the scene so that the median Gaussian is at the origin.
        means = means - means.median(dim=0).values

        # Rescale the scene so that most Gaussians are within range [-1, 1].
        scale_factor = means.abs().quantile(0.95, dim=0).max()
        means = means / scale_factor
        scales = scales / scale_factor

    # # Apply the rotation to the Gaussian rotations.
    # rotations = R.from_quat(rotations.detach().cpu().numpy()).as_matrix()
    # rotations = R.from_matrix(rotations).as_quat()
    # x, y, z, w = rearrange(rotations, "g xyzw -> xyzw g")
    # rotations = np.stack((w, x, y, z), axis=-1)

    # rotations are already in wxyz order (model convention); just move to cpu/numpy
    rotations_np = rotations.detach().cpu().numpy()

    # Since current model use SH_degree = 4,
    # which require large memory to store, we can only save the DC band to save memory.
    f_dc = harmonics[..., 0]
    f_rest = harmonics[..., 1:].flatten(start_dim=1)

    dtype_full = [
        (attribute, "f4")
        for attribute in construct_list_of_attributes(
            0 if save_sh_dc_only else f_rest.shape[1]
        )
    ]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = [
        means.detach().cpu().numpy(),
        torch.zeros_like(means).detach().cpu().numpy(),        # normals
        f_dc.detach().cpu().contiguous().numpy(),
        f_rest.detach().cpu().contiguous().numpy(),
        opacities[..., None].detach().cpu().numpy(),
        scales.log().detach().cpu().numpy(),
        rotations_np,                                          # wxyz
    ]
    if save_sh_dc_only:
        # remove f_rest from attributes
        attributes.pop(3)

    attributes = np.concatenate(attributes, axis=1)
    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)
    

def save_gaussian_ply(gaussians, visualization_dump, example, save_path):

    # v, _, h, w = example["context"]["image"].shape[1:]

    # # Transform means into camera space.
    # means = rearrange(
    #     gaussians.means, "() (v h w spp) xyz -> h w spp v xyz", v=v, h=h, w=w
    # )

    # # Create a mask to filter the Gaussians. throw away Gaussians at the
    # # borders, since they're generally of lower quality.
    # mask = torch.zeros_like(means[..., 0], dtype=torch.bool)
    # GAUSSIAN_TRIM = 8
    # mask[GAUSSIAN_TRIM:-GAUSSIAN_TRIM, GAUSSIAN_TRIM:-GAUSSIAN_TRIM, :, :] = 1

    # def trim(element):
    #     element = rearrange(
    #         element, "() (v h w spp) ... -> h w spp v ...", v=v, h=h, w=w
    #     )
    #     return element[mask][None]

    # # convert the rotations from camera space to world space as required
    # cam_rotations = trim(visualization_dump["rotations"])[0]
    # c2w_mat = repeat(
    #     example["context"]["extrinsics"][0, :, :3, :3],
    #     "v a b -> h w spp v a b",
    #     h=h,
    #     w=w,
    #     spp=1,
    # )
    # c2w_mat = c2w_mat[mask]  # apply trim

    # cam_rotations_np = R.from_quat(
    #     cam_rotations.detach().cpu().numpy()
    # ).as_matrix()
    # world_mat = c2w_mat.detach().cpu().numpy() @ cam_rotations_np
    # world_rotations = R.from_matrix(world_mat).as_quat()
    # world_rotations = torch.from_numpy(world_rotations).to(
    #     visualization_dump["scales"]
    # )

    # export_ply(
    #     example["context"]["extrinsics"][0, 0],
    #     trim(gaussians.means)[0],
    #     trim(visualization_dump["scales"])[0],
    #     world_rotations,
    #     trim(gaussians.harmonics)[0],
    #     trim(gaussians.opacities)[0],
    #     save_path,
    # )
    
    # # # the third element is fixed (corresponding to the normal direction)
    # ratio = 0.01
    # min_scale_2d, _ = gaussians.scales.min(dim=-1, keepdim=True)
    # # Enforce a minimum value (e.g., 0.01) to prevent it from becoming too small.
    # min_third_scale = 1.e-6
    # third_scale = torch.clamp(min_scale_2d * ratio, min=min_third_scale)
    # # Extend the 2D scales to 3D
    # scaling_extended = torch.cat([gaussians.scales, third_scale], dim=-1)
    
    
    # print("Exporting Gaussian PLY with the following tensor shapes:")
    # print("Means shape:", gaussians.means.shape)
    # print("Scales shape:", scaling_extended.shape)
    # print("Rotations shape:", gaussians.rotations.shape)
    # print("Harmonics shape:", gaussians.harmonics.shape)
    # print("Opacities shape:", gaussians.opacities.shape)
    # print("Save path:", save_path)
    
    export_ply(
        gaussians.means.squeeze(0),
        gaussians.scales.squeeze(0),
        # scaling_extended.squeeze(0),
        gaussians.rotations.squeeze(0),
        gaussians.harmonics.squeeze(0),
        gaussians.opacities.squeeze(0),
        save_path,
        shift_and_scale=False,
        save_sh_dc_only=True
    )



