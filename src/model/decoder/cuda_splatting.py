from math import isqrt
from typing import Literal

import torch
import torch.nn.functional as F

from diff_gaussian_rasterization import (GaussianRasterizationSettings as GaussianRasterizationSettings_3D, GaussianRasterizer as GaussianRasterizer_3D)
from diff_surfel_rasterization import (GaussianRasterizationSettings, GaussianRasterizer)

from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ...geometry.projection import get_fov, homogenize_points, depth_to_normal, depths_to_points
from ...misc.utils import inspect_depth_tensor

def get_projection_matrix(
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    fov_x: Float[Tensor, " batch"],
    fov_y: Float[Tensor, " batch"],
) -> Float[Tensor, "batch 4 4"]:
    """Maps points in the viewing frustum to (-1, 1) on the X/Y axes and (0, 1) on the Z
    axis. Differs from the OpenGL version in that Z doesn't have range (-1, 1) after
    transformation and that Z is flipped.
    """
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    top = tan_fov_y * near
    bottom = -top
    right = tan_fov_x * near
    left = -right

    (b,) = near.shape
    result = torch.zeros((b, 4, 4), dtype=torch.float32, device=near.device)
    result[:, 0, 0] = 2 * near / (right - left)
    result[:, 1, 1] = 2 * near / (top - bottom)
    result[:, 0, 2] = (right + left) / (right - left)
    result[:, 1, 2] = (top + bottom) / (top - bottom)
    result[:, 3, 2] = 1
    result[:, 2, 2] = far / (far - near)
    result[:, 2, 3] = -(far * near) / (far - near)
    return result


def render_cuda(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_scales: Float[Tensor, "batch gaussian 2"],
    gaussian_rotations: Float[Tensor, "batch gaussian 4"],
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    depth_ratio: int,
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"] | None = None,
    scale_invariant: bool = True,
    use_sh: bool = True,
    cam_rot_delta: Float[Tensor, "batch 3"] | None = None,
    cam_trans_delta: Float[Tensor, "batch 3"] | None = None,
    expected_depth: bool = True,
) -> tuple[Float[Tensor, "batch 3 height width"], Float[Tensor, "batch height width"] | None, Float[Tensor, "batch 3 height width"] | None,
           Float[Tensor, "batch height width"] | None, Float[Tensor, "batch height width"] | None, Float[Tensor, "batch 3 height width"] | None]:

    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    # Make sure everything is in a range where numerical issues don't appear.
    if scale_invariant:
        scale = 1 / near
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
        gaussian_scales = gaussian_scales * scale[:, None, None]
        # gaussian_covariances = gaussian_covariances * (scale[:, None, None, None] ** 2)
        gaussian_means = gaussian_means * scale[:, None, None]
        near = near * scale
        far = far * scale

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    b, _, _ = extrinsics.shape
    h, w = image_shape

    fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    all_images = []
    all_radii = []
    all_rend_alphas = []
    all_rend_normals = []
    all_rend_dists = []
    all_surf_depths = []
    all_surf_normals = []
    
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x[i].item(),
            tanfovy=tan_fov_y[i].item(),
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            # projmatrix_raw=projection_matrix[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        # used for covariance
        # row, col = torch.triu_indices(3, 3)

        # image, radii, depth, opacity, n_touched = rasterizer(
        image, radii, allmap = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[i, :, 0, :],
            opacities=gaussian_opacities[i, ..., None],
            scales=gaussian_scales[i],
            rotations=gaussian_rotations[i],
            # precomputed 3d covariance passed as None
            cov3D_precomp=None
            # cov3D_precomp=gaussian_covariances[i, :, row, col],
            # theta=cam_rot_delta[i] if cam_rot_delta is not None else None,
            # rho=cam_trans_delta[i] if cam_trans_delta is not None else None,
        )
        all_images.append(image)
                
        # additional regularizations
        render_alpha = allmap[1:2]

        # get normal map
        render_normal = allmap[2:5]
        # render_normal = render_normal  # keep normal in camera frame
        # transform normal from view space to world space
        render_normal = (render_normal.permute(1,2,0) @ (view_matrix[i][:3,:3].T)).permute(2,0,1)
        render_normal = F.normalize(render_normal, dim=0)
        
        # get median depth map
        render_depth_median = allmap[5:6]
        render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

        # Depth rendering mode switch:
        # - expected depth (default): normalize by alpha
        # - accumulated depth: use raw accumulated depth from rasterizer
        render_depth = allmap[0:1]
        if expected_depth:
            render_depth = (render_depth / render_alpha)
        render_depth = torch.nan_to_num(render_depth, 0, 0)
        
        # get depth distortion map
        render_dist = allmap[6:7]

        # pseudo surface attributes
        # surf depth is either median or expected by setting depth_ratio to 1 or 0
        surf_depth = render_depth * (1 - depth_ratio) + depth_ratio * render_depth_median
        
        # assume the depth points form the 'surface' and generate pseudo surface normal for regularizations.
        surf_normal = depth_to_normal(view_matrix[i], full_projection[i], w, h, surf_depth)
        # surf_normal = depth_to_normal(torch.eye(4, 4, dtype=view_matrix[i].dtype, device=view_matrix[i].device), projection_matrix[i], w, h, surf_depth)
        surf_normal = surf_normal.permute(2,0,1)
        # multiply with accum_alpha since render_normal is unnormalized.
        surf_normal = surf_normal * (render_alpha).detach()
        surf_normal = F.normalize(surf_normal, dim=0)
        
        # print(f'{render_alpha.shape=}')
        # inspect_depth_tensor(render_alpha)
        
        # print(f'{render_depth_median.shape=}')
        # inspect_depth_tensor(render_depth_median)

        # print(f'{render_depth.shape=}')
        # inspect_depth_tensor(render_depth)

        # print(f'{surf_normal.shape=}')
        # inspect_depth_tensor(torch.sqrt(torch.sum(surf_normal ** 2, dim=0)))

        # print(f'{render_normal.shape=}')
        # inspect_depth_tensor(torch.sqrt(torch.sum(render_normal ** 2, dim=0)))

        # import matplotlib.pyplot as plt
        # from ...visualization.normal import vis_normal
        # from ...misc.utils import vis_depth_map
        # surf_normal_img = vis_normal(surf_normal.permute(1, 2, 0).unsqueeze(0)).squeeze(0).detach().cpu().numpy()
        # render_normal_img = vis_normal(render_normal.permute(1, 2, 0).unsqueeze(0)).squeeze(0).detach().cpu().numpy()
        # surf_depth_img = vis_depth_map(surf_depth).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        # plt.imsave(f"results/surf_normal_img.png", surf_normal_img)
        # plt.imsave(f"results/render_normal_img.png", render_normal_img)
        # plt.imsave(f"results/surf_depth_img.png", surf_depth_img)
        
        # from ...geometry.surface_normal import surface_normal_from_depth
        # foc_x = intrinsics[0, 0, 0] * w
        # foc_y = intrinsics[0, 1, 1] * h
        # normal_pts = surface_normal_from_depth(surf_depth.unsqueeze(0), focal_x=foc_x[None], focal_y=foc_y[None], valid_mask=(surf_depth.unsqueeze(0) > 0))
        # normal_depth_vis = vis_normal(normal_pts.squeeze(0).permute(1, 2, 0).unsqueeze(0))[0].detach().cpu().numpy()
        # plt.imsave(f"surface_normal_from_surf_depths.png", normal_depth_vis)
        
        # # check normal direction: if ray dir and normal angle is smaller than 90, reverse normal
        # means3d = depths_to_points(view_matrix[i], full_projection[i], w, h, surf_depth)
        # cam_center = extrinsics[i, :3, 3].reshape(1, 3)
        # ray_dir = (means3d - cam_center).reshape(h, w, 3).permute(2, 0, 1)

        # normal_dir_not_correct = ((ray_dir * render_normal).sum(axis=0) > 0).unsqueeze(0).expand_as(render_normal)
        # render_normal[normal_dir_not_correct] = -render_normal[normal_dir_not_correct]

        # normal_dir_not_correct = ((ray_dir * surf_normal).sum(axis=0) > 0).unsqueeze(0).expand_as(surf_normal)
        # surf_normal[normal_dir_not_correct] = -surf_normal[normal_dir_not_correct]
        
        # all_radii.append(radii)
        all_rend_alphas.append(render_alpha.squeeze(0))
        all_rend_dists.append(render_dist.squeeze(0))
        all_surf_depths.append(surf_depth.squeeze(0))
        all_rend_normals.append(render_normal)
        all_surf_normals.append(surf_normal)
        
    all_images = torch.stack(all_images)
    all_rend_alphas = torch.stack(all_rend_alphas) if all_rend_alphas else None
    all_rend_normals = torch.stack(all_rend_normals) if all_rend_normals else None
    all_rend_dists = torch.stack(all_rend_dists) if all_rend_dists else None
    all_surf_depths = torch.stack(all_surf_depths) if all_surf_depths else None
    all_surf_normals = torch.stack(all_surf_normals) if all_surf_normals else None

    return all_images, all_rend_alphas, all_rend_normals, all_rend_dists, all_surf_depths, all_surf_normals


def render_cuda_orthographic(
    extrinsics: Float[Tensor, "batch 4 4"],
    width: Float[Tensor, " batch"],
    height: Float[Tensor, " batch"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_scales: Float[Tensor, "batch gaussian 3"],
    gaussian_rotations: Float[Tensor, "batch gaussian 4"],
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"] | None = None,
    fov_degrees: float = 0.1,
    use_sh: bool = True,
    dump: dict | None = None,
) -> Float[Tensor, "batch 3 height width"]:
    b, _, _ = extrinsics.shape
    h, w = image_shape
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    # Create fake "orthographic" projection by moving the camera back and picking a
    # small field of view.
    fov_x = torch.tensor(fov_degrees, device=extrinsics.device).deg2rad()
    tan_fov_x = (0.5 * fov_x).tan()
    distance_to_near = (0.5 * width) / tan_fov_x
    tan_fov_y = 0.5 * height / distance_to_near
    fov_y = (2 * tan_fov_y).atan()
    near = near + distance_to_near
    far = far + distance_to_near
    move_back = torch.eye(4, dtype=torch.float32, device=extrinsics.device)
    move_back[2, 3] = -distance_to_near
    extrinsics = extrinsics @ move_back

    # Escape hatch for visualization/figures.
    if dump is not None:
        dump["extrinsics"] = extrinsics
        dump["fov_x"] = fov_x
        dump["fov_y"] = fov_y
        dump["near"] = near
        dump["far"] = far

    projection_matrix = get_projection_matrix(
        near, far, repeat(fov_x, "-> b", b=b), fov_y
    )
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    all_images = []
    # all_radii = []
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings_3D(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x,
            tanfovy=tan_fov_y,
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            projmatrix_raw=projection_matrix[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
        rasterizer = GaussianRasterizer_3D(settings)

        row, col = torch.triu_indices(3, 3)

        # image, radii, allmap = rasterizer(
        image, radii, depth, opacity, n_touched = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[i, :, 0, :],
            opacities=gaussian_opacities[i, ..., None],
            # scales=gaussian_scales[i],
            # rotations=gaussian_rotations[i],
            # precomputed 3d covariance passed as None
            # cov3D_precomp=None
            cov3D_precomp=gaussian_covariances[i, :, row, col],
            # theta=cam_rot_delta[i] if cam_rot_delta is not None else None,
            # rho=cam_trans_delta[i] if cam_trans_delta is not None else None,
        )
        all_images.append(image)
        # all_radii.append(radii)
    return torch.stack(all_images)



def render_cuda_3d(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    scale_invariant: bool = True,
    use_sh: bool = True,
    cam_rot_delta: Float[Tensor, "batch 3"] | None = None,
    cam_trans_delta: Float[Tensor, "batch 3"] | None = None,
    expected_depth: bool = True,
) -> tuple[Float[Tensor, "batch 3 height width"], Float[Tensor, "batch height width"] | None, Float[Tensor, "batch 3 height width"] | None,
           Float[Tensor, "batch height width"] | None, Float[Tensor, "batch height width"] | None, Float[Tensor, "batch 3 height width"] | None]:
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    # Make sure everything is in a range where numerical issues don't appear.
    if scale_invariant:
        scale = 1 / near
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
        gaussian_covariances = gaussian_covariances * (scale[:, None, None, None] ** 2)
        gaussian_means = gaussian_means * scale[:, None, None]
        near = near * scale
        far = far * scale

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    b, _, _ = extrinsics.shape
    h, w = image_shape

    fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    all_images = []
    all_radii = []
    all_depths = []
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings_3D(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x[i].item(),
            tanfovy=tan_fov_y[i].item(),
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            projmatrix_raw=projection_matrix[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
        rasterizer = GaussianRasterizer_3D(settings)

        row, col = torch.triu_indices(3, 3)

        image, radii, depth, opacity, n_touched = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[i, :, 0, :],
            opacities=gaussian_opacities[i, ..., None],
            cov3D_precomp=gaussian_covariances[i, :, row, col],
            theta=cam_rot_delta[i] if cam_rot_delta is not None else None,
            rho=cam_trans_delta[i] if cam_trans_delta is not None else None,
        )
        all_images.append(image)
        all_radii.append(radii)
        # Depth rendering mode switch:
        # - expected depth (default): normalize by opacity
        # - accumulated depth: use raw accumulated depth from rasterizer
        if expected_depth:
            depth = (depth / opacity)
        depth = torch.nan_to_num(depth, 0, 0)
        all_depths.append(depth.squeeze(0))
    return torch.stack(all_images), None, None, None, torch.stack(all_depths), None


DepthRenderingMode = Literal["depth", "disparity", "relative_disparity", "log"]
