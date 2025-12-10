import cv2
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor


def decompose_extrinsic_RT(E: torch.Tensor):
    """
    Decompose the standard extrinsic matrix into RT.
    Batched I/O.
    """
    return E[:, :3, :]


def compose_extrinsic_RT(RT: torch.Tensor):
    """
    Compose the standard form extrinsic matrix from RT.
    Batched I/O.
    """
    return torch.cat([
        RT,
        torch.tensor([[[0, 0, 0, 1]]], dtype=RT.dtype, device=RT.device).repeat(RT.shape[0], 1, 1)
        ], dim=1)


def camera_normalization(pivotal_pose: torch.Tensor, poses: torch.Tensor):
    # [1, 4, 4], [N, 4, 4]

    canonical_camera_extrinsics = torch.tensor([[
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]], dtype=torch.float32, device=pivotal_pose.device)
    pivotal_pose_inv = torch.inverse(pivotal_pose)
    camera_norm_matrix = torch.bmm(canonical_camera_extrinsics, pivotal_pose_inv)

    # normalize all views
    poses = torch.bmm(camera_norm_matrix.repeat(poses.shape[0], 1, 1), poses)

    return poses


####### Pose update from delta

def rt2mat(R, T):
    mat = np.eye(4)
    mat[0:3, 0:3] = R
    mat[0:3, 3] = T
    return mat


def skew_sym_mat(x):
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm


def SO3_exp(theta):
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    else:
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )


def V(theta):
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V


def SE3_exp(tau):
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def update_pose(cam_trans_delta: Float[Tensor, "batch 3"],
                cam_rot_delta: Float[Tensor, "batch 3"],
                extrinsics: Float[Tensor, "batch 4 4"],
                # original_rot: Float[Tensor, "batch 3 3"],
                # original_trans: Float[Tensor, "batch 3"],
                # converged_threshold: float = 1e-4
                ):
    # extrinsics is c2w, here we need w2c as input, so we need to invert it
    bs = cam_trans_delta.shape[0]

    tau = torch.cat([cam_trans_delta, cam_rot_delta], dim=-1)
    T_w2c = extrinsics.inverse()

    new_w2c_list = []
    for i in range(bs):
        new_w2c = SE3_exp(tau[i]) @ T_w2c[i]
        new_w2c_list.append(new_w2c)

    new_w2c = torch.stack(new_w2c_list, dim=0)
    return new_w2c.inverse()

    # converged = tau.norm() < converged_threshold
    # camera.update_RT(new_R, new_T)
    #
    # camera.cam_rot_delta.data.fill_(0)
    # camera.cam_trans_delta.data.fill_(0)
    # return converged


#######  Pose estimation
def inv(mat):
    """ Invert a torch or numpy matrix
    """
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f'bad matrix type = {type(mat)}')


def get_pnp_pose(pts3d, opacity, K, H, W, opacity_threshold=0.3, return_inliers: bool = False, use_ransac: bool = True):
    """Estimate pose with PnP (RANSAC or plain iterative).

    Parameters
    ----------
    pts3d : torch.Tensor
        3D points (N,3) in world coordinates.
    opacity : torch.Tensor
        Opacity/confidence per point (N,).
    K : torch.Tensor
        Normalized intrinsics (fx,fy,cx,cy) scaled to [0,1] domain; will be scaled by image size.
    H, W : int
        Image height / width.
    opacity_threshold : float, default 0.3
        Minimum opacity to keep a 3D point for PnP.
    return_inliers : bool, default False
        If True, also return (inlier_count, inlier_ratio) from RANSAC result. For non-RANSAC mode these
        will be (num_points, 1.0) if success else (0,0.0).
    use_ransac : bool, default True
        If True use cv2.solvePnPRansac, else use cv2.solvePnP with SOLVEPNP_ITERATIVE on all surviving points.
    """
    pixels = np.mgrid[:W, :H].T.astype(np.float32)  # (H, W, 2)  [...,0]=x, [...,1]=y
    pts3d_np = pts3d.cpu().numpy()
    opacity_np = opacity.cpu().numpy()
    K_np = K.cpu().numpy()

    K_np[0, :] = K_np[0, :] * W
    K_np[1, :] = K_np[1, :] * H

    mask = opacity_np > opacity_threshold
    # if mask.sum() < 6:
    #     mask = opacity_np > (0.5 * opacity_threshold)
    if mask.sum() < 6:
        mask = np.ones_like(opacity_np, dtype=bool)

    if use_ransac:
        res = cv2.solvePnPRansac(pts3d_np[mask], pixels[mask], K_np, None,
                                 iterationsCount=100, reprojectionError=5, flags=cv2.SOLVEPNP_SQPNP)
        success, Rvec, T, inliers = res
        assert success, "cv2.solvePnPRansac failed to find a pose"
        R = cv2.Rodrigues(Rvec)[0]
        pose = inv(np.r_[np.c_[R, T], [(0, 0, 0, 1)]])
        pose_torch = torch.from_numpy(pose.astype(np.float32))
        if not return_inliers:
            return pose_torch
        inlier_count = int(0 if inliers is None else len(inliers))
        denom = int(mask.sum()) if mask is not None else 0
        inlier_ratio = float(inlier_count / denom) if denom > 0 else 0.0
        return pose_torch, inlier_count, inlier_ratio
    else:
        # Plain least-squares Gauss-Newton over all correspondences (no RANSAC)
        success, Rvec, T = cv2.solvePnP(pts3d_np[mask], pixels[mask], K_np, None, flags=cv2.SOLVEPNP_ITERATIVE)
        assert success, "cv2.solvePnP failed to find a pose"
        R = cv2.Rodrigues(Rvec)[0]
        pose = inv(np.r_[np.c_[R, T], [(0, 0, 0, 1)]])
        pose_torch = torch.from_numpy(pose.astype(np.float32))
        if not return_inliers:
            return pose_torch
        # All points considered inliers except those filtered by opacity threshold
        inlier_count = int(mask.sum())
        inlier_ratio = float(inlier_count / mask.size) if mask.size > 0 else 0.0
        return pose_torch, inlier_count, inlier_ratio


def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e) / t)
    return aucs
