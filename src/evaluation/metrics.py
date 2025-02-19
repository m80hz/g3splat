from functools import cache

import torch
from einops import reduce
from jaxtyping import Float
from lpips import LPIPS
from skimage.metrics import structural_similarity
from torch import Tensor


@torch.no_grad()
def compute_psnr(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    return -10 * mse.log10()


@cache
def get_lpips(device: torch.device) -> LPIPS:
    return LPIPS(net="vgg").to(device)


@torch.no_grad()
def compute_lpips(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    value = get_lpips(predicted.device).forward(ground_truth, predicted, normalize=True)
    return value[:, 0, 0, 0]


@torch.no_grad()
def compute_ssim(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ssim = [
        structural_similarity(
            gt.detach().cpu().numpy(),
            hat.detach().cpu().numpy(),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        for gt, hat in zip(ground_truth, predicted)
    ]
    return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)


def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch = m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).to(m1.device)))
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).to(m1.device)) * -1)

    theta = torch.acos(cos)

    # theta = torch.min(theta, 2*np.pi - theta)

    return theta


def angle_error_mat(R1, R2):
    cos = (torch.trace(torch.mm(R1.T, R2)) - 1) / 2
    cos = torch.clamp(cos, -1.0, 1.0)  # numerical errors can make it out of bounds
    return torch.rad2deg(torch.abs(torch.acos(cos)))


def angle_error_vec(v1, v2):
    n = torch.norm(v1) * torch.norm(v2)
    cos_theta = torch.dot(v1, v2) / n
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # numerical errors can make it out of bounds
    return torch.rad2deg(torch.acos(cos_theta))


def compute_translation_error(t1, t2):
    return torch.norm(t1 - t2)


@torch.no_grad()
def compute_pose_error(pose_gt, pose_pred):
    R_gt = pose_gt[:3, :3]
    t_gt = pose_gt[:3, 3]

    R = pose_pred[:3, :3]
    t = pose_pred[:3, 3]

    error_t = angle_error_vec(t, t_gt)
    error_t = torch.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_t_scale = compute_translation_error(t, t_gt)
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_t_scale, error_R


@torch.no_grad()
def compute_depth_errors(depth_gt, depth_pred, mask):
    """
    Compute depth error metrics per sample using only valid depths.
    
    Args:
        depth_gt (torch.Tensor): Ground truth depth tensor of shape (B, H, W).
        depth_pred (torch.Tensor): Predicted depth tensor of shape (B, H, W).
        mask (torch.Tensor): Mask of shape (B, H, W).
    
    Returns:
        Tuple of tensors (each of shape (B,)):
            abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
        where:
            - abs_rel: Absolute Relative Error.
            - sq_rel: Squared Relative Error.
            - rmse: Root Mean Squared Error.
            - rmse_log: Root Mean Squared Error in log-space.
            - a1, a2, a3: Accuracy metrics: fraction of valid depths where the ratio
              max(depth_gt/depth_pred, depth_pred/depth_gt) is below 1.25, 1.25^2, 1.25^3 respectively.
    """
    B = depth_gt.shape[0]
    
    abs_rel_list = []
    sq_rel_list = []
    rmse_list = []
    rmse_log_list = []
    a1_list = []
    a2_list = []
    a3_list = []
    
    for b in range(B):
        valid = mask[b] 
        # If no valid depth in this sample, record NaN for each metric.
        if valid.sum() == 0:
            device = depth_gt.device
            abs_rel_list.append(torch.tensor(float('nan'), device=device))
            sq_rel_list.append(torch.tensor(float('nan'), device=device))
            rmse_list.append(torch.tensor(float('nan'), device=device))
            rmse_log_list.append(torch.tensor(float('nan'), device=device))
            a1_list.append(torch.tensor(float('nan'), device=device))
            a2_list.append(torch.tensor(float('nan'), device=device))
            a3_list.append(torch.tensor(float('nan'), device=device))
        else:
            gt_sample = depth_gt[b][valid]
            pred_sample = depth_pred[b][valid]

            # Compute the threshold ratio per pixel.
            thresh = torch.max(gt_sample / pred_sample, pred_sample / gt_sample)
            a1 = (thresh < 1.25).float().mean()
            a2 = (thresh < 1.25**2).float().mean()
            a3 = (thresh < 1.25**3).float().mean()

            rmse = torch.sqrt(((gt_sample - pred_sample) ** 2).mean())
            rmse_log = torch.sqrt(((torch.log(gt_sample) - torch.log(pred_sample)) ** 2).mean())
            abs_rel = (torch.abs(gt_sample - pred_sample) / gt_sample).mean()
            sq_rel = (((gt_sample - pred_sample) ** 2) / gt_sample).mean()

            abs_rel_list.append(abs_rel)
            sq_rel_list.append(sq_rel)
            rmse_list.append(rmse)
            rmse_log_list.append(rmse_log)
            a1_list.append(a1)
            a2_list.append(a2)
            a3_list.append(a3)
    
    # Stack lists into tensors of shape (B,)
    abs_rel_tensor = torch.stack(abs_rel_list)
    sq_rel_tensor = torch.stack(sq_rel_list)
    rmse_tensor = torch.stack(rmse_list)
    rmse_log_tensor = torch.stack(rmse_log_list)
    a1_tensor = torch.stack(a1_list)
    a2_tensor = torch.stack(a2_list)
    a3_tensor = torch.stack(a3_list)
    
    return abs_rel_tensor, sq_rel_tensor, rmse_tensor, rmse_log_tensor, a1_tensor, a2_tensor, a3_tensor


@torch.no_grad
def depth_evaluation(
    predicted_depth_original,
    ground_truth_depth_original,
    max_depth=100,
    custom_mask=None,
    post_clip_min=None,
    post_clip_max=None,
    pre_clip_min=None,
    pre_clip_max=None,
    align_with_lstsq=False,
    align_with_lad=False,
    align_with_lad2=False,
    metric_scale=False,
    lr=1e-4,
    max_iters=1000,
    use_gpu=False,
    align_with_scale=False,
    disp_input=False,
):
    """
    Evaluate the depth map using various metrics and return a depth error parity map, 
    with an option for alignment.
    (Flattening is performed if input is 3D.)
    
    Returns:
        results (dict): A dictionary containing error metrics.
        depth_error_parity_map_full (torch.Tensor)
        predict_depth_map_full (torch.Tensor)
        gt_depth_map_full (torch.Tensor)
    """
    import numpy as np
    import torch

    if isinstance(predicted_depth_original, np.ndarray):
        predicted_depth_original = torch.from_numpy(predicted_depth_original)
    if isinstance(ground_truth_depth_original, np.ndarray):
        ground_truth_depth_original = torch.from_numpy(ground_truth_depth_original)
    if custom_mask is not None and isinstance(custom_mask, np.ndarray):
        custom_mask = torch.from_numpy(custom_mask)

    # --- Flatten if input is 3D ---
    if predicted_depth_original.dim() == 3:
        _, h, w = predicted_depth_original.shape
        predicted_depth_original = predicted_depth_original.view(-1, w)
        ground_truth_depth_original = ground_truth_depth_original.view(-1, w)
        if custom_mask is not None:
            custom_mask = custom_mask.view(-1, w)

    if use_gpu:
        predicted_depth_original = predicted_depth_original.cuda()
        ground_truth_depth_original = ground_truth_depth_original.cuda()

    if max_depth is not None:
        mask = (ground_truth_depth_original > 0) & (ground_truth_depth_original < max_depth)
    else:
        mask = ground_truth_depth_original > 0
    
    predicted_depth = predicted_depth_original[mask]
    ground_truth_depth = ground_truth_depth_original[mask]

    if pre_clip_min is not None:
        predicted_depth = torch.clamp(predicted_depth, min=pre_clip_min)
    if pre_clip_max is not None:
        predicted_depth = torch.clamp(predicted_depth, max=pre_clip_max)

    if disp_input:
        real_gt = ground_truth_depth.clone()
        ground_truth_depth = 1 / (ground_truth_depth + 1e-8)

    if metric_scale:
        pass
    elif align_with_lstsq:
        predicted_depth_np = predicted_depth.cpu().numpy().reshape(-1, 1)
        ground_truth_depth_np = ground_truth_depth.cpu().numpy().reshape(-1, 1)
        A = np.hstack([predicted_depth_np, np.ones_like(predicted_depth_np)])
        result = np.linalg.lstsq(A, ground_truth_depth_np, rcond=None)
        s, t = result[0][0], result[0][1]
        s = torch.tensor(s, device=predicted_depth_original.device)
        t = torch.tensor(t, device=predicted_depth_original.device)
        predicted_depth = s * predicted_depth + t
    elif align_with_lad:
        s, t = absolute_value_scaling(
            predicted_depth,
            ground_truth_depth,
            s=torch.median(ground_truth_depth) / torch.median(predicted_depth),
        )
        predicted_depth = s * predicted_depth + t
    elif align_with_lad2:
        s_init = (torch.median(ground_truth_depth) / torch.median(predicted_depth)).item()
        s, t = absolute_value_scaling2(
            predicted_depth,
            ground_truth_depth,
            s_init=s_init,
            lr=lr,
            max_iters=max_iters,
        )
        predicted_depth = s * predicted_depth + t
    elif align_with_scale:
        dot_pred_gt = torch.nanmean(ground_truth_depth)
        dot_pred_pred = torch.nanmean(predicted_depth)
        s = dot_pred_gt / dot_pred_pred
        for _ in range(10):
            residuals = s * predicted_depth - ground_truth_depth
            abs_residuals = residuals.abs() + 1e-8
            weights = 1.0 / abs_residuals
            weighted_dot_pred_gt = torch.sum(weights * predicted_depth * ground_truth_depth)
            weighted_dot_pred_pred = torch.sum(weights * predicted_depth**2)
            s = weighted_dot_pred_gt / weighted_dot_pred_pred
        s = s.clamp(min=1e-3).detach()
        predicted_depth = s * predicted_depth
    else:
        scale_factor = torch.median(ground_truth_depth) / torch.median(predicted_depth)
        predicted_depth *= scale_factor

    if disp_input:
        ground_truth_depth = real_gt
        predicted_depth = depth2disparity(predicted_depth)

    if post_clip_min is not None:
        predicted_depth = torch.clamp(predicted_depth, min=post_clip_min)
    if post_clip_max is not None:
        predicted_depth = torch.clamp(predicted_depth, max=post_clip_max)

    if custom_mask is not None:
        assert custom_mask.shape == ground_truth_depth_original.shape
        mask_within_mask = custom_mask[mask]
        predicted_depth = predicted_depth[mask_within_mask]
        ground_truth_depth = ground_truth_depth[mask_within_mask]

    abs_rel = torch.mean(torch.abs(predicted_depth - ground_truth_depth) / ground_truth_depth).item()
    sq_rel = torch.mean(((predicted_depth - ground_truth_depth) ** 2) / ground_truth_depth).item()
    rmse = torch.sqrt(torch.mean((predicted_depth - ground_truth_depth) ** 2)).item()
    predicted_depth = torch.clamp(predicted_depth, min=1e-5)
    log_rmse = torch.sqrt(torch.mean((torch.log(predicted_depth) - torch.log(ground_truth_depth)) ** 2)).item()
    max_ratio = torch.maximum(predicted_depth / ground_truth_depth, ground_truth_depth / predicted_depth)
    threshold_0 = torch.mean((max_ratio < 1.0).float()).item()
    threshold_1 = torch.mean((max_ratio < 1.25).float()).item()
    threshold_2 = torch.mean((max_ratio < 1.25**2).float()).item()
    threshold_3 = torch.mean((max_ratio < 1.25**3).float()).item()

    if metric_scale:
        predicted_depth_original_final = predicted_depth_original
        if disp_input:
            predicted_depth_original_final = depth2disparity(predicted_depth_original_final)
        depth_error_parity_map = torch.abs(predicted_depth_original_final - ground_truth_depth_original) / ground_truth_depth_original
    elif align_with_lstsq or align_with_lad or align_with_lad2:
        predicted_depth_original_final = predicted_depth_original * s + t
        if disp_input:
            predicted_depth_original_final = depth2disparity(predicted_depth_original_final)
        depth_error_parity_map = torch.abs(predicted_depth_original_final - ground_truth_depth_original) / ground_truth_depth_original
    elif align_with_scale:
        predicted_depth_original_final = predicted_depth_original * s
        if disp_input:
            predicted_depth_original_final = depth2disparity(predicted_depth_original_final)
        depth_error_parity_map = torch.abs(predicted_depth_original_final - ground_truth_depth_original) / ground_truth_depth_original
    else:
        predicted_depth_original_final = predicted_depth_original * scale_factor
        if disp_input:
            predicted_depth_original_final = depth2disparity(predicted_depth_original_final)
        depth_error_parity_map = torch.abs(predicted_depth_original_final - ground_truth_depth_original) / ground_truth_depth_original

    depth_error_parity_map_full = torch.zeros_like(ground_truth_depth_original)
    depth_error_parity_map_full = torch.where(mask, depth_error_parity_map, depth_error_parity_map_full)
    predict_depth_map_full = predicted_depth_original_final
    gt_depth_map_full = torch.zeros_like(ground_truth_depth_original)
    gt_depth_map_full = torch.where(mask, ground_truth_depth_original, gt_depth_map_full)

    num_valid_pixels = torch.sum(mask).item() if custom_mask is None else torch.sum(mask_within_mask).item()
    if num_valid_pixels == 0:
        abs_rel = sq_rel = rmse = log_rmse = threshold_0 = threshold_1 = threshold_2 = threshold_3 = 0

    results = {
        "Abs Rel": abs_rel,
        "Sq Rel": sq_rel,
        "RMSE": rmse,
        "Log RMSE": log_rmse,
        "δ < 1.": threshold_0,
        "δ < 1.25": threshold_1,
        "δ < 1.25^2": threshold_2,
        "δ < 1.25^3": threshold_3,
        "valid_pixels": num_valid_pixels,
    }

    return results, depth_error_parity_map_full, predict_depth_map_full, gt_depth_map_full
