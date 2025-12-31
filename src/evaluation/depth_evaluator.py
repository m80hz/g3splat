from pytorch_lightning.utilities.types import STEP_OUTPUT

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from lightning import LightningModule

# from ..misc.utils import get_overlap_tag
from .evaluation_cfg import EvaluationCfg
from ..misc.cam_utils import update_pose, get_pnp_pose
from ..loss.loss_ssim import ssim

import matplotlib.pyplot as plt

class DepthEvaluator(LightningModule):
    cfg: EvaluationCfg

    def __init__(self, cfg: EvaluationCfg, encoder, decoder, losses) -> None:
        super().__init__()
        self.cfg = cfg

        # our model
        self.encoder = encoder.to(self.device)
        self.decoder = decoder
        self.losses = nn.ModuleList(losses)

        self.data_shim = get_data_shim(self.encoder)
        
        
    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        # set to eval
        self.encoder.eval()
        # freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        b, _, _, h, w = batch["context"]["image"].shape
        assert b == 1
        if batch_idx % 100 == 0:
            print(f"Test step {batch_idx:0>6}.")

        # # get overlap.
        # overlap = batch["context"]["overlap"][0, 0]
        # overlap_tag = get_overlap_tag(overlap)
        # if overlap_tag == "ignore":
        #     return

        visualization_dump = {}

        # running encoder to obtain the 3DGS
        gaussians = self.encoder(
            batch["context"],
            self.global_step,
            visualization_dump=visualization_dump,
        )
        
        # pose refinement for the context and target views
        if self.cfg.use_pose_refinement:
            context_view_2_input_image = batch["context"]["image"][:, 1:2].clone()
            target_views_input_image = batch["target"]["image"][:, :].clone()
            
            ######## Context View 2 Pose Refinement ######################
            pose_opt = get_pnp_pose(visualization_dump['means'][0, 1].squeeze(),
                                    visualization_dump['opacities'][0, 1].squeeze(),
                                    batch["context"]["intrinsics"][0, 1], h, w)
            pose_opt = pose_opt.to(self.device)
            with torch.set_grad_enabled(True):
                cam_rot_delta = nn.Parameter(torch.zeros([b, 1, 3], requires_grad=True, device=self.device))
                cam_trans_delta = nn.Parameter(torch.zeros([b, 1, 3], requires_grad=True, device=self.device))

                opt_params = []
                opt_params.append(
                    {
                        "params": [cam_rot_delta],
                        "lr": 0.005,
                    }
                )
                opt_params.append(
                    {
                        "params": [cam_trans_delta],
                        "lr": 0.005,
                    }
                )

                pose_optimizer = torch.optim.Adam(opt_params)

                number_steps = 200
                extrinsics = pose_opt.unsqueeze(0).unsqueeze(0)  # initial pose use pose_opt
                for i in range(number_steps):
                    pose_optimizer.zero_grad()

                    output = self.decoder.forward(
                        gaussians,
                        extrinsics,
                        batch["context"]["intrinsics"][:, 1:2],
                        batch["context"]["near"][:, 1:2],
                        batch["context"]["far"][:, 1:2],
                        (h, w),
                        cam_rot_delta=cam_rot_delta,
                        cam_trans_delta=cam_trans_delta,
                        decoder_type="3D"  # Always use 3D for pose refinement (only 3D renderer returns camera pose gradients)
                    )

                    # Compute and log loss.
                    batch["target"]["image"] = context_view_2_input_image
                    total_loss = 0
                    for loss_fn in self.losses:
                        loss = loss_fn.forward(output, batch, gaussians, self.global_step)
                        total_loss = total_loss + loss

                    # add ssim structure loss
                    ssim_, _, _, structure = ssim(rearrange(batch["target"]["image"], "b v c h w -> (b v) c h w"),
                                        rearrange(output.color, "b v c h w -> (b v) c h w"),
                                        size_average=True, data_range=1.0, retrun_seprate=True, win_size=11)
                    ssim_loss = (1 - structure) * 1.0
                    total_loss = total_loss + ssim_loss

                    # back-propagate
                    # print(f"Step {i} - Loss: {total_loss.item()}")
                    total_loss.backward()
                    with torch.no_grad():
                        pose_optimizer.step()
                        new_extrinsic = update_pose(cam_rot_delta=rearrange(cam_rot_delta, "b v i -> (b v) i"),
                                                    cam_trans_delta=rearrange(cam_trans_delta, "b v i -> (b v) i"),
                                                    extrinsics=rearrange(extrinsics, "b v i j -> (b v) i j")
                                                    )
                        cam_rot_delta.data.fill_(0)
                        cam_trans_delta.data.fill_(0)

                        extrinsics = rearrange(new_extrinsic, "(b v) i j -> b v i j", b=b, v=1)
                        batch["context"]["extrinsics"][0, 1] = extrinsics[0, 0].clone()

            ######## Target Views Pose Refinement ######################
            batch["target"]["image"] = target_views_input_image            
            b, v, _, _, _ = batch["target"]["image"].shape
            
            with torch.set_grad_enabled(True):
                cam_rot_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=self.device))
                cam_trans_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=self.device))

                opt_params = []
                opt_params.append(
                    {
                        "params": [cam_rot_delta],
                        "lr": 0.005,
                    }
                )
                opt_params.append(
                    {
                        "params": [cam_trans_delta],
                        "lr": 0.005,
                    }
                )
                pose_optimizer = torch.optim.Adam(opt_params)

                number_steps = 200
                extrinsics = batch["target"]["extrinsics"].clone()
                for i in range(number_steps):
                    pose_optimizer.zero_grad()

                    output = self.decoder.forward(
                        gaussians,
                        extrinsics,
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h, w),
                        cam_rot_delta=cam_rot_delta,
                        cam_trans_delta=cam_trans_delta,
                        decoder_type="3D"  # Always use 3D for pose refinement (only 3D renderer returns camera pose gradients)
                    )

                    # Compute and log loss.
                    total_loss = 0
                    for loss_fn in self.losses:
                        loss = loss_fn.forward(output, batch, gaussians, self.global_step)
                        total_loss = total_loss + loss

                    # add ssim structure loss
                    ssim_, _, _, structure = ssim(rearrange(batch["target"]["image"], "b v c h w -> (b v) c h w"),
                                        rearrange(output.color, "b v c h w -> (b v) c h w"),
                                        size_average=True, data_range=1.0, retrun_seprate=True, win_size=11)
                    ssim_loss = (1 - structure) * 1.0
                    total_loss = total_loss + ssim_loss

                    # back-propagate
                    total_loss.backward()
                    with torch.no_grad():
                        pose_optimizer.step()
                        new_extrinsic = update_pose(cam_rot_delta=rearrange(cam_rot_delta, "b v i -> (b v) i"),
                                                    cam_trans_delta=rearrange(cam_trans_delta, "b v i -> (b v) i"),
                                                    extrinsics=rearrange(extrinsics, "b v i j -> (b v) i j")
                                                    )
                        cam_rot_delta.data.fill_(0)
                        cam_trans_delta.data.fill_(0)

                        extrinsics = rearrange(new_extrinsic, "(b v) i j -> b v i j", b=b, v=v)
                        batch["target"]["extrinsics"] = extrinsics.clone()


        # render context views
        output_context = self.decoder.forward(
            gaussians,
            batch["context"]["extrinsics"],
            batch["context"]["intrinsics"],
            batch["context"]["near"],
            batch["context"]["far"],
            (h, w),
        )

        # render target views
        output_target = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
        )
                
        context_img_rendered = output_context.color[0]
        context_depth_rendered = output_context.depth[0]
        
        target_img_rendered = output_target.color[0]
        target_depth_rendered = output_target.depth[0]

        # direct depth from gaussian means for context view 1
        gaussian_depths = visualization_dump["depth"][0].squeeze()   # (V, H, W, 1)
        if gaussian_depths.shape[-1] == 3:
            gaussian_depths = gaussian_depths.mean(dim=-1)

        context_depth_pointcloud = gaussian_depths
        #### Warp context view 2 pointcloud back -- from view 1 to 2
        # Helper: convert (B, H, W, 3) to homogeneous coordinates (B, 4, H, W)
        def pts3d_to_hom(pts: Tensor) -> Tensor:
            B, H, W, _ = pts.shape
            ones = torch.ones(B, H, W, 1, device=pts.device, dtype=pts.dtype)
            pts_h = torch.cat([pts, ones], dim=-1)  # (B, H, W, 4)
            return pts_h.permute(0, 3, 1, 2)         # (B, 4, H, W)
        
        all_pts3d = rearrange(gaussians.means, "b (v h w) d -> b v h w d", h=h, w=w)
        pts3d2 = all_pts3d[:, 1, ...]  # (b, h, w, 3)
        T1 = batch["context"]["extrinsics"][:, 0, :, :]     # view1 extrinsics (camera-to-world)
        T2 = batch["context"]["extrinsics"][:, 1, :, :]     # view2 extrinsics (camera-to-world)
        # Compute relative transformation: T_rel = T2^{-1} * T1 maps points from view1's frame to view2's camera coordinates.
        T2_inv = torch.inverse(T2)
        T_rel = torch.bmm(T2_inv, T1)  # (b=1, 4, 4)
        pts3d2_h = pts3d_to_hom(pts3d2)  # (b=1, 4, h, w)
        pts2_cam = torch.bmm(T_rel, pts3d2_h.view(1, 4, -1)).view(1, 4, h, w)
        depth2 = pts2_cam[:, 2, :, :].view(1, h, w)
        context_depth_pointcloud[1:2] = depth2
        
        context_depth_gt = batch["context"]["depth"][0].squeeze(1)
        context_valid_depth_gt = batch["context"]["valid_depth"][0].squeeze(1)
        
        target_depth_gt = batch["target"]["depth"][0].squeeze(1)
        target_valid_depth_gt = batch["target"]["valid_depth"][0].squeeze(1)
        
        # only evaluate on valid depth
        near_value_ctx = batch["context"]["near"].view(-1, 1, 1)  # shape (B, 1, 1)
        far_value_ctx  = batch["context"]["far"].view(-1, 1, 1)   # shape (B, 1, 1)
        near_value_tg = batch["target"]["near"].view(-1, 1, 1)  # shape (B, 1, 1)
        far_value_tg  = batch["target"]["far"].view(-1, 1, 1)   # shape (B, 1, 1)
        # depth_mask = None
        context_depth_mask = (context_depth_gt > near_value_ctx) & (context_depth_gt < far_value_ctx) & context_valid_depth_gt.bool()
        target_depth_mask = (target_depth_gt > near_value_tg) & (target_depth_gt < far_value_tg) & target_valid_depth_gt.bool()
        
        # --- Context 1 --- The one with identity pose
        results_ctx1_rendered, parity_map_ctx1_rendered, pred_full_ctx1_rendered, gt_full_ctx1_rendered = self.depth_evaluation(
            predicted_depth_original=context_depth_rendered[0:1],
            ground_truth_depth_original=context_depth_gt[0:1],
            max_depth=far_value_ctx[0:1].item(),
            custom_mask=context_depth_mask[0:1] if context_depth_mask is not None else None,
            # pre_clip_min=near_value_ctx[0:1].item(),
            use_gpu=True,
            align_with_lstsq=False,
            align_with_lad=False,
            align_with_lad2=False,
            metric_scale=False,
            align_with_scale=False,
            disp_input=False
        )

        results_ctx1_pointcloud, parity_map_ctx1_pointcloud, pred_full_ctx1_pointcloud, gt_full_ctx1_pointcloud = self.depth_evaluation(
            predicted_depth_original=context_depth_pointcloud[0:1],
            ground_truth_depth_original=context_depth_gt[0:1],
            max_depth=far_value_ctx[0:1].item(),
            custom_mask=context_depth_mask[0:1] if context_depth_mask is not None else None,
            # pre_clip_min=near_value_ctx[0:1].item(),
            use_gpu=True,
            align_with_lstsq=False,
            align_with_lad=False,
            align_with_lad2=False,
            metric_scale=False,
            align_with_scale=False,
            disp_input=False
        )

        # --- Context 2 ---
        results_ctx2_rendered, parity_map_ctx2_rendered, pred_full_ctx2_rendered, gt_full_ctx2_rendered = self.depth_evaluation(
            predicted_depth_original=context_depth_rendered[1:2],
            ground_truth_depth_original=context_depth_gt[1:2],
            max_depth=far_value_ctx[1:2].item(),
            custom_mask=context_depth_mask[1:2] if context_depth_mask is not None else None,
            # pre_clip_min=near_value_ctx[1:2].item(),
            use_gpu=True,
            align_with_lstsq=False,
            align_with_lad=False,
            align_with_lad2=False,
            metric_scale=False,
            align_with_scale=False,
            disp_input=False
        )

        results_ctx2_pointcloud, parity_map_ctx2_pointcloud, pred_full_ctx2_pointcloud, gt_full_ctx2_pointcloud = self.depth_evaluation(
            predicted_depth_original=context_depth_pointcloud[1:2],
            ground_truth_depth_original=context_depth_gt[1:2],
            max_depth=far_value_ctx[1:2].item(),
            custom_mask=context_depth_mask[1:2] if context_depth_mask is not None else None,
            # pre_clip_min=near_value_ctx[1:2].item(),
            use_gpu=True,
            align_with_lstsq=False,
            align_with_lad=False,
            align_with_lad2=False,
            metric_scale=False,
            align_with_scale=False,
            disp_input=False
        )
        
        
        # --- Target Views --- 
        results_targets_rendered, parity_map_targets_rendered, pred_full_targets_rendered, gt_full_targets_rendered = self.depth_evaluation(
            predicted_depth_original=target_depth_rendered,
            ground_truth_depth_original=target_depth_gt,
            max_depth=far_value_tg[0:1].item(),
            custom_mask=target_depth_mask if target_depth_mask is not None else None,
            # pre_clip_min=near_value_tg[0:1].item(),
            use_gpu=True,
            align_with_lstsq=False,
            align_with_lad=False,
            align_with_lad2=False,
            metric_scale=False,
            align_with_scale=False,
            disp_input=False
        )

        
        # Each call returns a "results" dictionary (with keys such as "Abs Rel", "Sq Rel", etc.).
        # For convenience, define error names in the order used by depth_evaluation:
        # error_names = ['Abs Rel', 'Sq Rel', 'RMSE', 'Log RMSE', 'δ < 1.10', 'δ < 1.25', 'δ < 1.25^2', 'δ < 1.25^3']
        error_names = ['Abs Rel', 'δ < 1.10', 'δ < 1.25']

        # Convert each results dictionary to a dictionary of mean values (they're global, since the function flattens the batch)
        metrics_ctx1_rendered = { f"context1_rendered_{name.replace(' ', '_')}": results_ctx1_rendered[name] for name in error_names }
        metrics_ctx1_pointcloud = { f"context1_pointcloud_{name.replace(' ', '_')}": results_ctx1_pointcloud[name] for name in error_names }
        metrics_ctx2_rendered = { f"context2_rendered_{name.replace(' ', '_')}": results_ctx2_rendered[name] for name in error_names }
        metrics_ctx2_pointcloud = { f"context2_pointcloud_{name.replace(' ', '_')}": results_ctx2_pointcloud[name] for name in error_names }

        metrics_targets_rendered = { f"targets_rendered_{name.replace(' ', '_')}": results_targets_rendered[name] for name in error_names }

        # Now update your running metrics
        self.print_preview_depth_metrics(metrics_ctx1_rendered, sub_tag="context1_rendered")
        self.print_preview_depth_metrics(metrics_ctx1_pointcloud, sub_tag="context1_pointcloud")
        self.print_preview_depth_metrics(metrics_ctx2_rendered, sub_tag="context2_rendered")
        self.print_preview_depth_metrics(metrics_ctx2_pointcloud, sub_tag="context2_pointcloud")
        self.print_preview_depth_metrics(metrics_targets_rendered, sub_tag="targets")

        return 0


    def print_preview_depth_metrics(self, metrics: dict[str, float], sub_tag: str | None = None) -> None:
        """
        Update and print running depth metrics for the given subgroup.

        Args:
            metrics (dict[str, float]): Dictionary mapping metric names (e.g. "context1_rendered_Abs_Rel")
                to scalar values (the mean over the current batch).
            sub_tag (str or None): Tag for this subgroup (e.g. "context1_rendered").
        """
        # We update only the subgroup metrics. Each sub_tag gets its own running metrics.
        if sub_tag is None:
            # If no sub_tag is provided, we can use a default key.
            sub_tag = "default"

        if not hasattr(self, "running_depth_metrics_sub"):
            self.running_depth_metrics_sub = {}
            self.running_depth_metric_steps_sub = {}
            self.all_depth_metrics_sub = {}
        
        if sub_tag not in self.running_depth_metrics_sub:
            self.running_depth_metrics_sub[sub_tag] = metrics.copy()
            self.running_depth_metric_steps_sub[sub_tag] = 1
            self.all_depth_metrics_sub[sub_tag] = {k: [metrics[k]] for k in metrics}
        else:
            s_sub = self.running_depth_metric_steps_sub[sub_tag]
            running_sub = self.running_depth_metrics_sub[sub_tag]
            updated_sub = {k: ((s_sub * running_sub.get(k, 0)) + metrics[k]) / (s_sub + 1)
                        for k in metrics}
            self.running_depth_metrics_sub[sub_tag] = updated_sub
            self.running_depth_metric_steps_sub[sub_tag] = s_sub + 1
            for k, v in metrics.items():
                self.all_depth_metrics_sub[sub_tag].setdefault(k, []).append(v)
        
        # Print the current subgroup metrics nicely.
        from tabulate import tabulate
        def print_table(running_metric: dict[str, float]):
            table = [[k, f"{v:.3f}"] for k, v in running_metric.items()]
            print(tabulate(table, headers=["Metric", "Value"]))
        
        print("\n" + "="*40)
        print(f"Current Depth Metrics (Sub Tag: {sub_tag}):")
        print_table(self.running_depth_metrics_sub[sub_tag])
        print("="*40 + "\n")


    def on_test_end(self) -> None:
        """
        Called at the end of testing to summarize depth metrics over all test samples.
        """
        import numpy as np
        from tabulate import tabulate

        print("====== Test End: Depth Evaluation ======\n")
        
        # Print and save each subgroup's metrics.
        if hasattr(self, "all_depth_metrics_sub"):
            for tag, metrics_dict in self.all_depth_metrics_sub.items():
                print(f"Subgroup Depth Metrics for Sub Tag: {tag}")
                print("-"*40)
                sub_avg = {k: np.mean(v) for k, v in metrics_dict.items()}
                print(tabulate([[k, f"{v:.3f}"] for k, v in sub_avg.items()], headers=["Metric", "Value"]))
                np.save(f"all_depth_metrics_sub_{tag}.npy", metrics_dict)
                print("\n" + "="*40 + "\n")
        else:
            print("No subgroup depth metrics recorded.")
        
        # Compute overall averages for rendered and pointcloud categories separately.
        # error_names = ['Abs_Rel', 'Sq_Rel', 'RMSE', 'Log_RMSE', 'δ_<_1.10', 'δ_<_1.25', 'δ_<_1.25^2', 'δ_<_1.25^3']
        error_names = ['Abs_Rel', 'δ_<_1.10', 'δ_<_1.25']
        
        # Gather subgroup keys for each category.
        rendered_keys = [tag for tag in self.all_depth_metrics_sub.keys() if "rendered" in tag.lower()]
        pointcloud_keys = [tag for tag in self.all_depth_metrics_sub.keys() if "pointcloud" in tag.lower()]
        targets_keys = [tag for tag in self.all_depth_metrics_sub.keys() if "targets" in tag.lower()]
        
        overall_rendered = {}
        overall_pointcloud = {}
        overall_targets = {}
        
        # For each error metric, combine values from all rendered subgroups.
        for err in error_names:
            rendered_vals = []
            for tag in rendered_keys:
                for key, vals in self.all_depth_metrics_sub[tag].items():
                    # We assume keys end with the error name (case-insensitive)
                    if key.lower().endswith(err.lower()):
                        rendered_vals.extend(vals)
            if rendered_vals:
                overall_rendered[err] = np.mean(rendered_vals)
        
        # For pointcloud subgroups.
        for err in error_names:
            pointcloud_vals = []
            for tag in pointcloud_keys:
                for key, vals in self.all_depth_metrics_sub[tag].items():
                    if key.lower().endswith(err.lower()):
                        pointcloud_vals.extend(vals)
            if pointcloud_vals:
                overall_pointcloud[err] = np.mean(pointcloud_vals)
        
        # For targets subgroups.
        for err in error_names:
            targets_vals = []
            for tag in targets_keys:
                for key, vals in self.all_depth_metrics_sub[tag].items():
                    if key.lower().endswith(err.lower()):
                        targets_vals.extend(vals)
            if targets_vals:
                overall_targets[err] = np.mean(targets_vals)
        
        # Print overall rendered metrics.
        if overall_rendered:
            print("Overall Rendered Depth Metrics (averaged over all context subgroups):")
            print(tabulate([[k, f"{v:.3f}"] for k, v in overall_rendered.items()],
                        headers=["Metric", "Value"]))
            np.save("overall_depth_metrics_rendered.npy", overall_rendered)
        else:
            print("No rendered depth subgroup metrics recorded.")
        
        # Print overall pointcloud metrics.
        if overall_pointcloud:
            print("Overall Pointcloud Depth Metrics (averaged over all context subgroups):")
            print(tabulate([[k, f"{v:.3f}"] for k, v in overall_pointcloud.items()],
                        headers=["Metric", "Value"]))
            np.save("overall_depth_metrics_pointcloud.npy", overall_pointcloud)
        else:
            print("No pointcloud depth subgroup metrics recorded.")

        if overall_targets:
            print("Overall Target Depth Metrics (averaged over all target subgroups):")
            print(tabulate([[k, f"{v:.3f}"] for k, v in overall_targets.items()],
                        headers=["Metric", "Value"]))
            np.save("overall_depth_metrics_targets.npy", overall_targets)
        else:
            print("No targets depth subgroup metrics recorded.")


    def depth2disparity(self, depth, return_mask=False):
        import numpy as np
        if isinstance(depth, torch.Tensor):
            disparity = torch.zeros_like(depth)
        elif isinstance(depth, np.ndarray):
            disparity = np.zeros_like(depth)
        non_negtive_mask = depth > 0
        disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
        if return_mask:
            return disparity, non_negtive_mask
        else:
            return disparity


    def absolute_error_loss(self, params, predicted_depth, ground_truth_depth):
        import numpy as np
        s, t = params

        predicted_aligned = s * predicted_depth + t

        abs_error = np.abs(predicted_aligned - ground_truth_depth)
        return np.sum(abs_error)

    
    def absolute_value_scaling(self, predicted_depth, ground_truth_depth, s=1, t=0):
        import numpy as np
        from scipy.optimize import minimize
        
        predicted_depth_np = predicted_depth.cpu().numpy().reshape(-1)
        ground_truth_depth_np = ground_truth_depth.cpu().numpy().reshape(-1)

        initial_params = torch.tensor([s, t]).cpu()  # s = 1, t = 0

        result = minimize(
            self.absolute_error_loss,
            initial_params,
            args=(predicted_depth_np, ground_truth_depth_np),
        )

        s, t = result.x
        return s, t
    
    
    def absolute_value_scaling2(self, predicted_depth, ground_truth_depth, s_init=1.0, t_init=0.0, lr=1e-4, max_iters=1000, tol=1e-6):
        # Initialize s and t as torch tensors with requires_grad=True
        s = torch.tensor(
            [s_init],
            requires_grad=True,
            device=predicted_depth.device,
            dtype=predicted_depth.dtype,
        )
        t = torch.tensor(
            [t_init],
            requires_grad=True,
            device=predicted_depth.device,
            dtype=predicted_depth.dtype,
        )

        optimizer = torch.optim.Adam([s, t], lr=lr)

        prev_loss = None

        for i in range(max_iters):
            optimizer.zero_grad()

            # Compute predicted aligned depth
            predicted_aligned = s * predicted_depth + t

            # Compute absolute error
            abs_error = torch.abs(predicted_aligned - ground_truth_depth)

            # Compute loss
            loss = torch.sum(abs_error)

            # Backpropagate
            loss.backward()

            # Update parameters
            optimizer.step()

            # Check convergence
            if prev_loss is not None and torch.abs(prev_loss - loss) < tol:
                break

            prev_loss = loss.item()

        return s.detach().item(), t.detach().item()

    
    # adapted from CUT3R (https://github.com/CUT3R/CUT3R/)
    @torch.no_grad
    def depth_evaluation(self, 
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
            s, t = self.absolute_value_scaling(
                predicted_depth,
                ground_truth_depth,
                s=torch.median(ground_truth_depth) / torch.median(predicted_depth),
            )
            predicted_depth = s * predicted_depth + t
        elif align_with_lad2:
            s_init = (torch.median(ground_truth_depth) / torch.median(predicted_depth)).item()
            s, t = self.absolute_value_scaling2(
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
            predicted_depth = self.depth2disparity(predicted_depth)

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
        threshold_0 = torch.mean((max_ratio < 1.10).float()).item()
        threshold_1 = torch.mean((max_ratio < 1.25).float()).item()
        threshold_2 = torch.mean((max_ratio < 1.25**2).float()).item()
        threshold_3 = torch.mean((max_ratio < 1.25**3).float()).item()

        if metric_scale:
            predicted_depth_original_final = predicted_depth_original
            if disp_input:
                predicted_depth_original_final = self.depth2disparity(predicted_depth_original_final)
            depth_error_parity_map = torch.abs(predicted_depth_original_final - ground_truth_depth_original) / ground_truth_depth_original
        elif align_with_lstsq or align_with_lad or align_with_lad2:
            predicted_depth_original_final = predicted_depth_original * s + t
            if disp_input:
                predicted_depth_original_final = self.depth2disparity(predicted_depth_original_final)
            depth_error_parity_map = torch.abs(predicted_depth_original_final - ground_truth_depth_original) / ground_truth_depth_original
        elif align_with_scale:
            predicted_depth_original_final = predicted_depth_original * s
            if disp_input:
                predicted_depth_original_final = self.depth2disparity(predicted_depth_original_final)
            depth_error_parity_map = torch.abs(predicted_depth_original_final - ground_truth_depth_original) / ground_truth_depth_original
        else:
            predicted_depth_original_final = predicted_depth_original * scale_factor
            if disp_input:
                predicted_depth_original_final = self.depth2disparity(predicted_depth_original_final)
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
            "δ < 1.10": threshold_0,
            "δ < 1.25": threshold_1,
            "δ < 1.25^2": threshold_2,
            "δ < 1.25^3": threshold_3,
            "valid_pixels": num_valid_pixels,
        }

        return results, depth_error_parity_map_full, predict_depth_map_full, gt_depth_map_full

