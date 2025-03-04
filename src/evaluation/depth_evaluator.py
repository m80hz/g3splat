import json
import os
import sys
from typing import Any

import math
from pytorch_lightning.utilities.types import STEP_OUTPUT

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample

import csv
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from lightning import LightningModule
from tabulate import tabulate

from ..loss.loss_ssim import ssim
from ..misc.image_io import load_image, save_image
from ..misc.utils import inverse_normalize, get_overlap_tag, vis_depth_map, inspect_depth_tensor
from ..visualization.annotation import add_label
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from .evaluation_cfg import EvaluationCfg
from .metrics import compute_depth_errors, depth_evaluation

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

        b, v, _, h, w = batch["context"]["image"].shape
        assert b == 1
        if batch_idx % 100 == 0:
            print(f"Test step {batch_idx:0>6}.")

        # get overlap.
        overlap = batch["context"]["overlap"][0, 0]
        overlap_tag = get_overlap_tag(overlap)
        # if overlap_tag == "ignore":
        #     return


        visualization_dump = {}

        # running encoder to obtain the 3DGS
        gaussians = self.encoder(
            batch["context"],
            self.global_step,
            visualization_dump=visualization_dump,
        )
       
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

        # direct depth from gaussian means
        gaussian_means = visualization_dump["depth"][0].squeeze()
        if gaussian_means.shape[-1] == 3:
            gaussian_means = gaussian_means.mean(dim=-1)

        context_depth_pointcloud = gaussian_means
        
        context_depth_gt = batch["context"]["depth"][0].squeeze(1)
        context_valid_depth_gt = batch["context"]["valid_depth"][0].squeeze(1)
        
        target_depth_gt = batch["target"]["depth"][0].squeeze(1)
        target_valid_depth_gt = batch["target"]["valid_depth"][0].squeeze(1)
        
        if self.cfg.save_depth_concat_img:
            fig, ax = plt.subplots(3, 6, figsize=(32, 21))

            # Top row titles
            ax[0, 0].set_title("GT Context Image (0)")
            ax[0, 1].set_title("Rendered Image (0)")
            ax[0, 2].set_title("GT Depth (0)")
            ax[0, 3].set_title("GT Valid Depth (0)")
            ax[0, 4].set_title("GS-Mean Depth (0)")
            ax[0, 5].set_title("Rendered Depth (0)")

            # Middle row titles
            ax[1, 0].set_title("GT Context Image (1)")
            ax[1, 1].set_title("Rendered Image (1)")
            ax[1, 2].set_title("GT Depth (1)")
            ax[1, 3].set_title("GT Valid Depth (1)")        
            ax[1, 4].set_title("GS-Mean Depth (1)")
            ax[1, 5].set_title("Rendered Depth (1)")

            # Bottom row titles
            ax[2, 0].set_title("GT Target Image")
            ax[2, 1].set_title("Rendered Target Image")
            ax[2, 2].set_title("GT Target Depth")
            ax[2, 3].set_title("GT Target Valid Depth")        
            ax[2, 4].set_title("Target Rendered Depth")
            ax[2, 5].set_title("Target Rendered Depth")


            # Top row images
            ax[0, 0].imshow(batch["context"]["image"][0, 0].cpu().permute(1, 2, 0) * 0.5 + 0.5)
            ax[0, 1].imshow(context_img_rendered[0].cpu().permute(1, 2, 0))
            ax[0, 2].imshow(vis_depth_map(context_depth_gt[0]).cpu().permute(1, 2, 0))
            ax[0, 3].imshow(context_valid_depth_gt[0].cpu(), cmap="grey")
            ax[0, 4].imshow(vis_depth_map(context_depth_pointcloud[0]).cpu().permute(1, 2, 0))
            ax[0, 5].imshow(vis_depth_map(context_depth_rendered[0]).cpu().permute(1, 2, 0))

            # Middle row images
            ax[1, 0].imshow(batch["context"]["image"][0, 1].cpu().permute(1, 2, 0) * 0.5 + 0.5)
            ax[1, 1].imshow(context_img_rendered[1].cpu().permute(1, 2, 0))
            ax[1, 2].imshow(vis_depth_map(context_depth_gt[1]).cpu().permute(1, 2, 0))
            ax[1, 3].imshow(context_valid_depth_gt[1].cpu(), cmap="grey")
            ax[1, 4].imshow(vis_depth_map(context_depth_pointcloud[1]).cpu().permute(1, 2, 0))
            ax[1, 5].imshow(vis_depth_map(context_depth_rendered[1]).cpu().permute(1, 2, 0))

            # Bottom row images
            ax[2, 0].imshow(batch["target"]["image"][0, 0].cpu().permute(1, 2, 0))
            ax[2, 1].imshow(target_img_rendered[0].cpu().permute(1, 2, 0))
            ax[2, 2].imshow(vis_depth_map(target_depth_gt[0]).cpu().permute(1, 2, 0))
            ax[2, 3].imshow(target_valid_depth_gt[0].cpu(), cmap="grey")
            ax[2, 4].imshow(vis_depth_map(target_depth_rendered[0]).cpu().permute(1, 2, 0))
            ax[2, 5].imshow(vis_depth_map(target_depth_rendered[0]).cpu().permute(1, 2, 0))

            plt.tight_layout()
            # plt.show()
            plt.savefig(f"depth_eval_{batch_idx}.png")

        # only evaluate on valid depth
        near_value_ctx = batch["context"]["near"].view(-1, 1, 1)  # shape (B, 1, 1)
        far_value_ctx  = batch["context"]["far"].view(-1, 1, 1)   # shape (B, 1, 1)
        near_value_tg = batch["target"]["near"].view(-1, 1, 1)  # shape (B, 1, 1)
        far_value_tg  = batch["target"]["far"].view(-1, 1, 1)   # shape (B, 1, 1)
        # depth_mask = None
        context_depth_mask = (context_depth_gt > near_value_ctx) & (context_depth_gt < far_value_ctx)
        target_depth_mask = (target_depth_gt > near_value_tg) & (target_depth_gt < far_value_tg)
        
       
        # --- Context 1 --- The one with identity pose
        results_ctx1_rendered, parity_map_ctx1_rendered, pred_full_ctx1_rendered, gt_full_ctx1_rendered = depth_evaluation(
            predicted_depth_original=context_depth_rendered[0:1],
            ground_truth_depth_original=context_depth_gt[0:1],
            max_depth=far_value_ctx[0:1].item(),
            custom_mask=context_depth_mask[0:1] if context_depth_mask is not None else None,
            pre_clip_min=near_value_ctx[0:1].item(),
            use_gpu=True,
            align_with_lstsq=False,
            align_with_lad=False,
            align_with_lad2=False,
            metric_scale=False,
            align_with_scale=False,
            disp_input=False
        )

        results_ctx1_pointcloud, parity_map_ctx1_pointcloud, pred_full_ctx1_pointcloud, gt_full_ctx1_pointcloud = depth_evaluation(
            predicted_depth_original=context_depth_pointcloud[0:1],
            ground_truth_depth_original=context_depth_gt[0:1],
            max_depth=far_value_ctx[0:1].item(),
            custom_mask=context_depth_mask[0:1] if context_depth_mask is not None else None,
            pre_clip_min=near_value_ctx[0:1].item(),
            use_gpu=True,
            align_with_lstsq=False,
            align_with_lad=False,
            align_with_lad2=False,
            metric_scale=False,
            align_with_scale=False,
            disp_input=False
        )

        # --- Context 2 ---
        results_ctx2_rendered, parity_map_ctx2_rendered, pred_full_ctx2_rendered, gt_full_ctx2_rendered = depth_evaluation(
            predicted_depth_original=context_depth_rendered[1:2],
            ground_truth_depth_original=context_depth_gt[1:2],
            max_depth=far_value_ctx[1:2].item(),
            custom_mask=context_depth_mask[1:2] if context_depth_mask is not None else None,
            pre_clip_min=near_value_ctx[1:2].item(),
            use_gpu=True,
            align_with_lstsq=False,
            align_with_lad=False,
            align_with_lad2=False,
            metric_scale=False,
            align_with_scale=False,
            disp_input=False
        )

        results_ctx2_pointcloud, parity_map_ctx2_pointcloud, pred_full_ctx2_pointcloud, gt_full_ctx2_pointcloud = depth_evaluation(
            predicted_depth_original=context_depth_pointcloud[1:2],
            ground_truth_depth_original=context_depth_gt[1:2],
            max_depth=far_value_ctx[1:2].item(),
            custom_mask=context_depth_mask[1:2] if context_depth_mask is not None else None,
            pre_clip_min=near_value_ctx[1:2].item(),
            use_gpu=True,
            align_with_lstsq=False,
            align_with_lad=False,
            align_with_lad2=False,
            metric_scale=False,
            align_with_scale=False,
            disp_input=False
        )
        
        
        # --- Target Views --- 
        results_targets_rendered, parity_map_targets_rendered, pred_full_targets_rendered, gt_full_targets_rendered = depth_evaluation(
            predicted_depth_original=target_depth_rendered,
            ground_truth_depth_original=target_depth_gt,
            max_depth=far_value_tg[0:1].item(),
            custom_mask=target_depth_mask if target_depth_mask is not None else None,
            pre_clip_min=near_value_tg[0:1].item(),
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
        error_names = ['Abs Rel', 'Sq Rel', 'RMSE', 'Log RMSE', 'δ < 1.', 'δ < 1.25', 'δ < 1.25^2', 'δ < 1.25^3']

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
        error_names = ['Abs_Rel', 'Sq_Rel', 'RMSE', 'Log_RMSE', 'δ_<_1.', 'δ_<_1.25', 'δ_<_1.25^2', 'δ_<_1.25^3']
        
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
