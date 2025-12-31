import json
import os
import sys
from typing import Any

import math
from pytorch_lightning.utilities.types import STEP_OUTPUT

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..misc.cam_utils import camera_normalization, pose_auc, update_pose, get_pnp_pose

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
from ..misc.utils import inverse_normalize, get_overlap_tag
from ..visualization.annotation import add_label
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from .evaluation_cfg import EvaluationCfg
from .metrics import compute_lpips, compute_psnr, compute_ssim, compute_pose_error


class PoseEvaluator(LightningModule):
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

        self.encoder.eval()
        for p in self.encoder.parameters(): p.requires_grad = False

        b, v, _, h, w = batch["context"]["image"].shape
        assert b == 1
        if batch_idx % 100 == 0:
            print(f"Test step {batch_idx:0>6}.")

        overlap = batch["context"]["overlap"][0, 0]
        overlap_tag = get_overlap_tag(overlap)
        if overlap_tag == "ignore": return

        input_images_view2 = batch["context"]["image"][:, 1:2].clone()
        input_images_view2 = input_images_view2 * 0.5 + 0.5
        visualization_dump = {}
        gaussians = self.encoder(
            batch["context"],
            self.global_step,
            visualization_dump=visualization_dump,
        )

        # Iterative PnP (no RANSAC)
        pose_pnp_iter = None
        pnp_iter_inliers = None
        pnp_iter_inlier_ratio = None
        if getattr(self.cfg, "use_pnp_iterative", False):
            pnp_iter_ret = get_pnp_pose(
                visualization_dump['means'][0, 1].squeeze(),
                visualization_dump['opacities'][0, 1].squeeze(),
                batch["context"]["intrinsics"][0, 1], h, w,
                return_inliers=True,
                use_ransac=False
            )
            pose_pnp_iter = pnp_iter_ret[0].to(self.device)
            pnp_iter_inliers = torch.tensor(pnp_iter_ret[1], device=self.device, dtype=torch.float32)
            pnp_iter_inlier_ratio = torch.tensor(pnp_iter_ret[2], device=self.device, dtype=torch.float32)


        # PnP init (RANSAC)
        pnp_ret = get_pnp_pose(
            visualization_dump['means'][0, 1].squeeze(),
            visualization_dump['opacities'][0, 1].squeeze(),
            batch["context"]["intrinsics"][0, 1], h, w,
            return_inliers=True,
            use_ransac=True
        )
        pose_pnp_ransac = pnp_ret[0].to(self.device)
        pnp_ransac_inliers = torch.tensor(pnp_ret[1], device=self.device, dtype=torch.float32)
        pnp_ransac_inlier_ratio = torch.tensor(pnp_ret[2], device=self.device, dtype=torch.float32)

        # PnP + photometric
        if self.cfg.use_pose_refinement:
            with torch.set_grad_enabled(True):
                cam_rot_delta = nn.Parameter(torch.zeros([b, 1, 3], requires_grad=True, device=self.device))
                cam_trans_delta = nn.Parameter(torch.zeros([b, 1, 3], requires_grad=True, device=self.device))
                pose_optimizer = torch.optim.Adam([cam_rot_delta, cam_trans_delta], lr=0.005)
                number_steps = 200
                extrinsics = pose_pnp_ransac.unsqueeze(0).unsqueeze(0)
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
                    batch["target"]["image"] = input_images_view2
                    total_loss = 0
                    for loss_fn in self.losses:
                        total_loss = total_loss + loss_fn.forward(output, batch, gaussians, self.global_step)
                    ssim_, _, _, structure = ssim(
                        rearrange(batch["target"]["image"], "b v c h w -> (b v) c h w"),
                        rearrange(output.color, "b v c h w -> (b v) c h w"),
                        size_average=True, data_range=1.0, retrun_seprate=True, win_size=11
                    )
                    total_loss = total_loss + (1 - structure) * 1.0
                    total_loss.backward()
                    with torch.no_grad():
                        pose_optimizer.step()
                        new_extrinsic = update_pose(
                            cam_rot_delta=rearrange(cam_rot_delta, "b v i -> (b v) i"),
                            cam_trans_delta=rearrange(cam_trans_delta, "b v i -> (b v) i"),
                            extrinsics=rearrange(extrinsics, "b v i j -> (b v) i j"),
                        )
                        cam_rot_delta.data.fill_(0)
                        cam_trans_delta.data.fill_(0)
                        extrinsics = rearrange(new_extrinsic, "(b v) i j -> b v i j", b=b, v=1)
            eval_pose_photo = extrinsics[0, 0]
        else:
            eval_pose_photo = None
            
        # Evaluate
        eval_pose_pnp_ransac = pose_pnp_ransac
        eval_pose_pnp_iter = pose_pnp_iter

        gt_pose = batch["context"]["extrinsics"][0, 1]

        if eval_pose_pnp_ransac is not None:
            error_t, _, error_R = compute_pose_error(gt_pose, eval_pose_pnp_ransac)
            self.print_preview_metrics({
                "e_t_pnp_ransac": error_t, "e_R_pnp_ransac": error_R, "e_pose_pnp_ransac": torch.max(error_t, error_R),
                "inliers_pnp_ransac": pnp_ransac_inliers, "inlier_ratio_pnp_ransac": pnp_ransac_inlier_ratio
            }, overlap_tag)
        if eval_pose_pnp_iter is not None:
            error_t_iter, _, error_R_iter = compute_pose_error(gt_pose, eval_pose_pnp_iter)
            self.print_preview_metrics({
                "e_t_pnp_iter": error_t_iter, "e_R_pnp_iter": error_R_iter, "e_pose_pnp_iter": torch.max(error_t_iter, error_R_iter),
                "inliers_pnp_iter": pnp_iter_inliers, "inlier_ratio_pnp_iter": pnp_iter_inlier_ratio
            }, overlap_tag)
        if eval_pose_photo is not None:
            e_t, _, e_R = compute_pose_error(gt_pose, eval_pose_photo)
            self.print_preview_metrics({
                "e_t_pnp_photo": e_t, "e_R_pnp_photo": e_R, "e_pose_pnp_photo": torch.max(e_t, e_R)
            }, overlap_tag)

        return 0


    def on_test_end(self) -> None:
        print("\n==================== Pose Evaluation Summary ====================")
        thresholds = [5, 10, 20, 30]

        def _maybe_get_list(store, key):
            if key not in store: return None
            arr = np.asarray(store[key]);  return None if arr.size == 0 else arr

        def _format_auc(auc_list):
            return " | ".join(f"@{t}:{a:.3f}" for t, a in zip(thresholds, auc_list))

        for method in self.cfg.methods:
            k = method.key
            pose_key = f"e_pose_{k}"
            t_key = f"e_t_{k}"
            r_key = f"e_R_{k}"
            pose_vals = _maybe_get_list(self.all_mertrics, pose_key)
            t_vals = _maybe_get_list(self.all_mertrics, t_key)
            r_vals = _maybe_get_list(self.all_mertrics, r_key)
            if pose_vals is None and t_vals is None and r_vals is None:
                continue
            print(f"\n--- Method: {k} (Overall) ---")
            if t_vals is not None:
                print("Translation AUC:", _format_auc(pose_auc(t_vals, thresholds)))
            else:
                print("Translation AUC: -")
            if r_vals is not None:
                print("Rotation AUC:    ", _format_auc(pose_auc(r_vals, thresholds)))
            else:
                print("Rotation AUC:     -")
            if pose_vals is not None:
                print("Max(pose) AUC:   ", _format_auc(pose_auc(pose_vals, thresholds)))
            else:
                print("Max(pose) AUC:    -")

            # Inlier stats (only meaningful for PnP / methods providing them)
            inlier_vals = _maybe_get_list(self.all_mertrics, f"inliers_{k}")
            inlier_ratio_vals = _maybe_get_list(self.all_mertrics, f"inlier_ratio_{k}")
            if inlier_vals is not None:
                print(f"Inliers (avg): {inlier_vals.mean():.2f}")
            if inlier_ratio_vals is not None:
                print(f"Inlier Ratio (avg): {100*inlier_ratio_vals.mean():.2f}%")

            if hasattr(self, 'all_mertrics_sub'):
                for overlap_tag, metrics_dict in self.all_mertrics_sub.items():
                    pose_vals_sub = _maybe_get_list(metrics_dict, pose_key)
                    t_vals_sub = _maybe_get_list(metrics_dict, t_key)
                    r_vals_sub = _maybe_get_list(metrics_dict, r_key)
                    if pose_vals_sub is None and t_vals_sub is None and r_vals_sub is None:
                        continue
                    print(f"    Overlap: {overlap_tag}")
                    if t_vals_sub is not None:
                        print("      Translation AUC:", _format_auc(pose_auc(t_vals_sub, thresholds)))
                    else:
                        print("      Translation AUC: -")
                    if r_vals_sub is not None:
                        print("      Rotation AUC:    ", _format_auc(pose_auc(r_vals_sub, thresholds)))
                    else:
                        print("      Rotation AUC:     -")
                    if pose_vals_sub is not None:
                        print("      Max(pose) AUC:   ", _format_auc(pose_auc(pose_vals_sub, thresholds)))
                    else:
                        print("      Max(pose) AUC:    -")
                    inlier_vals_sub = _maybe_get_list(metrics_dict, f"inliers_{k}")
                    inlier_ratio_vals_sub = _maybe_get_list(metrics_dict, f"inlier_ratio_{k}")
                    if inlier_vals_sub is not None:
                        print(f"      Inliers (avg): {inlier_vals_sub.mean():.2f}")
                    if inlier_ratio_vals_sub is not None:
                        print(f"      Inlier Ratio (avg): {100*inlier_ratio_vals_sub.mean():.2f}%")

        print("\nSaved raw metric arrays to all_metrics.npy / all_metrics_sub.npy")
        np.save("all_metrics.npy", self.all_mertrics)
        if hasattr(self, 'all_mertrics_sub'):
            np.save("all_metrics_sub.npy", self.all_mertrics_sub)

    def print_preview_metrics(self, metrics: dict[str, float], overlap_tag: str | None = None) -> None:
        if getattr(self, "running_metrics", None) is None:
            self.running_metrics = dict(metrics)
            self.running_metric_steps = 1
            self.all_mertrics = {k: [v.cpu().item()] for k, v in metrics.items()}
        else:
            s = self.running_metric_steps
            for k, v in metrics.items():
                if k in self.running_metrics:
                    self.running_metrics[k] = ((s * self.running_metrics[k]) + v) / (s + 1)
                else:
                    self.running_metrics[k] = v
            self.running_metric_steps += 1
            for k, v in metrics.items():
                if k not in self.all_mertrics:
                    self.all_mertrics[k] = []
                self.all_mertrics[k].append(v.cpu().item())

        if overlap_tag is not None:
            if getattr(self, "running_metrics_sub", None) is None:
                self.running_metrics_sub = {overlap_tag: dict(metrics)}
                self.running_metric_steps_sub = {overlap_tag: 1}
                self.all_mertrics_sub = {overlap_tag: {k: [v.cpu().item()] for k, v in metrics.items()}}
            elif overlap_tag not in self.running_metrics_sub:
                self.running_metrics_sub[overlap_tag] = dict(metrics)
                self.running_metric_steps_sub[overlap_tag] = 1
                self.all_mertrics_sub[overlap_tag] = {k: [v.cpu().item()] for k, v in metrics.items()}
            else:
                s = self.running_metric_steps_sub[overlap_tag]
                for k, v in metrics.items():
                    if k in self.running_metrics_sub[overlap_tag]:
                        self.running_metrics_sub[overlap_tag][k] = ((s * self.running_metrics_sub[overlap_tag][k]) + v) / (s + 1)
                    else:
                        self.running_metrics_sub[overlap_tag][k] = v
                self.running_metric_steps_sub[overlap_tag] += 1
                for k, v in metrics.items():
                    if k not in self.all_mertrics_sub[overlap_tag]:
                        self.all_mertrics_sub[overlap_tag][k] = []
                    self.all_mertrics_sub[overlap_tag][k].append(v.cpu().item())

        def print_metrics(runing_metric):
            table = []
            have_inlier_ratio = any(f"inlier_ratio_{m.key}" in runing_metric for m in self.cfg.methods)
            headers = ["Method", "e_t", "e_R", "e_pose"] + (["inlier_ratio"] if have_inlier_ratio else [])
            for method in self.cfg.methods:
                row = []
                for metric in ("e_t", "e_R", "e_pose"):
                    key = f"{metric}_{method.key}"
                    val = runing_metric.get(key)
                    if val is None:
                        row.append("-")
                    else:
                        try:
                            num = val.item() if hasattr(val, "item") else float(val)
                            row.append(f"{num:.3f}")
                        except Exception:
                            row.append(str(val))
                if have_inlier_ratio:
                    val = runing_metric.get(f"inlier_ratio_{method.key}")
                    if val is None:
                        row.append("-")
                    else:
                        try:
                            num = val.item() if hasattr(val, "item") else float(val)
                            row.append(f"{num:.3f}")
                        except Exception:
                            row.append(str(val))
                table.append((method.key, *row))
            print(tabulate(table, headers))

        print("Running Average (All Pairs):")
        print_metrics(self.running_metrics)
        if overlap_tag is not None:
            for k_sub, v_sub in self.running_metrics_sub.items():
                print(f"Running Average (Overlap: {k_sub}):")
                print_metrics(v_sub)
