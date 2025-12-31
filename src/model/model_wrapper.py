from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable, Any

import os
import moviepy.editor as mpy
import torch
import wandb
import numpy as np
from einops import pack, rearrange, repeat
from jaxtyping import Float
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from tabulate import tabulate
from torch import Tensor, nn, optim
import open3d as o3d

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..loss.loss_point import Regr3D
from ..loss.loss_ssim import ssim
from ..misc.benchmarker import Benchmarker
from ..misc.cam_utils import update_pose, get_pnp_pose
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.nn_module_tools import convert_to_buffer
from ..misc.step_tracker import StepTracker
from ..misc.utils import inverse_normalize, vis_depth_map, vis_scalar_map, confidence_map, get_overlap_tag
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode
from .decoder.decoder_splatting_cuda import DecoderType
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from ..visualization.normal import vis_normal
from ..geometry.surface_normal import surface_normal_from_depth, get_surface_normal
from ..geometry.projection import points_to_normal
from .encoder.common.gaussians import gaussian_orientation_from_scales
from .ply_export import save_gaussian_ply
from ..misc.mesh_utils import GaussianMeshExtractor, post_process_mesh

@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    backbone_lr_multiplier: float


@dataclass
class TestCfg:
    output_path: Path
    align_pose: bool
    pose_align_steps: int
    rot_opt_lr: float
    trans_opt_lr: float
    compute_scores: bool
    save_image: bool
    save_video: bool
    save_compare: bool
    save_gaussian: bool
    save_mesh: bool
    


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    distiller: str
    distill_max_steps: int


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
        distiller: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        self.distiller = distiller
        self.distiller_loss = None
        if self.distiller is not None:
            convert_to_buffer(self.distiller, persistent=False)
            self.distiller_loss = Regr3D()

        # This is used for testing.
        self.benchmarker = Benchmarker()

    def _decoder_type(self) -> DecoderType:
        gt = getattr(self.encoder.gaussian_adapter.cfg, "gaussian_type", "3d")
        return "3D" if str(gt).lower() == "3d" else "2D"

    def training_step(self, batch, batch_idx):
        # combine batch from different dataloaders
        if isinstance(batch, list):
            batch_combined = None
            for batch_per_dl in batch:
                if batch_combined is None:
                    batch_combined = batch_per_dl
                else:
                    for k in batch_combined.keys():
                        if isinstance(batch_combined[k], list):
                            batch_combined[k] += batch_per_dl[k]
                        elif isinstance(batch_combined[k], dict):
                            for kk in batch_combined[k].keys():
                                batch_combined[k][kk] = torch.cat([batch_combined[k][kk], batch_per_dl[k][kk]], dim=0)
                        else:
                            raise NotImplementedError
            batch = batch_combined
        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape

        # Run the model.
        visualization_dump = None
        if self.distiller is not None:
            visualization_dump = {}
        gaussians = self.encoder(batch["context"], self.global_step, visualization_dump=visualization_dump)
        output = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
            decoder_type=self._decoder_type(),
        )
        target_gt = batch["target"]["image"]

        # Compute metrics.
        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr_probabilistic", psnr_probabilistic.mean(), sync_dist=True)

        # Compute and log loss.
        total_loss = 0
        for loss_fn in self.losses:
            loss = loss_fn.forward(output, batch, gaussians, self.global_step)
            self.log(f"loss/{loss_fn.name}", loss, sync_dist=True)
            total_loss = total_loss + loss

        # distillation
        if self.distiller is not None and self.global_step <= self.train_cfg.distill_max_steps:
            with torch.no_grad():
                pseudo_gt1, pseudo_gt2 = self.distiller(batch["context"], False)
            distillation_loss = self.distiller_loss(pseudo_gt1['pts3d'], pseudo_gt2['pts3d'],
                                                    visualization_dump['means'][:, 0].squeeze(-2),
                                                    visualization_dump['means'][:, 1].squeeze(-2),
                                                    pseudo_gt1['conf'], pseudo_gt2['conf'], disable_view1=False) * 0.1
            self.log("loss/distillation_loss", distillation_loss)
            total_loss = total_loss + distillation_loss

        self.log("loss/total", total_loss, sync_dist=True)

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            print(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"loss = {total_loss:.6f}"
            )
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return total_loss

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1
        if batch_idx % 100 == 0:
            print(f"Test step {batch_idx:0>6}.")

        # Render Gaussians.
        visualization_dump = {}
        with self.benchmarker.time("encoder"):
            gaussians = self.encoder(
                batch["context"],
                self.global_step,
                visualization_dump=visualization_dump
            )

        # save gaussians
        if self.test_cfg.save_gaussian:
            (scene,) = batch["scene"]
            name = get_cfg()["wandb"]["name"]
            path = self.test_cfg.output_path / name
            save_path = Path(path) / scene / 'gaussians' / (scene + '.ply')
            save_gaussian_ply(gaussians, visualization_dump, batch, save_path)

        # align the target pose
        if self.test_cfg.align_pose:
            output = self.test_step_align(batch, gaussians)
        else:
            with self.benchmarker.time("decoder", num_calls=v):
                output = self.decoder.forward(
                    gaussians,
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                    decoder_type=self._decoder_type(),
                )

        # compute scores
        if self.test_cfg.compute_scores:
            overlap = batch["context"]["overlap"][0]
            overlap_tag = get_overlap_tag(overlap)

            rgb_pred = output.color[0]
            rgb_gt = batch["target"]["image"][0]
            all_metrics = {
                f"lpips_ours": compute_lpips(rgb_gt, rgb_pred).mean(),
                f"ssim_ours": compute_ssim(rgb_gt, rgb_pred).mean(),
                f"psnr_ours": compute_psnr(rgb_gt, rgb_pred).mean(),
            }
            methods = ['ours']

            self.log_dict(all_metrics)
            self.print_preview_metrics(all_metrics, methods, overlap_tag=overlap_tag)


        # direct depth from gaussian means (used for visualization only)
        gaussian_means = visualization_dump["depth"][0].squeeze()      # (v, h, w)
        if gaussian_means.shape[-1] == 3:
            gaussian_means = gaussian_means.mean(dim=-1)

        # surface normals derived from pointclouds - context views
        # all_pts_depth = gaussian_means.unsqueeze(-1)   # (v, h, w, 1)
        # foc_x = batch["context"]["intrinsics"][0, 0, 0, 0] * w
        # foc_y = batch["context"]["intrinsics"][0, 0, 1, 1] * h
        # normal_pts = surface_normal_from_depth(all_pts_depth.permute(0, 3, 1, 2), focal_x=foc_x[None], focal_y=foc_y[None], 
        #                                     valid_mask=(all_pts_depth > 0).permute(0, 3, 1, 2)) 
        # 
        # all_pts3d = rearrange(gaussians.means, "b (v h w) d -> b v h w d", h=h, w=w)
        # pts3d1 = all_pts3d[:, 0, ...]  # (B, H, W, 3)
        # pts3d2 = all_pts3d[:, 1, ...]  # (B, H, W, 3)
        # sn_batch = []
        # for i in range(2):
        #     xyz_i = all_pts3d[0, i, ...][None]  # (B=1, H, W, 3)
        #     # normal = get_surface_normal(xyz_i)    # using a smoother normal approximation
        #     sn_batch.append(normal)
        # sn_batch = torch.cat(sn_batch, dim=3).permute((3, 2, 0, 1))  # [v, c=3, h, w]
        # surf_normals_pts = sn_batch.permute(0, 2, 3, 1)   # (v, h, w, c=3)

        all_pts3d = rearrange(gaussians.means, "b (v h w) d -> b v h w d", h=h, w=w)
        # pts3d1 = all_pts3d[:, 0, ...]  # (B, H, W, 3)
        # pts3d2 = all_pts3d[:, 1, ...]  # (B, H, W, 3)
        surf_normals_pts, _ = points_to_normal(all_pts3d[0])   # (v, h, w, c=3)
       
        # Visualising depth and normals for target views
        # predictions for target views
        target_rendered_depth = vis_depth_map(output.depth[0])
        # surface_normal = vis_normal(output.surf_normal[0].permute(0, 2, 3, 1)).permute(0, 3, 1, 2).float() / 255
        # render_normal = vis_normal(output.rend_normal[0].permute(0, 2, 3, 1)).permute(0, 3, 1, 2).float() / 255
        # rend_dist = vis_depth_map(output.dist[0])
        # rend_alpha = vis_depth_map(output.alpha[0])

        # Visualisation of gaussians (predicted from context views) - for context 1 only 
        gaussian_rotations = visualization_dump["rotations"]
        gaussian_rotations = rearrange(gaussian_rotations, "b (v h w) d -> b v h w d", v=2, h=h, w=w)
        context1_gaussian_rotations = gaussian_rotations[:, 0, ...]     # shape (B, H, W, 4)

        gaussian_scales = visualization_dump["scales"]
        gaussian_scales = rearrange(gaussian_scales, "b (v h w) d -> b v h w d", v=2, h=h, w=w)
        context1_gaussian_scales = gaussian_scales[:, 0, ...]     # shape (B, H, W, 3)
        sorted_context1_gaussian_scales = torch.sort(context1_gaussian_scales, dim=-1, descending=True)[0]
        # Compute normalized scales by dividing every channel by the first channel (i.e., scale 0)
        epsilon = 1e-6
        context1_gaussian_scales_normalized = sorted_context1_gaussian_scales / (sorted_context1_gaussian_scales[..., :1] + epsilon)   # shape (B, H, W, 3)

        gaussian_opacities = visualization_dump['opacities']
        gaussian_opacities = rearrange(gaussian_opacities, "b v h w srf s -> b v h w (srf s)", v=2, h=h, w=w)
        context1_gaussian_opacities = gaussian_opacities[:, 0, ...]     # shape (B, H, W, 1)

        # Align normals with the smallest-scale axis of each Gaussian.
        gaussian_surfels_normals = gaussian_orientation_from_scales(
            context1_gaussian_rotations,
            context1_gaussian_scales,
        )  # shape: (B, H, W, 3)

        # Visualize the selected normals.
        gaussian_normal_vis = vis_normal(gaussian_surfels_normals).permute(0, 3, 1, 2).float() / 255.0

        # Save images.
        (scene,) = batch["scene"]
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name

        if self.test_cfg.save_image:
            context1_index = batch["context"]["index"][0, 0]
            save_image(gaussian_normal_vis[0], path / scene / f"context1_gaussian_normal/{context1_index:0>6}.png")
            
            # Save the opacities
            gaussian_opacity_map = context1_gaussian_opacities[..., 0]
            gaussian_opacity_vis = vis_scalar_map(gaussian_opacity_map)    # shape: (B, 3, H, W)
            save_image(gaussian_opacity_vis[0], path / scene / f"context1_gaussian_opacity/{context1_index:0>6}.png")
            
            # Save the scales: for each scale channel, save one image per batch.
            # sorted_context1_gaussian_scales has shape (B, H, W, 3)
            for scale_idx in range(3):
                # Extract one scale channel: shape (B, H, W)
                # gaussian_scale_map = context1_gaussian_scales[..., scale_idx]
                gaussian_scale_map = sorted_context1_gaussian_scales[..., scale_idx]
                # Visualize scale map using the depth visualization function.
                # gaussian_scale_vis = vis_scalar_map(gaussian_scale_map, norm_min=0.01, norm_max=0.1)    # shape: (B, 3, H, W)
                norm_min = torch.log(torch.tensor(0.001))
                norm_max = torch.log(torch.tensor(0.3))
                gaussian_scale_vis = vis_depth_map(gaussian_scale_map, norm_min=norm_min, norm_max=norm_max, colormap='turbo_r')    # shape: (B, 3, H, W)
                save_image(gaussian_scale_vis[0], path / scene / f"context1_gaussian_scale/{context1_index:0>6}_{scale_idx}.png")
            
                # Save the normalised scales for context 1: save one image per batch
                gaussian_scale_normalized_map = context1_gaussian_scales_normalized[..., scale_idx]       
                # gaussian_scale_normalized_vis = vis_scalar_map(gaussian_scale_normalized_map, colormap='turbo_r')    # shape: (B, 3, H, W)  
                gaussian_scale_normalized_vis = vis_scalar_map(gaussian_scale_normalized_map, norm_min=0.05, norm_max=0.8, colormap='turbo_r')    # shape: (B, 3, H, W)  
                # norm_min = torch.log(torch.tensor(0.05))
                # norm_max = torch.log(torch.tensor(0.6))
                # gaussian_scale_normalized_vis = vis_depth_map(gaussian_scale_normalized_map, norm_min=norm_min, norm_max=norm_max, colormap='turbo')    # shape: (B, 3, H, W)
                # gaussian_scale_normalized_vis = vis_depth_map(gaussian_scale_normalized_map, colormap='turbo_r')    # shape: (B, 3, H, W)
                save_image(gaussian_scale_normalized_vis[0], path / scene / f"context1_gaussian_scale_normalized/{context1_index:0>6}_{scale_idx}.png")
                
                
    
            

            # Save visualisations for context views
            context_img = inverse_normalize(batch["context"]["image"][0])
            for index, color in zip(batch["context"]["index"][0], context_img):
                save_image(color, path / scene / f"contexts_color/{index:0>6}.png")

            context_img_depth = vis_depth_map(gaussian_means)
            for index, depth in zip(batch["context"]["index"][0], context_img_depth):
                save_image(depth, path / scene / f"contexts_ptc_depth/{index:0>6}.png")

            context_img_normal_ptc = vis_normal(surf_normals_pts).permute(0, 3, 1, 2).float() / 255.0
            for index, normal in zip(batch["context"]["index"][0], context_img_normal_ptc):
                save_image(normal, path / scene / f"contexts_ptc_normal/{index:0>6}.png")

            # Save visualisations for target views
            for index, color in zip(batch["target"]["index"][0], output.color[0]):
                save_image(color, path / scene / f"targets_rendered_color/{index:0>6}.png")

            for index, color in zip(batch["target"]["index"][0], batch["target"]["image"][0]):
                save_image(color, path / scene / f"targets_gt_color/{index:0>6}.png")

            for index, depth in zip(batch["target"]["index"][0], target_rendered_depth):
                save_image(depth, path / scene / f"targets_rendered_depth/{index:0>6}.png")
                
            # if depth is in dataset, save the GT depth as well
            if "depth" in batch["target"].keys():
                target_gt_depth = batch["target"]["depth"][0].squeeze(1)  # (V, H, W)
                target_gt_depth_vis = vis_depth_map(target_gt_depth)
                for index, depth in zip(batch["target"]["index"][0], target_gt_depth_vis):
                    save_image(depth, path / scene / f"targets_gt_depth/{index:0>6}.png")

            # for index, normal in zip(batch["target"]["index"][0], surface_normal):
            #     save_image(normal, path / scene / f"targets_surface_normal/{index:0>6}.png")

            # for index, normal in zip(batch["target"]["index"][0], render_normal):
            #     save_image(normal, path / scene / f"targets_rendered_normal/{index:0>6}.png")

            # for index, alpha in zip(batch["target"]["index"][0], rend_alpha):
            #     save_image(alpha, path / scene / f"targets_rendered_alphas/{index:0>6}.png")

        
        if self.test_cfg.save_mesh:

            def trajectory_fn(t):
                extrinsics = interpolate_extrinsics(
                    batch["context"]["extrinsics"][0, 0],
                    (
                        batch["context"]["extrinsics"][0, 1]
                        if v == 2
                        else batch["target"]["extrinsics"][0, 0]
                    ),
                    t,
                )
                intrinsics = interpolate_intrinsics(
                    batch["context"]["intrinsics"][0, 0],
                    (
                        batch["context"]["intrinsics"][0, 1]
                        if v == 2
                        else batch["target"]["intrinsics"][0, 0]
                    ),
                    t,
                )
                return extrinsics[None], intrinsics[None]            

            # smooth trajectory
            num_frames_traj = 20
            t = torch.linspace(0, 1, num_frames_traj, dtype=torch.float32, device=self.device)
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

            extrinsics_traj, intrinsics_traj = trajectory_fn(t)
            
            near_traj = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames_traj)
            far_traj = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames_traj)
            output_traj = self.decoder.forward(gaussians, extrinsics_traj, intrinsics_traj, near_traj, far_traj, (h, w))
            
            mesh_extractor = GaussianMeshExtractor(output_traj, intrinsics_traj, extrinsics_traj, near_traj, far_traj, scale_invariant=True)
            mesh_extractor.estimate_bounding_sphere()

            voxel_size = 0.004
            sdf_trunc = 0.016
            depth_trunc = 3.0

            ## initialize the class to estimate the bounding sphere
            mesh = mesh_extractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)

            mesh_path = path / scene / "mesh" / (scene + '_bounded.ply')
            mesh_path.parent.mkdir(exist_ok=True, parents=True)
            o3d.io.write_triangle_mesh(mesh_path, mesh)

            # post-process the mesh and save, saving the largest N clusters
            mesh_post = post_process_mesh(mesh, cluster_to_keep=1)
            mesh_path = Path(path) / scene / "mesh" / (scene + '_bounded_post.ply')
            o3d.io.write_triangle_mesh(mesh_path, mesh_post)


        if self.test_cfg.save_video:
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
            save_video(
                [a for a in output.color[0]],
                path / scene / "video" / f"{scene}_frame_{frame_str}.mp4",
            )
            
            # Save interpolation and wobble videos locally during test step
            video_save_path = path / scene / "video"
            self.render_video_interpolation(batch, save_locally=True, save_path=video_save_path)
            self.render_video_wobble(batch, save_locally=True, save_path=video_save_path)
            

        if self.test_cfg.save_compare:
            # Construct comparison image.
            context_img = inverse_normalize(batch["context"]["image"][0])
            context_img_depth = vis_depth_map(gaussian_means)
            context_img_normal = vis_normal(surf_normals_pts).permute(0, 3, 1, 2).float() / 255.0
            vis_gaps = torch.ones_like(context_img)
            context = []
            context_normals = []
            for i in range(context_img.shape[0]):
                context.append(context_img[i])
                context.append(context_img_depth[i])
                context_normals.append(context_img_normal[i])
                context_normals.append(vis_gaps[i])
        
            comparison = hcat(
                add_label(vcat(*context), "Context"),
                add_label(vcat(*context_normals), "Context Surface Normal"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_pred), "Target (Prediction)"),
                add_label(vcat(*target_rendered_depth), "Depth (Prediction)"),
                # add_label(vcat(*surface_normal), "Surface Normal (Prediction)"),
                # add_label(vcat(*render_normal), "Rendered Normal (Prediction)"),
                # add_label(vcat(*rend_dist), "Depth Distortion (Prediction)"),
                # add_label(vcat(*rend_alpha), "Alpha (Prediction)"),
            )
            save_image(comparison, path / scene / "comparisons" / f"{scene}.png")

    def test_step_align(self, batch, gaussians):
        self.encoder.eval()
        # freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        b, v, _, h, w = batch["target"]["image"].shape
        with torch.set_grad_enabled(True):
            cam_rot_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=self.device))
            cam_trans_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=self.device))

            opt_params = []
            opt_params.append(
                {
                    "params": [cam_rot_delta],
                    "lr": self.test_cfg.rot_opt_lr,
                }
            )
            opt_params.append(
                {
                    "params": [cam_trans_delta],
                    "lr": self.test_cfg.trans_opt_lr,
                }
            )
            pose_optimizer = torch.optim.Adam(opt_params)

            extrinsics = batch["target"]["extrinsics"].clone()
            with self.benchmarker.time("optimize"):
                for i in range(self.test_cfg.pose_align_steps):
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
                        decoder_type="3D",  # Always use 3D for pose refinement (only 3D renderer returns camera pose gradients)
                    )

                    # Compute and log loss.
                    total_loss = 0
                    for loss_fn in self.losses:
                        loss = loss_fn.forward(output, batch, gaussians, self.global_step)
                        total_loss = total_loss + loss

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

        # Render Gaussians.
        output = self.decoder.forward(
            gaussians,
            extrinsics,
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
        )

        return output

    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
        self.benchmarker.dump_memory(
            self.test_cfg.output_path / name / "peak_memory.json"
        )
        self.benchmarker.summarize()

    @rank_zero_only
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1
        visualization_dump = {}
        gaussians = self.encoder(
            batch["context"],
            self.global_step,
            visualization_dump=visualization_dump,
        )
        output = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            depth_mode="depth",
            decoder_type=self._decoder_type(),
        )
        rgb_pred = output.color[0]
        depth_pred = vis_depth_map(output.depth[0])

        # direct depth from gaussian means (used for visualization only)
        gaussian_means = visualization_dump["depth"][0].squeeze()      # (v, h, w)
        if gaussian_means.shape[-1] == 3:
            gaussian_means = gaussian_means.mean(dim=-1)

        # surface normals derived from pointclouds - context views
        # all_pts_depth = gaussian_means.unsqueeze(-1)   # (v, h, w, 1)
        # foc_x = batch["context"]["intrinsics"][0, 0, 0, 0] * w
        # foc_y = batch["context"]["intrinsics"][0, 0, 1, 1] * h
        # normal_pts = surface_normal_from_depth(all_pts_depth.permute(0, 3, 1, 2), focal_x=foc_x[None], focal_y=foc_y[None], 
        #                                     valid_mask=(all_pts_depth > 0).permute(0, 3, 1, 2)) 
        # 
        # all_pts3d = rearrange(gaussians.means, "b (v h w) d -> b v h w d", h=h, w=w)
        # # pts3d1 = all_pts3d[:, 0, ...]  # (B, H, W, 3)
        # # pts3d2 = all_pts3d[:, 1, ...]  # (B, H, W, 3)
        # sn_batch = []
        # for i in range(2):
        #     xyz_i = all_pts3d[0, i, ...][None]  # (B=1, H, W, 3)
        #     normal = get_surface_normal(xyz_i)    # using a smoother normal approximation
        #     sn_batch.append(normal)
        # sn_batch = torch.cat(sn_batch, dim=3).permute((3, 2, 0, 1))  # [v, c=3, h, w]
        # surf_normals_pts = sn_batch.permute(0, 2, 3, 1)   # (v, h, w, c=3)
        
        all_pts3d = rearrange(gaussians.means, "b (v h w) d -> b v h w d", h=h, w=w)
        # pts3d1 = all_pts3d[:, 0, ...]  # (B, H, W, 3)
        # pts3d2 = all_pts3d[:, 1, ...]  # (B, H, W, 3)
        surf_normals_pts, _ = points_to_normal(all_pts3d[0])   # (v, h, w, c=3)

        
        # Compute validation metrics.
        rgb_gt = batch["target"]["image"][0]
        psnr = compute_psnr(rgb_gt, rgb_pred).mean()
        self.log(f"val/psnr", psnr)
        lpips = compute_lpips(rgb_gt, rgb_pred).mean()
        self.log(f"val/lpips", lpips)
        ssim = compute_ssim(rgb_gt, rgb_pred).mean()
        self.log(f"val/ssim", ssim)

        # surface_normal = vis_normal(output.surf_normal[0].permute(0, 2, 3, 1)).permute(0, 3, 1, 2).float() / 255
        # render_normal = vis_normal(output.rend_normal[0].permute(0, 2, 3, 1)).permute(0, 3, 1, 2).float() / 255
        # rend_dist = vis_depth_map(output.dist[0])
        # rend_alpha = vis_depth_map(output.alpha[0])

        # Visualisation of gaussians orientations (predicted from context views)
        gaussian_rotations = visualization_dump["rotations"]
        gaussian_rotations = rearrange(gaussian_rotations, "b (v h w) d -> b v h w d", v=2, h=h, w=w)
        contexts_gaussian_rotations = gaussian_rotations[0]     # shape (V, H, W, 4)

        gaussian_scales = visualization_dump["scales"]
        gaussian_scales = rearrange(gaussian_scales, "b (v h w) d -> b v h w d", v=2, h=h, w=w)
        contexts_gaussian_scales = gaussian_scales[0]     # shape (V, H, W, 3)
        # sorted_contexts_gaussian_scales = torch.sort(contexts_gaussian_scales, dim=-1, descending=True)[0]
        
        
        # gaussian_opacities = visualization_dump['opacities']
        # gaussian_opacities = rearrange(gaussian_opacities, "b v h w srf s -> b v h w (srf s)", v=2, h=h, w=w)
        # contexts_gaussian_opacities = gaussian_opacities[0]     # shape (V, H, W, 1)

        # Align normals with the smallest-scale axis of each Gaussian.
        gaussian_surfels_normals = gaussian_orientation_from_scales(
            contexts_gaussian_rotations,
            contexts_gaussian_scales,
        )  # shape: (V, H, W, 3)
        # Visualize the selected normals.
        gaussian_normal_vis = vis_normal(gaussian_surfels_normals).permute(0, 3, 1, 2).float() / 255.0


        # Construct comparison image.
        context_img = inverse_normalize(batch["context"]["image"][0])
        context_img_depth = vis_depth_map(gaussian_means)
        context_img_normal = vis_normal(surf_normals_pts).permute(0, 3, 1, 2).float() / 255.0
        context = []
        context_normals = []
        for i in range(context_img.shape[0]):
            context.append(context_img[i])
            context.append(context_img_depth[i])
            context_normals.append(context_img_normal[i])
            context_normals.append(gaussian_normal_vis[i])
       
        comparison = hcat(
            add_label(vcat(*context), "Context / Ptc Depth"),
            add_label(vcat(*context_normals), "Ctx Surface / GS Normal"),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb_pred), "Target (Prediction)"),
            add_label(vcat(*depth_pred), "Rendered Depth"),
            # add_label(vcat(*surface_normal), "Surface Normal"),
            # add_label(vcat(*render_normal), "Rendered Normal"),
            # add_label(vcat(*rend_dist), "Depth Distortion"),
            # add_label(vcat(*rend_alpha), "Alpha"),
        )

        if self.distiller is not None:
            with torch.no_grad():
                pseudo_gt1, pseudo_gt2 = self.distiller(batch["context"], False)
            depth1, depth2 = pseudo_gt1['pts3d'][..., -1], pseudo_gt2['pts3d'][..., -1]
            conf1, conf2 = pseudo_gt1['conf'], pseudo_gt2['conf']
            depth_dust = torch.cat([depth1, depth2], dim=0)
            depth_dust = vis_depth_map(depth_dust)
            conf_dust = torch.cat([conf1, conf2], dim=0)
            conf_dust = confidence_map(conf_dust)
            dust_vis = torch.cat([depth_dust, conf_dust], dim=0)
            comparison = hcat(add_label(vcat(*dust_vis), "Context"), comparison)

        self.logger.log_image(
            "comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )

        # Render projections and construct projection image.
        # These are disabled for now, since RE10k scenes are effectively unbounded.
        projections = hcat(
                *render_projections(
                    gaussians,
                    256,
                    extra_label="",
                )[0]
            )
        self.logger.log_image(
            "projection",
            [prep_image(add_border(projections))],
            step=self.global_step,
        )

        # Draw cameras.
        cameras = hcat(*render_cameras(batch, 256))
        self.logger.log_image(
            "cameras", [prep_image(add_border(cameras))], step=self.global_step
        )

        if self.encoder_visualizer is not None:
            for k, image in self.encoder_visualizer.visualize(
                batch["context"], self.global_step
            ).items():
                self.logger.log_image(k, [prep_image(image)], step=self.global_step)

        # Run video validation step.
        self.render_video_interpolation(batch)
        self.render_video_wobble(batch)
        if self.train_cfg.extended_visualization:
            self.render_video_interpolation_exaggerated(batch)

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample, save_locally: bool = False, save_path: Optional[Path] = None) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60, save_locally=save_locally, save_path=save_path)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample, save_locally: bool = False, save_path: Optional[Path] = None) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb", save_locally=save_locally, save_path=save_path)

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample, save_locally: bool = False, save_path: Optional[Path] = None) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
            save_locally=save_locally,
            save_path=save_path,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
        save_locally: bool = False,
        save_path: Optional[Path] = None,
    ) -> None:
        # Render probabilistic estimate of scene.
        gaussians = self.encoder(batch["context"], self.global_step)

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output = self.decoder.forward(
            gaussians, extrinsics, intrinsics, near, far, (h, w), "depth", decoder_type=self._decoder_type()
        )
        images = [
            vcat(rgb, depth)
            for rgb, depth in zip(output.color[0], vis_depth_map(output.depth[0]))
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        
        # Save locally if requested (useful for test step)
        if save_locally and save_path is not None:
            video_path = save_path / f"{name}.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            # Convert (T, C, H, W) uint8 tensor to list of (H, W, 3) frames for moviepy.
            frames = []
            for frame in video:  # frame: (C, H, W)
                if frame.ndim != 3:
                    continue  # skip invalid frames silently
                c, fh, fw = frame.shape
                if c == 1:
                    frame_hw3 = np.repeat(frame, 3, axis=0)
                elif c >= 3:
                    frame_hw3 = frame[:3]  # take first 3 channels
                else:
                    # Unexpected channel count; pad to 3
                    pad = np.zeros((3 - c, fh, fw), dtype=frame.dtype)
                    frame_hw3 = np.concatenate([frame, pad], axis=0)
                frame_hw3 = np.transpose(frame_hw3, (1, 2, 0))  # (H, W, 3)
                frames.append(frame_hw3)
            if len(frames) > 0:
                clip = mpy.ImageSequenceClip(frames, fps=30)
                clip.write_videofile(str(video_path), logger=None)
        
        # Log to wandb if not saving locally or if in training/validation
        if not save_locally:
            visualizations = {
                f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
            }

            # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
            try:
                wandb.log(visualizations)
            except Exception:
                assert isinstance(self.logger, LocalLogger)
                for key, value in visualizations.items():
                    tensor = value._prepare_video(value.data)
                    clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
                    dir = LOG_PATH / key
                    dir.mkdir(exist_ok=True, parents=True)
                    clip.write_videofile(
                        str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                    )

    def print_preview_metrics(self, metrics: dict[str, float | Tensor], methods: list[str] | None = None, overlap_tag: str | None = None) -> None:
        if getattr(self, "running_metrics", None) is None:
            self.running_metrics = metrics
            self.running_metric_steps = 1
        else:
            s = self.running_metric_steps
            self.running_metrics = {
                k: ((s * v) + metrics[k]) / (s + 1)
                for k, v in self.running_metrics.items()
            }
            self.running_metric_steps += 1

        if overlap_tag is not None:
            if getattr(self, "running_metrics_sub", None) is None:
                self.running_metrics_sub = {overlap_tag: metrics}
                self.running_metric_steps_sub = {overlap_tag: 1}
            elif overlap_tag not in self.running_metrics_sub:
                self.running_metrics_sub[overlap_tag] = metrics
                self.running_metric_steps_sub[overlap_tag] = 1
            else:
                s = self.running_metric_steps_sub[overlap_tag]
                self.running_metrics_sub[overlap_tag] = {k: ((s * v) + metrics[k]) / (s + 1)
                                                         for k, v in self.running_metrics_sub[overlap_tag].items()}
                self.running_metric_steps_sub[overlap_tag] += 1

        metric_list = ["psnr", "lpips", "ssim"]

        def print_metrics(runing_metric, methods=None):
            table = []
            if methods is None:
                methods = ['ours']

            for method in methods:
                row = [
                    f"{runing_metric[f'{metric}_{method}']:.3f}"
                    for metric in metric_list
                ]
                table.append((method, *row))

            headers = ["Method"] + metric_list
            table = tabulate(table, headers)
            print(table)

        print("All Pairs:")
        print_metrics(self.running_metrics, methods)
        if overlap_tag is not None:
            for k, v in self.running_metrics_sub.items():
                print(f"Overlap: {k}")
                print_metrics(v, methods)

    def configure_optimizers(self):
        new_params, new_param_names = [], []
        pretrained_params, pretrained_param_names = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if "gaussian_param_head" in name or "intrinsic_encoder" in name:
                new_params.append(param)
                new_param_names.append(name)
            else:
                pretrained_params.append(param)
                pretrained_param_names.append(name)

        param_dicts = [
            {
                "params": new_params,
                "lr": self.optimizer_cfg.lr,
             },
            {
                "params": pretrained_params,
                "lr": self.optimizer_cfg.lr * self.optimizer_cfg.backbone_lr_multiplier,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.optimizer_cfg.lr, weight_decay=0.05, betas=(0.9, 0.95))
        warm_up_steps = self.optimizer_cfg.warm_up_steps
        warm_up = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            1 / warm_up_steps,
            1,
            total_iters=warm_up_steps,
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=get_cfg()["trainer"]["max_steps"], eta_min=self.optimizer_cfg.lr * 0.1)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warm_up, lr_scheduler], milestones=[warm_up_steps])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
