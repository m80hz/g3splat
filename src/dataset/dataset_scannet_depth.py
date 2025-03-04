import json
import os
import os.path as osp

from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms as tf
import numpy as np
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset
from torch.distributed import get_rank, get_world_size

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler


@dataclass
class DatasetScannetDepthCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    skip_bad_shape: bool


@dataclass
class DatasetScannetDepthCfgWrapper:
    scannet_depth: DatasetScannetDepthCfg


class DatasetScannetDepth(IterableDataset):
    cfg: DatasetScannetDepthCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.5
    far: float = 10.0

    def __init__(
        self,
        cfg: DatasetScannetDepthCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        # Collect data.
        self.data_root = cfg.roots[0]

        # List scene directories (names starting with "scene")
        self.scenes = sorted([
            os.path.join(self.data_root, d)
            for d in os.listdir(self.data_root)
            if os.path.isdir(os.path.join(self.data_root, d)) and d.startswith("scene")
        ])
        
    def load_intrinsics(self, scene_path, modality="color"):
        """
        Load intrinsic parameters from the scene's intrinsic folder.
        Expects a file named "intrinsic_color.txt" or "intrinsic_depth.txt".
        """
        intrinsic_file = os.path.join(scene_path, "intrinsic", f"intrinsic_{modality}.txt")
        with open(intrinsic_file, "r") as f:
            lines = f.readlines()
        K = np.stack([np.array([float(i) for i in r.split()]) for r in lines if r.strip() != ""])
        return K  # Expected shape (4,4)

    def load_pose(self, scene_path, index):
        """Load the pose (extrinsics) for a given image index as a 4x4 matrix from the pose folder."""
        pose_file = os.path.join(scene_path, "pose", f"{index}.txt")
        pose = np.loadtxt(pose_file)
        return pose

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]
    
    def center_principal_point(self, image, cx, cy, h, w):
        """
        Shift the image (a tensor of shape (N, C, H, W)) so that the principal point (cx, cy)
        is centered in the output image.
        Returns the shifted image and the new principal point coordinates.
        """
        cx = round(cx)
        cy = round(cy)
        center_x, center_y = w // 2, h // 2
        shift_x = center_x - cx
        shift_y = center_y - cy
        new_w = max(w, w - 2 * shift_x)
        new_h = max(h, h - 2 * shift_y)
        new_w = round(new_w)
        new_h = round(new_h)
        new_image = torch.zeros((image.shape[0], image.shape[1], new_h, new_w), dtype=torch.float32)
        pad_left = max(0, -shift_x)
        pad_top = max(0, -shift_y)
        src_left = max(0, shift_x)
        src_top = max(0, shift_y)
        src_right = min(w, w + shift_x)
        src_bottom = min(h, h + shift_y)
        new_image[:, :, pad_top:pad_top + (src_bottom - src_top),
                  pad_left:pad_left + (src_right - src_left)] = image[:, :, src_top:src_bottom, src_left:src_right]
        new_cx = new_w // 2
        new_cy = new_h // 2
        return new_image, new_cx, new_cy

    def center_principal_point_depth(self, depth, cx, cy, h, w):
        """
        Similar to center_principal_point but for depth maps.
        Expects depth tensor of shape (N, 1, H, W).
        """
        cx = round(cx)
        cy = round(cy)
        center_x, center_y = w // 2, h // 2
        shift_x = center_x - cx
        shift_y = center_y - cy
        new_w = max(w, w - 2 * shift_x)
        new_h = max(h, h - 2 * shift_y)
        new_w = round(new_w)
        new_h = round(new_h)
        new_depth = torch.zeros((depth.shape[0], depth.shape[1], new_h, new_w), dtype=torch.float32)
        pad_left = max(0, -shift_x)
        pad_top = max(0, -shift_y)
        src_left = max(0, shift_x)
        src_top = max(0, shift_y)
        src_right = min(w, w + shift_x)
        src_bottom = min(h, h + shift_y)
        new_depth[:, :, pad_top:pad_top + (src_bottom - src_top),
                  pad_left:pad_left + (src_right - src_left)] = depth[:, :, src_top:src_bottom, src_left:src_right]
        new_cx = new_w // 2
        new_cy = new_h // 2
        return new_depth, new_cx, new_cy


    def __iter__(self):
        # Loop over each scene.
        for scene_path in self.scenes:
            scene_name = os.path.basename(scene_path)
            color_dir = os.path.join(scene_path, "color")
            # Extract image indices from filenames (e.g., "105.jpg")
            files = [f for f in os.listdir(color_dir) if f.endswith(".jpg")]
            indices = []
            for f in files:
                base = os.path.splitext(f)[0]
                try:
                    num = int(base)
                    indices.append(num)
                except ValueError:
                    continue
            indices = sorted(indices)
            if len(indices) < 5:
                print(f"Skipping {scene_name}: only {len(indices)} images (need at least 5).")
                continue


            # Load intrinsic parameters for color and depth (original copies).
            K_color_orig = self.load_intrinsics(scene_path, modality="color")  # shape (4,4)
            K_depth_orig = self.load_intrinsics(scene_path, modality="depth")  # shape (4,4)

            # Create separate copies for context and target.
            K_color_context = K_color_orig.copy()
            K_color_target  = K_color_orig.copy()
            K_depth_context = K_depth_orig.copy()
            K_depth_target  = K_depth_orig.copy()

            # Generate 3 examples per scene using a sliding window of 5 consecutive images.
            num_examples = 3
            max_start = len(indices) - 5
            starts = np.linspace(0, max_start, num=num_examples, dtype=int)

            for start in starts:
                window_indices = indices[start: start + 5]
                # For each window: use the middle two as context and one before plus two after as targets.
                context_indices = window_indices[1:3]       # 2 images
                target_indices = [window_indices[0], window_indices[3], window_indices[4]]  # 3 images

                # Build file paths.
                context_color_files = [os.path.join(scene_path, "color", f"{idx}.jpg") for idx in context_indices]
                target_color_files = [os.path.join(scene_path, "color", f"{idx}.jpg") for idx in target_indices]
                context_depth_files = [os.path.join(scene_path, "depth", f"{idx}.png") for idx in context_indices]
                target_depth_files = [os.path.join(scene_path, "depth", f"{idx}.png") for idx in target_indices]

                # Load color and depth data.
                context_images = self.convert_images(context_color_files)   # (2, 3, H, W)
                target_images = self.convert_images(target_color_files)     # (3, 3, H, W)
                context_depths = self.convert_depths(context_depth_files)     # (2, 1, H_d, W_d)
                target_depths = self.convert_depths(target_depth_files)       # (3, 1, H_d, W_d)

                # Load poses.
                context_pose = [self.load_pose(scene_path, idx) for idx in context_indices]
                target_pose = [self.load_pose(scene_path, idx) for idx in target_indices]
                context_pose = torch.tensor(context_pose, dtype=torch.float32)  # (2, 4, 4)
                target_pose = torch.tensor(target_pose, dtype=torch.float32)    # (3, 4, 4)

                # --- Process color intrinsics and images separately ---
                # For context images:
                h_ctx, w_ctx = context_images.shape[-2:]
                context_images, new_cx_ctx, new_cy_ctx = self.center_principal_point(context_images,
                                                                                     K_color_context[0, 2],
                                                                                     K_color_context[1, 2],
                                                                                     h_ctx, w_ctx)
                K_color_context[0, 2] = new_cx_ctx
                K_color_context[1, 2] = new_cy_ctx

                # For target images:
                h_t, w_t = target_images.shape[-2:]
                target_images, new_cx_t, new_cy_t = self.center_principal_point(target_images,
                                                                                 K_color_target[0, 2],
                                                                                 K_color_target[1, 2],
                                                                                 h_t, w_t)
                K_color_target[0, 2] = new_cx_t
                K_color_target[1, 2] = new_cy_t

                # --- Process depth intrinsics and depths separately ---
                h_d_ctx, w_d_ctx = context_depths.shape[-2:]
                context_depths, new_cx_d_ctx, new_cy_d_ctx = self.center_principal_point_depth(context_depths,
                                                                                                K_depth_context[0, 2],
                                                                                                K_depth_context[1, 2],
                                                                                                h_d_ctx, w_d_ctx)
                K_depth_context[0, 2] = new_cx_d_ctx
                K_depth_context[1, 2] = new_cy_d_ctx

                h_d_t, w_d_t = target_depths.shape[-2:]
                target_depths, new_cx_d_t, new_cy_d_t = self.center_principal_point_depth(target_depths,
                                                                                           K_depth_target[0, 2],
                                                                                           K_depth_target[1, 2],
                                                                                           h_d_t, w_d_t)
                K_depth_target[0, 2] = new_cx_d_t
                K_depth_target[1, 2] = new_cy_d_t
              
                # Compute valid depth masks.
                context_valid_depths = (context_depths > 0).type(torch.float32)
                target_valid_depths = (target_depths > 0).type(torch.float32)

            
                # --- Transform poses so that the first context view becomes the world frame ---
                first_context_pose = context_pose[0]
                C0_inv = torch.inverse(first_context_pose)
                new_context_pose = torch.stack([C0_inv @ p for p in context_pose], dim=0)
                new_target_pose = torch.stack([C0_inv @ p for p in target_pose], dim=0)
                
                # --- Normalize intrinsics ---
                # Normalize using the context image dimensions for context intrinsics,
                # and target image dimensions for target intrinsics.
                K_norm_ctx = K_color_context.copy()
                K_norm_ctx = K_norm_ctx[:3, :3]
                K_norm_ctx[0, :3] /= w_ctx
                K_norm_ctx[1, :3] /= h_ctx
                intrinsics_context = torch.tensor(K_norm_ctx, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)

                K_norm_t = K_color_target.copy()
                K_norm_t = K_norm_t[:3, :3]
                K_norm_t[0, :3] /= w_t
                K_norm_t[1, :3] /= h_t
                intrinsics_target = torch.tensor(K_norm_t, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)

                # --- Additional parameters ---
                overlap = torch.tensor([0.5], dtype=torch.float32)
                scale = torch.tensor([1.0], dtype=torch.float32)
                context_idx_tensor = torch.tensor([0, 1], dtype=torch.int64)
                target_idx_tensor = torch.tensor([0, 1, 2], dtype=torch.int64)

                # --- Build final example ---
                example = {
                    "context": {
                        "extrinsics": new_context_pose,        # (2, 4, 4)
                        "intrinsics": intrinsics_context,      # (2, 3, 3)
                        "image": context_images,               # (2, 3, H, W)
                        "depth": context_depths,               # (2, 1, H_d, W_d)
                        "valid_depth": context_valid_depths,   # (2, 1, H_d, W_d)
                        "near": self.get_bound("near", 2),
                        "far": self.get_bound("far", 2),
                        "index": context_idx_tensor,
                        "overlap": overlap,
                        "scale": scale,
                    },
                    "target": {
                        "extrinsics": new_target_pose,           # (3, 4, 4)
                        "intrinsics": intrinsics_target,         # (3, 3, 3)
                        "image": target_images,                  # (3, 3, H, W)
                        "depth": target_depths,                  # (3, 1, H_d, W_d)
                        "valid_depth": target_valid_depths,      # (3, 1, H_d, W_d)
                        "near": self.get_bound("near", 3),
                        "far": self.get_bound("far", 3),
                        "index": target_idx_tensor,
                    },
                    "scene": scene_name,
                }

                example = apply_crop_shim(example, tuple(self.cfg.input_image_shape))
                yield example


    def convert_images(
        self,
        images,
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(image)
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def convert_depths(
        self,
        depths,
    ) -> Float[Tensor, "batch 1 height width"]:
        torch_depths = []
        for depth in depths:
            depth = np.array(Image.open(depth)).astype(np.float32) / 1000.0
            depth[~np.isfinite(depth)] = 0  # invalid
            torch_depths.append(self.to_tensor(depth))
        return torch.stack(torch_depths)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage
