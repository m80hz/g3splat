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
class DatasetScannetPoseCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool
    context_pair_file: str


@dataclass
class DatasetScannetPoseCfgWrapper:
    scannet_pose: DatasetScannetPoseCfg


class DatasetScannetPose(IterableDataset):
    cfg: DatasetScannetPoseCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 100.0

    def __init__(
        self,
        cfg: DatasetScannetPoseCfg,
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
        # following BA-Net's splits
        pair_file = os.path.join(cfg.roots[0], self.cfg.context_pair_file)
        # Load the text file; each row is [scene_name, image1, image2]
        data_pairs = np.loadtxt(pair_file, delimiter=" ", dtype=str)
        
        # save the pairs path from the text file
        self.pairs = data_pairs  # Each row: [scene_name, image1, image2]

        # shape of relative pose (3, 4): [R | t]
        # dummy_rel_pose = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
        self.rel_pose = None
        

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __iter__(self):
        # When testing, the data loaders alternate data.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.pairs = [
                pair
                for pair_index, pair in enumerate(self.pairs)
                if pair_index % worker_info.num_workers == worker_info.id
            ]
            if self.rel_pose is not None:
                self.rel_pose = [
                    pose
                    for pose_index, pose in enumerate(self.rel_pose)
                    if pose_index % worker_info.num_workers == worker_info.id
                ]

        for scene_row in self.pairs:
            # scene_row is something like: ["scene0688_00", "000346", "000350"]
            scene_name = scene_row[0]  # already complete scene name
            # remove zero padding from file names
            im_A_num = int(scene_row[1])
            im_B_num = int(scene_row[2])

            # Build image file paths
            im_A_path = os.path.join(self.data_root, scene_name, "color", f"{im_A_num}.jpg")
            im_B_path = os.path.join(self.data_root, scene_name, "color", f"{im_B_num}.jpg")

            # Build the corresponding depth file paths
            depth_A_path = os.path.join(self.data_root, scene_name, "depth", f"{im_A_num}.png")
            depth_B_path = os.path.join(self.data_root, scene_name, "depth", f"{im_B_num}.png")

            # Build the corresponding pose file paths
            pose_A_path = os.path.join(self.data_root, scene_name, "pose", f"{im_A_num}.txt")
            pose_B_path = os.path.join(self.data_root, scene_name, "pose", f"{im_B_num}.txt")

            context_images = [im_A_path, im_B_path]
            context_images = self.convert_images(context_images)      # returns a tensor (2, 3, H, W)

            context_depths = [depth_A_path, depth_B_path]
            context_depths = self.convert_depths(context_depths)      # returns a tensor (2, 1, H, W)
            
            # Load the pose files (each should contain a 4x4 matrix)
            T_A = np.loadtxt(pose_A_path)  # camera-to-world transformation for image A
            T_B = np.loadtxt(pose_B_path)  # camera-to-world transformation for image B

            # Compute the relative pose from B to A is: T_rel = inv(T_A) @ T_B
            T_rel = np.linalg.inv(T_A) @ T_B

            h, w = context_images.shape[-2:]
            K = np.stack(
                [
                    np.array([float(i) for i in r.split()])
                    for r in open(osp.join(self.data_root, scene_name, "intrinsic", "intrinsic_color.txt"), "r")
                    .read()
                    .split("\n")
                    if r
                ]
            )

            h_depth, w_depth = context_depths.shape[-2:]
            K_depth = np.stack(
                [
                    np.array([float(i) for i in r.split()])
                    for r in open(osp.join(self.data_root, scene_name, "intrinsic", "intrinsic_depth.txt"), "r")
                    .read()
                    .split("\n")
                    if r
                ]
            )

            # crop the image to make the principal point in the center of the image
            def center_principal_point(image, cx, cy, h, w):
                cx = round(cx)
                cy = round(cy)

                # Calculate the desired center
                center_x, center_y = w // 2, h // 2

                # Calculate the shift needed
                shift_x = center_x - cx
                shift_y = center_y - cy

                # Calculate new image dimensions
                new_w = max(w, w - 2 * shift_x)
                new_h = max(h, h - 2 * shift_y)

                # convert to int
                new_w = round(new_w)
                new_h = round(new_h)

                # Create a new blank image
                new_image = torch.zeros((2, 3, new_h, new_w), dtype=torch.float32)

                # Calculate padding
                pad_left = max(0, -shift_x)
                pad_top = max(0, -shift_y)

                # Calculate the region of the original image to copy
                src_left = max(0, shift_x)
                src_top = max(0, shift_y)
                src_right = min(w, w + shift_x)
                src_bottom = min(h, h + shift_y)

                # Copy the shifted image to the new image
                new_image[:, :, pad_top:pad_top + src_bottom - src_top,
                pad_left:pad_left + src_right - src_left] = image[:, :, src_top:src_bottom, src_left:src_right]

                # Calculate new intrinsic parameters
                new_cx = new_w // 2
                new_cy = new_h // 2

                return new_image, new_cx, new_cy
            
            def center_principal_point_depth(depth, cx, cy, h, w):
                cx = round(cx)
                cy = round(cy)

                # Desired center coordinates for the new image.
                center_x, center_y = w // 2, h // 2

                # Calculate shifts to center the principal point.
                shift_x = center_x - cx
                shift_y = center_y - cy

                # Calculate new dimensions.
                new_w = max(w, w - 2 * shift_x)
                new_h = max(h, h - 2 * shift_y)
                new_w = round(new_w)
                new_h = round(new_h)

                # Create new depth image filled with zeros (invalid depth).
                new_depth = torch.zeros((depth.shape[0], depth.shape[1], new_h, new_w), dtype=torch.float32)

                # Compute padding and source region.
                pad_left = max(0, -shift_x)
                pad_top = max(0, -shift_y)
                src_left = max(0, shift_x)
                src_top = max(0, shift_y)
                src_right = min(w, w + shift_x)
                src_bottom = min(h, h + shift_y)

                # Copy valid region from original depth to new image.
                new_depth[:, :, pad_top:pad_top + (src_bottom - src_top),
                        pad_left:pad_left + (src_right - src_left)] = depth[:, :, src_top:src_bottom, src_left:src_right]

                # Update principal point to be at the center of the new image.
                new_cx = new_w // 2
                new_cy = new_h // 2

                return new_depth, new_cx, new_cy

            # tgt_cx, tgt_cy = w // 2, h // 2
            context_images, tgt_cx, tgt_cy = center_principal_point(
                context_images, K[0, 2], K[1, 2], h, w
            )
            K[0, 2] = tgt_cx
            K[1, 2] = tgt_cy

            h, w = context_images.shape[-2:]
            target_images = context_images.clone()

            context_depths, tgt_cx_depth, tgt_cy_depth = center_principal_point_depth(
                context_depths, K_depth[0, 2], K_depth[1, 2], h_depth, w_depth
            )
            K_depth[0, 2] = tgt_cx_depth
            K_depth[1, 2] = tgt_cy_depth

            context_valid_depths = (context_depths > 0).type(torch.float32)  # invalid depth is 0     
            
            target_depths = context_depths.clone()
            target_valid_depths = context_valid_depths.clone()
            
            # Here, we set image A's frame as the world (i.e. extrinsics for A are identity).
            extrinsic_A = torch.eye(4, dtype=torch.float32)
            extrinsic_B = torch.tensor(T_rel, dtype=torch.float32)
            extrinsics = torch.stack((extrinsic_A, extrinsic_B), dim=0)

            # pose1 = torch.eye(4)
            # pose2 = torch.eye(4)
            # pose2[:3, :4] = torch.tensor(rel_pose.reshape(3, 4)).to(torch.float32)
            # pose2 = torch.inverse(pose2)
            # extrinsics = torch.stack((pose1, pose2), dim=0)

            # normalize K
            K = K[:3, :3]
            K[0, :3] /= w
            K[1, :3] /= h

            intrinsics = torch.tensor(K, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)

            # Resize the world to make the baseline 1.
            if self.cfg.make_baseline_1:
                baseline = torch.norm(extrinsics[0, :3, 3] - extrinsics[1, :3, 3])
                scale_factor = 1.0 / baseline
                extrinsics[:, :3, 3] *= scale_factor
            else:
                scale_factor = 1.0


            overlap = torch.tensor([-1.0], dtype=torch.float32)
            scale = torch.tensor([scale_factor], dtype=torch.float32)
            context_indices = torch.tensor([0, 1], dtype=torch.int64)

            example = {
                "context": {
                    "extrinsics": extrinsics,
                    "intrinsics": intrinsics,
                    "image": context_images,
                    "depth": context_depths,
                    "valid_depth": context_valid_depths,
                    "near": self.get_bound("near", 2) * scale_factor,
                    "far": self.get_bound("far", 2) * scale_factor,
                    "index": context_indices,
                    "overlap": overlap,
                    "scale": scale,
                },
                "target": {
                    "extrinsics": extrinsics,
                    "intrinsics": intrinsics,
                    "image": target_images,
                    "depth": target_depths,
                    "valid_depth": target_valid_depths,
                    "near": self.get_bound("near", 2) * scale_factor,
                    "far": self.get_bound("far", 2) * scale_factor,
                    "index": context_indices,
                },
                "scene": f"{scene_name}_{context_indices[0]}_{context_indices[1]}",
            }
            yield apply_crop_shim(example, tuple(self.cfg.input_image_shape))

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
