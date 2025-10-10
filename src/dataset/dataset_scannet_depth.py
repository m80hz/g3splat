import json
import os
import os.path as osp
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torchvision.transforms as tf
from einops import repeat
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

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
    context_pair_file: str
    num_target_views: int


@dataclass
class DatasetScannetDepthCfgWrapper:
    scannet_depth: DatasetScannetDepthCfg


class DatasetScannetDepth(IterableDataset):
    cfg: DatasetScannetDepthCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
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
        self.data_root = cfg.roots[0]


        # List scene directories (names starting with "scene")
        self.scenes = sorted([
            os.path.join(self.data_root, d)
            for d in os.listdir(self.data_root)
            if osp.isdir(osp.join(self.data_root, d)) and d.startswith("scene")
        ])

        # Load context pairs from the provided text file.
        # Each line is expected to have: scene_name context1 context2
        context_pair_file_path = os.path.join(self.data_root, self.cfg.context_pair_file)
        self.context_pairs = {}
        with open(context_pair_file_path, "r") as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    scene = parts[0]
                    try:
                        c1 = int(parts[1])
                        c2 = int(parts[2])
                    except ValueError:
                        continue
                    # Ensure c1 is the smaller one.
                    if c1 > c2:
                        c1, c2 = c2, c1
                    if scene not in self.context_pairs:
                        self.context_pairs[scene] = []
                    self.context_pairs[scene].append((c1, c2))

        # Build evaluation indices as a list of examples.
        # Each example is a dict with "scene", "context": [c1, c2], and "target": [target indices]
        self.eval_indices = []
        for scene_path in self.scenes:
            scene_name = osp.basename(scene_path)
            if scene_name not in self.context_pairs:
                continue  # No context pairs for this scene.
            # List available image indices in the color folder.
            color_dir = osp.join(scene_path, "color")
            files = [f for f in os.listdir(color_dir) if f.endswith(".jpg")]
            available = []
            for f in files:
                base = osp.splitext(f)[0]
                try:
                    available.append(int(base))
                except ValueError:
                    continue
            available = sorted(available)
            # For each context pair for the scene:
            for (c1, c2) in self.context_pairs[scene_name]:
                if c1 not in available or c2 not in available:
                    print(f"Skipping {scene_name} pair ({c1}, {c2}): not found in available images.")
                    continue
                # Candidate target indices are those strictly between c1 and c2.
                candidates = [idx for idx in available if c1 < idx < c2]
                if len(candidates) < self.cfg.num_target_views:
                    print(f"Skipping {scene_name} pair ({c1}, {c2}): only {len(candidates)} candidates (need at least {self.cfg.num_target_views}).")
                    continue
                # Uniformly select target indices.
                linspace = np.linspace(0, len(candidates) - 1, self.cfg.num_target_views)
                target_idxs = [candidates[int(round(i))] for i in linspace]
                self.eval_indices.append({
                    "scene": scene_name,
                    "context": [c1, c2],
                    "target": target_idxs
                })
                
        print(f"Total examples in eval_indices: {len(self.eval_indices)}")

        # Save the evaluation indices as a JSON file.
        eval_index_file = osp.join("evaluation_index_scannetv1.json")
        with open(eval_index_file, "w") as f:
            json.dump(self.eval_indices, f, indent=4)
        print(f"Saved evaluation indices to {eval_index_file}")

    def load_intrinsics(self, scene_path, modality="color"):
        intrinsic_file = osp.join(scene_path, "intrinsic", f"intrinsic_{modality}.txt")
        with open(intrinsic_file, "r") as f:
            lines = f.readlines()
        K = np.stack([np.array([float(i) for i in r.split()]) for r in lines if r.strip() != ""])
        return K  # Expected shape (4,4)

    def load_pose(self, scene_path, index):
        pose_file = osp.join(scene_path, "pose", f"{index}.txt")
        pose = np.loadtxt(pose_file)
        return pose

    def center_principal_point(self, image, cx, cy, h, w):
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

    def get_bound(self, bound: Literal["near", "far"], num_views: int) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    def convert_images(self, images) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            img = Image.open(image).convert("RGB")
            torch_images.append(self.to_tensor(img))
        return torch.stack(torch_images)

    def convert_depths(self, depths) -> Float[Tensor, "batch 1 height width"]:
        torch_depths = []
        for depth in depths:
            d = np.array(Image.open(depth)).astype(np.float32) / 1000.0
            d[~np.isfinite(d)] = 0
            d = Image.fromarray(d)
            torch_depths.append(self.to_tensor(d))
        return torch.stack(torch_depths)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Divide the dataset among workers
            total_examples = len(self.eval_indices)
            per_worker = int(np.ceil(total_examples / worker_info.num_workers))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, total_examples)
            eval_indices = self.eval_indices[start:end]
        else:
            eval_indices = self.eval_indices

        # Now iterate over each example in the evaluation indices list.
        for example_dict in eval_indices:
            scene_name = example_dict["scene"]
            context_indices = example_dict["context"]  # e.g. [346, 350]
            target_indices = example_dict["target"]      # uniformly chosen targets

            scene_path = osp.join(self.data_root, scene_name)
            color_dir = osp.join(scene_path, "color")
            depth_dir = osp.join(scene_path, "depth")
            context_color_files = [osp.join(scene_path, "color", f"{idx}.jpg") for idx in context_indices]
            target_color_files = [osp.join(scene_path, "color", f"{idx}.jpg") for idx in target_indices]
            context_depth_files = [osp.join(scene_path, "depth", f"{idx}.png") for idx in context_indices]
            target_depth_files = [osp.join(scene_path, "depth", f"{idx}.png") for idx in target_indices]

            context_images = self.convert_images(context_color_files)  # (2, 3, H, W)
            target_images = self.convert_images(target_color_files)    # (num_target, 3, H, W)
            context_depths = self.convert_depths(context_depth_files)    # (2, 1, H_d, W_d)
            target_depths = self.convert_depths(target_depth_files)      # (num_target, 1, H_d, W_d)

            context_pose = [self.load_pose(scene_path, idx) for idx in context_indices]
            target_pose = [self.load_pose(scene_path, idx) for idx in target_indices]
            context_pose = torch.tensor(context_pose, dtype=torch.float32)  # (2, 4, 4)
            target_pose = torch.tensor(target_pose, dtype=torch.float32)    # (num_target, 4, 4)

            # Process intrinsics separately for context and target.
            K_color_orig = self.load_intrinsics(scene_path, modality="color")  # (4,4)
            K_depth_orig = self.load_intrinsics(scene_path, modality="depth")  # (4,4)
            K_color_context = K_color_orig.copy()
            K_color_target  = K_color_orig.copy()
            K_depth_context = K_depth_orig.copy()
            K_depth_target  = K_depth_orig.copy()

            h_ctx, w_ctx = context_images.shape[-2:]
            context_images, new_cx_ctx, new_cy_ctx = self.center_principal_point(
                context_images, K_color_context[0, 2], K_color_context[1, 2], h_ctx, w_ctx)
            K_color_context[0, 2] = new_cx_ctx
            K_color_context[1, 2] = new_cy_ctx

            h_t, w_t = target_images.shape[-2:]
            target_images, new_cx_t, new_cy_t = self.center_principal_point(
                target_images, K_color_target[0, 2], K_color_target[1, 2], h_t, w_t)
            K_color_target[0, 2] = new_cx_t
            K_color_target[1, 2] = new_cy_t

            h_d_ctx, w_d_ctx = context_depths.shape[-2:]
            context_depths, new_cx_d_ctx, new_cy_d_ctx = self.center_principal_point_depth(
                context_depths, K_depth_context[0, 2], K_depth_context[1, 2], h_d_ctx, w_d_ctx)
            K_depth_context[0, 2] = new_cx_d_ctx
            K_depth_context[1, 2] = new_cy_d_ctx

            h_d_t, w_d_t = target_depths.shape[-2:]
            target_depths, new_cx_d_t, new_cy_d_t = self.center_principal_point_depth(
                target_depths, K_depth_target[0, 2], K_depth_target[1, 2], h_d_t, w_d_t)
            K_depth_target[0, 2] = new_cx_d_t
            K_depth_target[1, 2] = new_cy_d_t

            context_valid_depths = (context_depths > 0).type(torch.float32)
            target_valid_depths = (target_depths > 0).type(torch.float32)

            # Transform poses so that the first context becomes the world frame.
            first_context_pose = context_pose[0]
            C0_inv = torch.inverse(first_context_pose)
            new_context_pose = torch.stack([C0_inv @ p for p in context_pose], dim=0)
            new_target_pose = torch.stack([C0_inv @ p for p in target_pose], dim=0)

            K_norm_ctx = K_color_context.copy()[:3, :3]
            K_norm_ctx[0, :3] /= w_ctx
            K_norm_ctx[1, :3] /= h_ctx
            intrinsics_context = torch.tensor(K_norm_ctx, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
            K_norm_t = K_color_target.copy()[:3, :3]
            K_norm_t[0, :3] /= w_t
            K_norm_t[1, :3] /= h_t
            intrinsics_target = torch.tensor(K_norm_t, dtype=torch.float32).unsqueeze(0).repeat(len(target_indices), 1, 1)

            # Resize the world to make the baseline 1.
            if self.cfg.make_baseline_1:
                baseline = torch.norm(new_context_pose[0, :3, 3] - new_context_pose[1, :3, 3])
                scale_factor = 1.0 / baseline
                new_context_pose[:, :3, 3] *= scale_factor
                new_target_pose[:, :3, 3] *= scale_factor
            else:
                scale_factor = 1.0

            overlap = torch.tensor([-1.0], dtype=torch.float32)
            scale = torch.tensor([scale_factor], dtype=torch.float32)
            context_idx_tensor = torch.tensor([0, 1], dtype=torch.int64)
            target_idx_tensor = torch.tensor(list(range(len(target_indices))), dtype=torch.int64)

            example = {
                "context": {
                    "extrinsics": new_context_pose,
                    "intrinsics": intrinsics_context,
                    "image": context_images,
                    "depth": context_depths,
                    "valid_depth": context_valid_depths,
                    "near": self.get_bound("near", 2) * scale_factor,
                    "far": self.get_bound("far", 2) * scale_factor,
                    "index": context_idx_tensor,
                    "overlap": overlap,
                    "scale": scale,
                },
                "target": {
                    "extrinsics": new_target_pose,
                    "intrinsics": intrinsics_target,
                    "image": target_images,
                    "depth": target_depths,
                    "valid_depth": target_valid_depths,
                    "near": self.get_bound("near", len(target_indices)) * scale_factor,
                    "far": self.get_bound("far", len(target_indices)) * scale_factor,
                    "index": target_idx_tensor,
                },
                "scene": f"{scene_name}_{context_indices[0]}_{context_indices[1]}",
            }

            example = apply_crop_shim(example, tuple(self.cfg.input_image_shape))
            yield example

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage