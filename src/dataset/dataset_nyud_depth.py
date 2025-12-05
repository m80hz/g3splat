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
class DatasetNYUDepthCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    skip_bad_shape: bool
    # num_target_views: int


@dataclass
class DatasetNYUDepthCfgWrapper:
    nyud_depth: DatasetNYUDepthCfg


# Used for single-view depth evaluation on context
class DatasetNYUDepth(IterableDataset):
    cfg: DatasetNYUDepthCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    near: float = 0.1
    far: float = 10.0

    def __init__(
        self,
        cfg: DatasetNYUDepthCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        self.data_root = cfg.roots[0]
        orig_h, orig_w = self.cfg.original_image_shape
        # Normalized crop bounds (top, bottom, left, right) to ignore edges in NYU depth evaluation.
        self.crop_bounds_norm = (
            45 / orig_h,
            471 / orig_h,
            41 / orig_w,
            601 / orig_w,
        )

        # New structure: single root with color/depth/intrinsic_color.txt
        # Build eval indices from all color images under root/color.
        color_root = osp.join(self.data_root, "color")
        depth_root = osp.join(self.data_root, "depth")
        if not osp.isdir(color_root):
            raise FileNotFoundError(f"Color directory not found: {color_root}")
        if not osp.isdir(depth_root):
            raise FileNotFoundError(f"Depth directory not found: {depth_root}")

        color_files = sorted([
            f for f in os.listdir(color_root)
            if f.lower().endswith(".png")
        ])

        self.eval_indices = []
        base_scene_name = osp.basename(self.data_root)
        for fname in color_files:
            stem, _ = osp.splitext(fname)
            # build a unique scene name per frame, e.g. <scene>_<frameid>
            scene_name = f"{base_scene_name}_{stem}"
            # assume matching depth file exists under depth/<stem>.png
            # Use 2 context views (duplicated) and 1 target view.
            self.eval_indices.append({
                "scene": scene_name,
                "context": [stem, stem],   # duplicate as two contexts
                "target": [stem],          # single target view
            })

        print(f"Total examples in eval_indices: {len(self.eval_indices)} from {color_root}")

    def load_intrinsics(self, scene_path):
        """Load 3x3 color intrinsics matrix from intrinsic_color.txt at root (not normalized)."""
        intrinsic_file = osp.join(scene_path, "intrinsic_color.txt")
        if not osp.isfile(intrinsic_file):
            raise FileNotFoundError(f"intrinsic_color.txt not found for scene/root: {scene_path}")
        K = np.loadtxt(intrinsic_file).astype(np.float32)
        if K.shape == (3, 3):
            return K
        # Fallback: try to reshape if provided line-by-line
        K = np.array(K, dtype=np.float32).reshape(3, 3)
        return K

    def load_pose(self, scene_path, index=None):
        """Single-view: no poses available; use identity extrinsics."""
        return np.eye(4, dtype=np.float32)

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
            d = np.array(Image.open(depth)).astype(np.float32) / 5000.0
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
        color_root = osp.join(self.data_root, "color")
        depth_root = osp.join(self.data_root, "depth")

        for example_dict in eval_indices:
            scene_name = example_dict["scene"]
            context_indices = example_dict["context"]  # list of strings (len=2)
            target_indices = example_dict["target"]    # list of strings (len=1)
            scene_path = self.data_root

            # Resolve file paths with new flat structure
            context_color_files = [osp.join(color_root, f"{idx}.png") for idx in context_indices]
            target_color_files = [osp.join(color_root, f"{idx}.png") for idx in target_indices]
            context_depth_files = [osp.join(depth_root, f"{idx}.png") for idx in context_indices]
            target_depth_files = [osp.join(depth_root, f"{idx}.png") for idx in target_indices]

            # Load RGB and depth
            context_images = self.convert_images(context_color_files)  # (2, 3, H, W)
            target_images = self.convert_images(target_color_files)    # (1, 3, H, W)
            context_depths = self.convert_depths(context_depth_files)  # (2, 1, H_d, W_d)
            target_depths = self.convert_depths(target_depth_files)    # (1, 1, H_d, W_d)

            # Extrinsics: identity for both context and targets
            context_pose = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
            target_pose = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(1, 1, 1)

            # Intrinsics (same for context/target), loaded from intrinsic_color.txt
            K_orig = self.load_intrinsics(scene_path)  # (3,3)
            K_color_context = K_orig.copy()
            K_color_target = K_orig.copy()
            K_depth_context = K_orig.copy()
            K_depth_target = K_orig.copy()

            # Center principal point for images/depths using their respective intrinsics
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

            # Valid depth masks
            context_valid_depths = (context_depths > 0).type(torch.float32)
            target_valid_depths = (target_depths > 0).type(torch.float32)

            # Misc fields to match the evaluator expectations
            overlap = torch.tensor([0.5], dtype=torch.float32)
            scale = torch.tensor([1.0], dtype=torch.float32)
            context_idx_tensor = torch.tensor([0, 1], dtype=torch.int64)
            target_idx_tensor = torch.tensor([0], dtype=torch.int64)

            # Normalize intrinsics by image size (like ScanNet loader does)
            K_norm_ctx = K_color_context[:3, :3].copy()
            K_norm_ctx[0, :3] /= w_ctx
            K_norm_ctx[1, :3] /= h_ctx
            intrinsics_context = torch.tensor(K_norm_ctx, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)

            K_norm_t = K_color_target[:3, :3].copy()
            K_norm_t[0, :3] /= w_t
            K_norm_t[1, :3] /= h_t
            intrinsics_target = torch.tensor(K_norm_t, dtype=torch.float32).unsqueeze(0).repeat(1, 1, 1)

            example = {
                "context": {
                    "extrinsics": context_pose,
                    "intrinsics": intrinsics_context,
                    "image": context_images,
                    "depth": context_depths,
                    "valid_depth": context_valid_depths,
                    "near": self.get_bound("near", 2),
                    "far": self.get_bound("far", 2),
                    "index": context_idx_tensor,
                    "overlap": overlap,
                    "scale": scale,
                },
                "target": {
                    "extrinsics": target_pose,
                    "intrinsics": intrinsics_target,
                    "image": target_images,
                    "depth": target_depths,
                    "valid_depth": target_valid_depths,
                    "near": self.get_bound("near", len(target_indices)),
                    "far": self.get_bound("far", len(target_indices)),
                    "index": target_idx_tensor,
                },
                "scene": f"{scene_name}",
            }

            example = apply_crop_shim(example, tuple(self.cfg.input_image_shape))
            example = self.apply_depth_crop_mask(example)
            yield example

    def apply_depth_crop_mask(self, example: dict) -> dict:
        """Apply NYU depth evaluation crop by zeroing valid_depth outside region."""
        top_n, bot_n, left_n, right_n = self.crop_bounds_norm

        def apply_mask(tensor: Tensor) -> Tensor:
            mask = torch.zeros_like(tensor)
            _, _, h, w = tensor.shape
            top = int(round(top_n * h))
            bottom = int(round(bot_n * h))
            left = int(round(left_n * w))
            right = int(round(right_n * w))
            mask[..., top:bottom, left:right] = 1.0
            return mask

        ctx_mask = apply_mask(example["context"]["valid_depth"])
        tgt_mask = apply_mask(example["target"]["valid_depth"])

        example["context"]["valid_depth"] = example["context"]["valid_depth"] * ctx_mask
        example["context"]["depth"] = example["context"]["depth"] * ctx_mask
        example["target"]["valid_depth"] = example["target"]["valid_depth"] * tgt_mask
        example["target"]["depth"] = example["target"]["depth"] * tgt_mask
        return example

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage