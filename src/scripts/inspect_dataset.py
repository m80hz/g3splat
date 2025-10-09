"""
Inspect a RE10K/DTU-like dataset after dataloader processing.

This script constructs DatasetRE10k with an evaluation view sampler using a
JSON index (e.g., assets/evaluation_index_re10k.json), iterates a few examples,
and saves the processed example content to the output directory for inspection.

Saved outputs per example:
- Images: context/ and target/ as PNGs.
- Depths: saved as colorized PNGs via vis_depth_map (depth_color_XXX.png).
- Non-image tensors (intrinsics, extrinsics, near, far, indices, overlap): saved as JSON.

Usage (run from repo root):
  python -m src.scripts.inspect_dataset \
    --roots datasets/re10k \
    --index assets/evaluation_index_re10k.json \
    --output outputs/inspect_re10k \
    --num 2

For DTU converted into RE10k-like format:
  python -m src.scripts.inspect_dataset \
    --roots /path/to/converted_dtu_root \
    --index assets/evaluation_index_dtu.json \
    --original-shape 512 640 \
    --output outputs/inspect_dtu \
    --num 2
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from ..dataset.dataset_re10k import DatasetRE10k, DatasetRE10kCfg
from ..dataset.view_sampler import get_view_sampler
from ..dataset.view_sampler.view_sampler_evaluation import (
    ViewSamplerEvaluationCfg,
)
from ..dataset.types import Stage
from ..misc.image_io import save_image
from ..misc.utils import vis_depth_map


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect processed dataset examples")
    p.add_argument(
        "--roots",
        type=str,
        nargs="+",
        required=True,
        help="One or more dataset roots containing <stage>/*.torch and index.json",
    )
    p.add_argument(
        "--index",
        type=str,
        required=True,
        help="Evaluation index JSON (e.g., assets/evaluation_index_dtu.json)",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to save inspection outputs",
    )
    p.add_argument(
        "--num",
        type=int,
        default=2,
        help="Number of examples to save",
    )
    p.add_argument(
        "--stage",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset stage to iterate (default: test)",
    )
    p.add_argument(
        "--input-shape",
        type=int,
        nargs=2,
        default=[256, 256],
        metavar=("H", "W"),
        help="Model input image shape after crop (H W)",
    )
    p.add_argument(
        "--original-shape",
        type=int,
        nargs=2,
        default=[360, 640],
        metavar=("H", "W"),
        help="Original image shape before crop (H W)",
    )
    p.add_argument(
        "--num-context-views",
        type=int,
        default=2,
        help="Number of context views expected by evaluation sampler",
    )
    # Defaults: True, with paired flags to disable.
    p.add_argument(
        "--make-baseline-1",
        dest="make_baseline_1",
        action="store_true",
        default=True,
        help="Enable baseline normalization to 1 (default: on)",
    )
    p.add_argument(
        "--no-make-baseline-1",
        dest="make_baseline_1",
        action="store_false",
        help="Disable baseline normalization to 1",
    )
    p.add_argument(
        "--relative-pose",
        dest="relative_pose",
        action="store_true",
        default=True,
        help="Use relative pose normalization (default: on)",
    )
    p.add_argument(
        "--no-relative-pose",
        dest="relative_pose",
        action="store_false",
        help="Disable relative pose normalization",
    )
    p.add_argument(
        "--skip-bad-shape",
        dest="skip_bad_shape",
        action="store_true",
        default=True,
        help="Skip examples whose images deviate from original_shape (default: on)",
    )
    p.add_argument(
        "--no-skip-bad-shape",
        dest="skip_bad_shape",
        action="store_false",
        help="Do not skip examples whose images deviate from original_shape",
    )
    p.add_argument(
        "--scenes",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of scene keys to include (others skipped)",
    )
    return p.parse_args()


def build_cfg(
    roots: List[str],
    input_shape: List[int],
    original_shape: List[int],
    make_baseline_1: bool,
    relative_pose: bool,
    skip_bad_shape: bool,
    index_path: str,
    num_context_views: int,
    stage: Stage,
) -> tuple[DatasetRE10kCfg, object]:
    # Dataset config roughly mirrors config/dataset/re10k.yaml & base_dataset.yaml
    ds_cfg = DatasetRE10kCfg(
        name="re10k",
        roots=[Path(r) for r in roots],
        baseline_min=1e-3,
        baseline_max=1e10,
        max_fov=100.0,
        make_baseline_1=make_baseline_1,
        augment=False if stage != "train" else True,
        relative_pose=relative_pose,
        skip_bad_shape=skip_bad_shape,
        original_image_shape=list(original_shape),
        input_image_shape=list(input_shape),
        background_color=[0.0, 0.0, 0.0],
        cameras_are_circular=False,
        overfit_to_scene=None,
        view_sampler=None,  # populated below
    )

    # Evaluation sampler uses a precomputed index.
    vs_cfg = ViewSamplerEvaluationCfg(
        name="evaluation",
        index_path=Path(index_path),
        num_context_views=num_context_views,
    )
    ds_cfg.view_sampler = vs_cfg  # type: ignore
    return ds_cfg, vs_cfg


def save_numpy(path: Path, array: np.ndarray) -> None:
    """Save numpy arrays to .npy (used for image-like tensors such as depth)."""
    path.parent.mkdir(exist_ok=True, parents=True)
    np.save(path, array)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def save_example(out_dir: Path, example: dict, idx: int) -> None:
    scene = example["scene"]
    scene_dir = out_dir / f"{scene}__sample_{idx:03d}"
    scene_dir.mkdir(exist_ok=True, parents=True)

    # Save raw example dict for maximal fidelity.
    torch.save(example, scene_dir / "example.pt")

    def save_views(tag: str, views: dict):
        vdir = scene_dir / tag
        vdir.mkdir(exist_ok=True, parents=True)
        images = views["image"]  # (V, 3, H, W)
        extr = views["extrinsics"]  # (V, 4, 4)
        intr = views["intrinsics"]  # (V, 3, 3)
        near = views.get("near", None)  # (V,)
        far = views.get("far", None)  # (V,)
        indices = views.get("index", None)  # (V,)
        overlap = views.get("overlap", None)  # (1,) on context if present

        # Save non-image tensors as JSON for readability
        save_json(vdir / "intrinsics.json", tensor_to_numpy(intr).tolist())
        save_json(vdir / "extrinsics.json", tensor_to_numpy(extr).tolist())
        if near is not None:
            save_json(vdir / "near.json", tensor_to_numpy(near).tolist())
        if far is not None:
            save_json(vdir / "far.json", tensor_to_numpy(far).tolist())
        if indices is not None:
            save_json(vdir / "indices.json", tensor_to_numpy(indices).tolist())
        if overlap is not None:
            # overlap can be a scalar or tensor; convert accordingly
            arr = tensor_to_numpy(overlap)
            save_json(vdir / "overlap.json", arr.tolist() if hasattr(arr, "tolist") else float(arr))

        # Optional depth
        depths = views.get("depth", None)  # (V, 1, H, W)
        valid_depths = views.get("valid_depth", None)
        if depths is not None:
            # Convert to (V, H, W)
            d = depths.squeeze(1)
            # Colorize in a consistent way across this set using vis_depth_map
            colored = vis_depth_map(d)  # (V, 3, H, W)
            Vd = colored.shape[0]
            for i in range(Vd):
                save_image(colored[i], vdir / f"depth_color_{i:03d}.png")
            # Optionally save valid_depths as PNG for quick inspection
            if valid_depths is not None:
                vm = valid_depths.squeeze(1).float().clamp(0, 1)  # (V, H, W)
                for i in range(vm.shape[0]):
                    # Repeat to 3 channels for saving
                    save_image(vm[i].repeat(3, 1, 1), vdir / f"valid_depth_{i:03d}.png")

        # Save per-view images for quick inspection
        V = images.shape[0]
        for i in range(V):
            save_image(images[i], vdir / f"{i:03d}.png")

    save_views("context", example["context"])
    save_views("target", example["target"])

    # Save a tiny manifest
    manifest = {
        "scene": scene,
        "context_views": int(example["context"]["image"].shape[0]),
        "target_views": int(example["target"]["image"].shape[0]),
        "input_shape": list(example["context"]["image"].shape[-2:]),
    }
    (scene_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main():
    args = parse_args()

    stage: Stage = args.stage  # type: ignore
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    ds_cfg, vs_cfg = build_cfg(
        roots=args.roots,
        input_shape=list(args.input_shape),
        original_shape=list(args.original_shape),
        make_baseline_1=args.make_baseline_1,
        relative_pose=args.relative_pose,
        skip_bad_shape=args.skip_bad_shape,
        index_path=args.index,
        num_context_views=args.num_context_views,
        stage=stage,
    )

    view_sampler = get_view_sampler(
        vs_cfg, stage=stage, overfit=False, cameras_are_circular=ds_cfg.cameras_are_circular, step_tracker=None
    )
    dataset = DatasetRE10k(cfg=ds_cfg, stage=stage, view_sampler=view_sampler)

    saved = 0
    for ex in dataset:
        if args.scenes is not None and ex["scene"] not in args.scenes:
            continue
        save_example(output_dir, ex, saved)
        saved += 1
        if saved >= args.num:
            break

    print(f"Saved {saved} example(s) to {output_dir}")


if __name__ == "__main__":
    main()
