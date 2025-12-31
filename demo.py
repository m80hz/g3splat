#!/usr/bin/env python3
"""
G³Splat Interactive Demo

A Gradio-based web demo for visualizing G³Splat outputs including:
- Novel view synthesis
- Depth estimation
- Surface normals and Gaussian normals
- Interactive 3D Gaussian splat visualization
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import re

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from einops import rearrange
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_typed_root_config
from src.dataset.data_module import get_data_shim
from src.model.decoder import get_decoder
from src.model.encoder import get_encoder
from src.misc.utils import vis_depth_map
from src.visualization.normal import vis_normal
from src.geometry.projection import points_to_normal
from src.model.encoder.common.gaussians import gaussian_orientation_from_scales
from src.misc.cam_utils import get_pnp_pose, update_pose
from src.visualization.camera_trajectory.interpolation import interpolate_extrinsics, interpolate_intrinsics
from src.loss import get_losses
from src.loss.loss_ssim import ssim


# Default approximate intrinsics per experiment config (normalized)
EXPERIMENT_INTRINSICS = {
    "re10k_align_orient": {
        "fx": 0.86,
        "fy": 0.86,
        "cx": 0.5,
        "cy": 0.5,
    },
    "acid_align_orient": {
        "fx": 0.6,
        "fy": 0.6,
        "cx": 0.5,
        "cy": 0.5,
    },
    "scannet_depth_align_orient": {
        "fx": 1.21,
        "fy": 1.21,
        "cx": 0.5,
        "cy": 0.5,
    },
}

# Map scene name prefixes to experiment configs
SCENE_PREFIX_TO_CONFIG = {
    "re10k": "re10k_align_orient",
    "acid": "acid_align_orient",
    "scannet": "scannet_depth_align_orient",
}


def get_intrinsics_for_scene(scene_name: str) -> Tuple[dict, str]:
    """Get intrinsics based on scene name prefix."""
    scene_lower = scene_name.lower()
    for prefix, config in SCENE_PREFIX_TO_CONFIG.items():
        if scene_lower.startswith(prefix):
            return EXPERIMENT_INTRINSICS[config], config
    # Default to re10k
    return EXPERIMENT_INTRINSICS["re10k_align_orient"], "re10k_align_orient"


def get_dataset_name_from_scene(scene_name: str) -> str:
    """Extract dataset name from scene name (e.g., 're10k_001' -> 'RealEstate10K')."""
    dataset_display_names = {
        "re10k": "RealEstate10K",
        "acid": "ACID",
        "scannet": "ScanNet",
    }
    scene_lower = scene_name.lower()
    for prefix, display_name in dataset_display_names.items():
        if scene_lower.startswith(prefix):
            return display_name
    return "Unknown"


def format_intrinsics_display(intr: dict) -> str:
    """Format intrinsics for display."""
    return f"fx={intr['fx']:.2f}, fy={intr['fy']:.2f}, cx={intr['cx']:.2f}, cy={intr['cy']:.2f}"


# Global model state
_model_state = {
    "encoder": None,
    "decoder": None,
    "data_shim": None,
    "device": None,
    "gaussian_type": "3d",
    "losses": None,  # For pose refinement
    "experiment_config": "re10k_align_orient",  # Current experiment config
}

# Available experiment configs for the dropdown
EXPERIMENT_CONFIGS = [
    "re10k_align_orient",
    "acid_align_orient",
    "scannet_depth_align_orient",
]


def detect_gaussian_type_from_checkpoint(checkpoint_path: str) -> str:
    """Auto-detect gaussian type from checkpoint filename."""
    filename = Path(checkpoint_path).name.lower()
    if "2dgs" in filename:
        return "2d"
    elif "3dgs" in filename:
        return "3d"
    return "3d"  # default


def load_model(checkpoint_path: str, experiment_config: str, gaussian_type: str, device: str = "cuda"):
    """Load the G³Splat model from checkpoint."""
    
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return f"❌ Checkpoint not found: {checkpoint_path}", gaussian_type
    
    # Auto-detect gaussian type from checkpoint name if not explicitly set differently
    detected_type = detect_gaussian_type_from_checkpoint(checkpoint_path)
    
    # Load config using Hydra compose API
    config_dir = str(Path(__file__).parent / "config")
    
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()
    
    # Initialize Hydra and compose config
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        overrides = [
            f"+experiment={experiment_config}",
            f"model.encoder.gaussian_adapter.gaussian_type={gaussian_type}",
        ]
        cfg_dict = compose(config_name="main", overrides=overrides)
    
    cfg = load_typed_root_config(cfg_dict)
    
    # Initialize encoder and decoder
    encoder, _ = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder)
    
    # Load checkpoint weights
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    
    # Remove "encoder." prefix if present
    encoder_state = {k[8:]: v for k, v in state_dict.items() if k.startswith("encoder.")}
    
    encoder.load_state_dict(encoder_state, strict=False)
    encoder = encoder.to(device).eval()
    decoder = decoder.to(device)
    
    # Load losses for pose refinement
    losses = get_losses(cfg.loss)
    losses = torch.nn.ModuleList(losses).to(device)
    
    data_shim = get_data_shim(encoder)
    
    _model_state["encoder"] = encoder
    _model_state["decoder"] = decoder
    _model_state["data_shim"] = data_shim
    _model_state["device"] = device
    _model_state["gaussian_type"] = gaussian_type
    _model_state["losses"] = losses
    _model_state["experiment_config"] = experiment_config
    
    status = (
        f"✅ Model loaded successfully!\n"
        f"- Checkpoint: {Path(checkpoint_path).name}\n"
        f"- Experiment: {experiment_config}\n"
        f"- Gaussian Type: {gaussian_type.upper()}"
        f"{' (auto-detected)' if gaussian_type == detected_type else ''}"
    )
    
    return status, detected_type


def on_checkpoint_change(checkpoint_path: str):
    """Update gaussian type when checkpoint path changes."""
    detected_type = detect_gaussian_type_from_checkpoint(checkpoint_path)
    return detected_type


def preprocess_images(img1, img2, target_size: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """Preprocess input images for the model. Returns normalized tensors.
    
    Args:
        img1: First image (can be numpy array or file path)
        img2: Second image (can be numpy array or file path)
        target_size: Target size for resizing
    """
    
    device = _model_state["device"]
    
    def process_single(img) -> torch.Tensor:
        # Handle file path input
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        # Convert numpy to PIL
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        
        # Resize to target size
        img = img.resize((target_size, target_size), Image.LANCZOS)
        
        # Convert to tensor [0, 1]
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        
        # Handle grayscale
        if img_tensor.ndim == 2:
            img_tensor = img_tensor.unsqueeze(-1).repeat(1, 1, 3)
        elif img_tensor.shape[-1] == 4:  # RGBA
            img_tensor = img_tensor[..., :3]
        
        # [H, W, C] -> [C, H, W]
        img_tensor = img_tensor.permute(2, 0, 1)
        
        return img_tensor
    
    img1_tensor = process_single(img1).to(device)
    img2_tensor = process_single(img2).to(device)
    
    return img1_tensor, img2_tensor


def create_batch_with_dummy_poses(img1_tensor: torch.Tensor, img2_tensor: torch.Tensor, 
                                   scene_name: Optional[str] = None,
                                   custom_intrinsics: Optional[dict] = None) -> dict:
    """Create a batch with dummy poses for initial Gaussian prediction.
    
    Args:
        img1_tensor: First image tensor (C, H, W)
        img2_tensor: Second image tensor (C, H, W)
        scene_name: Optional scene name to determine intrinsics (e.g., 're10k_001', 'scannet_002')
        custom_intrinsics: Optional dict with fx, fy, cx, cy (overrides scene_name and experiment config)
    """
    
    device = _model_state["device"]
    h, w = img1_tensor.shape[1], img1_tensor.shape[2]
    
    # Stack as context views [B, V, C, H, W]
    images = torch.stack([img1_tensor, img2_tensor], dim=0).unsqueeze(0)
    
    # Get intrinsics: priority is custom_intrinsics > scene_name > experiment_config
    if custom_intrinsics:
        intr_cfg = custom_intrinsics
    elif scene_name:
        intr_cfg, _ = get_intrinsics_for_scene(scene_name)
    else:
        experiment_config = _model_state.get("experiment_config", "re10k_align_orient")
        intr_cfg = EXPERIMENT_INTRINSICS.get(experiment_config, EXPERIMENT_INTRINSICS["re10k_align_orient"])
    
    intrinsics = torch.tensor([
        [intr_cfg["fx"], 0.0, intr_cfg["cx"]],
        [0.0, intr_cfg["fy"], intr_cfg["cy"]],
        [0.0, 0.0, 1.0]
    ], device=device).float()
    intrinsics = intrinsics.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1)
    
    # Create dummy extrinsics (identity for both - will be updated after pose estimation)
    extrinsics = torch.eye(4, device=device).float()
    extrinsics = extrinsics.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1)
    
    batch = {
        "context": {
            "image": images,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "near": torch.tensor([[0.1, 0.1]], device=device),
            "far": torch.tensor([[100.0, 100.0]], device=device),
        },
        "target": {
            "image": images,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "near": torch.tensor([[0.1, 0.1]], device=device),
            "far": torch.tensor([[100.0, 100.0]], device=device),
        }
    }
    
    return batch


def _pnp_pose_from_flat(pts3d: torch.Tensor, opacity: torch.Tensor, 
                         K: torch.Tensor, h: int, w: int, 
                         opacity_threshold: float = 0.3) -> Tuple[torch.Tensor, int, float]:
    """
    Custom PnP implementation for flattened point arrays.
    
    Args:
        pts3d: (H*W, 3) 3D points
        opacity: (H*W,) opacity/confidence per point
        K: (3, 3) normalized intrinsics
        h, w: image dimensions
        opacity_threshold: minimum opacity to keep a point
    
    Returns:
        pose: (4, 4) camera-to-world transformation
        inlier_count: number of inliers
        inlier_ratio: ratio of inliers
    """
    import cv2
    
    # Create pixel coordinates grid and flatten
    pixels_y, pixels_x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing='ij'
    )
    pixels = torch.stack([pixels_x, pixels_y], dim=-1).reshape(-1, 2)  # (H*W, 2)
    
    pts3d_np = pts3d.cpu().numpy()
    opacity_np = opacity.cpu().numpy()
    pixels_np = pixels.numpy()
    K_np = K.cpu().numpy().copy()
    
    # Scale intrinsics from normalized to pixel coordinates
    K_np[0, :] = K_np[0, :] * w
    K_np[1, :] = K_np[1, :] * h
    
    # Filter by opacity
    mask = opacity_np > opacity_threshold
    if mask.sum() < 6:
        mask = np.ones_like(opacity_np, dtype=bool)
    
    pts3d_masked = pts3d_np[mask]
    pixels_masked = pixels_np[mask]
    
    try:
        res = cv2.solvePnPRansac(
            pts3d_masked, pixels_masked, K_np, None,
            iterationsCount=100, reprojectionError=5, flags=cv2.SOLVEPNP_SQPNP
        )
        success, Rvec, T, inliers = res
        
        if not success:
            raise RuntimeError("solvePnPRansac failed")
        
        R = cv2.Rodrigues(Rvec)[0]
        # Construct w2c matrix and invert to get c2w
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = T.flatten()
        pose = np.linalg.inv(w2c)
        
        inlier_count = 0 if inliers is None else len(inliers)
        inlier_ratio = inlier_count / mask.sum() if mask.sum() > 0 else 0.0
        
        return torch.from_numpy(pose.astype(np.float32)), inlier_count, float(inlier_ratio)
        
    except Exception as e:
        raise RuntimeError(f"PnP failed: {e}")


def estimate_poses(gaussians, visualization_dump: dict, intrinsics: torch.Tensor, 
                   h: int, w: int, use_refinement: bool = False,
                   img1_tensor: torch.Tensor = None, img2_tensor: torch.Tensor = None) -> Tuple[torch.Tensor, str]:
    """
    Estimate camera poses for the two context views.
    
    View 1 is set as the canonical pose (identity).
    View 2's pose is estimated using PnP RANSAC on the predicted 3D points.
    
    Args:
        gaussians: Predicted Gaussians from the encoder
        visualization_dump: Dict containing depth predictions
        intrinsics: (1, 2, 3, 3) intrinsics tensor
        h, w: Image dimensions
        use_refinement: Whether to refine pose with photometric optimization
        img1_tensor: (C, H, W) tensor for view 1 (needed for pose refinement)
        img2_tensor: (C, H, W) tensor for view 2 (needed for pose refinement)
    
    Returns:
        extrinsics: (2, 4, 4) tensor of estimated poses
        status_msg: str describing the estimation result
    """
    device = _model_state["device"]
    decoder = _model_state["decoder"]
    losses = _model_state["losses"]
    gaussian_type = _model_state["gaussian_type"]
    decoder_type = "3D" if gaussian_type.lower() == "3d" else "2D"
    
    # View 1: canonical pose (identity)
    pose_view1 = torch.eye(4, device=device, dtype=torch.float32)
    
    # View 2: estimate using PnP RANSAC
    # visualization_dump['means'] shape: (B, V, H, W, srf*spp, 3)
    # visualization_dump['opacities'] shape: (B, V, H, W, srf, s)
    pts3d_view2 = visualization_dump['means'][0, 1]  # (H, W, srf*spp, 3)
    opacity_view2 = visualization_dump['opacities'][0, 1]  # (H, W, srf, s)
    
    # Squeeze out the surface/sample dimensions to get (H, W, 3) and (H, W)
    pts3d_view2 = pts3d_view2.squeeze(-2)  # (H, W, 3)
    opacity_view2 = opacity_view2.squeeze(-1).squeeze(-1)  # (H, W)
    
    # Reshape to (H*W, 3) and (H*W,) for PnP - pixels will also be reshaped inside get_pnp_pose
    pts3d_flat = pts3d_view2.reshape(-1, 3)
    opacity_flat = opacity_view2.reshape(-1)
    
    try:
        # Use custom PnP implementation that handles flattened arrays
        pose_view2, inlier_count, inlier_ratio = _pnp_pose_from_flat(
            pts3d_flat, opacity_flat, intrinsics[0, 1], h, w
        )
        pose_view2 = pose_view2.to(device)
        
        status_msg = f"PnP RANSAC: {inlier_count} inliers ({inlier_ratio:.1%})"
        
    except Exception as e:
        # Fallback: use a small baseline translation
        pose_view2 = torch.eye(4, device=device, dtype=torch.float32)
        pose_view2[0, 3] = 0.1  # Small translation in x
        status_msg = f"PnP failed, using fallback pose: {str(e)}"
        use_refinement = False  # Skip refinement if PnP failed
    
    # Optional: Pose refinement with photometric loss
    if use_refinement and losses is not None and img1_tensor is not None and img2_tensor is not None:
        try:
            with torch.set_grad_enabled(True):
                cam_rot_delta = torch.nn.Parameter(torch.zeros([1, 1, 3], requires_grad=True, device=device))
                cam_trans_delta = torch.nn.Parameter(torch.zeros([1, 1, 3], requires_grad=True, device=device))
                pose_optimizer = torch.optim.Adam([cam_rot_delta, cam_trans_delta], lr=0.005)
                
                # Initialize view 2's pose that we'll optimize
                extrinsics_v2_opt = pose_view2.unsqueeze(0).unsqueeze(0)  # (1, 1, 4, 4)
                
                # Build context batch with 2 views to match gaussian structure (V=2)
                # View 1: identity pose (canonical frame)
                # View 2: pose being optimized
                pose_view1_fixed = torch.eye(4, device=device, dtype=torch.float32)
                
                # Stack images for both views: (1, 2, C, H, W)
                context_images = torch.stack([img1_tensor, img2_tensor], dim=0).unsqueeze(0)
                
                # Target is view 2's image (what we render and compare against)
                target_image = img2_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, C, H, W)
                
                num_steps = 300
                for i in range(num_steps):
                    pose_optimizer.zero_grad()
                    
                    # Build full extrinsics with both views for the loss batch
                    # View 1: fixed identity, View 2: current optimized pose
                    context_extrinsics = torch.cat([
                        pose_view1_fixed.unsqueeze(0).unsqueeze(0),  # (1, 1, 4, 4)
                        extrinsics_v2_opt  # (1, 1, 4, 4)
                    ], dim=1)  # (1, 2, 4, 4)
                    
                    # The context has V=2 views to match gaussians (which have 2*H*W points)
                    loss_batch = {
                        "target": {
                            "image": target_image,  # (1, 1, C, H, W) - render target is view 2
                            "intrinsics": intrinsics[:, 1:2],  # (1, 1, 3, 3)
                            "extrinsics": extrinsics_v2_opt,  # (1, 1, 4, 4)
                            "near": torch.tensor([[0.1]], device=device),
                            "far": torch.tensor([[100.0]], device=device),
                        },
                        "context": {
                            "image": context_images,  # (1, 2, C, H, W) - both views
                            "intrinsics": intrinsics,  # (1, 2, 3, 3) - both views
                            "extrinsics": context_extrinsics,  # (1, 2, 4, 4) - view1=identity, view2=optimized
                            "near": torch.tensor([[0.1, 0.1]], device=device),
                            "far": torch.tensor([[100.0, 100.0]], device=device),
                        }
                    }
                    
                    # Render view 2 only (the view we're optimizing)
                    output = decoder.forward(
                        gaussians,
                        extrinsics_v2_opt,
                        intrinsics[:, 1:2],
                        torch.tensor([[0.1]], device=device),
                        torch.tensor([[100.0]], device=device),
                        (h, w),
                        cam_rot_delta=cam_rot_delta,
                        cam_trans_delta=cam_trans_delta,
                        decoder_type="3D",  # Always use 3D for pose refinement (only 3D renderer returns camera pose gradients)
                    )
                    
                    # Compute total loss using all configured losses
                    total_loss = 0
                    for loss_fn in losses:
                        loss = loss_fn.forward(output, loss_batch, gaussians, 0)
                        total_loss = total_loss + loss
                    
                    # Add SSIM structure loss
                    ssim_val, _, _, structure = ssim(
                        rearrange(target_image, "b v c h w -> (b v) c h w"),
                        rearrange(output.color, "b v c h w -> (b v) c h w"),
                        size_average=True, data_range=1.0, retrun_seprate=True, win_size=11
                    )
                    total_loss = total_loss + (1 - structure) * 0.5
                    
                    total_loss.backward()
                    
                    with torch.no_grad():
                        pose_optimizer.step()
                        new_extrinsic = update_pose(
                            cam_rot_delta=rearrange(cam_rot_delta, "b v i -> (b v) i"),
                            cam_trans_delta=rearrange(cam_trans_delta, "b v i -> (b v) i"),
                            extrinsics=rearrange(extrinsics_v2_opt, "b v i j -> (b v) i j"),
                        )
                        cam_rot_delta.data.fill_(0)
                        cam_trans_delta.data.fill_(0)
                        extrinsics_v2_opt = rearrange(new_extrinsic, "(b v) i j -> b v i j", b=1, v=1)
                
                pose_view2 = extrinsics_v2_opt[0, 0]
                status_msg += f" + refined ({num_steps} steps)"
                
        except Exception as e:
            status_msg += f" (refinement failed: {str(e)})"
    
    # Stack poses
    extrinsics = torch.stack([pose_view1, pose_view2], dim=0)  # (2, 4, 4)
    
    return extrinsics, status_msg


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy image."""
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.shape[0] in [1, 3]:
        tensor = tensor.permute(1, 2, 0)
    
    img = tensor.detach().cpu().numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    
    if img.shape[-1] == 1:
        img = img.squeeze(-1)
    
    return img


@torch.no_grad()
def run_inference(
    img1: np.ndarray,
    img2: np.ndarray,
    novel_view_angle: float = 0.5,
    use_pose_refinement: bool = False,
    fx: float = 0.86,
    fy: float = 0.86,
    cx: float = 0.5,
    cy: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Run G³Splat inference on input image pair with pose estimation."""
    
    if _model_state["encoder"] is None:
        return None, None, None, None, None, None, None, "❌ Please load a model first!"
    
    encoder = _model_state["encoder"]
    decoder = _model_state["decoder"]
    data_shim = _model_state["data_shim"]
    device = _model_state["device"]
    gaussian_type = _model_state["gaussian_type"]
    
    # Build intrinsics dict from user inputs
    custom_intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    
    try:
        # Preprocess images
        img1_tensor, img2_tensor = preprocess_images(img1, img2)
        h, w = img1_tensor.shape[1], img1_tensor.shape[2]
        
        # Create batch with dummy poses for initial Gaussian prediction
        batch = create_batch_with_dummy_poses(img1_tensor, img2_tensor, custom_intrinsics=custom_intrinsics)
        batch = data_shim(batch)
        
        # Run encoder to get Gaussians (pose-free prediction)
        visualization_dump = {}
        gaussians = encoder(batch["context"], global_step=0, visualization_dump=visualization_dump)
        
        # Estimate poses using PnP RANSAC
        estimated_extrinsics, pose_status = estimate_poses(
            gaussians, 
            visualization_dump, 
            batch["context"]["intrinsics"],
            h, w,
            use_refinement=use_pose_refinement,
            img1_tensor=img1_tensor,
            img2_tensor=img2_tensor
        )
        
        # Determine decoder type
        decoder_type = "3D" if gaussian_type.lower() == "3d" else "2D"
        
        # Render context view 1 (after pose estimation (Identity pose))
        output_ctx1 = decoder.forward(
            gaussians,
            estimated_extrinsics[0:1].unsqueeze(0),  # (1, 1, 4, 4)
            batch["context"]["intrinsics"][:, 0:1],
            batch["context"]["near"][:, 0:1],
            batch["context"]["far"][:, 0:1],
            (h, w),
            depth_mode="depth",
            decoder_type=decoder_type,
        )
        
        # Get rendered color (for context view 1)
        rendered_color = tensor_to_image(output_ctx1.color[0, 0])
        
        # Get rendered depth visualization
        depth_vis = vis_depth_map(output_ctx1.depth[0])
        rendered_depth_img = tensor_to_image(depth_vis[0])
        
        # Get surface normals from point cloud (derivative of Gaussian means)
        all_pts3d = rearrange(gaussians.means, "b (v h w) d -> b v h w d", h=h, w=w)
        surf_normals, _ = points_to_normal(all_pts3d[0])
        surface_normal_vis = vis_normal(surf_normals[0:1])  # Returns uint8 [0, 255]
        surface_normal_img = surface_normal_vis[0].cpu().numpy()  # Already uint8
        
        # Get Gaussian normals (smallest scale eigenvalue direction)
        gaussian_rotations = visualization_dump.get("rotations", None)
        gaussian_scales = visualization_dump.get("scales", None)
        
        if gaussian_rotations is not None and gaussian_scales is not None:
            gaussian_rotations = rearrange(gaussian_rotations, "b (v h w) d -> b v h w d", v=2, h=h, w=w)
            gaussian_scales = rearrange(gaussian_scales, "b (v h w) d -> b v h w d", v=2, h=h, w=w)
            
            # Get Gaussian normals for context view 1
            context1_rotations = gaussian_rotations[:, 0, ...]  # (B, H, W, 4)
            context1_scales = gaussian_scales[:, 0, ...]  # (B, H, W, 3)
            
            gaussian_surfels_normals = gaussian_orientation_from_scales(
                context1_rotations,
                context1_scales,
            )  # shape: (B, H, W, 3)
            
            gaussian_normal_vis = vis_normal(gaussian_surfels_normals)  # Returns uint8 [0, 255]
            gaussian_normal_img = gaussian_normal_vis[0].cpu().numpy()  # Already uint8
        else:
            gaussian_normal_img = surface_normal_img  # fallback
        
        # Interpolated novel view using estimated poses
        t = torch.tensor([novel_view_angle], dtype=torch.float32, device=device)
        
        # Smooth interpolation
        t_smooth = (torch.cos(torch.pi * (t + 1)) + 1) / 2
        
        # Interpolate between the two estimated poses
        interp_extrinsics = interpolate_extrinsics(
            estimated_extrinsics[0],  # View 1 pose (identity)
            estimated_extrinsics[1],  # View 2 pose (estimated)
            t_smooth
        )  # (1, 4, 4)
        
        interp_intrinsics = interpolate_intrinsics(
            batch["context"]["intrinsics"][0, 0],
            batch["context"]["intrinsics"][0, 1],
            t_smooth
        )  # (1, 3, 3)
        
        novel_output = decoder.forward(
            gaussians,
            interp_extrinsics.unsqueeze(0),  # (1, 1, 4, 4)
            interp_intrinsics.unsqueeze(0),  # (1, 1, 3, 3)
            batch["context"]["near"][:, :1],
            batch["context"]["far"][:, :1],
            (h, w),
            depth_mode="depth",
            decoder_type=decoder_type,
        )
        novel_view = tensor_to_image(novel_output.color[0, 0])
        
        # Novel view depth
        novel_depth_vis = vis_depth_map(novel_output.depth[0])
        novel_depth_img = tensor_to_image(novel_depth_vis[0])
        
        status = (
            f"✅ Inference complete!\n"
            f"- Gaussians: {gaussians.means.shape[1]:,}\n"
            f"- Image size: {h}×{w}\n"
            f"- Pose: {pose_status}"
        )
        
        return rendered_color, rendered_depth_img, surface_normal_img, gaussian_normal_img, novel_view, novel_depth_img, status
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error during inference:\n{str(e)}\n\n{traceback.format_exc()}"
        return None, None, None, None, None, None, error_msg


@torch.no_grad()
def export_ply(img1: np.ndarray, img2: np.ndarray,
               fx: float = 0.86, fy: float = 0.86, 
               cx: float = 0.5, cy: float = 0.5) -> Tuple[str, str, str]:
    """Export Gaussians as PLY file."""
    
    if _model_state["encoder"] is None:
        return None, None, "❌ Please load a model first!"
    
    encoder = _model_state["encoder"]
    device = _model_state["device"]
    data_shim = _model_state["data_shim"]
    
    # Build intrinsics dict from user inputs
    custom_intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    
    try:
        from src.model.ply_export import export_ply as _export_ply
        
        img1_tensor, img2_tensor = preprocess_images(img1, img2)
        batch = create_batch_with_dummy_poses(img1_tensor, img2_tensor, custom_intrinsics=custom_intrinsics)
        batch = data_shim(batch)
        
        visualization_dump = {}
        gaussians = encoder(batch["context"], global_step=0, visualization_dump=visualization_dump)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            ply_path = Path(f.name)
        
        # Export PLY using the existing export function
        _export_ply(
            means=gaussians.means.squeeze(0),
            scales=gaussians.scales.squeeze(0),
            rotations=gaussians.rotations.squeeze(0),
            harmonics=gaussians.harmonics.squeeze(0),
            opacities=gaussians.opacities.squeeze(0),
            path=ply_path,
            shift_and_scale=True,
            save_sh_dc_only=True,
        )
        
        return str(ply_path), str(ply_path), f"✅ PLY exported: {ply_path.name}"
        
    except Exception as e:
        import traceback
        return None, None, f"❌ Export failed: {str(e)}\n{traceback.format_exc()}"


# Global mapping from example image paths to scene names (set during demo creation)
_example_path_to_scene = {}


def get_scene_name_from_path(img_path: str) -> Optional[str]:
    """Extract scene name from image path.
    
    Handles both direct paths (assets/examples/re10k_001/context_0.png)
    and Gradio cached paths by checking the global mapping.
    """
    if not img_path:
        return None
    
    # First check if this exact path is in our mapping (for Gradio cached files)
    if img_path in _example_path_to_scene:
        return _example_path_to_scene[img_path]
    
    # Check if any known scene name pattern appears in the path
    path_str = str(img_path).lower()
    scene_prefixes = ["re10k_", "acid_", "scannet_"]
    
    for prefix in scene_prefixes:
        if prefix in path_str:
            # Extract the scene name (e.g., 're10k_001' from the path)
            import re
            pattern = rf"({prefix}\d+)"
            match = re.search(pattern, path_str)
            if match:
                return match.group(1)
    
    # Try parent directory approach (for direct paths)
    path = Path(img_path)
    parent_name = path.parent.name
    if parent_name and parent_name != "examples" and not parent_name.startswith("gradio"):
        # Check if parent looks like a valid scene name
        for prefix in ["re10k", "acid", "scannet"]:
            if parent_name.lower().startswith(prefix):
                return parent_name
    
    return None


def get_dataset_from_config(config_name: str) -> str:
    """Get dataset name from experiment config."""
    config_to_dataset = {
        "re10k_align_orient": "RealEstate10K",
        "acid_align_orient": "ACID",
        "scannet_depth_align_orient": "ScanNet",
    }
    return config_to_dataset.get(config_name, "Unknown")


def check_dataset_mismatch(image_dataset: str, current_config: str) -> str:
    """Check if the image dataset matches the current experiment config.
    
    Args:
        image_dataset: Dataset name from the image (e.g., 'RealEstate10K', 'ScanNet')
        current_config: Current experiment config name
    
    Returns:
        Warning message if datasets don't match, empty string otherwise.
    """
    if not image_dataset or image_dataset == "Unknown":
        return ""
    
    # Get dataset from current experiment config
    config_dataset = get_dataset_from_config(current_config)
    
    if image_dataset != config_dataset:
        # Find the matching config for this dataset
        dataset_to_config = {
            "RealEstate10K": "re10k_align_orient",
            "ACID": "acid_align_orient",
            "ScanNet": "scannet_depth_align_orient",
        }
        matching_config = dataset_to_config.get(image_dataset, "re10k_align_orient")
        return f"⚠️ Dataset mismatch: Image is from {image_dataset}, but config is for {config_dataset}.\nPlease change Experiment Config to '{matching_config}' and reload the model."
    
    return ""


def create_demo(default_checkpoint: str = "", default_device: str = "cuda"):
    """Create the Gradio demo interface."""
    
    # Custom CSS
    css = """
    .gradio-container {
        max-width: 1400px !important;
    }
    .output-image {
        border-radius: 8px;
    }
    .status-box {
        font-family: monospace;
        font-size: 12px;
    }
    """
    
    # Get example images with dataset labels
    examples_dir = Path(__file__).parent / "assets" / "examples"
    example_pairs = []
    example_labels = []
    
    # Clear and rebuild the path-to-scene mapping
    global _example_path_to_scene
    _example_path_to_scene = {}
    
    if examples_dir.exists():
        for scene_dir in sorted(examples_dir.iterdir()):
            if scene_dir.is_dir():
                ctx0 = scene_dir / "context_0.png"
                ctx1 = scene_dir / "context_1.png"
                if ctx0.exists() and ctx1.exists():
                    scene_name = scene_dir.name
                    dataset_name = get_dataset_name_from_scene(scene_name)
                    example_pairs.append([str(ctx0), str(ctx1), dataset_name])
                    example_labels.append(f"{dataset_name}: {scene_name}")
                    # Store mapping from path to scene name
                    _example_path_to_scene[str(ctx0)] = scene_name
                    _example_path_to_scene[str(ctx1)] = scene_name
    
    # Auto-detect gaussian type from default checkpoint
    default_gaussian_type = detect_gaussian_type_from_checkpoint(default_checkpoint)
    
    # Store device in model state for use by preprocess_images
    _model_state["device"] = default_device
    
    with gr.Blocks(css=css, title="G³Splat Demo") as demo:
        gr.Markdown("""
        # 🌟 G³Splat: Geometrically Consistent Generalizable Gaussian Splatting
        
        Upload two images or select from examples to generate 3D Gaussian splats, novel views, depth maps, and normals.
        
        [📄 Paper](https://arxiv.org/abs/2512.17547) | [💻 Code](https://github.com/m80hz/g3splat) | [🤗 Models](https://huggingface.co/m80hz/g3splat)
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Model Configuration")
                
                checkpoint_input = gr.Textbox(
                    label="Checkpoint Path",
                    value=default_checkpoint,
                    placeholder="Path to .ckpt file"
                )
                
                experiment_config = gr.Dropdown(
                    choices=EXPERIMENT_CONFIGS,
                    value="re10k_align_orient",
                    label="Experiment Config",
                    info="Click 'Load Model' after changing to apply."
                )
                
                gaussian_type = gr.Radio(
                    choices=["3d", "2d"],
                    value=default_gaussian_type,
                    label="Gaussian Type",
                    info="Select 3DGS or 2DGS variant (can be detected from checkpoint name)"
                )
                
                load_btn = gr.Button("🚀 Load Model", variant="primary")
                model_status = gr.Textbox(label="Model Status", interactive=False, lines=4)
        
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📸 Input Images")
                
                with gr.Row():
                    img1_input = gr.Image(label="Context View 1", type="filepath")
                    img2_input = gr.Image(label="Context View 2", type="filepath")
                    dataset_display = gr.Textbox(
                        label="Dataset",
                        value="",
                        interactive=False,
                        scale=0,
                        min_width=120
                    )
                
                # Intrinsics inputs (editable)
                gr.Markdown("""📐 **Camera Intrinsics** *(approximate normalized values, edit if you have accurate intrinsics)*""")
                
                default_intr = EXPERIMENT_INTRINSICS['re10k_align_orient']
                with gr.Row():
                    fx_input = gr.Number(label="fx", value=default_intr['fx'], precision=3, minimum=0.1, maximum=2.0)
                    fy_input = gr.Number(label="fy", value=default_intr['fy'], precision=3, minimum=0.1, maximum=2.0)
                    cx_input = gr.Number(label="cx", value=default_intr['cx'], precision=3, minimum=0.0, maximum=1.0)
                    cy_input = gr.Number(label="cy", value=default_intr['cy'], precision=3, minimum=0.0, maximum=1.0)
                
                # Warning display for dataset mismatch
                dataset_warning = gr.Textbox(
                    label="",
                    value="",
                    interactive=False,
                    visible=True,
                    lines=2,
                    show_label=False
                )
                
                # Examples section - moved here, below Input Images
                if example_pairs:
                    gr.Markdown("### 📂 Example Images")
                    gr.Markdown("*Click an example to load.*")
                    gr.Examples(
                        examples=example_pairs,
                        inputs=[img1_input, img2_input, dataset_display],
                        label="Click to load example",
                        examples_per_page=10
                    )
                
                novel_view_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="Novel View Position",
                    info="Interpolation between view 1 (0.0) and view 2 (1.0)"
                )
                
                pose_refinement_checkbox = gr.Checkbox(
                    label="Enable Test-time Pose Refinement",
                    value=False,
                    info="Refine estimated pose with the same training losses (slower but more accurate)"
                )
                
                with gr.Row():
                    run_btn = gr.Button("🎨 Run Inference", variant="primary", scale=2)
                    export_btn = gr.Button("💾 Visualize Gaussians and Export PLY", scale=1)
                
                inference_status = gr.Textbox(label="Status", interactive=False, lines=4)
        
        gr.Markdown("---")
        
        with gr.Row():
            gr.Markdown("### 🖼️ Results")
        
        with gr.Row():
            with gr.Column():
                rendered_output = gr.Image(label="Rendered RGB (Context 1)", elem_classes="output-image")
            with gr.Column():
                novel_output = gr.Image(label="Rendered RGB (Novel View)", elem_classes="output-image")
        
        with gr.Row():
            with gr.Column():
                depth_output = gr.Image(label="Rendered Depth (Context 1)", elem_classes="output-image")
            with gr.Column():
                novel_depth_output = gr.Image(label="Rendered Depth (Novel View)", elem_classes="output-image")
        
        with gr.Row():
            with gr.Column():
                surface_normal_output = gr.Image(label="Surface Normals (Context 1)", elem_classes="output-image")
            with gr.Column():
                gaussian_normal_output = gr.Image(label="Gaussian Normals (Context 1)", elem_classes="output-image")
        
        gr.Markdown("---")
        gr.Markdown("### 🎮 3D Visualization")
        
        with gr.Row():
            with gr.Column():
                model_3d_output = gr.Model3D(label="Gaussian Splat (PLY)", elem_classes="output-image")
            with gr.Column():
                ply_output = gr.File(label="Download PLY")
        
        # Footer
        gr.Markdown("""
        ---
        ### 📝 Notes
        - **3DGS**: Standard 3D Gaussian Splatting
        - **2DGS**: 2D Gaussian Splatting (Gaussian surfels)
        - **Surface Normals**: Derived from the 3D point cloud (gradient of Gaussian means)
        - **Gaussian Normals**: Direction of the smallest scale eigenvalue of each Gaussian (surfel orientation)
        - For best results, use images with overlapping views of the same scene
        - The model expects images at 256x256 resolution
        """)
        
        # Event handlers
        
        # Auto-update gaussian type when checkpoint changes
        checkpoint_input.change(
            fn=on_checkpoint_change,
            inputs=[checkpoint_input],
            outputs=[gaussian_type]
        )
        
        # Update intrinsics inputs when experiment config changes
        def update_intrinsics_from_config(config_name):
            intr_cfg = EXPERIMENT_INTRINSICS.get(config_name, EXPERIMENT_INTRINSICS["re10k_align_orient"])
            return intr_cfg["fx"], intr_cfg["fy"], intr_cfg["cx"], intr_cfg["cy"], ""  # Clear warning
        
        experiment_config.change(
            fn=update_intrinsics_from_config,
            inputs=[experiment_config],
            outputs=[fx_input, fy_input, cx_input, cy_input, dataset_warning]
        )
        
        load_btn.click(
            fn=load_model,
            inputs=[checkpoint_input, experiment_config, gaussian_type],
            outputs=[model_status, gaussian_type]
        )
        
        run_btn.click(
            fn=run_inference,
            inputs=[img1_input, img2_input, novel_view_slider, pose_refinement_checkbox, fx_input, fy_input, cx_input, cy_input],
            outputs=[rendered_output, depth_output, surface_normal_output, gaussian_normal_output, novel_output, novel_depth_output, inference_status]
        )
        
        export_btn.click(
            fn=export_ply,
            inputs=[img1_input, img2_input, fx_input, fy_input, cx_input, cy_input],
            outputs=[ply_output, model_3d_output, inference_status]
        )
        
        # Check for dataset mismatch when dataset_display changes (from Examples selection)
        dataset_display.change(
            fn=check_dataset_mismatch,
            inputs=[dataset_display, experiment_config],
            outputs=[dataset_warning]
        )
    
    return demo


def main():
    parser = argparse.ArgumentParser(description="G³Splat Interactive Demo")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="pretrained_weights/g3splat_mast3r_3dgs_align_orient_re10k.ckpt",
        help="Default checkpoint path (shown in UI, not pre-loaded)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Server port"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public Gradio link"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # Create and launch demo (no pre-loading - user clicks Load Model button)
    demo = create_demo(
        default_checkpoint=args.checkpoint,
        default_device=args.device
    )
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == "__main__":
    main()
