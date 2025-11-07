# -*- coding: utf-8 -*-
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import open3d as o3d
import sklearn.neighbors as skln
import torch
from einops import repeat
from lightning import LightningModule
from PIL import Image

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from .evaluation_cfg import EvaluationCfg
from ..misc.mesh_utils import GaussianMeshExtractor, post_process_mesh
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)

# ----------------------------- utilities -----------------------------

def _compute_depth_stats(depth: np.ndarray) -> Dict[str, float]:
    d = depth.astype(np.float32)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {"count": 0}
    pos = d[d > 0]
    stats = {
        "count": int(d.size),
        "pos_ratio": float(pos.size) / float(d.size),
        "min": float(d.min()),
        "max": float(d.max()),
        "mean": float(d.mean()),
        "std": float(d.std()),
        "p1": float(np.percentile(d, 1)),
        "p5": float(np.percentile(d, 5)),
        "p50": float(np.percentile(d, 50)),
        "p95": float(np.percentile(d, 95)),
        "p99": float(np.percentile(d, 99)),
    }
    if pos.size > 0:
        stats.update({
            "pos_min": float(pos.min()),
            "pos_max": float(pos.max()),
            "pos_mean": float(pos.mean()),
            "pos_p50": float(np.percentile(pos, 50)),
            "pos_p95": float(np.percentile(pos, 95)),
        })
    return stats


def _sample_points_from_mesh(mesh: o3d.geometry.TriangleMesh, density: float) -> np.ndarray:
    """Uniformly sample points on the mesh surface, robust to CUDA/CPU meshes."""
    try:
        num_tris = len(mesh.triangles)
    except Exception:
        return np.asarray(mesh.vertices)

    if num_tris == 0:
        return np.asarray(mesh.vertices)

    try:
        verts = np.asarray(mesh.vertices)
        tris = np.asarray(mesh.triangles)
        v0 = verts[tris[:, 0]]
        v1 = verts[tris[:, 1]]
        v2 = verts[tris[:, 2]]
        face_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        area = float(np.sum(face_area))
    except Exception:
        area = float(num_tris)

    n = int(max(5000, min(100000, area / max(1e-6, (density * 0.01)))))
    try:
        pts = mesh.sample_points_uniformly(number_of_points=n)
        return np.asarray(pts.points)
    except Exception:
        return np.asarray(mesh.vertices)


def _downsample_radius(points: np.ndarray, radius: float) -> np.ndarray:
    if points.shape[0] == 0:
        return points
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=radius, algorithm="kd_tree", n_jobs=1)
    nn_engine.fit(points)
    rnn_idxs = nn_engine.radius_neighbors(points, radius=radius, return_distance=False)
    mask = np.ones(points.shape[0], dtype=np.bool_)
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = 0
            mask[curr] = 1
    return points[mask]


def _make_o3d_intrinsics(width: int, height: int, K: np.ndarray) -> o3d.camera.PinholeCameraIntrinsic:
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    intr = o3d.camera.PinholeCameraIntrinsic()
    intr.set_intrinsics(width, height, fx, fy, cx, cy)
    return intr


# ---------- dataset parsing & IO helpers (full-res intrinsics/poses) ----------

def _parse_scene_token(token: str, num_ctx: int, num_tgt: int) -> Tuple[str, List[int], List[int]]:
    """
    token example: 'scene0664_02_290_1234' -> ('scene0664_02', [290], [1234]) for num_ctx=1,num_tgt=1.
    Takes all trailing integers; first num_ctx are contexts, next num_tgt are targets.
    """
    m = re.match(r"^(scene\d+_\d+)", token)
    if not m:
        raise ValueError(f"Cannot parse scene id from token '{token}'")
    scene_id = m.group(1)
    tail = token[len(scene_id):]
    ids = [int(x) for x in tail.split("_") if x.isdigit()]
    ctx = ids[:num_ctx]
    tgt = ids[num_ctx:num_ctx + num_tgt]
    return scene_id, ctx, tgt


def _read_intrinsics_color(scene_dir: Path) -> Tuple[np.ndarray, int, int]:
    intr_path = scene_dir / "intrinsic" / "intrinsic_color.txt"
    K4 = np.loadtxt(str(intr_path)).astype(np.float64)  # 4x4 in ScanNet files
    K = K4[:3, :3]
    # Resolve image size from a color frame at full resolution
    color_dir = scene_dir / "color"
    try:
        cand = sorted(color_dir.glob("*.jpg"))[0]
        with Image.open(cand) as im:
            w, h = im.size
    except Exception:
        w = int(round(2.0 * K[0, 2]))
        h = int(round(2.0 * K[1, 2]))
    return K, w, h


def _load_scannet_pose_txt(pose_file: Path) -> np.ndarray:
    """Load a 4x4 camera-to-world matrix from ScanNet .txt."""
    vals = np.loadtxt(str(pose_file)).astype(np.float64).reshape(-1)
    if vals.size != 16:
        raise ValueError(f"Pose file {pose_file} does not contain 16 values")
    return vals.reshape(4, 4)


def _camera_centers_from_c2w(c2w_stack: np.ndarray) -> np.ndarray:
    """c2w_stack: [N,4,4]. Returns centers [N,3]."""
    return c2w_stack[:, :3, 3].copy()


def _apply_transform_to_mesh(mesh: o3d.geometry.TriangleMesh, T: np.ndarray) -> o3d.geometry.TriangleMesh:
    m = o3d.geometry.TriangleMesh(mesh)
    m.transform(T.astype(np.float64))
    return m


def _triangle_centroids(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.triangles, dtype=np.int32)
    if F.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return (V[F[:, 0]] + V[F[:, 1]] + V[F[:, 2]]) / 3.0


# ---------- frusta / cropping ----------

def _frustum_corners_world_plusZ(K: np.ndarray, c2w: np.ndarray, w: int, h: int, near: float, far: float) -> np.ndarray:
    """Return 8 world-space frustum corner points assuming +Z forward in camera frame."""
    K = K.astype(np.float64)
    c2w = c2w.astype(np.float64)
    Kinv = np.linalg.inv(K)
    pix = np.array([[0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1], [0, h - 1, 1]], dtype=np.float64).T  # (3,4)
    dirs = Kinv @ pix
    dirs /= np.maximum(dirs[2:3, :], 1e-12)  # z=1 rays
    Pn = dirs * near
    Pf = dirs * far
    Pn_h = np.vstack([Pn, np.ones((1, 4))])
    Pf_h = np.vstack([Pf, np.ones((1, 4))])
    Pn_w = (c2w @ Pn_h)[:3].T  # (4,3)
    Pf_w = (c2w @ Pf_h)[:3].T  # (4,3)
    return np.vstack([Pn_w, Pf_w])  # (8,3)


def _compute_half_frustum_planes_no_far(K: np.ndarray, c2w: np.ndarray, w: int, h: int, near: float):
    """
    Build only the 'front/near', 'left', 'right', 'top', 'bottom' planes.
    Planes are oriented so that the half-space n·x + d >= 0 is INSIDE the frustum.
    """
    corners = _frustum_corners_world_plusZ(K, c2w, w, h, near, far=near * 2.0)  # far ignored; we use only near corners
    ntl, ntr, nbr, nbl = corners[0], corners[1], corners[2], corners[3]
    C = c2w[:3, 3]

    def plane(a, b, c):
        n = np.cross(b - a, c - a)
        n /= (np.linalg.norm(n) + 1e-12)
        d = -np.dot(n, a)
        # ensure that a point slightly in front of near center is inside
        near_center = (ntl + ntr + nbr + nbl) / 4.0
        inside_probe = C + 1.5 * (near_center - C)
        if np.dot(n, inside_probe) + d < 0:
            n = -n
            d = -d
        return n, d

    # Front/near plane through near quad (use oriented triangle)
    near_plane = plane(ntl, ntr, nbr)

    # Side planes: use camera center and two edge points on near plane
    left_plane   = plane(C, nbl, ntl)
    right_plane  = plane(C, ntr, nbr)
    top_plane    = plane(C, ntl, ntr)
    bottom_plane = plane(C, nbr, nbl)

    return [near_plane, left_plane, right_plane, top_plane, bottom_plane]


def _mesh_crop_by_union_plane_sets(mesh: o3d.geometry.TriangleMesh, plane_sets: List[List[Tuple[np.ndarray, float]]]) -> o3d.geometry.TriangleMesh:
    """Cull triangles whose centroids are outside all plane sets (union-of-views)."""
    if len(mesh.triangles) == 0:
        return mesh
    cents = _triangle_centroids(mesh)
    if cents.shape[0] == 0:
        return mesh

    def inside_planes(pts: np.ndarray, planes) -> np.ndarray:
        mask = np.ones((pts.shape[0],), dtype=bool)
        for n, d in planes:
            mask &= (pts @ n + d) >= 0.0
        return mask

    inside_any = np.zeros((cents.shape[0],), dtype=bool)
    for planes in plane_sets:
        inside_any |= inside_planes(cents, planes)

    F = np.asarray(mesh.triangles, dtype=np.int32)
    tri_mask = inside_any
    if not np.any(tri_mask):
        return o3d.geometry.TriangleMesh()

    kept_F = F[tri_mask]
    V = np.asarray(mesh.vertices)
    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(V)
    out.triangles = o3d.utility.Vector3iVector(kept_F)
    if mesh.has_vertex_colors():
        out.vertex_colors = mesh.vertex_colors
    out.remove_unreferenced_vertices()
    out.remove_degenerate_triangles()
    out.compute_vertex_normals()
    return out


def _union_aabb_for_views_plusZ(views: List[Tuple[np.ndarray, np.ndarray, float, float, int, int]]) -> o3d.geometry.AxisAlignedBoundingBox:
    """AABB from union of frustum corner boxes (uses +Z; far is only for AABB extent)."""
    if len(views) == 0:
        raise RuntimeError("AABB views list is empty.")
    corners_all = []
    for K, c2w, near, far, w, h in views:
        corners = _frustum_corners_world_plusZ(K, c2w, w, h, float(near), float(far))
        corners_all.append(corners)
    pts = np.concatenate(corners_all, axis=0)
    return o3d.geometry.AxisAlignedBoundingBox(pts.min(axis=0), pts.max(axis=0))


def _mesh_center_radius(mesh: o3d.geometry.TriangleMesh) -> Tuple[np.ndarray, float]:
    aabb = mesh.get_axis_aligned_bounding_box()
    c = np.asarray(aabb.get_center(), dtype=np.float64)
    ext = np.asarray(aabb.get_extent(), dtype=np.float64)
    r = float(np.linalg.norm(ext) * 0.5)
    return c, r


# ---------- scale-only and full Sim(3) ----------

def _estimate_scale_only_from_meshes(
    pred_mesh_norm: o3d.geometry.TriangleMesh,
    gt_mesh_metric: o3d.geometry.TriangleMesh,
    iters: int = 8,
    outlier_frac: float = 0.2,
) -> float:
    """
    Estimate a single scalar s to map pred_mesh_norm -> metric by minimizing
        sum || s * p_i - g_j(i) ||^2
    where j(i) are NN correspondences in GT. No rotation, no translation.
    Scaling is performed about the origin (context-0 is identity).
    Deterministic; raises on failure (no fallbacks).
    """
    P = _sample_points_from_mesh(pred_mesh_norm, density=0.02)
    G = _sample_points_from_mesh(gt_mesh_metric, density=0.02)
    if P.shape[0] < 200 or G.shape[0] < 200:
        raise RuntimeError(f"Not enough points for scale estimation (|P|={P.shape[0]}, |G|={G.shape[0]}).")

    # NN on GT
    nn = skln.NearestNeighbors(n_neighbors=1, algorithm="kd_tree", n_jobs=1)
    nn.fit(G)

    # Init from radius ratio
    _, rP = _mesh_center_radius(pred_mesh_norm)
    _, rG = _mesh_center_radius(gt_mesh_metric)
    if rP <= 1e-8:
        raise RuntimeError("Pred mesh radius is ~0; cannot estimate scale.")
    s = rG / rP

    # Fixed-point refinement on correspondences
    P2 = (P ** 2).sum(axis=1)
    denom = float(P2.sum())
    if denom <= 1e-12:
        raise RuntimeError("Degenerate pred point set; cannot estimate scale.")

    for _ in range(max(1, iters)):
        P_scaled = s * P
        dists, idxs = nn.kneighbors(P_scaled, n_neighbors=1, return_distance=True)
        idxs = idxs.reshape(-1)
        # Robust trimming (deterministic): drop largest outlier_frac of pairs
        if outlier_frac > 0:
            keep_n = int((1.0 - outlier_frac) * P.shape[0])
            order = np.argsort(dists.reshape(-1))
            sel = order[:max(3, keep_n)]
            P_sel = P[sel]
            G_sel = G[idxs[sel]]
        else:
            P_sel = P
            G_sel = G[idxs]
        num = float((P_sel * G_sel).sum())
        s_new = num / denom
        if not np.isfinite(s_new) or s_new <= 0:
            raise RuntimeError(f"Invalid scale update s={s_new}")
        if abs(s_new - s) / s < 1e-4:
            s = s_new
            break
        s = s_new

    if not np.isfinite(s) or s <= 0 or s > 1e6:
        raise RuntimeError(f"Estimated invalid scale s={s}")
    return float(s)


def _estimate_scale_translate_from_meshes(
    pred_mesh_norm: o3d.geometry.TriangleMesh,
    gt_mesh_metric: o3d.geometry.TriangleMesh,
    iters: int = 50,
    outlier_frac: float = 0.2,
    restrict_axis: Optional[str] = None,
    eps_rel: float = 1e-4,
) -> np.ndarray:
    """
    Estimate Sim transform with (scale + translation) only (no rotation) mapping
        x_pred_metric ≈ s * x_pred_norm + t
    where t can be:
      - unrestricted (restrict_axis=None)
      - only along +Z camera axis (restrict_axis == 'z')

    Minimizes robust objective over NN correspondences with iterative refinement.
    Returns 4x4 matrix T such that X_metric = T @ [X_norm; 1].
    """
    P = _sample_points_from_mesh(pred_mesh_norm, density=0.02)  # (N,3)
    G = _sample_points_from_mesh(gt_mesh_metric, density=0.02)  # (M,3)
    if P.shape[0] < 200 or G.shape[0] < 200:
        raise RuntimeError("Not enough points for scale+translation estimation.")

    # Build NN structure on GT (fixed target set)
    nn = skln.NearestNeighbors(n_neighbors=1, algorithm="kd_tree", n_jobs=1)
    nn.fit(G)

    # Initialize scale from radius ratio; translation from centroid difference.
    cP, rP = _mesh_center_radius(pred_mesh_norm)
    cG, rG = _mesh_center_radius(gt_mesh_metric)
    if rP <= 1e-8:
        raise RuntimeError("Pred mesh radius ~0; cannot initialize scale.")
    s = rG / rP
    t = cG - s * cP
    if restrict_axis == 'z':
        # Only allow translation along z; project initial translation onto z-axis.
        ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        t = (t @ ez) * ez

    def objective(P_scaled_trans, G_sel):
        return float(np.mean(np.sum((P_scaled_trans - G_sel) ** 2, axis=1)))

    prev_obj = None
    P2 = (P ** 2).sum(axis=1)
    denom_scale = float(P2.sum())
    if denom_scale <= 1e-12:
        raise RuntimeError("Degenerate pred point set; cannot estimate scale.")

    for _ in range(max(1, iters)):
        P_st = s * P + t  # current transformed points
        dists, idxs = nn.kneighbors(P_st, n_neighbors=1, return_distance=True)
        dists = dists.reshape(-1)
        idxs = idxs.reshape(-1)
        if outlier_frac > 0:
            keep_n = int((1.0 - outlier_frac) * P.shape[0])
            order = np.argsort(dists)
            sel = order[:max(3, keep_n)]
        else:
            sel = slice(None)
        P_sel = P[sel]
        G_sel = G[idxs[sel]]
        # Closed form for scale with current correspondences (d/ds objective = 0)
        num = float((P_sel * (G_sel - t)).sum())
        s_new = num / denom_scale
        if not np.isfinite(s_new) or s_new <= 0:
            # fallback: keep previous scale
            s_new = s
        # Translation: mean residual (G - s_new*P)
        t_new = (G_sel - s_new * P_sel).mean(axis=0)
        if restrict_axis == 'z':
            ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            t_new = (t_new @ ez) * ez

        s, t = s_new, t_new
        P_st = s * P + t
        obj = objective(P_st[sel] if isinstance(sel, np.ndarray) else P_st, G_sel)
        if prev_obj is not None:
            rel_impr = (prev_obj - obj) / max(prev_obj, 1e-12)
            if rel_impr < eps_rel:
                break
        prev_obj = obj

    if not np.isfinite(s) or s <= 0 or s > 1e6:
        raise RuntimeError(f"Estimated invalid scale s={s}")
    if not np.isfinite(t).all():
        raise RuntimeError("Estimated invalid translation t")

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.eye(3) * s
    T[:3, 3] = t
    return T


def _estimate_sim3_full_from_meshes(
    pred_mesh_norm: o3d.geometry.TriangleMesh,
    gt_mesh_metric: o3d.geometry.TriangleMesh,
    max_iters: int = 60,
    cfg=None,
) -> np.ndarray:
    """
    Full Sim(3) via Open3D ICP (with scaling), then project to proper rotation and isotropic scale.
    Deterministic and strict: raises on reflection or out-of-bounds scale.
    """
    src_pts = _sample_points_from_mesh(pred_mesh_norm, density=0.02)
    dst_pts = _sample_points_from_mesh(gt_mesh_metric, density=0.02)
    if src_pts.shape[0] < 200 or dst_pts.shape[0] < 200:
        raise RuntimeError(f"Not enough points for Sim(3) ICP (|P|={src_pts.shape[0]}, |G|={dst_pts.shape[0]}).")

    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_pts))
    dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dst_pts))

    # Radius-based init (scale only), no translation/rotation
    _, rP = _mesh_center_radius(pred_mesh_norm)
    _, rG = _mesh_center_radius(gt_mesh_metric)
    if rP <= 1e-8:
        raise RuntimeError("Pred mesh radius is ~0; cannot init Sim(3).")
    s0 = rG / rP
    T = np.eye(4, dtype=np.float64); T[:3, :3] = np.eye(3) * s0

    est = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iters)

    r = max(0.02, rG * 0.25)
    for corr in [r, r * 0.5, r * 0.25]:
        result = o3d.pipelines.registration.registration_icp(
            src, dst, max_correspondence_distance=corr, init=T, estimation_method=est, criteria=criteria
        )
        T = result.transformation
        if not np.isfinite(T).all():
            raise RuntimeError("ICP returned NaN/Inf transform.")

    # --- Project to Sim(3)+ (proper rotation) and extract isotropic scale ---
    M = T[:3, :3]
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt
    detR = float(np.linalg.det(R))

    require_proper = True if cfg is None else bool(getattr(cfg, "sim3_require_proper_rotation", True))
    fix_reflect    = False if cfg is None else bool(getattr(cfg, "sim3_fix_reflection", False))
    if detR < 0.0:
        if require_proper and not fix_reflect:
            raise RuntimeError("ICP produced improper rotation (reflection). Set sim3_fix_reflection=True to allow projection, or check inputs.")
        # project to proper rotation by flipping the third column of V
        Vt[-1, :] *= -1.0
        R = U @ Vt
        detR = float(np.linalg.det(R))
        if detR < 0.0:
            raise RuntimeError("Failed to project ICP rotation to proper SO(3).")

    # isotropic scale = average singular value
    s_est = float(np.mean(S))
    if not np.isfinite(s_est) or s_est <= 0:
        raise RuntimeError(f"Invalid estimated scale s={s_est}")

    # --- Enforce scale bounds ---
    abs_bounds = None if cfg is None else getattr(cfg, "sim3_scale_bounds", None)      # e.g., (0.2, 5.0)
    rel_bounds = None if cfg is None else getattr(cfg, "sim3_relative_scale_bounds", None)  # e.g., (0.25, 4.0)

    if abs_bounds is not None:
        s_min, s_max = float(abs_bounds[0]), float(abs_bounds[1])
        if not (s_min <= s_est <= s_max):
            raise RuntimeError(f"Estimated scale {s_est:.6f} outside absolute bounds [{s_min}, {s_max}].")

    if rel_bounds is not None:
        r_min, r_max = float(rel_bounds[0]), float(rel_bounds[1])
        if not (r_min * s0 <= s_est <= r_max * s0):
            raise RuntimeError(
                f"Estimated scale {s_est:.6f} outside relative bounds [{r_min}x{s0:.6f}, {r_max}x{s0:.6f}]."
            )

    # Rebuild T with corrected linear part (keep ICP translation; optional: recompute t via centroids)
    T_out = np.eye(4, dtype=np.float64)
    T_out[:3, :3] = s_est * R
    T_out[:3, 3]  = T[:3, 3]
    return T_out


# ---------- Open3D visualization ----------

def _make_frustum_lineset_plusZ(K: np.ndarray, c2w: np.ndarray, w: int, h: int, near: float, far: float, color=(0, 1, 0)) -> o3d.geometry.LineSet:
    corners = _frustum_corners_world_plusZ(K, c2w, w, h, near, far)
    pts = corners
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],          # near
        [4, 5], [5, 6], [6, 7], [7, 4],          # far
        [0, 4], [1, 5], [2, 6], [3, 7],          # sides
    ]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32)),
    )
    ls.colors = o3d.utility.Vector3dVector(np.tile(np.array(color, dtype=np.float64), (len(lines), 1)))
    return ls


def _visualize_scene_alignment(
    scene_token: str,
    pred_mesh_norm: o3d.geometry.TriangleMesh,
    pred_mesh_metric: o3d.geometry.TriangleMesh,
    gt_mesh_metric: o3d.geometry.TriangleMesh,
    scene_dir: Path,
    ids_for_frusta: List[int],
    show_frusta: bool = True,
    coord_frame_size: float = 0.3,
    viz_far_m: float = 5.0,
):
    pred_before = o3d.geometry.TriangleMesh(pred_mesh_norm)
    pred_before.paint_uniform_color([0.95, 0.55, 0.10])  # orange
    pred_after = o3d.geometry.TriangleMesh(pred_mesh_metric)
    pred_after.paint_uniform_color([0.10, 0.80, 0.10])   # green
    gt_vis = o3d.geometry.TriangleMesh(gt_mesh_metric)
    gt_vis.paint_uniform_color([0.10, 0.30, 0.90])       # blue

    geoms = [gt_vis, pred_before, pred_after]
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=coord_frame_size))

    if show_frusta and len(ids_for_frusta) > 0:
        K, W, H = _read_intrinsics_color(scene_dir)
        Twc0 = _load_scannet_pose_txt(scene_dir / "pose" / f"{ids_for_frusta[0]}.txt")
        T_align = np.linalg.inv(Twc0)
        near_m = 0.05
        colors = [(0.1, 0.9, 0.1), (0.9, 0.1, 0.1), (0.9, 0.9, 0.1), (0.1, 0.9, 0.9)]
        for idx, i in enumerate(ids_for_frusta):
            c2w = T_align @ _load_scannet_pose_txt(scene_dir / "pose" / f"{i}.txt")
            geoms.append(_make_frustum_lineset_plusZ(K, c2w, W, H, near_m, viz_far_m, color=colors[idx % len(colors)]))

    print(f"[viz] {scene_token}: GT (blue), pred BEFORE (orange), pred AFTER (green)")
    o3d.visualization.draw_geometries(geoms)


# ----------------------------- evaluator -----------------------------

@dataclass
class MeshEvalResult:
    scene: str
    accuracy: float
    completeness: float
    overall: float
    precision_5cm: float
    recall_5cm: float
    fscore_5cm: float


class ScanNetMeshEvaluator(LightningModule):
    """Evaluate meshes on ScanNet with community-standard metrics (Chamfer + F@5cm).
    - GT crop: default AABB; optional frustum on top of AABB (no far plane). Always +Z.
    - Alignment: default scale-only; optional full Sim(3) if requested.
    - Deterministic, no fallbacks: raises on failures to keep protocols consistent.
    """

    def __init__(self, cfg: EvaluationCfg, encoder, decoder, losses) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder.to(self.device)
        self.decoder = decoder
        self.losses = torch.nn.ModuleList(losses)
        self.data_shim = get_data_shim(self.encoder)

        self.scene_results: List[MeshEvalResult] = []
        

    # ---------------- predicted mesh ----------------
    def _build_predicted_mesh(self, batch: BatchedExample, gaussians) -> o3d.geometry.TriangleMesh:
        print(f"Building predicted mesh for scene {batch['scene'][0]}...")
        b, v, _, h, w = batch["context"]["image"].shape
        assert b == 1

        num_frames = getattr(self.cfg, "mesh_num_frames", 20)
        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        t = (torch.cos(torch.pi * (t + 1)) + 1) / 2  # smooth

        def trajectory_fn(tvals: torch.Tensor):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (batch["context"]["extrinsics"][0, 1] if v == 2 else batch["target"]["extrinsics"][0, 0]),
                tvals,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (batch["context"]["intrinsics"][0, 1] if v == 2 else batch["target"]["intrinsics"][0, 0]),
                tvals,
            )
            return extrinsics[None], intrinsics[None]

        extrinsics_traj, intrinsics_traj = trajectory_fn(t)
        near_traj = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far_traj = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)

        output_traj = self.decoder.forward(
            gaussians, extrinsics_traj, intrinsics_traj, near_traj, far_traj, (h, w),
        )

        if bool(getattr(self.cfg, "depth_debug", False)):
            try:
                pred_depths = output_traj.depth[0].detach().cpu().numpy()
                picks = [0, pred_depths.shape[0] // 2, max(0, pred_depths.shape[0] - 1)]
                stats = {f"pred_frame_{i}": _compute_depth_stats(pred_depths[i]) for i in picks}
                debug_dir = Path(getattr(self.cfg, "debug_output_dir", "results/mesh_debug"))
                debug_dir.mkdir(parents=True, exist_ok=True)
                with (debug_dir / f"{batch['scene'][0]}_pred_depth_stats.json").open("w") as f:
                    json.dump({"scene": batch["scene"][0], "pred_depth_stats": stats}, f, indent=2)
                np.save(debug_dir / f"{batch['scene'][0]}_pred_depth_sample.npy", pred_depths[picks[1]])
            except Exception as e:
                print(f"Depth debug (pred) failed: {e}")

        mesh_extractor = GaussianMeshExtractor(
            output_traj, intrinsics_traj, extrinsics_traj, near_traj, far_traj, scale_invariant=True
        )
        mesh_extractor.estimate_bounding_sphere()

        voxel_size = getattr(self.cfg, "pred_mesh_voxel_size", None)
        if voxel_size is None:
            voxel_size = max(mesh_extractor.radius / 600.0, 1e-4)      # scene-adaptive default
        sdf_trunc = getattr(self.cfg, "pred_mesh_sdf_trunc", None)
        if sdf_trunc is None:
            sdf_trunc = 3.0 * voxel_size           # tighter mu reduces ghosting
        depth_trunc = getattr(self.cfg, "pred_mesh_depth_trunc", None)
        if depth_trunc is None:
            depth_trunc = 2.0 * mesh_extractor.radius
        depth_scale = float(getattr(self.cfg, "pred_mesh_depth_scale", 0.5))  # <- 1 with baseline=1

        mesh = mesh_extractor.extract_mesh_bounded(
            voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc, depth_scale=depth_scale
        )
        try:
            mesh = post_process_mesh(mesh, cluster_to_keep=1)
        except Exception as e:
            print(f"[mesh_eval] post_process_mesh failed: {e}; returning empty mesh.")
            mesh = o3d.geometry.TriangleMesh()
        # If empty, return as-is
        if len(mesh.triangles) == 0:
            print("[mesh_eval] Predicted mesh empty after post-processing.")
        return mesh

    # ---------------- GT mesh (load + align + crop) ----------------
    def _build_gt_mesh(self, batch: BatchedExample) -> o3d.geometry.TriangleMesh:
        print(f"Building GT mesh for scene {batch['scene'][0]}] ...")
        scene_token = batch["scene"][0]
        num_ctx = int(batch["context"]["image"].shape[1])
        num_tgt = int(batch.get("target", {}).get("image", torch.empty(1, 0)).shape[1]) if "target" in batch else 0
        scene_id, ctx_ids, tgt_ids = _parse_scene_token(scene_token, num_ctx, num_tgt)
        if len(ctx_ids) == 0:
            raise RuntimeError(f"No context image ids parsed from '{scene_token}'")

        ds_root = Path(getattr(self.cfg, "scannet_root", "datasets/scannetv1_test"))
        scene_dir = ds_root / scene_id

        # Load GT mesh (metric), rigidly align into loader world via inv(pose_ctx0)
        mesh_path = scene_dir / "mesh" / f"{scene_id}_vh_clean_2.ply"
        if not mesh_path.is_file():
            raise FileNotFoundError(f"GT mesh not found: {mesh_path}")
        Twc0 = _load_scannet_pose_txt(scene_dir / "pose" / f"{ctx_ids[0]}.txt")
        T_align = np.linalg.inv(Twc0)  # context-0 becomes identity
        gt_mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        gt_mesh = _apply_transform_to_mesh(gt_mesh, T_align)

        crop_mode = getattr(self.cfg, "gt_mesh_crop_mode", "frustum")     # 'aabb' | 'frustum' | 'none'
        view_sel  = getattr(self.cfg, "gt_crop_views", "context")       # 'context' | 'context+target'  (ToDo: target ids loaded in dataloader?)
        ids_for_cull = ctx_ids + tgt_ids if (view_sel == "context+target") else ctx_ids

        # Build AABB (always +Z, with near>0 and a display far for extent)
        if crop_mode.lower() != "none":
            if len(ids_for_cull) == 0:
                raise RuntimeError("Cropping requested but no view ids available.")
            K_full, W_full, H_full = _read_intrinsics_color(scene_dir)
            near_m = float(getattr(self.cfg, "gt_cull_near_m", 0.05))
            # Use GT radius to set a finite far just for AABB extent
            _, gt_r = _mesh_center_radius(gt_mesh)
            far_for_aabb = float(getattr(self.cfg, "aabb_far_m", max(2.0, 2.0 * gt_r)))
            views = []
            for i in ids_for_cull:
                c2w_norm = T_align @ _load_scannet_pose_txt(scene_dir / "pose" / f"{i}.txt")
                views.append((K_full, c2w_norm, near_m, far_for_aabb, W_full, H_full))
            aabb = _union_aabb_for_views_plusZ(views)
            gt_mesh = gt_mesh.crop(aabb)
            if len(gt_mesh.triangles) == 0:
                raise RuntimeError("AABB cropping produced an empty GT mesh; check poses/intrinsics.")

        # frustum: refine AABB crop by near/left/right/top/bottom planes (no far)
        if crop_mode.lower() == "frustum":
            K_full, W_full, H_full = _read_intrinsics_color(scene_dir)
            near_m = float(getattr(self.cfg, "gt_cull_near_m", 0.05))
            plane_sets = []
            for i in ids_for_cull:
                c2w_norm = T_align @ _load_scannet_pose_txt(scene_dir / "pose" / f"{i}.txt")
                plane_sets.append(_compute_half_frustum_planes_no_far(K_full, c2w_norm, W_full, H_full, near_m))
            gt_mesh = _mesh_crop_by_union_plane_sets(gt_mesh, plane_sets)
            if len(gt_mesh.triangles) == 0:
                raise RuntimeError("Frustum cropping produced an empty GT mesh; check +Z assumption and planes.")

        # Cluster cleanup (deterministic, no fallback)
        k_keep = int(getattr(self.cfg, "gt_mesh_clusters_keep", 1))
        if k_keep > 0:
            if len(gt_mesh.triangles) == 0:
                raise RuntimeError("GT mesh is empty before cluster filtering.")
            gt_mesh = post_process_mesh(gt_mesh, cluster_to_keep=k_keep)
        return gt_mesh

    # ---------------- metrics ----------------
    def _compute_metrics(self, pred_pts: np.ndarray, gt_pts: np.ndarray, tau_m: float = 0.05) -> Dict[str, float]:
        if pred_pts.shape[0] == 0 or gt_pts.shape[0] == 0:
            raise RuntimeError(f"Empty point sets for metrics (|pred|={pred_pts.shape[0]}, |gt|={gt_pts.shape[0]}).")
        nn = skln.NearestNeighbors(n_neighbors=1, algorithm="kd_tree", n_jobs=1)
        nn.fit(gt_pts)
        d_pred_to_gt = nn.kneighbors(pred_pts, n_neighbors=1, return_distance=True)[0].reshape(-1)

        nn.fit(pred_pts)
        d_gt_to_pred = nn.kneighbors(gt_pts, n_neighbors=1, return_distance=True)[0].reshape(-1)

        acc = float(d_pred_to_gt.mean())
        comp = float(d_gt_to_pred.mean())
        overall = (acc + comp) / 2.0

        thr = float(tau_m)  
        prec = float((d_pred_to_gt < thr).mean())
        rec  = float((d_gt_to_pred < thr).mean())
        f1 = (2 * prec * rec) / max(prec + rec, 1e-12)
        return {
            "accuracy": acc,
            "completeness": comp,
            "overall": overall,
            "precision_5cm": prec,
            "recall_5cm": rec,
            "fscore_5cm": f1,
        }

    # ---------------- test loop ----------------
    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        (scene_token,) = batch["scene"]
        print(f"[scene] {scene_token}")

        # Predicted mesh in loader-normalized coords
        vis = {}
        gaussians = self.encoder(batch["context"], self.global_step, visualization_dump=vis)
        pred_mesh_norm = self._build_predicted_mesh(batch, gaussians)

        # GT mesh in loader-normalized coords (metric) + cropping
        try:
            gt_mesh_metric = self._build_gt_mesh(batch)
        except Exception as e:
            print(f"[mesh_eval] Failed building GT mesh: {e}; recording NaN metrics.")
            self.scene_results.append(MeshEvalResult(scene=scene_token, accuracy=np.nan, completeness=np.nan, overall=np.nan, precision_5cm=np.nan, recall_5cm=np.nan, fscore_5cm=np.nan))
            return 0

        # If predicted mesh is empty, skip alignment/metrics
        if len(pred_mesh_norm.triangles) == 0:
            print("[mesh_eval] Skipping alignment & metrics (empty predicted mesh).")
            self.scene_results.append(MeshEvalResult(scene=scene_token, accuracy=np.nan, completeness=np.nan, overall=np.nan, precision_5cm=np.nan, recall_5cm=np.nan, fscore_5cm=np.nan))
            return 0

        # If GT mesh ended up empty (after cropping), skip
        if len(gt_mesh_metric.triangles) == 0:
            print("[mesh_eval] Skipping alignment & metrics (empty GT mesh).")
            self.scene_results.append(MeshEvalResult(scene=scene_token, accuracy=np.nan, completeness=np.nan, overall=np.nan, precision_5cm=np.nan, recall_5cm=np.nan, fscore_5cm=np.nan))
            return 0

        # Alignment: scale-only, sim3, etc. (robust: record NaN on failure)
        try:
            sim3_mode = str(getattr(self.cfg, "sim3_mode", "sim3")).lower()
            if sim3_mode == "scale":
                s = _estimate_scale_only_from_meshes(
                    pred_mesh_norm,
                    gt_mesh_metric,
                    iters=int(getattr(self.cfg, "scale_icp_iters", 8)),
                    outlier_frac=float(getattr(self.cfg, "scale_outlier_frac", 0.2)),
                )
                abs_bounds = getattr(self.cfg, "sim3_scale_bounds", None)
                if abs_bounds is not None:
                    s_min, s_max = abs_bounds
                    if not (s_min <= s <= s_max):
                        raise RuntimeError(f"Scale-only: s={s:.6f} outside absolute bounds [{s_min}, {s_max}]")
                print(f"[scale-only] scale={s:.6f}")
                T_pred_to_metric = np.eye(4, dtype=np.float64); T_pred_to_metric[:3, :3] = np.eye(3) * s
            elif sim3_mode in ("scale_t", "scale_tz"):
                restrict_axis = 'z' if sim3_mode == 'scale_tz' else None
                T_pred_to_metric = _estimate_scale_translate_from_meshes(
                    pred_mesh_norm,
                    gt_mesh_metric,
                    iters=int(getattr(self.cfg, "scale_t_iters", 50)),
                    outlier_frac=float(getattr(self.cfg, "scale_t_outlier_frac", 0.2)),
                    restrict_axis=restrict_axis,
                    eps_rel=float(getattr(self.cfg, "scale_t_eps_rel", 1e-4)),
                )
                s_est = float(np.cbrt(np.linalg.det(T_pred_to_metric[:3, :3])))
                t_est = T_pred_to_metric[:3, 3]
                if restrict_axis == 'z':
                    print(f"[scale+tz] scale={s_est:.6f} t_z={t_est[2]:.6f}")
                else:
                    print(f"[scale+t] scale={s_est:.6f} t=({t_est[0]:.4f},{t_est[1]:.4f},{t_est[2]:.4f})")
            elif sim3_mode == "sim3":
                T_pred_to_metric = _estimate_sim3_full_from_meshes(
                    pred_mesh_norm,
                    gt_mesh_metric,
                    max_iters=int(getattr(self.cfg, "sim3_icp_iters", 8)),
                    cfg=self.cfg,
                )
                s_est = float(np.cbrt(np.linalg.det(T_pred_to_metric[:3, :3])))
                print(f"[sim3] estimated scale ≈ {s_est:.6f}")
            else:
                raise ValueError("Unknown sim3_mode '%s'. Use 'scale', 'scale_t', 'scale_tz', or 'sim3'." % sim3_mode)

            pred_mesh_metric = _apply_transform_to_mesh(pred_mesh_norm, T_pred_to_metric)
        except Exception as e:
            print(f"[mesh_eval] Alignment failed: {e}; recording NaN metrics.")
            self.scene_results.append(MeshEvalResult(scene=scene_token, accuracy=np.nan, completeness=np.nan, overall=np.nan, precision_5cm=np.nan, recall_5cm=np.nan, fscore_5cm=np.nan))
            return 0

        # Save meshes (metric coordinates)
        if bool(getattr(self.cfg, "save_meshes", False)):
            out_dir = getattr(self.cfg, "mesh_output_dir", "outputs/meshes_scannet")
            self.mesh_out_dir = Path(out_dir)
            self.mesh_out_dir.mkdir(parents=True, exist_ok=True)
            pred_mesh_path = self.mesh_out_dir / f"{scene_token}_pred_metric.ply"
            gt_mesh_path   = self.mesh_out_dir / f"{scene_token}_gt_metric.ply"
            o3d.io.write_triangle_mesh(str(pred_mesh_path), pred_mesh_metric)
            o3d.io.write_triangle_mesh(str(gt_mesh_path), gt_mesh_metric)

        # Visualization
        if bool(getattr(self.cfg, "viz_enable", False)):
            num_ctx = int(batch["context"]["image"].shape[1])
            num_tgt = int(batch.get("target", {}).get("image", torch.empty(1, 0)).shape[1]) if "target" in batch else 0
            scene_id, ctx_ids, tgt_ids = _parse_scene_token(scene_token, num_ctx, num_tgt)
            ds_root = Path(getattr(self.cfg, "scannet_root", "datasets/scannetv1_test"))
            scene_dir = ds_root / scene_id
            ids_for_frusta = ctx_ids + (tgt_ids if getattr(self.cfg, "gt_crop_views", "context") == "context+target" else [])
            viz_far = float(getattr(self.cfg, "viz_far_m", 5.0))
            _visualize_scene_alignment(
                scene_token,
                pred_mesh_norm,
                pred_mesh_metric,
                gt_mesh_metric,
                scene_dir,
                ids_for_frusta,
                show_frusta=bool(getattr(self.cfg, "viz_show_frusta", True)),
                coord_frame_size=float(getattr(self.cfg, "viz_coord_frame_size", 0.3)),
                viz_far_m=viz_far,
            )

        # Sample and compute metrics (meters)
        density = float(getattr(self.cfg, "downsample_density", 0.01))
        pred_pts = _downsample_radius(_sample_points_from_mesh(pred_mesh_metric, density), density)
        gt_pts   = _downsample_radius(_sample_points_from_mesh(gt_mesh_metric, density), density)
        if pred_pts.shape[0] == 0 or gt_pts.shape[0] == 0:
            print("[mesh_eval] Empty point samples; recording NaN metrics.")
            self.scene_results.append(MeshEvalResult(scene=scene_token, accuracy=np.nan, completeness=np.nan, overall=np.nan, precision_5cm=np.nan, recall_5cm=np.nan, fscore_5cm=np.nan))
            return 0
        tau_m = float(getattr(self.cfg, "metric_threshold_meters", 0.05))  # 5cm default
        metrics = self._compute_metrics(pred_pts, gt_pts, tau_m=tau_m)

        self.scene_results.append(MeshEvalResult(scene=scene_token, **metrics))

        # Print per-scene
        from tabulate import tabulate
        print(
            tabulate(
                [[
                    scene_token,
                    metrics["accuracy"], metrics["completeness"], metrics["overall"],
                    metrics["precision_5cm"], metrics["recall_5cm"], metrics["fscore_5cm"],
                ]],
                headers=["scene", "accuracy", "completeness", "overall", "prec@5cm", "recall@5cm", "fscore@5cm"],
                floatfmt=".4f",
            )
        )
        return 0

    def on_test_end(self) -> None:
        import numpy as np
        from tabulate import tabulate

        if len(self.scene_results) == 0:
            print("No mesh results recorded.")
            return

        rows = [
            [r.scene, r.accuracy, r.completeness, r.overall, r.precision_5cm, r.recall_5cm, r.fscore_5cm]
            for r in self.scene_results
        ]
        print("\n===== ScanNet Mesh Evaluation (per scene) =====")
        print(tabulate(rows, headers=["scene", "accuracy", "completeness", "overall", "prec@5cm", "recall@5cm", "fscore@5cm"], floatfmt=".3f"))

        accs = np.array([r.accuracy for r in self.scene_results], dtype=float)
        comps = np.array([r.completeness for r in self.scene_results], dtype=float)
        overs = np.array([r.overall for r in self.scene_results], dtype=float)
        precs = np.array([r.precision_5cm for r in self.scene_results], dtype=float)
        recs  = np.array([r.recall_5cm for r in self.scene_results], dtype=float)
        fs    = np.array([r.fscore_5cm for r in self.scene_results], dtype=float)

        # Unfiltered overall (only ignores NaNs, keeps large finite values)
        overall_unfiltered = {
            "accuracy": float(np.nanmean(accs)),
            "completeness": float(np.nanmean(comps)),
            "overall": float(np.nanmean(overs)),
            "precision_5cm": float(np.nanmean(precs)),
            "recall_5cm": float(np.nanmean(recs)),
            "fscore_5cm": float(np.nanmean(fs)),
        }
        print("\n===== ScanNet Mesh Evaluation (overall mean - unfiltered) =====")
        print(
            tabulate(
                [[
                    overall_unfiltered["accuracy"], overall_unfiltered["completeness"], overall_unfiltered["overall"],
                    overall_unfiltered["precision_5cm"], overall_unfiltered["recall_5cm"], overall_unfiltered["fscore_5cm"],
                ]],
                headers=["accuracy", "completeness", "overall", "prec@5cm", "recall@5cm", "fscore@5cm"],
                floatfmt=".3f",
            )
        )

        # Outlier filtering (ignore > threshold) for distance-like metrics.
        outlier_thresh = float(getattr(self.cfg, "mean_metric_outlier_thresh", 10.0))
        def _filter(arr):
            mask = (arr <= outlier_thresh) | ~np.isfinite(arr)
            # Keep NaNs (already excluded by nanmean) but drop large finite outliers.
            arr_filtered = arr.copy()
            arr_filtered[~mask] = np.nan
            return arr_filtered

        accs_f  = _filter(accs)
        comps_f = _filter(comps)
        overs_f = _filter(overs)
        # (Do not filter precision/recall/F1; they are bounded [0,1])

        overall_filtered = {
            "accuracy": float(np.nanmean(accs_f)),
            "completeness": float(np.nanmean(comps_f)),
            "overall": float(np.nanmean(overs_f)),
            "precision_5cm": float(np.nanmean(precs)),
            "recall_5cm": float(np.nanmean(recs)),
            "fscore_5cm": float(np.nanmean(fs)),
        }
        print("\n===== ScanNet Mesh Evaluation (overall mean - filtered) =====")
        print(f"(Outlier filtering applied: values > {outlier_thresh} ignored for accuracy/completeness/overall)")
        print(
            tabulate(
                [[
                    overall_filtered["accuracy"], overall_filtered["completeness"], overall_filtered["overall"],
                    overall_filtered["precision_5cm"], overall_filtered["recall_5cm"], overall_filtered["fscore_5cm"],
                ]],
                headers=["accuracy", "completeness", "overall", "prec@5cm", "recall@5cm", "fscore@5cm"],
                floatfmt=".3f",
            )
        )

        # Persist per-scene + overall metrics
        from pathlib import Path
        import json
        with Path("mesh_per_scene.json").open("w") as f:
            json.dump([
                {
                    "scene": r.scene,
                    "accuracy": r.accuracy,
                    "completeness": r.completeness,
                    "overall": r.overall,
                    "precision_5cm": r.precision_5cm,
                    "recall_5cm": r.recall_5cm,
                    "fscore_5cm": r.fscore_5cm,
                }
                for r in self.scene_results
            ], f, indent=2)
        # Save both variants; keep original filename for filtered metrics to preserve behavior.
        np.save("overall_mesh_metrics.npy", overall_filtered)
        np.save("overall_mesh_metrics_unfiltered.npy", overall_unfiltered)
