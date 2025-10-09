import torch
import numpy as np
import os
import math
import open3d as o3d
from tqdm import tqdm
from ..geometry.projection import get_fov
from ..model.decoder.cuda_splatting import get_projection_matrix
from einops import rearrange
from typing import NamedTuple


def focus_point_fn(poses: np.ndarray) -> np.ndarray:
  """Calculate nearest point to all focal axes in poses."""
  directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
  m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
  mt_m = np.transpose(m, [0, 2, 1]) @ m
  focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
  return focus_pt


def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    # Handle degenerate / empty meshes gracefully
    if cluster_n_triangles.size == 0:
        print("[post_process_mesh] No triangle clusters found; returning empty mesh.")
        empty = o3d.geometry.TriangleMesh()
        return empty

    keep_k = min(cluster_to_keep, cluster_n_triangles.size)
    try:
        n_cluster = np.sort(cluster_n_triangles.copy())[-keep_k]
    except Exception:
        print("[post_process_mesh] Failed to compute cluster threshold; returning empty mesh.")
        return o3d.geometry.TriangleMesh()
    n_cluster = max(int(n_cluster), 50)  # min triangle threshold

    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    if triangles_to_remove.size == 0:
        print("[post_process_mesh] triangles_to_remove empty; returning original mesh copy.")
        return mesh_0

    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0


class ViewpointStack(NamedTuple):
    projection_matrix: torch.Tensor #(4,4)
    world_view_transform: torch.Tensor #(4,4)
    image_width: int
    image_height: int
    
    

def to_cam_open3d(viewpoint_stack):
    camera_traj = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        W = viewpoint_cam.image_width
        H = viewpoint_cam.image_height
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, 0, 1]]).float().cuda().T
        intrins =  (viewpoint_cam.projection_matrix @ ndc2pix)[:3,:3].T
        intrinsic=o3d.camera.PinholeCameraIntrinsic(
            width=viewpoint_cam.image_width,
            height=viewpoint_cam.image_height,
            cx = intrins[0,2].item(),
            cy = intrins[1,2].item(), 
            fx = intrins[0,0].item(), 
            fy = intrins[1,1].item()
        )

        extrinsic=np.asarray((viewpoint_cam.world_view_transform.T).cpu().numpy())
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj


class GaussianMeshExtractor(object):
    def __init__(self, gaussian_decoder_output, intrinsics, extrinsics, near, far, scale_invariant=True, resolution=1024, bg_color=None):
        """
        gaussian_decoder_output (class Decoder [color, depth])
        intrinsics (torch.Tensor [B,N,H,W])
        extrinsics (torch.Tensor [B,N,H,W])
        near (torch.Tensor [N,H,W])
        far (torch.Tensor[N,H,W])
        """
        self.rgbmaps = gaussian_decoder_output.color[0]  #(N,3,H,W)
        self.depthmaps = gaussian_decoder_output.depth[0] #(N,H,W)
        self.intrinsics = intrinsics[0] #(N,3,3)
        self.extrinsics = extrinsics[0] #(N,4,4)
        self.resolution = resolution
        self.near = near[0]
        self.far = far[0]
        self.scale_invariant = scale_invariant

        if bg_color is None:
            bg_color = [0, 0, 0]


    def estimate_bounding_sphere(self):
        """
        Estimate the bounding sphere given camera poses
        
        Args:
            extrinsic (torch.Tensor[B,N,4,4]): camera-to-world transformation
        """
        self.c2ws = np.array([(np.asarray((cam).cpu().numpy())) for cam in self.extrinsics]) #(N,4,4)
        ## opencv to opengl transformations
        poses = self.c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
        ## computes centroid 
        center = (focus_point_fn(poses)) 
        self.radius = np.linalg.norm(self.c2ws[:,:3,3] - center, axis=-1).min()
        self.center = torch.from_numpy(center).float().cuda()
        print(f"The estimated bounding radius is {self.radius:.2f}")
        print(f"Use at least {2.0 * self.radius:.2f} for depth_trunc")



    @torch.no_grad()
    def extract_mesh_bounded(self, voxel_size=0.004, sdf_trunc=0.016, depth_trunc=100.0, mask_background=True, depth_scale=20.0):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.
        
        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_backgrond: whether to mask backgroud, only works when the dataset have masks

        return o3d.mesh
        """
        print ("Running tsdf volume integration...")
        print(f'voxel_size: {voxel_size}')
        print(f'sdf_trunc: {sdf_trunc}')
        print(f'depth_truc: {depth_trunc}')

        # Make sure everything is in a range where numerical issues don't appear.
        if self.scale_invariant:
            scale = 1 / self.near
            #scale = 1
            extrinsics = self.extrinsics.clone()
            extrinsics[..., :3, 3] = extrinsics[..., :3, 3] #* scale[:, None]
            near = self.near #* scale
            far = self.far #* scale

            fov_x, fov_y = get_fov(self.intrinsics).unbind(dim=-1)
            ## get projection matrix ##
            projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
            projection_matrix = rearrange(projection_matrix, "b i j -> b j i") # convert to opengl
            world_view_transform = rearrange(extrinsics.inverse(), "b i j -> b j i") # convert to opengl
        
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        viewpoint_stack = [
            ViewpointStack(
            projection_matrix=projection_matrix[i],
            world_view_transform=world_view_transform[i],
            image_width=self.rgbmaps.shape[-1],
            image_height=self.rgbmaps.shape[2]
        ) for i in range(projection_matrix.shape[0])
        ]
        for i, cam_o3d in tqdm(enumerate(to_cam_open3d(viewpoint_stack)), desc="TSDF intergration"):
            rgb = self.rgbmaps[i]
            depth = self.depthmaps[i].unsqueeze(0)

            # Prepare Open3D RGBD with C-contiguous buffers
            rgb_np = rgb.permute(1, 2, 0).detach().cpu().numpy()
            rgb_u8 = np.ascontiguousarray((np.clip(rgb_np, 0.0, 1.0) * 255.0).astype(np.uint8))
            depth_np = depth.squeeze(0).detach().cpu().numpy()  # H,W
            depth_f32 = np.ascontiguousarray(depth_np.astype(np.float32))

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rgb_u8),
                o3d.geometry.Image(depth_f32),
                depth_scale=depth_scale,
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False,
            )

            volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

        mesh = volume.extract_triangle_mesh()
        return mesh
    