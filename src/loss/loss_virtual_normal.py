from dataclasses import dataclass

import torch
import numpy as np
from einops import reduce, rearrange
from jaxtyping import Float
from torch import Tensor
import torch.nn.functional as F

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss

@dataclass
class LossNormalCfg:
    lambda_normal: float
    lambda_distortion: float
    apply_normal_after_step: int
    apply_distortion_after_step: int
    normal_valid_threshold: float = 1e-6  # threshold to consider a normal valid


@dataclass
class LossNormalCfgWrapper:
    normal: LossNormalCfg


class LossVirtualNormal(Loss[LossNormalCfg, LossNormalCfgWrapper]):
    
    def __init__(self, cfg: LossNormalCfgWrapper) -> None:
        super().__init__(cfg)
        
        # TODO: read from config - temp param init
        W, H = 256, 256
        self.input_size = (H, W)
        self.fx = torch.tensor([W], dtype=torch.float32).cuda()
        self.fy = torch.tensor([H], dtype=torch.float32).cuda()
        self.u0 = torch.tensor(W // 2, dtype=torch.float32).cuda()
        self.v0 = torch.tensor(H // 2, dtype=torch.float32).cuda()
        
        self.init_image_coor()
        self.delta_cos = 0.867
        self.delta_diff_x = 0.01
        self.delta_diff_y = 0.01
        self.delta_diff_z = 0.01
        self.delta_z = 0.0001
        self.sample_ratio = 0.15
        self.select = True
    
    def init_image_coor(self):
        x_row = np.arange(0, self.input_size[1])
        x = np.tile(x_row, (self.input_size[0], 1))
        x = x[np.newaxis, :, :]
        x = x.astype(np.float32)
        x = torch.from_numpy(x.copy()).cuda()
        self.u_u0 = x - self.u0

        y_col = np.arange(0, self.input_size[0])  # y_col = np.arange(0, height)
        y = np.tile(y_col, (self.input_size[1], 1)).T
        y = y[np.newaxis, :, :]
        y = y.astype(np.float32)
        y = torch.from_numpy(y.copy()).cuda()
        self.v_v0 = y - self.v0

    def transfer_xyz(self, depth):
        x = self.u_u0 * torch.abs(depth) / self.fx
        y = self.v_v0 * torch.abs(depth) / self.fy
        z = depth
        pw = torch.cat([x, y, z], 1).permute(0, 2, 3, 1) # [b, h, w, c]
        return pw
    
    def select_index(self):
        valid_width = self.input_size[1]
        valid_height = self.input_size[0]
        num = valid_width * valid_height
        p1 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p1)
        p2 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p2)
        p3 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p3)

        p1_x = p1 % self.input_size[1]
        p1_y = (p1 / self.input_size[1]).astype(int)

        p2_x = p2 % self.input_size[1]
        p2_y = (p2 / self.input_size[1]).astype(int)

        p3_x = p3 % self.input_size[1]
        p3_y = (p3 / self.input_size[1]).astype(int)
        p123 = {'p1_x': p1_x, 'p1_y': p1_y, 'p2_x': p2_x, 'p2_y': p2_y, 'p3_x': p3_x, 'p3_y': p3_y}
        return p123
    
    def form_pw_groups(self, p123, pw):
        """
        Form 3D points groups, with 3 points in each grouup.
        :param p123: points index
        :param pw: 3D points
        :return:
        """
        p1_x = p123['p1_x']
        p1_y = p123['p1_y']
        p2_x = p123['p2_x']
        p2_y = p123['p2_y']
        p3_x = p123['p3_x']
        p3_y = p123['p3_y']

        pw1 = pw[:, p1_y, p1_x, :]
        pw2 = pw[:, p2_y, p2_x, :]
        pw3 = pw[:, p3_y, p3_x, :]
        # [B, N, 3(x,y,z), 3(p1,p2,p3)]
        pw_groups = torch.cat([pw1[:, :, :, np.newaxis], pw2[:, :, :, np.newaxis], pw3[:, :, :, np.newaxis]], 3)
        return pw_groups


    def form_pw_groups_for_normals(self, p123, pred_normal):
        """
        Select predicted normals based on the indices of already selecetd 3D points in p123
        :param p123: points index
        :param pw: structured pred_normal (b, h, w, c)
        :return:
        """
        p1_x = p123['p1_x']
        p1_y = p123['p1_y']
        p2_x = p123['p2_x']
        p2_y = p123['p2_y']
        p3_x = p123['p3_x']
        p3_y = p123['p3_y']

        pw1 = pred_normal[:, p1_y, p1_x, :]
        pw2 = pred_normal[:, p2_y, p2_x, :]
        pw3 = pred_normal[:, p3_y, p3_x, :]
        # [B, N, 3(x,y,z), 3(p1,p2,p3)]
        pw_normal_groups = torch.cat([pw1[:, :, :, np.newaxis], pw2[:, :, :, np.newaxis], pw3[:, :, :, np.newaxis]], 3)
        return pw_normal_groups


    def filter_mask(self, p123, gt_xyz, delta_cos=0.867,
                    delta_diff_x=0.005,
                    delta_diff_y=0.005,
                    delta_diff_z=0.005):
        pw = self.form_pw_groups(p123, gt_xyz)
        pw12 = pw[:, :, :, 1] - pw[:, :, :, 0]
        pw13 = pw[:, :, :, 2] - pw[:, :, :, 0]
        pw23 = pw[:, :, :, 2] - pw[:, :, :, 1]
        ###ignore linear
        pw_diff = torch.cat([pw12[:, :, :, np.newaxis], pw13[:, :, :, np.newaxis], pw23[:, :, :, np.newaxis]],
                            3)  # [b, n, 3, 3]
        m_batchsize, groups, coords, index = pw_diff.shape
        proj_query = pw_diff.view(m_batchsize * groups, -1, index).permute(0, 2, 1)  # (B* X CX(3)) [bn, 3(p123), 3(xyz)]
        proj_key = pw_diff.view(m_batchsize * groups, -1, index)  # B X  (3)*C [bn, 3(xyz), 3(p123)]
        q_norm = proj_query.norm(2, dim=2)
        nm = torch.bmm(q_norm.view(m_batchsize * groups, index, 1), q_norm.view(m_batchsize * groups, 1, index)) #[]
        energy = torch.bmm(proj_query, proj_key)  # transpose check [bn, 3(p123), 3(p123)]
        norm_energy = energy / (nm + 1e-8)
        norm_energy = norm_energy.view(m_batchsize * groups, -1)
        mask_cos = torch.sum((norm_energy > delta_cos) + (norm_energy < -delta_cos), 1) > 3  # igonre
        mask_cos = mask_cos.view(m_batchsize, groups)
        ##ignore padding and invilid depth
        mask_pad = torch.sum(pw[:, :, 2, :] > self.delta_z, 2) == 3

        ###ignore near
        mask_x = torch.sum(torch.abs(pw_diff[:, :, 0, :]) < delta_diff_x, 2) > 0
        mask_y = torch.sum(torch.abs(pw_diff[:, :, 1, :]) < delta_diff_y, 2) > 0
        mask_z = torch.sum(torch.abs(pw_diff[:, :, 2, :]) < delta_diff_z, 2) > 0

        mask_ignore = (mask_x & mask_y & mask_z) | mask_cos
        mask_near = ~mask_ignore
        mask = mask_pad & mask_near

        return mask, pw

    def select_points_groups(self, gt_depth, pred_normal):
        pw_gt = self.transfer_xyz(gt_depth)     # (b, h, w, c)
        pred_normal = pred_normal.permute(0, 2, 3, 1)     # (b, h, w, c)
        # make the depth corresponding to invalid prediction normals zero (invalid) to ignore them later in the filtering process
        invalid_rend = torch.norm(pred_normal, dim=3, keepdim=True) < self.cfg.normal_valid_threshold
        invalid_rend = invalid_rend.repeat(1, 1, 1, 3)
        pw_gt[invalid_rend] = 0.0
        B, C, H, W = gt_depth.shape
        p123 = self.select_index()
        # mask:[b, n], pw_groups_gt: [b, n, 3(x,y,z), 3(p1,p2,p3)]
        mask, pw_groups_gt = self.filter_mask(p123, pw_gt,
                                              delta_cos=0.867,
                                              delta_diff_x=0.005,
                                              delta_diff_y=0.005,
                                              delta_diff_z=0.005)
        # [b, n, 3, 3]
        pw_normal_groups = self.form_pw_groups_for_normals(p123, pred_normal)
        mask_broadcast = mask.repeat(1, 9).reshape(B, 3, 3, -1).permute(0, 3, 1, 2)
        pw_groups_gt_not_ignore = pw_groups_gt[mask_broadcast].reshape(1, -1, 3, 3)
        pw_normal_groups_not_ignore = pw_normal_groups[mask_broadcast].reshape(1, -1, 3, 3)
        return pw_groups_gt_not_ignore, pw_normal_groups_not_ignore


    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        """
        Virtual Normal Loss.
        :param pred_depth: predicted depth map
        :param data: target label, ground truth depth, [B, C=1, H, W], padding region [padding_up, padding_down]
        :return:
        """

        lambda_normal = self.cfg.lambda_normal if global_step > self.cfg.apply_normal_after_step else 0.0

        depth = prediction.depth  # shape: (B, V, H, W)
        rend_normal = prediction.rend_normal    # shape: (B, V, C=3, H, W)

        gt_depth = rearrange(depth, "b v h w -> (b v) h w").unsqueeze(1) 
        # Reshape to [b, -1] to compute min and max for each sample
        b = gt_depth.shape[0]
        depth_flat = gt_depth.view(b, -1)
        depth_min = depth_flat.min(dim=1)[0].view(b, 1, 1, 1)
        depth_max = depth_flat.max(dim=1)[0].view(b, 1, 1, 1)
        normalized_depth = (gt_depth - depth_min) / (depth_max - depth_min + 1e-8)
        pred_normal = rearrange(rend_normal, "b v c h w -> (b v) c h w")            
        # output shape - 3D either for points or normals for 3 points (B, N, 3(x,y,z), 3(p1,p2,p3))
        gt_points, pred_rend_normal = self.select_points_groups(normalized_depth, pred_normal)

        gt_p12 = gt_points[:, :, :, 1] - gt_points[:, :, :, 0]
        gt_p13 = gt_points[:, :, :, 2] - gt_points[:, :, :, 0]
        gt_normal = torch.cross(gt_p12, gt_p13, dim=2)

        # TODO: review later to use all predicted normals for 3 points
        # for now pick the pred_normal of point 1
        dt_normal = pred_rend_normal[:, :, :, 0]
        
        dt_norm = torch.norm(dt_normal, 2, dim=2, keepdim=True)
        gt_norm = torch.norm(gt_normal, 2, dim=2, keepdim=True)
        dt_mask = dt_norm == 0.0
        gt_mask = gt_norm == 0.0
        dt_mask = dt_mask.to(torch.float32)
        gt_mask = gt_mask.to(torch.float32)
        dt_mask *= 0.01
        gt_mask *= 0.01
        gt_norm = gt_norm + gt_mask
        dt_norm = dt_norm + dt_mask
        gt_normal = gt_normal / gt_norm
        dt_normal = dt_normal / dt_norm
        loss = torch.abs(gt_normal - dt_normal)
        loss = torch.sum(torch.sum(loss, dim=2), dim=0)
        if self.select:
            loss, indices = torch.sort(loss, dim=0, descending=False)
            loss = loss[int(loss.size(0) * 0.25):]
        
        total_normal_loss = lambda_normal * torch.mean(loss)

        return total_normal_loss
            
