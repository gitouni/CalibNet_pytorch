import torch
import torch.nn.functional as F
from torch import nn
from math import sqrt
from utils import so3
from losses.chamfer_loss import chamfer_distance
from scipy.spatial.transform import Rotation
import numpy as np
    
class Photo_Loss(nn.Module):
    def __init__(self,scale=1.0,reduction='mean'):
        super(Photo_Loss, self).__init__()
        assert reduction in ['sum','mean','none'], 'Unknown or invalid reduction'
        self.scale = scale
        self.reduction = reduction
    def forward(self,input:torch.Tensor,target:torch.Tensor):
        """Photo loss

        Args:
            input (torch.Tensor): (B,H,W)
            target (torch.Tensor): (B,H,W)

        Returns:
            torch.Tensor: scaled mse loss between input and target
        """
        return F.mse_loss(input/self.scale,target/self.scale,reduction=self.reduction)
    def __call__(self,input:torch.Tensor,target:torch.Tensor)->torch.Tensor:
        return self.forward(input,target)
    
class ChamferDistanceLoss(nn.Module):
    def __init__(self,scale=1.0,reduction='mean'):
        super(ChamferDistanceLoss, self).__init__()
        assert reduction in ['sum','mean','none'], 'Unknown or invalid reduction'
        self.reduction = reduction
        self.scale = scale
    def forward(self, template, source):
        p0 = template/self.scale
        p1 = source/self.scale
        if self.reduction == 'none':
            return chamfer_distance(p0, p1)
        elif self.reduction == 'mean':
            return torch.mean(chamfer_distance(p0, p1),dim=0)
        elif self.reduction == 'sum':
            return torch.sum(chamfer_distance(p0, p1),dim=0)
    def __call__(self,template:torch.Tensor,source:torch.Tensor)->torch.Tensor:
        return self.forward(template,source)


def geodesic_distance(x:torch.Tensor,gt:torch.Tensor)->tuple:
    """geodesic distance for evaluation

    Args:
        x (torch.Tensor): (B,4,4)
        gt (torch.Tensor): (B,4,4)

    Returns:
        torch.Tensor(1),torch.Tensor(1): distance of component R and T 
    """
    R = x[:,:3,:3]  # (B,3,3)
    T = x[:,:3,3]  # (B,3)
    gtR = gt[:,:3,:3]  # (B,3,3)
    gtT = gt[:,:3,3]  # (B,3)
    dR = so3.log(R.transpose(1,2).bmm(gtR)) # (B,3)
    dR = F.mse_loss(dR,torch.zeros_like(dR).to(dR),reduction='none').mean(dim=1)  # (B,3) -> (B,1)
    dR = torch.sqrt(dR).mean(dim=0)  # (B,1) -> (1,)  Rotation RMSE (mean in batch)
    dT = F.mse_loss(T,gtT,reduction='none').mean(dim=1) # (B,3) -> (B,1)
    dT = torch.sqrt(dT).mean(dim=0)  # (B,1) -> (1,) Translation RMSE (mean in batch)
    return dR, dT

def gt2euler(gt:np.ndarray):
    """gt transformer to euler anlges and translation

    Args:
        gt (np.ndarray): 4x4

    Returns:
        angle_gt, trans_gt: (3,1),(3,1)
    """
    R_gt = gt[:3, :3]
    euler_angle = Rotation.from_matrix(R_gt)
    anglez_gt, angley_gt, anglex_gt = euler_angle.as_euler('zyx')
    angle_gt = np.array([anglex_gt, angley_gt, anglez_gt])
    trans_gt_t = -R_gt @ gt[:3, 3]
    return angle_gt, trans_gt_t