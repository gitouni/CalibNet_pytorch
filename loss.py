import torch
import torch.nn.functional as F
from torch import nn
from math import sqrt
from utils import so3
from losses.chamfer_loss import chamfer_distance
  
    
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
    dR = torch.linalg.norm(1/sqrt(2)*so3.log(R.transpose(1,2).bmm(gtR)).mean(dim=0))  # (B,)
    dT = F.mse_loss(T,gtT,reduction='mean')  # (B,)
    return dR, dT

