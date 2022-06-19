import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.dirname(__file__))
from chamfer_distance import ChamferDistance
# copied from https://github.com/vinits5/pcrnet_pytorch/tree/master/pcrnet/losses

def chamfer_distance(template: torch.Tensor, source: torch.Tensor):
	cost_p0_p1, cost_p1_p0 = ChamferDistance()(template, source)
	cost_p0_p1 = torch.mean(torch.sqrt(cost_p0_p1),dim=-1)
	cost_p1_p0 = torch.mean(torch.sqrt(cost_p1_p0),dim=-1)
	chamfer_loss = (cost_p0_p1 + cost_p1_p0)/2.0
	return chamfer_loss


class ChamferDistanceLoss(nn.Module):
	def __init__(self,reduction='mean'):
		super(ChamferDistanceLoss, self).__init__()
		self.reduction = reduction
	def forward(self, template, source):
		if self.reduction == 'none':
			return chamfer_distance(template, source)
		elif self.reduction == 'mean':
			return torch.mean(chamfer_distance(template, source),dim=0)
		elif self.reduction == 'sum':
			return torch.sum(chamfer_distance(template, source),dim=0)
	def __call__(self,template,source):
		return self.forward(self,template,source)

if __name__ == "__main__":
    a = torch.rand(2,3,100)
    b = torch.rand(2,3,100)
    loss1 = ChamferDistanceLoss('none')
    loss2 = ChamferDistanceLoss('mean')
    loss3 = ChamferDistanceLoss('sum')
    print(loss1(a,b),loss2(a,b),loss3(a,b))