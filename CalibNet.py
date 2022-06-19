import torch
from torch import nn
import torch.nn.functional as F
from Modules import resnet18

class Aggregation(nn.Module):
    def __init__(self,inplanes=768,planes=96):
        super(Aggregation,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes,out_channels=planes*4,kernel_size=3,stride=2,padding=1)
        self.bn1 = nn.BatchNorm2d(planes*4)
        self.conv2 = nn.Conv2d(in_channels=planes*4,out_channels=planes*4,kernel_size=3,stride=2,padding=1)
        self.bn2 = nn.BatchNorm2d(planes*4)
        self.conv3 = nn.Conv2d(in_channels=planes*4,out_channels=planes*2,kernel_size=(2,1),stride=2)
        self.bn3 = nn.BatchNorm2d(planes*2)
        self.tr_conv = nn.Conv2d(in_channels=planes*2,out_channels=planes,kernel_size=1,stride=1)
        self.tr_bn = nn.BatchNorm2d(planes)
        self.rot_conv = nn.Conv2d(in_channels=planes*2,out_channels=planes,kernel_size=1,stride=1)
        self.rot_bn = nn.BatchNorm2d(planes)
        self.tr_drop = nn.Dropout2d(p=0.1)
        self.rot_drop = nn.Dropout2d(p=0.1)
        self.tr_pool = nn.AdaptiveAvgPool2d(output_size=(6,2))
        self.rot_pool = nn.AdaptiveAvgPool2d(output_size=(6,2))
        self.fc1 = nn.Linear(1152,3)  # 96*12=1152
        self.fc2 = nn.Linear(1152,3)  # 96*12=1152

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x_tr = self.tr_conv(x)
        x_tr = self.tr_bn(x_tr)
        x_tr = self.tr_drop(x_tr)
        x_tr = self.tr_pool(x_tr)  # (19,6)
        x_tr = self.fc1(x_tr.view(x_tr.shape[0],-1))
        x_rot = self.rot_conv(x)
        x_rot = self.rot_bn(x_rot)
        x_rot = self.rot_drop(x_rot)  # (19.6)
        x_rot = self.rot_pool(x_rot)
        x_rot = self.fc2(x_rot.view(x_rot.shape[0],-1))
        return x_tr, x_rot

class CalibNet(nn.Module):
    def __init__(self,backbone_pretrained=True):
        super(CalibNet,self).__init__()
        self.rgb_resnet = resnet18(inplanes=3,planes=64)  # outplanes = 512
        self.depth_resnet = nn.Sequential(
            nn.MaxPool2d(kernel_size=5,stride=1,padding=2),  # outplanes = 256
            resnet18(inplanes=1,planes=32),
        )
        self.aggregation = Aggregation(inplanes=512+256,planes=96)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if backbone_pretrained:
            self.rgb_resnet.load_state_dict(torch.load("resnetV1C.pth")['state_dict'],strict=False)
        self.to(self.device)
    def forward(self,rgb:torch.Tensor,depth:torch.Tensor):
        # rgb: [B,3,H,W]
        # depth: [B,1,H,W]
        x1,x2 = rgb,depth
        x1 = self.rgb_resnet(x1)[-1]
        x2 = self.depth_resnet(x2)[-1]
        feat = torch.cat((x1,x2),dim=1)  # [B,C1+C2,H,W]
        x_tr, x_rot =  self.aggregation(feat)
        return x_tr, x_rot 
if __name__=="__main__":
    x = (torch.rand(2,3,1242,375).cuda(),torch.rand(2,1,1242,375).cuda())
    model = CalibNet(backbone_pretrained=False).cuda()
    model.eval()
    translation,rotation = model(*x)
    print("translation size:{}".format(translation.size()))
    print("rotation size:{}".format(rotation.size()))


