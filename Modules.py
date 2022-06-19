# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 23:39:37 2021

@author: 17478
"""
import torch.nn as nn
from torch.nn import functional as F
import torch


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    """
        3x3 convolution with padding
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, dilation=dilation, bias=False)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,
                 dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride,
                             padding=dilation, dilation=dilation,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class ConvModule(nn.Module):
    def __init__(self,inplanes, planes, **kwargs):
        super(ConvModule,self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, **kwargs)
        self.bn = nn.BatchNorm2d(planes)
        self.activate = nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        out = self.activate(x)
        return out
        
class ASPPHead(nn.Module):
    def __init__(self,num_classes):
        super(ASPPHead,self).__init__()
        self.dropout = nn.Dropout2d(p=0.1)
        self.conv_seg = nn.Conv2d(128,num_classes,kernel_size=1,stride=1)
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            ConvModule(512,128,kernel_size=1,stride=1,bias=False),
            ) 
        self.aspp_modules = nn.ModuleList([
            ConvModule(512, 128, kernel_size=1,stride=1,bias=False),
            ConvModule(512, 128, kernel_size=3,stride=1,padding=12,dilation=12,bias=False),
            ConvModule(512, 128, kernel_size=3,stride=1,padding=24,dilation=24,bias=False),
            ConvModule(512, 128, kernel_size=3,stride=1,padding=36,dilation=36,bias=False),
            ])
        self.bottleneck = ConvModule(640, 128, kernel_size=3,stride=1,padding=1,bias=False)
    def forward(self,feature_map):
        feature_map_h = feature_map.size()[2] # (== h/16)
        feature_map_w = feature_map.size()[3] # (== w/16)
        out_1x1 = self.aspp_modules[0](feature_map) # (shape: (batch_size, 128, h/16, w/16))
        out_3x3_1 = self.aspp_modules[1](feature_map) # (shape: (batch_size, 128, h/16, w/16))
        out_3x3_2 = self.aspp_modules[2](feature_map) # (shape: (batch_size, 128, h/16, w/16))
        out_3x3_3 = self.aspp_modules[3](feature_map) # (shape: (batch_size, 128, h/16, w/16))
        out_img = self.image_pool(feature_map) # (shape: (batch_size, 128, h/16, w/16))
        out_img = F.interpolate(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") # (shape: (batch_size, 128, h/16, w/16))
        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) # (shape: (batch_size, 640, h/16, w/16))
        out = self.bottleneck(out) # (shape: (batch_size, 128, h/16, w/16))
        out = self.dropout(out) # (shape: (batch_size, 128, h/16, w/16))
        out = self.conv_seg(out) # (shape: (batch_size, num_classes, h/16, w/16))
        return out

class FCNHead(nn.Module):
    def __init__(self,num_classes=2,inplanes=256):
        super(FCNHead,self).__init__()
        planes = inplanes // 4
        self.conv_seg = nn.Conv2d(planes,num_classes,kernel_size=1,stride=1)
        self.dropout = nn.Dropout2d(p=0.1)
        self.convs = nn.Sequential(
            ConvModule(inplanes, planes, kernel_size=3,stride=1,padding=1,bias=False)
            )
    def forward(self,x):
        x = self.convs(x)
        x = self.dropout(x)
        x = self.conv_seg(x)
        return x
    
class resnet18(nn.Module):
    def __init__(self, inplanes=3, planes=64):
        super(resnet18,self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(inplanes, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, planes, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            BasicBlock(planes, planes, stride=1, dilation=1),
            BasicBlock(planes, planes, stride=1),
            )
        self.layer2 = nn.Sequential(
            BasicBlock(planes, planes*2, stride=2, dilation=1, downsample=nn.Sequential(
                nn.Conv2d(planes, planes*2, 1, stride=2, bias=False),
                nn.BatchNorm2d(planes*2),
                )),
            BasicBlock(planes*2, planes*2, stride=1),
            )
        self.layer3 = nn.Sequential(
            BasicBlock(planes*2, planes*4, stride=1,downsample=nn.Sequential(
                nn.Conv2d(planes*2, planes*4, 1, stride=1, bias=False),
                nn.BatchNorm2d(planes*4),
                )),
            BasicBlock(planes*4, planes*4, stride=1, dilation=2),
            )
        self.layer4 = nn.Sequential(
            BasicBlock(planes*4, planes*8, stride=1,dilation=2,downsample=nn.Sequential(
                nn.Conv2d(planes*4, planes*8, 1, stride=1, bias=False),
                nn.BatchNorm2d(planes*8),
                )),
            BasicBlock(planes*8, planes*8, stride=1,dilation=4),
            )
    def forward(self,x):
        out = self.stem(x)
        out = self.maxpool(out)
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        return out1, out2, out3, out4
    
class EncoderDecoder(nn.Module):
    def __init__(self,num_classes=2,auxiliary_loss=True, backbone_pretrained=True):
        super(EncoderDecoder,self).__init__()
        self.backbone = resnet18()
        self.decode_head = ASPPHead(num_classes=num_classes)
        self.auxiliary_loss = auxiliary_loss
        if auxiliary_loss:
            self.auxiliary_head = FCNHead(num_classes=num_classes,inplanes=256)
        else:
            self.auxiliary_head = None
        if backbone_pretrained:
            backbone_state = torch.load("resnetV1C.pth")['state_dict']
            for key in self.backbone.state_dict().keys():
                assert key in backbone_state.keys(), "backbone state-dict mismatch"
            self.backbone.load_state_dict(backbone_state,strict=False)
            print("pretrained model loaded!")
    def forward(self,x):
        input_shape = x.shape[-2:]
        feat = self.backbone(x)
        decode_seg = self.decode_head(feat[-1])
        decode_seg = F.interpolate(decode_seg, size=input_shape, mode='bilinear', align_corners=False)
        if (not self.auxiliary_loss) or (not self.training):
            return decode_seg
        else:
            aux_seg = self.auxiliary_head(feat[-2])
            aux_seg = F.interpolate(aux_seg, size=input_shape, mode='bilinear', align_corners=False)
            return decode_seg, aux_seg
    
if __name__ == "__main__":
    model = resnet18()
    x = torch.rand(1,3,32,32)
    outs = model(x)
    print(outs[0].size(),outs[1].size(),outs[2].size(),outs[3].size())
        