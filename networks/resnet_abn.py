
#!/usr/bin/python
# -*- encoding: utf-8 -*-
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from inplace_abn import InPlaceABN

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
}

def conv3x3(in_chan, out_chan, stride=1):
    return nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_chan, out_chan, stride=1):
    return nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, padding=0, bias=False)

class BasicBlock(nn.Module):
    expansion=1
    def __init__(self, in_chan, out_chan, stride, norm_layer=None):
        super(BasicBlock, self).__init__()
        self.norm_layer = norm_layer
        self.conv1 = conv3x3(in_chan=in_chan, out_chan=out_chan, stride=stride)
        if norm_layer == "InPlaceABN":
            self.bn1 = InPlaceABN(num_features=out_chan, activation='leaky_relu')
        elif norm_layer == "BatchNorm2d":
            self.bn1 = nn.BatchNorm2d(num_features=out_chan)
        self.conv2 = conv3x3(in_chan=out_chan, out_chan=out_chan)
        if norm_layer == "InPlaceABN":
            self.bn2 = InPlaceABN(num_features=out_chan, activation='identity')
        elif norm_layer == "BatchNorm2d":
            self.bn2 = nn.BatchNorm2d(num_features=out_chan)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            if norm_layer == "InPlaceABN":
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels=in_chan, out_channels=out_chan, bias=False, stride=stride, kernel_size=1),
                    InPlaceABN(out_chan, activation='identity'),
                )
            elif norm_layer == "BatchNorm2d":
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels=in_chan, out_channels=out_chan, bias=False, stride=stride, kernel_size=1),
                    nn.BatchNorm2d(out_chan),
                )

    def forward(self, x):

        out = self.conv1(x)
        if self.norm_layer == "InPlaceABN":
            out = self.bn1(out)
        elif self.norm_layer == "BatchNorm2d":
            out = self.bn1(out)
            out = self.leakyrelu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        shortcut = x

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out_ = shortcut + out
        out_ = self.relu(out_)
        return out_

class ResNet(nn.Module):
    def __init__(self, block, layers, strides, norm_layer=None):
        super(ResNet, self).__init__()
        self.norm_layer = norm_layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if norm_layer == "InPlaceABN":
            self.bn1 = InPlaceABN(num_features=64, activation='leaky_relu')
        elif norm_layer == "BatchNorm2D":
            self.bn1 = nn.BatchNorm2d(num_features=64)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inplanes = 64
        self.layer1 = self.create_layer(block, 64, bnum=layers[0], stride=strides[0], norm_layer=norm_layer)
        self.layer2 = self.create_layer(block, 128, bnum=layers[1], stride=strides[1], norm_layer=norm_layer)
        self.layer3 = self.create_layer(block, 256, bnum=layers[2], stride=strides[2], norm_layer=norm_layer)
        self.layer4 = self.create_layer(block, 512, bnum=layers[3], stride=strides[3], norm_layer=norm_layer)

    def create_layer(self, block, out_chan, bnum, stride=1, norm_layer=None):
        layers = [block(self.inplanes, out_chan, stride=stride, norm_layer=norm_layer)]
        self.inplanes = out_chan*block.expansion
        for i in range(bnum-1):
            layers.append(block(self.inplanes, out_chan, stride=1, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        if self.norm_layer == "InPlaceABN":
            x = self.bn1(x)
        elif self.norm_layer == "BatchNorm2D":
            x = self.bn1(x)
            x = self.leaky_relu(x)
        else:
            raise Exception("Accepted norm_functions are 'InPlaceABN' or 'BatchNorm2D'")
        x = self.maxpool(x)

        feat4 = self.layer1(x)
        feat8 = self.layer2(feat4)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat4, feat8, feat16, feat32

    def init_weight(self, state_dict):
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            if 'fc' in k:continue
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

def Resnet18(pretrained=True, norm_layer=None, **kwargs):
    model = ResNet(BasicBlock, [2,2,2,2],[2,2,2,2], norm_layer=norm_layer)
    if pretrained:
        model.init_weight(model_zoo.load_url(model_urls['resnet18']))
    return model

def Resnet34(pretrained=True, norm_layer=None, **kwargs):
    model = ResNet(BasicBlock, [3,4,6,3],[2,2,2,2], norm_layer=norm_layer)
    if pretrained:
        model.init_weight(model_zoo.load_url(model_urls['resnet34']))
    return model

class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, activation='identity'):
        super(BatchNorm2d, self).__init__(num_features=num_features)
        self.bn = nn.BatchNorm2d(num_features)
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(inplace=True)
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'identity':
            self.activation = lambda x:x
        else:
            raise Exception("Accepted activations 'leaky_relu', 'relu', 'identity'")

    def forward(self, x):
        abn = self.bn(x)
        abn = self.activation(abn)
        return abn



if __name__ == "__main__":
    net = Resnet18(norm_layer="BatchNorm2D")
    x = torch.randn(16, 3, 1024, 2048)
    out = net(x)
    print(out[0].size())
    print(out[1].size())
    print(out[2].size())
    print(out[3].size())








