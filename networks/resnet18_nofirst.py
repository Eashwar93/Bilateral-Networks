

import torch
import torch.nn as nn
import torch.utils.model_zoo as modelzoo


from ptflops import get_model_complexity_info
import time

resnet18_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'

from torch.nn import BatchNorm2d

def conv3x3(in_planes, out_planes, stride=1):
       return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1 = BatchNorm2d(out_chan)
        self.conv2 = conv3x3(out_chan, out_chan)
        self.bn2 = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(out_chan),
            )
    def forward(self,x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = shortcut + residual
        out = self.relu(out)
        return out

def create_layer_basic(in_chan, out_chan, bnum, stride=1):
    layers = [BasicBlock(in_chan, out_chan, stride=stride)]
    for i in range (bnum-1):
        layers.append(BasicBlock(out_chan, out_chan, stride=1))
    return nn.Sequential(*layers)

class Resnet18_nofirst(nn.Module):
    def __init__(self):
        super(Resnet18_nofirst, self).__init__()
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = create_layer_basic(64, 64, bnum=2, stride=1)
        self.layer2 = create_layer_basic(64, 128, bnum=2, stride=2)
        self.layer3 = create_layer_basic(128,256, bnum=2, stride=2)
        self.layer4 = create_layer_basic(256, 512, bnum=2, stride=2)
        self.init_weight()

    def forward(self, x):
        # first_conv = self.conv1(x)
        # x = self.bn1(first_conv)
        # x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        feat8 = self.layer2(x)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat8, feat16, feat32

    def init_weight(self):
        state_dict = modelzoo.load_url(resnet18_url)
        self_state_dict = self.state_dict()
        for k,v in state_dict.items():
            if 'fc' in k: continue
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict,strict=False)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


if __name__ == "__main__":
    net = Resnet18_nofirst().cuda()
    x = torch.randn(1, 64, 240, 320).cuda()
    net.eval()
    net.init_weight()
    with torch.no_grad():
        torch.cuda.synchronize()
        out8,out16,out32 = net(x)
        torch.cuda.synchronize()
        start_ts = time.time()
        for i in range(100):
            out8, out16, out32,  = net(x)
        torch.cuda.synchronize()
        end_ts = time.time()
        t_diff = end_ts-start_ts
        print("FPS: %f" % (100 / t_diff))
    macs, params = get_model_complexity_info(net, (64, 240, 320), as_strings=True,
                                             print_per_layer_stat=False, verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    out8, out16, out32 = net(x)
    print("Output size 8: ", out8.size())
    print("Output size 16: ", out16.size())
    print("Output size 32: ", out32.size())




