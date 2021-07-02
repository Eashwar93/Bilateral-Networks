

import torch
import torch.nn as nn


from prettytable import PrettyTable
from ptflops import get_model_complexity_info
import time



from torch.nn import BatchNorm2d

def conv3x3(in_planes, out_planes, stride=1, groups=1):
       return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)

class BasicBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1, num_group=32):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1 = BatchNorm2d(out_chan)
        self.conv2 = conv3x3(out_chan, out_chan, groups=num_group)
        self.bn2 = BatchNorm2d(out_chan)
        self.mix = nn.Conv2d(in_channels=out_chan, out_channels=out_chan, kernel_size=1, stride=1, bias=False)
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
        residual = self.relu(residual)
        residual = self.mix(residual)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = shortcut + residual
        out = self.relu(out)
        return out

def create_layer_basic(in_chan, out_chan, bnum, stride=1):
    layers = [BasicBlock(in_chan, out_chan, stride=stride, num_group=8)]  ##################
    for i in range (bnum-1):
        layers.append(BasicBlock(out_chan, out_chan, stride=1, num_group=8)) ################
    return nn.Sequential(*layers)

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")

class ResneXt18_mix(nn.Module):
    def __init__(self):
        super(ResneXt18_mix, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = create_layer_basic(64, 64, bnum=2, stride=1)
        self.layer2 = create_layer_basic(64, 128, bnum=2, stride=2)
        self.layer3 = create_layer_basic(128,256, bnum=2, stride=2)
        self.layer4 = create_layer_basic(256, 512, bnum=2, stride=2)
        self.init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        feat8 = self.layer2(x)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat8, feat16, feat32

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

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
    net = ResneXt18_mix().cuda()
    x = torch.randn(1, 3, 480, 640).cuda()
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
    macs, params = get_model_complexity_info(net, (3, 480, 640), as_strings=True,
                                             print_per_layer_stat=False, verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    out8, out16, out32 = net(x)
    print("Output size 8: ", out8.size())
    print("Output size 16: ", out16.size())
    print("Output size 32: ", out32.size())
    count_parameters(net)




