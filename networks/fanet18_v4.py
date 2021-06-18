#!/usr/bin/python
# -*- encoding: utf-8 -*-
import sys
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_fanet18_v4 import Resnet18

from prettytable import PrettyTable
from ptflops import get_model_complexity_info
import time


up_kwargs = {'mode':'bilinear', 'align_corners':True}

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, activation='leaky_relu',*args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.activation = activation
        self.conv = nn.Conv2d(in_chan, out_chan, stride=stride, kernel_size=ks, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.init_weight()

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation == 'leaky_relu':
            x = self.leaky_relu(x)
        elif self.activation == 'none':
            pass
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class FastAttModule(nn.Module):
    def __init__(self, in_chan, mid_chn=256, out_chan=128,  *args, **kwargs):
        super(FastAttModule, self).__init__()
        self._up_kwargs = up_kwargs
        mid_chn = int(in_chan/2)
        self.w_qs = ConvBNReLU(in_chan=in_chan, out_chan=32, ks=1, stride=1, padding=0, activation='none')

        self.w_ks = ConvBNReLU(in_chan=in_chan, out_chan=32, ks=1, stride=1, padding=0, activation='none')

        self.w_vs = ConvBNReLU(in_chan=in_chan, out_chan=in_chan, ks=1, stride=1, padding=0, activation='leaky_relu')

        self.latlayer3 = ConvBNReLU(in_chan=in_chan, out_chan=in_chan, ks=1, stride=1, padding=0, activation='leaky_relu')

        self.up = ConvBNReLU(in_chan=in_chan, out_chan=mid_chn, ks=1, stride=1, padding=0, activation='leaky_relu')

        self.smooth = ConvBNReLU(in_chan=in_chan, out_chan=out_chan, ks=3, stride=1, padding=1, activation='leaky_relu')

        self.init_weight()

    def forward(self, feat, up_fea_in, up_flag, smf_flag):

        query = self.w_qs(feat)
        key = self.w_ks(feat)
        value = self.w_vs(feat)

        N, C, H, W = feat.size()

        query_ = query.view(N, 32, -1).permute(0,2,1)
        query = torch.softmax(query_, dim=1)
        # query = F.normalize(query_, p=2, dim=2, eps=1e-12)

        key_ = key.view(N, 32, -1)
        key = torch.softmax(key_, dim=1)
        # key = F.normalize(key_, p=2, dim=1, eps=1e-12)

        value = value.view(N, C, -1).permute(0,2,1)

        f = torch.matmul(key, value)
        y = torch.matmul(query, f)
        y = y.permute(0, 2, 1).contiguous()

        y = y.view(N,C,H,W)
        W_y = self.latlayer3(y)
        p_feat = W_y + feat

        if up_flag and smf_flag:
            if up_fea_in is not None:
                p_feat = self._upsample_add(up_fea_in, p_feat)
            up_feat = self.up(p_feat)
            smooth_feat = self.smooth(p_feat)
            return up_feat, smooth_feat

        if up_flag and not smf_flag:
            if up_fea_in is not None:
                p_feat = self._upsample_add(up_fea_in, p_feat)
            up_feat = self.up(p_feat)
            return up_feat

        if not up_flag and smf_flag:
            if up_fea_in is not None:
                p_feat = self._upsample_add(up_fea_in, p_feat)
            smooth_feat = self.smooth(p_feat)
            return smooth_feat

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x,(H,W), **self._up_kwargs) + y


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

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class FPNOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(FPNOutput, self).__init__()
        self._up_kwargs=up_kwargs
        self.conv = ConvBNReLU(in_chan=in_chan, out_chan=mid_chan, ks=3, padding=1, activation='leaky_relu')
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self,x, H, W):
        x = self.conv(x)
        x = self.conv_out(x)
        x1 = F.interpolate(x, (H, W), **self._up_kwargs)
        return x1

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

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class FANet18_v4(nn.Module):
    def __init__(self, n_classes=2, backbone='resnet18', aux_output=False, export=False):
        super(FANet18_v4, self).__init__()

        self._up_kwargs = up_kwargs
        self.nclass = n_classes
        self.aux_output = aux_output
        self.export = export
        if backbone == "resnet18":
            self.resnet = Resnet18()
        else:
            raise RuntimeError('unknown backnone: {}'.format(backbone))
        self.fam_32 = FastAttModule(in_chan=512, mid_chn=256, out_chan=128)
        self.fam_16 = FastAttModule(in_chan=256, mid_chn=256, out_chan=128)
        self.fam_8 = FastAttModule(in_chan=128, mid_chn=256, out_chan=128)

        self.clslayer = FPNOutput(256, 256, n_classes)
        self.aux0 = FPNOutput(256,256, n_classes)
        self.aux1 = FPNOutput(128,256, n_classes)
        self.init_weight()

    def forward(self, x, lbl=None):

        _, _, H, W = x.size()

        _, feat8, feat16, feat32 = self.resnet(x)

        upfeat_32, smfeat_32 = self.fam_32(feat32, None, True, True)
        upfeat_16 = self.fam_16(feat16, upfeat_32, True, False)
        smfeat_8 = self.fam_8(feat8, upfeat_16, False, True)
        x = self._upsample_cat(smfeat_32, smfeat_8)
        if self.aux_output:
            aux0 = self.aux0(upfeat_32, H, W)
            aux1 = self.aux1(upfeat_16, H, W)
            output = self.clslayer(x, H, W)
            return output, aux0, aux1
        else:
            output = self.clslayer(x, H, W)
            if self.export:
                output = output.argmax(dim=1)
            return output

    def _upsample_cat(self, x1, x2):
        _,_,H,W = x2.size()
        x1 = F.interpolate(x1, (H,W), **self._up_kwargs)
        x = torch.cat([x1,x2], dim=1)
        return x

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (FastAttModule, FPNOutput)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

if __name__ == "__main__":
    net = FANet18_v4(2).cuda()
    x = torch.randn(1, 3, 480, 640).cuda()
    net.eval()
    net.init_weight()
    with torch.no_grad():
        torch.cuda.synchronize()
        out = net(x)
        torch.cuda.synchronize()
        start_ts = time.time()
        for i in range(100):
            out = net(x)
        torch.cuda.synchronize()
        end_ts = time.time()
        t_diff = end_ts-start_ts
        print("FPS: %f" % (100 / t_diff))
    macs, params = get_model_complexity_info(net, (3, 480, 640), as_strings=True,
                                             print_per_layer_stat=False, verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    out = net(x)
    print("Output size: ", out.size())
    count_parameters(net)

