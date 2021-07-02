import argparse
import os.path as osp
import sys
sys.path.insert(0, '.')

import torch
from networks import model_factory
from configs import cfg_factory

torch.set_grad_enabled(False)

parse = argparse.ArgumentParser()
parse.add_argument('--model', dest='model', type=str, default='bisenetv1',)
parse.add_argument('--weight-path', dest='weight_pth', type=str, default='./res/bisenet_v1.pth')
parse.add_argument('--outpath', dest='out_pth', type=str, default='./res/bisenetv1.onnx')
args = parse.parse_args()

cfg = cfg_factory[args.model]
if cfg.use_sync_bn: cfg.us_sync_bn = False

net = model_factory[cfg.model_type](cfg.categories, aux_output=False, export=True)
net.load_state_dict(torch.load(args.weight_pth), strict=False)
net.eval()

dummy_input = torch.randn(1, 3, 480, 640)
input_names = ['input_image']
output_names = ['preds',]

torch.onnx.export(net, dummy_input, args.out_pth,
                  input_names=input_names, output_names=output_names,
                  verbose=False, opset_version=11)