import sys
sys.path.insert(0, '.')
import argparse
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2

import dataload.transform_cv2 as T
from networks import model_factory
from configs import cfg_factory

torch.set_grad_enabled(False)
np.random.seed(123)

parse = argparse.ArgumentParser()
parse.add_argument('--model', dest='model', type=str, default='bisenetv1',)
parse.add_argument('--weight-path', dest='weight_path', type=str, default='./res/model_monorail_rotated.pth',)
parse.add_argument('--img-path', dest='img_path', type=str, default='./152.png',)
args = parse.parse_args()
cfg = cfg_factory[args.model]

palette = np.random.randint(0,256, (256, 3), dtype=np.uint8)

net = model_factory[cfg.model_type](cfg.categories)
net.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
net.eval()
net.cuda()

to_tensor = T.ToTensor(
    mean=(0.3257, 0.3690, 0.3223),
    std=(0.2112, 0.2148, 0.2115),
)
im = cv2.imread(args.img_path)[:, :, ::-1]
im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()

out = net(im)[0]
print(out.shape)
out = out.argmax(dim=1).squeeze().detach().cpu().numpy()
pred = palette[out]
cv2.imwrite('./res_152.png', pred)
