import sys
sys.path.insert(0, '.')
import torch
import random
import numpy as np
import argparse
import torchvision
from networks import model_factory
from configs import cfg_factory
import torch.nn as nn
import torch.distributed as dist
from dataload.rexroth_cv2 import get_data_loader
from train.ohem_ce_loss import OhemCELoss
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.deterministic = True

def parse_args():
    parse = argparse.ArgumentParser()
    parse = argparse.ArgumentParser()
    parse.add_argument('--local_rank', dest='local_rank', type=int, default=-1, )
    parse.add_argument('--port', dest='port', type=int, default=44554, )
    parse.add_argument('--model', dest='model', type=str, default='bisenetv1', )
    parse.add_argument('--fintune-from',dest='finetune_from', type=str, default=None, )
    return parse.parse_args()


args = parse_args()
cfg = cfg_factory[args.model]


def set_model():
    net = model_factory[cfg.model_type](cfg.categories)
    if not args.finetune_from is None:
        net.load_state_dict(torch.load(args.finetune_from, map_location='cpu'))
    if cfg.use_sync_bn: net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net.cuda()
    net.train()
    criteria_pre = OhemCELoss(0.7)
    criteria_aux = [OhemCELoss(0.7) for _ in range(cfg.num_aux_heads)]
    return net, criteria_pre, criteria_aux

def set_model_dist(net):
    local_rank = dist.get_rank()
    net = nn.parallel.DistributedDataParallel(
        net, device_ids=[local_rank, ], output_device=local_rank
    )
    return net


def matplotlib_imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))





def write():
    writer = SummaryWriter('./tensorboard')
    net, criteria_pre, criteria_aux = set_model()
    is_dist = dist.is_initialized()
    net = set_model_dist(net)

    dl = get_data_loader(
        cfg.im_root, cfg.train_im_anns,
        cfg.ims_per_gpu, cfg.scales, cfg.cropsize,
        cfg.max_iter, mode='train', distributed=is_dist
    )
    dataiter = iter(dl)
    images, labels = dataiter.next()
    img_grid = torchvision.utils.make_grid(images)
    matplotlib_imshow(img_grid)
    writer.add_image('4 images', img_grid)
    writer.add_graph(net, images)
    writer.close()




if __name__ == "__main__":
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:{}'.format(args.port),
        world_size=torch.cuda.device_count(),
        rank=args.local_rank

    )
    write()