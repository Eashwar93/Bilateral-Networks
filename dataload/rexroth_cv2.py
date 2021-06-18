

# import sys
# sys.path.insert(0, '.')
import os
import os.path as osp
import json
import torch
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
import cv2
import numpy as np

import dataload.transform_cv2 as T
from dataload.sampler import RepeatedDistSampler
from dataload.base_dataset import BaseDataset, TransformationTrain, TransformationVal


labels_info = [
    {"hasInstances": False, "category": "void", "catid": 0, "name": "background", "ignoreInEval": True, "id": 0, "color": [0, 0, 0], "trainId": 0},
    {"hasInstances": True, "category": "void", "catid": 1, "name": "monorail", "ignoreInEval": False, "id": 1, "color": [128, 64,128], "trainId": 1},
]


class Rexroth(BaseDataset):
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(Rexroth, self).__init__(dataroot, annpath, trans_func, mode)
        self.n_cats = 2
        self.lb_ignore = 255
        self.lb_map = np.arange(256).astype(np.uint8)
        for el in labels_info:
            self.lb_map[el['id']] = el['trainId']

        self.to_tensor = T.ToTensor(
            mean=(0.3257, 0.3690, 0.3223),
            std=(0.2112, 0.2148, 0.2115)
        )

def get_data_loader(datapth, annpath, ims_per_gpu, scales, cropsize, max_iter=None, mode='train', distributed=True):
    if mode == 'train':
        trans_func = TransformationTrain(scales, cropsize)
        print("scales in dataloader:", scales)
        batchsize = ims_per_gpu
        shuffle = True
        drop_last = True
    elif mode == 'val':
        trans_func = TransformationVal()
        batchsize = ims_per_gpu
        shuffle = False
        drop_last = False

    ds = Rexroth(datapth, annpath, trans_func=trans_func, mode=mode)

    if distributed:
        assert dist.is_available(), "dist should be initialized"
        if mode == 'train':
            assert not max_iter is None
            n_train_imgs = ims_per_gpu * dist.get_world_size()*max_iter
            sampler = RepeatedDistSampler(ds, n_train_imgs, shuffle=shuffle)

        else:
            sampler = torch.utils.data.distributed.DistributedSampler(
                ds, shuffle=shuffle)
        batchsampler = torch.utils.data.sampler.BatchSampler(
            sampler, batchsize, drop_last=drop_last
        )
        dl = DataLoader(ds, batch_sampler=batchsampler, num_workers=4, pin_memory=True)
    else:
        dl = DataLoader(ds, batch_size=batchsize, shuffle=shuffle, drop_last=drop_last, num_workers=4, pin_memory=True)

    return dl

if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    ds = Rexroth(dataroot='./datasets/Rexroth',annpath='./datasets/Rexroth/train.txt', mode='val')
    dl = DataLoader(ds,
                    batch_size = 4,
                    shuffle = False,
                    num_workers = 4,
                    drop_last = True)
    # f = open("dim_check.txt", "a")
    for it, (imgs, label) in enumerate(dl):
        # f.write("it:"+it+"\n")
        # f.write("batchesize:" + len(imgs) + "\n")
        # f.write("image tensor:" + imgs.size()+ "\n")
        # f.write("label tensor:" + label.size() + "\n")
        print(it)
        print(len(imgs))
        for el in imgs:
           print(el.size())
        print(imgs.size())
        print(label.size())




