#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=1 train/train.py --model bisenet_v1_g6
python -m torch.distributed.launch --nproc_per_node=1 train/train.py --model fanet18_v1
python -m torch.distributed.launch --nproc_per_node=1 train/train.py --model fanet18_v4