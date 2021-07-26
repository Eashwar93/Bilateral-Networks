#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=1 train/train.py --model fanet18_v1_se1
python -m torch.distributed.launch --nproc_per_node=1 train/train.py --model fanet18_v1_se2
python -m torch.distributed.launch --nproc_per_node=1 train/train.py --model fanet18_v1_se3
python -m torch.distributed.launch --nproc_per_node=1 train/train.py --model fanet18_v4_se1
python -m torch.distributed.launch --nproc_per_node=1 train/train.py --model fanet18_v4_se2