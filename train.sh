#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=1 train/train.py --model fanet18_v4_se2
python -m torch.distributed.launch --nproc_per_node=1 train/train.py --model fanet18_v1_se2_c1
python -m torch.distributed.launch --nproc_per_node=1 train/train.py --model fanet18_v1_se3_c1