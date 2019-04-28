#!/usr/bin/env python
# python check_version.py
# (pytorch_p36) ubuntu@ip-172-31-6-107:~$ python check_versions.py
# PyTorch version: 1.0.1
# PyTorch cuda version:  10.0.130
# NCCL version 2.3.7+cuda10.0

import argparse
import os
import sys
import time
import torch.optim as optim
import torch.distributed as dist

import torch.nn as nn
# import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel


import numpy as np

import ncluster
import torch
from ncluster import aws_util

# local imports
import util

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='bandwidth_test')
parser.add_argument('--seed', type=int, default=1)

parser.add_argument("--aws", action="store_true", help="enable to run on AWS")
parser.add_argument('--machines', type=int, default=1)
parser.add_argument('--nproc_per_node', type=int, default=1)
parser.add_argument('--image_name', type=str, default='reference01')
parser.add_argument('--nospot', action='store_true',
                    help='use regular instead of spot instances')

parser.add_argument('--fp16', action='store_true')
parser.add_argument('--skip_setup', action='store_true')


parser.add_argument('--num_rings', type=int, default=16)
parser.add_argument('--num_layers', type=int, default=16)
parser.add_argument('--bucket_cap', type=int, default=25)

# worker params
parser.add_argument('--logdir', type=str, default='/tmp')

# distributed params
parser.add_argument('--role', type=str, default='worker',
                    help='internal flag, launcher or worker')
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--master_addr', type=str, default='127.0.0.1',
                    help='address of master node')
parser.add_argument('--master_port', type=int, default=-1,
                    help='port of master node')
args = parser.parse_args()

os.environ['NCCL_DEBUG']='VERSION'
os.environ['NCCL_MIN_NRINGS']='16'
os.environ['NCCL_SOCKET_IFNAME']='ens5'
os.environ['RANK']='0'
os.environ['WORLD_SIZE']='1'
os.environ['MASTER_ADDR']='127.0.0.1'
os.environ['MASTER_PORT']='6016'

class SimpleNet(nn.Module):
    def __init__(self, num_layers, dim):
        super(SimpleNet, self).__init__()
        self.layers = []
        
        for i in range(num_layers):
            param0 = torch.normal(torch.zeros((dim, dim)), 0.001)
            param = nn.Parameter(param0)
            self.layers.append(param)
            setattr(self, 'W'+str(i), param)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer
        return x

log = None
def test_optimize():
    global log

    recv_bytes, transmit_bytes = util.network_bytes()
    
    device = 'cuda'
    fp16 = True

    dim = 2 ** 12  # multiple of 8, about 67MB matrix in fp32

    model = SimpleNet(args.num_layers, dim)
    model = model.to(device)
    if fp16:
        model = model.half()
        bytes_per_number = 2
    else:
        bytes_per_number = 4

    gradient_size = args.num_layers * (dim * dim) * bytes_per_number
    size_mb = gradient_size / 1e6

    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=util.get_world_size())

    model = DistributedDataParallel(model,
                                    device_ids=[args.local_rank],
                                    output_device=args.local_rank)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    x = torch.eye(dim)
    x = x.to(device)
    if fp16:
        x = x.half()
    time_list = []
    start_time = time.perf_counter()
    start_time0 = start_time
    for i in range(1):
        optimizer.zero_grad()

        output = model(x)

        def sqr(a): return a*a
        loss = sqr(output-x).sum()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        elapsed_time_sec = (time.perf_counter() - start_time)
        start_time = time.perf_counter()
        
        elapsed_time_ms = elapsed_time_sec * 1000
        time_list.append(elapsed_time_ms)
        rate = size_mb / elapsed_time_sec


def main():
    global log
    if args.role == "launcher":
        launcher()
    elif args.role == "worker":
        print("PyTorch version:", torch.__version__)
        print("PyTorch cuda version: ", torch.version.cuda)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        torch.cuda.set_device(args.local_rank)
        #      test_p2p()
        test_optimize()
    else:
        assert False, "Unknown role " + args.role


if __name__ == '__main__':
    main()
