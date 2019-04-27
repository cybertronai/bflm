#!/usr/bin/env python
# python launch_network_test.py --instance_type=p3dn.24xlarge
#
# Local gloo 1000MB tensors p2p:
# min:   204.96, median:   308.84, mean:   322.97
#

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

parser.add_argument("--aws", action="store_true", help="enable to run on AWS")
parser.add_argument('--instance_type', type=str, default="p3.2xlarge")
parser.add_argument('--machines', type=int, default=2)
parser.add_argument('--nproc_per_node', type=int, default=8)
parser.add_argument('--image_name', type=str, default='reference01')
parser.add_argument('--nospot', action='store_true',
                    help='use regular instead of spot instances')

parser.add_argument('--num_layers', type=int, default=10)
# parser.add_argument('--layer_size_mb', type=int, default=10)
parser.add_argument('--iters', type=int, default=10,
                    help='how many iterations')
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--skip_setup', action='store_true')

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


def _get_nccl_params():
    params = f'NCCL_DEBUG=VERSION '
    if args.machines > 1:
        params += f'NCCL_MIN_NRINGS=16 '
    if aws_util.instance_supports_100gbps_network(args.instance_type):
        params += f'NCCL_SOCKET_IFNAME=ens5 '

    return params


def launcher():
    if args.aws:
        ncluster.set_backend('aws')

    ncluster.set_logdir_root('/ncluster/runs.network')

    # todo: flag for skip setup

    job = ncluster.make_job(name=args.name,
                            run_name=args.name,
                            num_tasks=args.machines,
                            image_name=args.image_name,
                            instance_type=args.instance_type,
                            spot=not args.nospot,
                            skip_setup=args.skip_setup)
    print(f"Logging to {job.logdir}")

    nccl_params = _get_nccl_params()

    # pass through launcher params to worker script
    assert '--role=launcher' in sys.argv, "how did you get here?"
    worker_params = sys.argv[1:]
    worker_params.remove('--role=launcher')
    worker_params.extend([f'--logdir {job.logdir}'])

    worker_params = ' '.join(worker_params)  # pass through all args

    dist_params0 = (f'--nproc_per_node={args.nproc_per_node} '
                    f'--nnodes={args.machines} '
                    f'--master_addr={job.tasks[0].ip} '
                    f'--master_port={6016} ')

    #    worker_script_fn = os.path.abspath(__file__)  # local location
    job.upload(__file__)
    job.upload('util.py')
    worker_script_fn = os.path.basename(__file__)  # remote location

    job.run('killall python || echo fail && source activate pytorch_p36')

    for i, task in enumerate(job.tasks):
        dist_params = dist_params0 + f'--node_rank={i} '
        cmd = (f'{nccl_params} python -m torch.distributed.launch {dist_params} {worker_script_fn} '
               f'{worker_params} ')
        task.run(f'echo {cmd} > {job.logdir}/task-{i}.cmd')  # save command-line
        task.run(cmd, non_blocking=True)

    print(f"Logging to {job.logdir}")


def test_p2p():
    import torch
    import torch.distributed as dist
    import numpy as np

    log = util.FileLogger(args.logdir + f'/worker-{util.get_global_rank()}',
                          mirror=(args.local_rank == 0))

    os.environ['MASTER_ADDR'] = str(args.master_addr)
    os.environ['MASTER_PORT'] = str(args.master_port)
    # Use TCP backend. Gloo needs nightly, where it currently fails with
    #     dist.init_process_group('gloo', rank=args.rank,
    #   AttributeError: module 'torch.distributed' has no attribute 'init_process_group'

    log("Initializing distributed pytorch")
    # nccl backend does not support send
    dist.init_process_group('gloo', rank=util.get_global_rank(),
                            world_size=util.get_world_size())

    rank = util.get_global_rank()
    tensor = torch.ones(args.layer_size_mb * 250 * 1000) * (rank + 1)
    time_list = []
    for i in range(args.iters):
        start_time = time.perf_counter()
        if args.local_rank == 0:
            dist.send(tensor=tensor, dst=1)
        elif args.local_rank == 1:
            dist.recv(tensor=tensor, src=0)

        elapsed_time_ms = (time.perf_counter() - start_time) * 1000
        time_list.append(elapsed_time_ms)

        #        rate = args.size_mb/(elapsed_time_ms/1000)
        rate = 0

        log('%03d/%d added %d MBs in %.1f ms: %.2f MB/second' % (
        i, args.iters, args.layer_size_mb, elapsed_time_ms, rate))

    min_time = np.min(time_list)
    median = np.median(time_list)
    log(f"min: {min_time:8.2f}, median: {median:8.2f}, mean: {np.mean(time_list):8.2f}")


class SimpleNet(nn.Module):
    def __init__(self, num_layers, layer_dim):
        super(SimpleNet, self).__init__()
        self.layers = []
        for i in range(num_layers):
            self.layers.append(nn.Linear(layer_dim, layer_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return (x * x).sum()


def test_optimize():
    device = 'cuda'
    fp16 = True

    # args.layer_size_mb*250*1000
    # 10 layers, 100MB per layer in fp32
    log = util.FileLogger(args.logdir + f'/worker-{util.get_global_rank()}', mirror=(args.local_rank == 0))

    dim = 2 ** 12  # multiple of 8, about 67MB matrix in fp32
    num_layers = 10


    #    model = SimpleNet(num_layers=num_layers, layer_dim=dim)
    layers = list(nn.Linear(dim, dim) for _ in range(num_layers))
    model = nn.Sequential(*layers)
    model = model.to(device)
    if fp16:
        model = model.half()
        bytes_per_number = 2
    else:
        bytes_per_number = 4

    size_mb = (dim ** 2) * bytes_per_number / 1e6

    log('initializing process group')
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=util.get_world_size())


    log('calling DDP')
    model = DistributedDataParallel(model,
                                    device_ids=[args.local_rank],
                                    output_device=args.local_rank)
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    x = torch.ones((dim, dim))
    time_list = []
    for i in range(args.iters):
        start_time = time.perf_counter()
        optimizer.zero_grad()

        output = model(x)
        loss = (output * output).mean()
        loss.backward()
        optimizer.step()

        elapsed_time_sec = (time.perf_counter() - start_time)
        elapsed_time_ms = elapsed_time_sec * 1000
        time_list.append(elapsed_time_ms)
        rate = size_mb / elapsed_time_sec

        log('%03d/%d added %d MBs in %.1f ms: %.2f MB/second %.1f' % (
        i, args.iters, args.layer_size_mb, elapsed_time_ms,
        rate, loss))

    min_time = np.min(time_list)
    median = np.median(time_list)
    log(f"min: {min_time:8.2f}, median: {median:8.2f}, mean: {np.mean(time_list):8.2f}")


def test_end2end():
    pass


def main():
    if args.role == "launcher":
        launcher()
    elif args.role == "worker":
        #      test_p2p()
        test_optimize()
    else:
        assert False, "Unknown role " + args.role


if __name__ == '__main__':
    main()
