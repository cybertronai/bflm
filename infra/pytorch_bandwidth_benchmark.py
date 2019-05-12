#!/usr/bin/env python
# python launch_network_test.py --instance_type=p3dn.24xlarge
#
# # 41 Gbps
# # pytorch 1.0, nccl 2.3.7+cuda10.0
# python pytorch_bandwidth_benchmark.py --nospot --conda_env=pytorch_p36 --role=launcher --name=nt --skip_setup
#
# # 10.7
# # PyTorch 1.1.0a0+3803d1c with nccl 2.3.7
# python pytorch_bandwidth_benchmark.py --nospot --conda_env=pytorch_april --role=launcher --name=nt --skip_setup
#
# # 12.8
# #  PyTorch 1.1 with nccl 2.4.7ms0
# python pytorch_bandwidth_benchmark.py --nospot --conda_env=pytorch_april_patched --role=launcher --name=nt --skip_setup

# 16 Rings, 8 Processes, 151-153, 53 Gbps, received 20.9
# 16 rings, 8 processes, 173-178, 46 Gbps, received 20.9
# 171-177ms, 39.8 Gbps
# with nccl 2.4.6 12 Gbps
# python pytorch_bandwidth_benchmark.py --role=launcher --machines=2 --aws --instance_type=p3dn.24xlarge --nospot --nproc_per_node=8 --num_rings=16 --skip_setup

# 185ms, average bw=28
# python pytorch_bandwidth_benchmark.py --role=launcher --method=allreduce --machines=2 --aws --instance_type=p3dn.24xlarge --nospot --nproc_per_node=8 --num_rings=16 --skip_setup

# 170ms, average bw=45
# python pytorch_bandwidth_benchmark.py --role=launcher --machines=2 --aws --instance_type=p3dn.24xlarge --nospot --nproc_per_node=8 --num_rings=16 --skip_setup
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
parser.add_argument('--seed', type=int, default=1)

parser.add_argument("--aws", action="store_true", help="enable to run on AWS")
parser.add_argument('--instance_type', type=str, default="p3dn.24xlarge")
parser.add_argument('--machines', type=int, default=2)
parser.add_argument('--nproc_per_node', type=int, default=8)

# pytorch 1.0.1/2.3.7+cuda10.0
parser.add_argument('--conda_env', type=str, default='pytorch_p36')

# pytorch latest/2.3.7+cuda10.0
# parser.add_argument('--conda_env', type=str, default='pytorch_april_nccl237')

# pytorch latest/2.4.6+cuda10.0
# parser.add_argument('--conda_env', type=str, default='pytorch_april')


parser.add_argument('--image_name', type=str, default='cybertronai01')

parser.add_argument('--method', type=str, default='optimize')

parser.add_argument('--nospot', action='store_true',
                    help='use regular instead of spot instances')

parser.add_argument('--iters', type=int, default=20,
                    help='how many iterations')
parser.add_argument('--skip_setup', action='store_true')


parser.add_argument('--num_rings', type=int, default=16)
parser.add_argument('--num_layers', type=int, default=16)
parser.add_argument('--bucket_cap', type=int, default=25)

parser.add_argument('--use_latest_nccl', action='store_true')

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

fp16 = True

def _get_nccl_params():
    params = f'NCCL_DEBUG=INFO '
    params = f'NCCL_DEBUG=VERSION '
    if args.machines > 1:
        params += f'NCCL_MIN_NRINGS={args.num_rings} NCCL_MAX_NRINGS={args.num_rings} '
    if aws_util.instance_supports_100gbps_network(args.instance_type):
        params += f'NCCL_SOCKET_IFNAME=ens5 '

    return params


def launcher():
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

        
    job.run(f'killall -9 python || echo fail && source activate {args.conda_env}')

    for i, task in enumerate(job.tasks):
        dist_params = dist_params0 + f'--node_rank={i} '
        cmd = (f'{nccl_params} python -m torch.distributed.launch {dist_params} {worker_script_fn} '
               f'{worker_params} ')
        task.run(f'echo {cmd} > {job.logdir}/task-{i}.cmd')  # save command-line
        task.run(cmd, non_blocking=True)

    print(f"Logging to {job.logdir}")


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

    log('initializing process group')
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=util.get_world_size())

    log('calling DDP')
    model = DistributedDataParallel(model,
                                    device_ids=[args.local_rank],
                                    output_device=args.local_rank)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    x = torch.eye(dim)
    x = x.to(device)
    if fp16:
        x = x.half()
    time_list = []


    # force initialization of NCCL
    dist.all_reduce(torch.ones(()).cuda())
    dist.barrier()
    
    log("Start timing")
    start_time = time.perf_counter()
    start_time0 = start_time
    for i in range(args.iters):
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

        log('%03d/%d added %d MBs in %.1f ms: %.2f MB/second %.1f' % (
            i, args.iters, size_mb, elapsed_time_ms, rate, loss))

    del time_list[0]   # first measurement is off because of syncing
    min_time = np.min(time_list)
    median = np.median(time_list)
    log(f"min: {min_time:8.2f}, median: {median:8.2f}, mean: {np.mean(time_list):8.2f}")

    dist.barrier()
    elapsed_time = time.perf_counter() - start_time0
    recv_bytes1, transmit_bytes1 = util.network_bytes()
    log(f"Received {(recv_bytes1-recv_bytes)/1e9:.1f}, transmitted {(transmit_bytes1-transmit_bytes)/1e9:.1f} in {elapsed_time:.1f} seconds")
    log(f"predicted {gradient_size*args.iters/1e9:.1f}")

    log(f"average bw: {(recv_bytes1-recv_bytes)*8/elapsed_time/1e9:.1f} Gbps")

def test_allreduce():
    global log

    recv_bytes, transmit_bytes = util.network_bytes()
    
    device = 'cuda'

    dim = 2 ** 12  # multiple of 8, about 67MB matrix in fp32

    if fp16:
        bytes_per_number = 2
    else:
        bytes_per_number = 4

    gradient_size = args.num_layers * (dim * dim) * bytes_per_number
    size_mb = gradient_size / 1e6

    log('initializing process group')
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=util.get_world_size())

    xs = [torch.ones((dim, dim)) for i in range(args.num_layers)]
    xs = [x.to(device) for x in xs]
    if fp16:
        xs = [x.half() for x in xs]
    time_list = []


    # force initialization of NCCL
    dist.all_reduce(torch.ones(()).cuda())
    dist.barrier()
    
    log("Start timing")
    start_time = time.perf_counter()
    start_time0 = start_time
    for i in range(args.iters):
        
        [dist.all_reduce(x, async_op=True) for x in xs]
        
        torch.cuda.synchronize()
        elapsed_time_sec = (time.perf_counter() - start_time)
        start_time = time.perf_counter()
        
        elapsed_time_ms = elapsed_time_sec * 1000
        time_list.append(elapsed_time_ms)
        rate = size_mb / elapsed_time_sec

        # could do barrier, but didn't have effect on timing
        # dist.barrier()   
        new_result = xs[0]
        log('%03d/%d added %d MBs in %.1f ms: %.2f MB/second %.1f' % (
            i, args.iters, size_mb, elapsed_time_ms, rate, new_result[0,0]))

    del time_list[0]   # first measurement is off because of syncing
    min_time = np.min(time_list)
    median = np.median(time_list)
    log(f"min: {min_time:8.2f}, median: {median:8.2f}, mean: {np.mean(time_list):8.2f}")

    dist.barrier()
    elapsed_time = time.perf_counter() - start_time0
    recv_bytes1, transmit_bytes1 = util.network_bytes()
    log(f"Received {(recv_bytes1-recv_bytes)/1e9:.1f}, transmitted {(transmit_bytes1-transmit_bytes)/1e9:.1f} in {elapsed_time:.1f} seconds")
    log(f"predicted {gradient_size*args.iters/1e9:.1f}")

    log(f"average bw: {(recv_bytes1-recv_bytes)*8/elapsed_time/1e9:.1f} Gbps")


def main():
    global log
    if args.role == "launcher":
        launcher()
    elif args.role == "worker":
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        log = util.FileLogger(args.logdir + f'/worker-{util.get_global_rank()}', mirror=(args.local_rank == 0))

        torch.cuda.set_device(args.local_rank)
        #      test_p2p()
        if args.method == 'optimize':
            test_optimize()
        elif args.method == 'allreduce':
            test_allreduce()
        else:
            assert False, 'unknown arg'
    else:
        assert False, "Unknown role " + args.role


if __name__ == '__main__':
    main()
