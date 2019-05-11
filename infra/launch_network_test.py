#!/usr/bin/env python
# python launch_network_test.py --nospot
#
#

import argparse

import ncluster

import util

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='nt')
parser.add_argument('--instance_type', type=str, default="p3dn.24xlarge")
parser.add_argument('--machines', type=int, default=2)
parser.add_argument('--image_name', type=str, default='cybertronai00')
parser.add_argument('--nospot', action='store_true',
                    help='use regular instead of spot instances')
parser.add_argument('--multiproc', action='store_true')
parser.add_argument('--flows_per_proc', type=int, default=10)
parser.add_argument('--duration_sec', type=int, default=600)
parser.add_argument('--num_procs', type=int, default=8)
args = parser.parse_args()


def main():
    if args.instance_type == 'p3.16xlarge':
        instance_short_name = 'p3'
    elif args.instance_type == 'p3dn.24xlarge':
        instance_short_name = 'p4'
    elif args.instance_type == 'c5.18xlarge':
        instance_short_name = 'c5'
    else:
        assert False, 'unsupported instance '+args.instance_type

    job = ncluster.make_job(name=args.name,
                            num_tasks=args.machines,
                            image_name=args.image_name,
                            instance_type=args.instance_type,
                            spot=not args.nospot)
    print(f"Logging to {job.logdir}")
    tasks = job.tasks

    for i in range(args.num_procs):
        ip = tasks[0].ip
        port = 6006+i
        tag = f"s{i}"
        tasks[0].switch_window(i)
        tasks[0].run(f'sudo iperf3 -s -p {port}', non_blocking=True)
        tasks[1].switch_window(i)
        tasks[1].run(f'sudo iperf3 -T {tag} -c {ip} -P {args.flows_per_proc} -i 1 -t {args.duration_sec} -V -p {port}',
                     non_blocking=True)



if __name__ == '__main__':
    main()
