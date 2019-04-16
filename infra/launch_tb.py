#!/usr/bin/env python
# script to launch tensorboard job for a single run


import argparse
import ncluster

parser = argparse.ArgumentParser(description='Launch TensorBoard')
parser.add_argument('--name', type=str, default='tensorboard',
                    help="instance name")
parser.add_argument('--instance-type', type=str, default='r5.large',
                    help="type of instance")
parser.add_argument('--image-name', type=str,
                    default='Deep Learning AMI (Ubuntu) Version 22.0',
                    help="name of AMI to use ")
parser.add_argument('--logdir_root', type=str, default='/ncluster/runs', help="where logs and events go")

args = parser.parse_args()


ncluster.set_backend('aws')


def main():
  task = ncluster.make_task(name=args.name,
                            instance_type=args.instance_type,
                            disk_size=100,
                            image_name=args.image_name)

  task.run('source activate tensorflow_p36')
  task.run(f'tensorboard --logdir={args.logdir_root} --port=6006', non_blocking=True)
  print(f'TensorBoard at http://{task.public_ip}:6006')


if __name__=='__main__':
  main()
