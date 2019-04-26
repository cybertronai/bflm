#!/usr/bin/env python

import argparse
import ncluster

parser = argparse.ArgumentParser()
parser.add_argument('--image-name', type=str,
                    default='Deep Learning AMI (Ubuntu) Version 22.0',
                    help="name of AMI to use ")

args = parser.parse_args()

ncluster.set_backend('aws')


def main():
  task = ncluster.make_task(name='p3-billing-test',
                            instance_type='p3.16xlarge',
                            disk_size=1000,
                            image_name=args.image_name)
  task.run('sudo shutdown now', non_blocking=True)
  task = ncluster.make_task(name='p3-dn-billing-test',
                            instance_type='p3dn.24xlarge',
                            disk_size=1000,
                            image_name=args.image_name)
  task.run('sudo shutdown now', non_blocking=True)


if __name__ == '__main__':
  main()
