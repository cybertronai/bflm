#!/usr/bin/env python

import argparse
import ncluster

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='pair',
                    help="instance name")
parser.add_argument('--image-name', type=str,
                    default='cybertronai01',
                    help="name of AMI to use ")
parser.add_argument('--instance-type', type=str, default='p3.8xlarge',
                    help="type of instance")

args = parser.parse_args()


def main():
    job = ncluster.make_job(name=args.name,
                            instance_type=args.instance_type,
                            num_tasks=2,
                            disk_size=1000,
                            image_name=args.image_name)
    public_keys = {}
    for task in job.tasks:
        key_fn = '~/.ssh/id_rsa'
        task.run(f"yes | ssh-keygen -t rsa -f {key_fn} -N ''")
        public_keys[task] = task.read(key_fn + '.pub')

    for task1 in job.tasks:
        task1.run("""sudo bash -c 'echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config'""")
        for task2 in job.tasks:
            # allow passwordless SSH from task1 to task2
            task2.run(f'echo "{public_keys[task1]}" >> ~/.ssh/authorized_keys')


if __name__ == '__main__':
    main()
