#!/usr/bin/env python
#
# Script to move tiny or "deleteme" logdirs out of main logdir root
#

import os
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--logdir_root', type=str, default='/ncluster/runs', help="where logs and events go")
parser.add_argument('--dryrun', help="only print actions, don't don anything", action='store_true')
args = parser.parse_args()


def get_directory_size(start_path='.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def main():
    os.system(f'mkdir -p {args.logdir_root}.old')
    for d in os.listdir(args.logdir_root):
        logdir = f'{args.logdir_root}/{d}'
        moved_logdir = f'{args.logdir_root}.old/{d}'
        dir_size = get_directory_size(logdir)
        
        if dir_size < 100 or 'deleteme' in logdir:
            print(f"Moving {logdir} to {moved_logdir}")
            if not args.dryrun:
                os.system(f'mv {logdir} {moved_logdir}')


if __name__=='__main__':
    main()