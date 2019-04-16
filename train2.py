#!/usr/bin/env python
#
# coding: utf-8

import argparse
import os
import time
from collections import OrderedDict

import torch
from tensorboardX import SummaryWriter
from data_loader import get_data_loader
from model_sampler import print_samples

from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer, OpenAIAdam

# global variables
global_timeit_dict = OrderedDict()
global_example_count = 0
global_token_count = 0
event_writer = None
logdir = None


class timeit:
    """Decorator to measure length of time spent in the block in millis and log
  it to TensorBoard."""

    def __init__(self, tag=""):
        self.tag = tag

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        interval_ms = 1000 * (self.end - self.start)
        global_timeit_dict.setdefault(self.tag, []).append(interval_ms)
        newtag = 'times/' + self.tag
        log_tb(newtag, interval_ms)


def log_tb(tag, val):
    """Log value to tensorboard (relies on global_example_count rather than step count to give comparable graphs across
    batch sizes)"""
    global global_token_count, event_writer
    event_writer.add_scalar(tag, val, global_token_count)


def checkpoint(model, args):
    ts = int(time.time()) - 1555360224  # ts relative to Apr 15, 2019
    output_model_file = os.path.join(logdir, f"{ts}.bin")
    print('saving checkpoint to', output_model_file)
    # Only save the model itself
    model_to_save = model.module if hasattr(model, 'module') else model
    with timeit('checkpoint'):
        torch.save(model_to_save.state_dict(), output_model_file)


def current_timestamp() -> str:
    # timestamp format like 2019-04-15_11-29-51
    current_seconds = time.time()

    # correct to local timezone (PDT) if running on AWS (which is UTC)
    import datetime
    from pytz import reference
    localtime = reference.LocalTimezone()
    today = datetime.datetime.now()
    timezone = localtime.tzname(today)

    # TODO(y): use pytz for proper timezone conversion instead of -=
    if timezone == 'UTC':
        current_seconds -= 7 * 3600
    else:
        assert timezone == 'PDT'
    time_str = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(current_seconds))
    return time_str


def parse_args():
    parser = argparse.ArgumentParser(description='LM training')
    parser.add_argument('--data', metavar='DIR', help='path to dataset')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--fp16', action='store_true', help='Run model fp16 mode. Default True')
    parser.add_argument('--loss-scale', type=float, default=1024,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--distributed', action='store_true', help='Run distributed training. Default True')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='Used for multi-process training. Can either be manually set ' +
                             'or automatically set by using \'python -m multiproc\'.')
    parser.add_argument('--short-epoch', action='store_true',
                        help='make epochs short (for debugging)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_name_or_path', type=str, default='gpt2',
                        help='pretrained model name')
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    # TODO: rename to "temporary dir" since it's only for npz files
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--context_length', type=int, default=1024)
    parser.add_argument('--num_train_epochs', type=int, default=99999)
    parser.add_argument('--batch_size', type=int, default=16)

    # optimizer params
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--warmup_proportion', type=float, default=0.002)
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parser.add_argument('--logdir_root', type=str, default='/ncluster/runs', help="where logs and events go")
    parser.add_argument('--run_name', type=str, default='default', help="name of run")

    parser.add_argument('--print_freq', '-p', default=10, type=int,
                        metavar='N', help='log/print every this many steps (default: 5)')

    parser.add_argument('--min_file_len', type=int, help="When loading dataset, throw out files with fewer than this many characters")
    parser.add_argument('--max_file_len', type=int, help="When loading dataset, throw out files with greater than this many characters")

    args = parser.parse_args()
    return args


def main():
    global global_example_count, global_token_count, event_writer, logdir

    args = parse_args()

    logdir = f'{args.logdir_root}/{args.run_name}-{current_timestamp()}'
    os.system(f'mkdir -p {logdir}')
    os.system(f'mkdir -p {args.output_dir}')
    assert os.path.exists(args.data), f"Didn't find {args.data}"

    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    enc = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
    model.to(device)

    # setup TensorBoard logging
    global_example_count = 0
    global_token_count = 0
    print(f"Logging to {logdir}")
    event_writer = SummaryWriter(logdir)
    log_tb("first", time.time())

    data_loader = get_data_loader(args.data, enc, args.batch_size, args)

    # ## Prep optimizer
    # We use OpenAIAdam because that's what run_openai_gpt used
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    num_train_optimization_steps = len(data_loader) * args.num_train_epochs

    optimizer = OpenAIAdam(optimizer_grouped_parameters,
                           lr=args.learning_rate,
                           warmup=args.warmup_proportion,
                           max_grad_norm=args.max_grad_norm,
                           weight_decay=args.weight_decay,
                           t_total=num_train_optimization_steps)

    # Reset all model weights so we can train from scratch.
    model.apply(model.init_weights)
    model.train()

    for current_epoch in range(args.num_train_epochs):
        data_loader_iter = iter(data_loader)
        for step in range(len(data_loader)):
            start_batch_ts = time.time()
            with timeit('dataloader'):
                batch = next(data_loader_iter)
            with timeit('batch.to'):
                batch = batch.to(device)
            with timeit('loss'):
                loss = model(batch, lm_labels=batch)
            with timeit('loss.backward'):
                loss.backward()
            with timeit('optimizer.step'):
                optimizer.step()
            optimizer.zero_grad()
            end_batch_ts = time.time()

            # time to do single batch
            batch_time = end_batch_ts - start_batch_ts

            total_tokens = args.context_length * args.batch_size
            time_per_token = batch_time / total_tokens
            time_per_sample = batch_time / args.batch_size
            log_tb('times/tokens_per_sec', 1 / time_per_token)
            log_tb('times/samples_per_sec', 1 / time_per_sample)
            log_tb('times/step', 1000 * batch_time)

            if step % args.print_freq == 0:
                log_tb("memory/allocated_gb", torch.cuda.memory_allocated() / 1e9)
                log_tb("memory/max_allocated_gb", torch.cuda.max_memory_allocated() / 1e9)
                log_tb("memory/cached_gb", torch.cuda.memory_cached() / 1e9)
                log_tb("memory/max_cached_gb", torch.cuda.max_memory_cached() / 1e9)

                print('loss', loss.item())
                print('lr', optimizer.get_lr()[0])
                log_tb('loss', loss.item())
                log_tb('lr', optimizer.get_lr()[0])

                with timeit('sample'):
                    sample = print_samples(
                        model, enc, args,
                        # Context is a random sample from the dataset.
                        context_tokens=next(iter(data_loader)),
                        batch_size=1, length=20, nsamples=1,
                        temperature=1, top_k=40)
                event_writer.add_text('sample', sample, global_example_count)

            # TODO: replace with len(batch)
            global_example_count += args.batch_size
            global_token_count += total_tokens

        # checkpoint at the end of each epoch
        print("Checkpointing at epoch ", current_epoch)
        checkpoint(model, args)


if __name__ == '__main__':
    main()
