#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader, Dataset
from tqdm import trange

import pytorch_pretrained_bert
from data_loader import DataSampler, load_dataset
from model_sampler import print_samples
from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer


def checkpoint(model, args):
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    print('saving checkpoint to', output_model_file)
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    torch.save(model_to_save.state_dict(), output_model_file)

class SampledDataset(Dataset):
    def __init__(self, sampler, length):
        self.sampler = sampler
        self.length = length
    def __len__(self):
        # This is a lie
        return self.sampler.total_size
    def __getitem__(self, i):
        # TODO: use the index
        return self.sampler.sample(self.length)
    

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_name_or_path', type=str, default='gpt2',
                        help='pretrained model name')
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--train_dataset', type=str, default='')
    parser.add_argument('--eval_dataset', type=str, default='')
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument('--warmup_proportion', type=float, default=0.002)
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--n_valid', type=int, default=374)
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    enc = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
    model.to(device)

    # ## Prep dataset
    cache_path = f'{args.output_dir}/{os.path.basename(args.train_dataset)}.npz'
    if not os.path.exists(cache_path):
        train_data = load_dataset(enc, args.train_dataset)
        # Cache encoded data.
        print(f'caching data to {cache_path}')
        np.savez_compressed(cache_path, *train_data)
    else:
        train_data = load_dataset(enc, cache_path)
    assert len(train_data) > 0
    
    print(f'loaded {len(train_data)} lines')

    sampler = DataSampler(train_data)
    s = sampler.sample(1024)
    decoded = enc.decode(s)
    print('data sample:', decoded[:100])


    ds = SampledDataset(sampler, 128)
    data_loader = DataLoader(ds, batch_size=args.train_batch_size)
    print('batch shape:', next(iter(data_loader)).shape)
    print('num samples:', sampler.total_size)


    # ## Prep optimizer
    # We use OpenAIAdam because that's what run_openai_gpt used

    from pytorch_pretrained_bert import OpenAIAdam

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    num_train_optimization_steps = len(data_loader) * args.num_train_epochs // args.train_batch_size
    optimizer = OpenAIAdam(optimizer_grouped_parameters,
                        lr=args.learning_rate,
                        warmup=args.warmup_proportion,
                        max_grad_norm=args.max_grad_norm,
                        weight_decay=args.weight_decay,
                        t_total=num_train_optimization_steps)


    # ## Train loop
    # Based on `run_openai_gpt.py`
    nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None

    # Reset all model weights so we can train from scratch.
    model.apply(model.init_weights)

    # Put model in training mode.
    model.train()
    try:
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_steps = 0
            tqdm_bar = tqdm.tqdm(data_loader, desc="Training")
            for step, batch in enumerate(tqdm_bar):
                batch = batch.to(device)
                # input_ids, position_ids=None, token_type_ids=None, lm_labels=None, past=None
                # if lm_labels, outputs loss
                loss = model(batch, lm_labels=batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                tr_loss += loss.item()
                exp_average_loss = loss.item() if exp_average_loss is None else 0.7*exp_average_loss+0.3*loss.item()
                nb_tr_steps += 1
                tqdm_bar.desc = "Training loss: {:.2e} lr: {:.2e}".format(exp_average_loss, optimizer.get_lr()[0])

    except KeyboardInterrupt:
        print_samples(model, enc, args, batch_size=1, length=20, nsamples=1, 
                temperature=1, top_k=40)
        checkpoint(model, args)


if __name__ == '__main__':
    main()
