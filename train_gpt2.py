#!/usr/bin/env python
# coding: utf-8

import pytorch_pretrained_bert
from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer
import argparse
import logging
from tqdm import trange
import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import os
from data_loader import load_dataset, Sampler
from torch.utils.data import Dataset, DataLoader

from sampler import print_samples


def checkpoint(model, args):
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(args['output_dir'], "pytorch_model.bin")
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
    parser.add_argument('--model_name', type=str, default='openai-gpt',
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
    parser.add_argument('--lm_coef', type=float, default=0.9)
    parser.add_argument('--n_valid', type=int, default=374)
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()
    args = vars(args)
    args.update(dict(batch_size=1, length=200, 
                model_name_or_path='gpt2', nsamples=1, 
                seed=0, temperature=1, top_k=40, unconditional=False))

    torch.random.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = GPT2Tokenizer.from_pretrained(args['model_name_or_path'])
    model = GPT2LMHeadModel.from_pretrained(args['model_name_or_path'])
    model.to(device)


    # TODO: add args here
    # print_samples()

    # ## Prep dataset
    # `load_dataset` stolen from https://github.com/nshepperd/gpt-2/blob/finetuning/src/load_dataset.py


    #train_data = load_dataset(enc, '/ncluster/data/wikiextracted/AA/*')
    #train_data = load_dataset(enc, '/home/ubuntu/data/wikiAA.npz')
    train_data = load_dataset(enc, '/Users/ben/data/wikitext-2/wiki.train.tokens')
    assert len(train_data) > 0

    # Cache encoded data.
    print('saving data')
    SAVE_PATH = '/Users/ben/data/wikitext-2/wiki.train.npz'
    np.savez_compressed(SAVE_PATH, *train_data)
    print(len(train_data))


    sampler = Sampler(train_data)
    s = sampler.sample(1024)
    decoded = enc.decode(s)
    print(len(decoded), decoded[:100])


    ds = SampledDataset(sampler, 128)
    batch_size = 4
    data_loader = DataLoader(ds, batch_size=batch_size)
    print(next(iter(data_loader)).shape)
    print(sampler.total_size)


    # ## Prep optimizer
    # We use OpenAIAdam because that's what run_openai_gpt used

    num_train_epochs, train_batch_size, warmup_proportion, max_grad_norm, weight_decay, learning_rate = 3, batch_size, .002, 1, .01, 6e-5
    from pytorch_pretrained_bert import OpenAIAdam

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    num_train_optimization_steps = len(data_loader) * num_train_epochs // train_batch_size
    optimizer = OpenAIAdam(optimizer_grouped_parameters,
                        lr=learning_rate,
                        warmup=warmup_proportion,
                        max_grad_norm=max_grad_norm,
                        weight_decay=weight_decay,
                        t_total=num_train_optimization_steps)


    # ## Train loop
    # Based on `run_openai_gpt.py`

    lm_coef = 0.9
    nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None
    model.apply(model.init_weights)
    model.train()
    try:
        for _ in trange(int(num_train_epochs), desc="Epoch"):
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
        checkpoint(model, args)



if __name__ == '__main__':
    main()