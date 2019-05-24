#!/usr/bin/env python
# Evaluate GPT-2 model on lambada dataset.
# This compares match in the last BPE token rather than match in last predicted word. A heuristic --ignore-fragments will throw out any examples where last word doesn't encode to single BPE token
#
#
# Example usage:
#
# python eval_lambada.py --batch=20 --path=/ncluster/data/lambada/lambada_test_plain_text.txt --ignore-fragments --preprocess
# Accuracy: 0.3128
#
# python eval_lambada.py --batch=20 --path=/ncluster/data/lambada/lambada_test_plain_text.txt --ignore-fragments
# Accuracy: 0.3093
#
# python eval_lambada.py --batch=20 --path=/ncluster/data/lambada/lambada_test_plain_text.txt
# Accuracy: 0.61
#
# python eval_lambada.py --batch=20 --path=/ncluster/data/lambada/lambada_test_plain_text.txt --ignore-fragments
# Accuracy: 0.3093

# python eval_lambada.py --ignore-fragments --jeff_suggestion --ignore-fragments
# Accuracy: 0.3171
#
# python eval_lambada.py --batch=20 --path=/ncluster/data/lambada/lambada_control_test_data_plain_text.txt
# Accuracy: 0.35

import argparse
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from tqdm import trange

import pytorch_pretrained_bert
from data_loader import get_data_loader
from model_sampler import print_samples
from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer, OpenAIAdam
from torch.utils.data import DataLoader, Dataset, Subset

model_name = 'gpt2'
enc = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='/ncluster/data/lambada/lambada_test_plain_text.txt', help='location of lambada dataset')
parser.add_argument('--batch', type=int, default=4, help='batch size')
parser.add_argument('--max-batches', type=int, default=0, help='batch size')
parser.add_argument('--ignore-fragments',  action='store_true', help="Whether to run training.")
parser.add_argument('--preprocess',  action='store_true', help="strip quotes")
parser.add_argument('--jeff_suggestion',  action='store_true', help="use jeff's suggestion of prepending \n to each example")
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device

model_name = 'gpt2'
enc = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.to(device)


def argmax(t):
    return int(torch.argmax(t).item())

# from https://github.com/openai/gpt-2/issues/131#issuecomment-492786058
def preprocess(text):
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("''", '"')
    text = text.replace("``", '"')
    return '\n'+text.strip()

def score_batch(batch):
    """Return number of last-word mismatches in a batch."""
    batch_encoded = []
    lengths = []
    fragments = []
    for line in batch:
        line = line.strip()
        if args.jeff_suggestion:
            line = '\n'+line
        line_encoded = enc.encode(line)
        encoded_last_word = enc.decode(line_encoded[-1:]).strip()
        actual_last_word = line.split()[-1].strip()
        if encoded_last_word != actual_last_word:
            fragments.append(True)
        else:
            fragments.append(False)
        batch_encoded.append(line_encoded)

    # array is ragged, so pad to turn into rectangular tensor
    max_len = max(len(encoded) for encoded in batch_encoded)
    batch_padded = []
    for encoded in batch_encoded:
        batch_padded.append(encoded+[0]*(max_len - len(encoded)))
        lengths.append(len(encoded))

    batch_padded = torch.tensor(batch_padded)
    batch_padded = batch_padded.to(device)
    logits, presents = model(batch_padded)

    errors = 0
    total = 0
    for i in range(args.batch):
        # break on small last batch
        if i >= len(batch_padded):
            break
        last_idx = lengths[i]-1
        observed = batch_encoded[i][last_idx]
        predicted = argmax(logits[i][last_idx-1])
        if args.ignore_fragments and fragments[i]:
            continue
        total+=1
        errors += 0 if (observed == predicted) else 1

    return errors, total


def main():
    ds_raw = open(f'{args.path}').read()
    if args.preprocess:
        ds_raw = preprocess(ds_raw)
        
    ds = ds_raw.strip().split('\n')
        
    data_loader = DataLoader(ds, batch_size=args.batch, shuffle=False)
    
    errors = 0
    total = 0
    for i, batch in enumerate(data_loader):
        errors_batch, total_batch = score_batch(batch)
        errors += errors_batch
        total += total_batch
        if args.max_batches and i>=args.max_batches-1:
            break

    print("Accuracy: %.4f"%(1-errors/total,))
        

if __name__=='__main__':
    main()
