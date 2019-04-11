#!/usr/bin/env python
# coding: utf-8

import pytorch_pretrained_bert
from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer
import argparse
import logging
from tqdm import trange

import torch
import torch.nn.functional as F
import numpy as np


args = dict(batch_size=1, length=200, 
            model_name_or_path='gpt2', nsamples=1, 
            seed=0, temperature=1, top_k=40, unconditional=False)


def top_k_logits(logits, k):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)


def sample_sequence(model, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, device='cuda', sample=True):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
    prev = context
    output = context
    past = None
    with torch.no_grad():
        for i in trange(length):
            logits, past = model(prev, past=past)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            output = torch.cat((output, prev), dim=1)
    return output


torch.random.manual_seed(args['seed'])
torch.cuda.manual_seed(args['seed'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

enc = GPT2Tokenizer.from_pretrained(args['model_name_or_path'])
model = GPT2LMHeadModel.from_pretrained(args['model_name_or_path'])
model.to(device)

def print_samples():
    model.eval()

    generated = 0
    print(args)
    for _ in range(args['nsamples'] // args['batch_size']):
        context_tokens = context_tokens = enc.encode('This is a test.')
        out = sample_sequence(
            model=model, length=args['length'],
            context=context_tokens if not args['unconditional'] else None,
            start_token=enc.encoder['<|endoftext|>'] if args['unconditional'] else None,
            batch_size=args['batch_size'],
            temperature=args['temperature'], top_k=args['top_k'], device=device
        )
        out = out[:, len(context_tokens):].tolist()
        for i in range(args['batch_size']):
            generated += 1
            text = enc.decode(out[i])
            print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
            print(text)

#print_samples()

# ## Prep dataset
# `load_dataset` stolen from https://github.com/nshepperd/gpt-2/blob/finetuning/src/load_dataset.py

import glob
import numpy as np
import os
import random
import tqdm

def load_dataset(enc, path, combine=50000):
    paths = []
    if os.path.isfile(path):
        # Simple file
        paths.append(path)
    elif os.path.isdir(path):
        # Directory
        for (dirpath, _, fnames) in os.walk(path):
            for fname in fnames:
                paths.append(os.path.join(dirpath, fname))
    else:
        # Assume glob
        paths = glob.glob(path)

    token_chunks = []
    raw_text = ''
    for path in tqdm.tqdm(paths):
        if path.endswith('.npz'):
            # Pre-encoded
            with np.load(path) as npz:
                for item in npz.files:
                    token_chunks.append(npz[item])
        else:
            # Plain text
            with open(path, 'r') as fp:
                raw_text += fp.read()
            if len(raw_text) >= combine:
                tokens = np.stack(enc.encode(raw_text))
                token_chunks.append(tokens)
                raw_text = ''
            else:
                raw_text += '<|endoftext|>'
    if raw_text:
        tokens = np.stack(enc.encode(raw_text))
        token_chunks.append(tokens)
    return token_chunks

#train_data = load_dataset(enc, '/ncluster/data/wikiextracted/AA/*')
train_data = load_dataset(enc, '/home/ubuntu/data/wikiAA.npz')
#train_data = load_dataset(enc, '/Users/ben/data/wikitext-2/wiki.train.tokens')
assert len(train_data) > 0

# Cache encoded data.
# np.savez_compressed('/home/ubuntu/data/wikiAA.npz', *train_data)
print(len(train_data))

def binary_search(f, lo, hi):
    if f(lo) or not f(hi):
        return None
    while hi > lo + 1:
        mid = (lo + hi) // 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return hi


class Sampler(object):
    """Fairly samples a slice from a set of variable sized chunks.

    'Fairly' means that the distribution is the same as sampling from one concatenated chunk,
    but without crossing chunk boundaries."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.total_size = sum(chunk.shape[0] for chunk in chunks)
        self.boundaries = [0]
        for i in range(len(chunks)):
            self.boundaries.append(self.boundaries[-1] + chunks[i].shape[0])

    def sample(self, length):
        assert length < self.total_size // len(
            self.chunks
        ), "Dataset files are too small to sample {} tokens at a time".format(
            length)
        while True:
            index = random.randint(0, self.total_size - length - 1)
            i = binary_search(lambda j: self.boundaries[j] > index, 0,
                              len(self.boundaries) - 1) - 1
            if self.boundaries[i + 1] > index + length:
                within_chunk = index - self.boundaries[i]
                return self.chunks[i][within_chunk:within_chunk + length]

sampler = Sampler(train_data)
s = sampler.sample(1024)
decoded = enc.decode(s)
print(len(decoded), decoded[:100])
from torch.utils.data import Dataset, DataLoader

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



model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
config = model.config
torch.save(model_to_save.state_dict(), output_model_file)

