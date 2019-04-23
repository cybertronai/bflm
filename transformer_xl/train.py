import torch
from pytorch_pretrained_bert import TransfoXLTokenizer, TransfoXLModel, TransfoXLLMHeadModel, GPT2Tokenizer, TransfoXLConfig
import pytorch_pretrained_bert
from model_sampler import print_samples

import os
import time
from collections import OrderedDict
import data_loader

import logging
logging.basicConfig(level=logging.INFO)

tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')
#config = TransfoXLConfig()
#model = TransfoXLLMHeadModel(config)
model.to('cuda')


from collections import namedtuple
from torch.utils.data import Subset, DataLoader
import importlib
import numpy as np
import time

ARGS = namedtuple('args',['min_file_len','max_file_len',
                          'context_length','output_dir','batch_size',
                         'print_freq','device'])
args = ARGS(None,None,129,'models/',2,100,'cuda')


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
        #global_timeit_dict.setdefault(self.tag, []).append(interval_ms)
        newtag = 'times/' + self.tag
        #print(self.tag, ": ", interval_ms)
        


def checkpoint(model, args):
    output_model_file = os.path.join(args.output_dir, pytorch_pretrained_bert.modeling_transfo_xl.WEIGHTS_NAME)
    print('saving checkpoint to', output_model_file)
    # Only save the model itself
    model_to_save = model.module if hasattr(model, 'module') else model  
    torch.save(model_to_save.state_dict(), output_model_file)
    with open(os.path.join(args.output_dir, pytorch_pretrained_bert.modeling_gpt2.CONFIG_NAME), 'w', encoding='utf-8') as f:
        f.write(model_to_save.config.to_json_string())

        
loader = data_loader.get_data_loader("data/med.txt",tokenizer,args.batch_size,args)
num_epochs = 1000
iterations = len(loader)* num_epochs
print("Number of iterations: %d" % iterations)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = torch.optim.Adam(optimizer_grouped_parameters,lr=1e-4)

#model.apply(model.init_weights)

for current_epoch in range(num_epochs):
    data_loader_iter = iter(loader)
    for step in range(len(loader)):
        model.train()
        start_batch_ts = time.time()
        with timeit('loader'):
            batch = next(data_loader_iter)
        with timeit('batch.to'):
            #print(batch.shape)
            batch = batch.to("cuda")
        with timeit('loss'):
            #print(batch[:,:-1].shape, batch[:,:-1].contiguous().shape)
            loss, _ = model(batch[:,:-1].contiguous(),target=batch[:,1:].contiguous())
            #loss, _ = model(batch.view(1,-1),target=batch.view(1,-1))
            loss = loss.mean()
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

        
        #log_tb('times/tokens_per_sec', 1 / time_per_token)
        #log_tb('times/samples_per_sec', 1 / time_per_sample)
        #log_tb('times/step', 1000 * batch_time)

        if step % 1000 == 0:
            checkpoint(model,args)
            
        if step % args.print_freq == 0:
            print("tokens_per_sec: ", 1/ time_per_token)
            print("samples_per_sec: ", 1/time_per_sample)
            print("step: ", 1000 * batch_time)

            print("memory/allocated_gb", torch.cuda.memory_allocated() / 1e9)
            print("memory/max_allocated_gb", torch.cuda.max_memory_allocated() / 1e9)
            print("memory/cached_gb", torch.cuda.memory_cached() / 1e9)
            print("memory/max_cached_gb", torch.cuda.max_memory_cached() / 1e9)

            print('loss', loss.item())
            # FP16Optimizer doesn't support get_lr
            #                print('lr', optimizer.get_lr()[0])
            print('loss', loss.item())
            #                log_tb('lr', optimizer.get_lr()[0])

            with timeit('sample'):
                sample = print_samples(
                    model, tokenizer, args, unconditional=False,
                    # Context is a random sample from the dataset.
                    context_tokens=next(iter(loader))[0],
                    #context_tokens=tokenizer.convert_to_tensor(tokenizer.tokenize("He peered sideways up and gave a long")),
                    batch_size=1, length=20, nsamples=1,
                    temperature=1, top_k=40)