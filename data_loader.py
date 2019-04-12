""" Stolen from https://github.com/nshepperd/gpt-2/blob/finetuning/src/load_dataset.py"""

import argparse
import glob
import os
import random

import numpy as np
import tqdm
from torch.utils.data import DataLoader, Dataset, Subset

from pytorch_pretrained_bert import GPT2Tokenizer


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


def get_data_loader(dataset_path, enc, batch_size, args, verbose=True):
    data = lazy_load(dataset_path, enc, args)[0]

    # Chunk data by context_length
    ds = Subset(data, [
        slice(i, i+args.context_length) 
        for i in range(0, len(data) - (len(data) % args.context_length), args.context_length)])
    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    if verbose:
        print(f'loaded {len(data)} tokens, {len(ds)} samples')
        decoded = enc.decode(ds[0])
        print('data sample:', decoded[:100])
        print('batch shape:', next(iter(data_loader)).shape)

    return data_loader

def lazy_load(dataset_path, enc, args):
    cache_path = f'{args.output_dir}/{os.path.basename(dataset_path)}.{abs(hash(dataset_path)) % (10 ** 8)}.npz'
    if not os.path.exists(cache_path):
        print('loading data from', dataset_path)
        # Set combine to a huge number so everything is 1 vector
        data = load_dataset(enc, dataset_path, combine=1e99)
        # Cache encoded data.
        print(f'caching data to {cache_path}')
        np.savez_compressed(cache_path, *data)
    else:
        data = load_dataset(enc, cache_path)
    assert len(data) > 0
    return data

def main():
    """Preprocess a dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                    help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--model_name_or_path', type=str, default='gpt2',
                        help='pretrained model name')
    args = parser.parse_args()
    enc = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    _ = lazy_load(args.dataset_path, enc, args)
    
if __name__ == '__main__':
    main()
