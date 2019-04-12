""" Stolen from https://github.com/nshepperd/gpt-2/blob/finetuning/src/load_dataset.py"""

import glob
import numpy as np
import os
import random
import tqdm
from torch.utils.data import DataLoader, Dataset, Subset

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


class DataSampler():
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
    cache_path = f'{args.output_dir}/{os.path.basename(dataset_path)}.npz'
    if not os.path.exists(cache_path):
        # Set combine to a huge number so everything is 1 vector
        data = load_dataset(enc, dataset_path, combine=1e99)
        # Cache encoded data.
        print(f'caching data to {cache_path}')
        np.savez_compressed(cache_path, *data)
    else:
        data = load_dataset(enc, cache_path)
    assert len(data) > 0
    return data

    