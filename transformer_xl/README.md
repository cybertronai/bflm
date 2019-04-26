### HuggingFace Transformer-XL Training

Data loading, sampling, and training scripts are modifications of the GPT2 scripts to work with transformer-xl tokenizer and sampling. Since the data loading function does not return an (x,y) tuple, train.py uses a "context length" of (n+1), where the first n tokens are treated as input and last n tokens are treated as target.  
