"""Collection of functions based on huggingface examples/run_gpt2.py to sample from a GPT-2 model."""

import torch
import torch.nn.functional as F
from tqdm import trange

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

def print_samples(model, enc, args, context_tokens=[], unconditional=True, **kwargs):
    print('generating samples')
    model.eval()
    generated = 0
    for _ in range(kwargs['nsamples'] // kwargs['batch_size']):
        out = sample_sequence(
            model=model, length=kwargs['length'],
            context=context_tokens if not unconditional else None,
            start_token=enc.encoder['<|endoftext|>'] if unconditional else None,
            batch_size=kwargs['batch_size'],
            temperature=kwargs['temperature'], top_k=kwargs['top_k'], device=args.device
        )
        out = out[:, len(context_tokens):].tolist()
        for i in range(kwargs['batch_size']):
            generated += 1
            print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
            text = enc.decode(out[i])
            print(text)
    return text