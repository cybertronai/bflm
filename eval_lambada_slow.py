#!/usr/bin/env python
# Evaluate GPT-2 model on lambada dataset.

#  ...
#  5000/ 5153, acc: 0.31
#  5100/ 5153, acc: 0.32
# Final accuracy
# acc: 0.31
#
# Before stopword filtering got acc: 0.26
#
# Does line-by-line prediction of several BPE tokens, and compares the last
# word.
#
#
# First 3 mispredictions: true \n predicted
# in my palm is a clear stone , and inside it is a small ivory statuette . a guardian angel . `` figured if you 're going to be out at night getting hit by cars , you might as well have some backup . '' i look at him , feeling stunned . like this is some sort of sign . but as i stare at harlin , his mouth curved in a confident grin , i do n't care about signs
# in my palm is a clear stone , and inside it is a small ivory statuette . a guardian angel . `` figured if you 're going to be out at night getting hit by cars , you might as well have some backup . '' i look at him , feeling stunned . like this is some sort of sign . but as i stare at harlin , his mouth curved in a confident grin , i do n't care about the

# give me a minute to change and i 'll meet you at the docks . '' she 'd forced those words through her teeth . `` no need to change . we wo n't be that long . '' shane gripped her arm and started leading her to the dock . `` i can make it there on my own , shane
# give me a minute to change and i 'll meet you at the docks . '' she 'd forced those words through her teeth . `` no need to change . we wo n't be that long . '' shane gripped her arm and started leading her to the dock . `` i can make it there on my own , but

# helen 's heart broke a little in the face of miss mabel 's selfless courage . she thought that because she was old , her life was of less value than the others ' . for all helen knew , miss mabel had a lot more years to live than she did . `` not going to happen , '' replied helen
# helen 's heart broke a little in the face of miss mabel 's selfless courage . she thought that because she was old , her life was of less value than the others ' . for all helen knew , miss mabel had a lot more years to live than she did . `` not going to happen , '' replied Miss

import argparse

import torch
from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer
from pytorch_pretrained_bert.tokenization import BasicTokenizer

model_name = 'gpt2'
enc = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = BasicTokenizer()


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='lambada_test_plain_text.txt',
                    help='location of lambada dataset')
parser.add_argument('--batch', type=int, default=4, help='batch size')
parser.add_argument('--max-batches', type=int, default=0, help='batch size')
parser.add_argument('--ignore-fragments',  action='store_true', help="Whether to run training.")
parser.add_argument('--word-eval',  action='store_true', help="whether to do evaluation on words rather than BPE "
                                                              "tokens.")
parser.add_argument('--print-every-n',  type=int, default=100, help='print results every n lines')
parser.add_argument('--beam-width',  type=int, default=128, help='predict this many results before stopword filtering')


args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using ", device)
args.device = device

model_name = 'gpt2'
enc = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.to(device)

stopwords = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'}


def argmax(t):
    return int(torch.argmax(t).item())

def to_list(tensor):
    return list(tensor.cpu().numpy())


def remove_last_word(line):
  line = line.strip()
  toks = tokenizer.tokenize(line)
  length_of_word = len(toks[-1])
  assert length_of_word>0
  return line[:-length_of_word].strip(), toks[-1]


def predict(line, max_predictions):
    """Give continuation of the line with at most max_predictions BPE tokens. Returns line extended with predictions of
     the model."""
    line_encoded = enc.encode(line)
    line_encoded = torch.tensor(line_encoded)
    line_encoded = line_encoded.unsqueeze_(0) # batch of size 1
    line_encoded_list = list(line_encoded[0].numpy())
    line_encoded = line_encoded.to(device)
    state = None

    for i in range(max_predictions):
        logits, state = model(line_encoded, past=state)
        
        #        predicted = argmax(logits[0,-1,:])

        # [[idx1, idx2, ...]] 
        _, line_encoded_candidates = torch.topk(logits[:,-1,:], k=args.beam_width, dim=-1)

        # determine which candidates are stopwords by decoding them and
        # comparing against NLTK stopword list

        line_encoded_candidates = to_list(line_encoded_candidates[0])
        is_stopword = []
        for s in line_encoded_candidates:
            is_stopword.append(enc.decode([s.item()]).strip() in stopwords)

        # find first prediction which is not a stopword
        predicted = None
        for (idx, candidate) in enumerate(line_encoded_candidates):
            if is_stopword[idx]:
                #                print('skipping stopword ', idx)
                continue
            else:
                predicted = candidate
                break
        assert predicted is not None
        line_encoded = torch.tensor([[predicted]]).to(device)
        line_encoded_list.append(predicted)

    return enc.decode(line_encoded_list)


def main():
    lines = open(f'{args.path}').readlines()

    predictions_file = open('/ncluster/data/lambada_predictions.txt', 'w')
    errors = 0
    total = 0
    for i, line in enumerate(lines):
        line = line.strip()
        context, last_word = remove_last_word(line)

        # because BPE tokens can span words, predict several BPE tokens
        # and then identify the single word
        prediction = predict(context, 3)
        # string generated by the model
        predicted_part = prediction[len(context):].strip()
        # first word in the generated string
        predicted_word = tokenizer.tokenize(predicted_part)[0]

        is_error = predicted_word != last_word
        if is_error:
            errors += 1
        total+=1

        predictions_file.write(f"{line}\n{predicted_word}\n{is_error}\n\n")

        if i%args.print_every_n == 0:
            print(f"{i:5d}/{len(lines):5d}, acc: {1-errors/total:.2f}")
            predictions_file.flush()

    predictions_file.close()
    print("Final accuracy")
    print(f"acc: {1-errors/total:.2f}")


if __name__=='__main__':
    main()
