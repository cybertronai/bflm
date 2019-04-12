"""Detokenizes a file. Removes newlines 😣."""
import sys
from sacremoses import MosesTokenizer, MosesDetokenizer

def main():
    assert len(sys.argv) == 3, 'Usage: detokenizer.py $input $output'
    with open(sys.argv[1]) as f:
        tokens = f.read().split(' ')
        md = MosesDetokenizer(lang='en')
        with open(sys.argv[2], 'w') as out:
            out.write(md.detokenize(tokens, unescape=False))

if __name__ == '__main__':
    main()
