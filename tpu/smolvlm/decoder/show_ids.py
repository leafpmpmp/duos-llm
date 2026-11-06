"""Print vlm output lines as text. Usage: python show_ids.py TOKENIZER.json ANSWERS.txt"""
import sys

from tokenizers import Tokenizer

tok = Tokenizer.from_file(sys.argv[1])
for line in open(sys.argv[2]):
    p = line.split()
    print(p[0], p[1], p[2], repr(tok.decode([int(x) for x in p[3:]], skip_special_tokens=True)))
