"""Make a test prompt file for vlm from the SmolLM2 tokenizer and embedding table.

Three copies of one prompt: token ids only, the same tokens as image rows, and token ids
again. All three must give the same answer.

Usage: python make_test.py TOKENIZER.json EMBED.f16 OUT.bin
"""
import sys

import numpy as np
from tokenizers import Tokenizer

tok = Tokenizer.from_file(sys.argv[1])
embed = np.memmap(sys.argv[2], dtype=np.float16, mode="r").reshape(-1, 576)
TEXT = "<|im_start|>user\nWhat is a TPU?<|im_end|>\n<|im_start|>assistant\n"
ids = np.array(tok.encode(TEXT, add_special_tokens=False).ids, dtype=np.int32)
print(len(ids), ids.tolist())

FAKE = len(embed) + 500  # an id outside the vocabulary marks the rows sent as image rows
span = np.arange(2, 10)
marked = ids.copy()
marked[span] = FAKE
rows = embed[ids[span]].astype(np.float32)

with open(sys.argv[3], "wb") as f:
    for x, r in ((ids, None), (marked, rows), (ids, None)):
        np.array([len(x)], dtype=np.int32).tofile(f)
        x.tofile(f)
        if r is not None:
            r.tofile(f)
print("image id", FAKE)
