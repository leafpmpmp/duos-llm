# SPDX-License-Identifier: MIT
# Float reference perplexity for the runtime's --ppl mode: the same tokens,
# positions 0..n-1, NLL of token i+1 from the logits at i. Writes the tokens
# as int32 for the board.
# With PREFIX_ID set, that token goes first and position 0 is not scored
# (the runtime gets its K/V from make_prefix.py).
# Usage: [PREFIX_ID=1] python ref_ppl.py HF_DIR TEXT OUT.i32 [N]
import os
import sys

import numpy as np
import torch
from tokenizers import Tokenizer

from model import Decoder

path, text, out = sys.argv[1], sys.argv[2], sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 256
pre = [int(os.environ["PREFIX_ID"])] if "PREFIX_ID" in os.environ else []
ids = (pre + Tokenizer.from_file(f"{path}/tokenizer.json").encode(open(text).read()).ids)[:n]
np.array(ids, dtype=np.int32).tofile(out)
dec = Decoder(path, S=n)
nll, first = 0.0, len(pre)
for i in range(n - 1):
    x = dec.step(ids[i]).double()
    if i >= first:
        nll += float(torch.logsumexp(x, 0) - x[ids[i + 1]])
m = n - 1 - first
print(f"tokens {n} (scored {m}): float PPL {np.exp(nll / m):.4f} (mean NLL {nll / m:.4f})")
