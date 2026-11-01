# SPDX-License-Identifier: MIT
# Float K/V for a fixed prompt prefix (e.g. <|im_start|> or a system prompt),
# loaded by the runtime with -x: the TPU then starts at position P. This also
# keeps the attention-sink token (activations ~26000) out of the INT8 parts.
# File: int32 P, then int32 token ids [P], then float32 k, v [layers][nkv][P][hd].
# Usage: python make_prefix.py HF_DIR OUT.kv (TEXT | --ids ID[,ID...])
import sys

import numpy as np
from tokenizers import Tokenizer

from model import Decoder

path, out = sys.argv[1], sys.argv[2]
if sys.argv[3] == "--ids":
    ids = [int(x) for x in sys.argv[4].split(",")]
else:
    ids = Tokenizer.from_file(f"{path}/tokenizer.json").encode(sys.argv[3]).ids
P = len(ids)
dec = Decoder(path, S=max(P, 1))
for t in ids:
    dec.step(t)
k = dec.k[:, 0, :, :P].numpy().astype(np.float32)  # [layers, nkv, P, hd]
v = dec.v[:, 0, :, :P].numpy().astype(np.float32)
with open(out, "wb") as f:
    np.array([P] + ids, dtype=np.int32).tofile(f)
    k.tofile(f)
    v.tofile(f)
print(f"prefix of {P} tokens {ids[:8]}{'...' if P > 8 else ''} -> {out}")
