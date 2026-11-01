# SPDX-License-Identifier: MIT
# Check DecodeLayerT against DecodeLayer: T tokens in one call must give the
# same dh / k / v rows as T single-token calls.
# Usage: python check_t.py HF_DIR TEXT [LAYER]
import sys

import torch
from tokenizers import Tokenizer

from model import Decoder, DecodeLayerT, multi_inputs

path, text = sys.argv[1], sys.argv[2]
L = int(sys.argv[3]) if len(sys.argv) > 3 else 15
S = 256
ids = Tokenizer.from_file(f"{path}/tokenizer.json").encode(open(text).read()).ids[:64]
dec = Decoder(path, S)
rec = {L: []}
for t in ids:
    dec.step(t, record=rec)
layer = dec.layers[L]
layer.delta = True
names = ["h", "cos", "sin", "k_cache", "v_cache", "mask"]
with torch.no_grad():
    for T in (4, 8):
        lt = DecodeLayerT(layer, T)
        for p in (0, 5, 40):
            single = [layer(*[rec[L][p + t][k] for k in names]) for t in range(T)]
            h = torch.cat([rec[L][p + t]["h"] for t in range(T)], dim=1)
            cq, sq, ck, sk, mask = multi_inputs(dec.cfg, p, T, S)
            dh, k, v = lt(h, cq, sq, ck, sk, rec[L][p]["k_cache"], rec[L][p]["v_cache"], mask)
            e = max((dh[0, t] - single[t][0][0, 0]).abs().max().item() for t in range(T))
            ek = max((k[0, :, t] - single[t][1][0, :, 0]).abs().max().item() for t in range(T))
            ev = max((v[0, :, t] - single[t][2][0, :, 0]).abs().max().item() for t in range(T))
            print(f"T={T} pos {p:2d}: max|d dh| {e:.2e}  k {ek:.2e}  v {ev:.2e}")
