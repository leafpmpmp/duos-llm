# SPDX-License-Identifier: MIT
# Compare the runtime's LLM_DUMP (residual after each layer, first 8
# positions) with the float model on the same tokens.
# Usage: python compare_dump.py HF_DIR TOKENS.i32 DUMP.f32
import sys

import numpy as np
import torch

from model import Decoder

path, tokfile, dump = sys.argv[1:4]
ids = np.fromfile(tokfile, dtype=np.int32)[:8]
got = np.fromfile(dump, dtype=np.float32).reshape(8, 30, 576)
dec = Decoder(path, S=256)
print("cos(h) per layer; rows = position, columns = layers 0,3,6,...,27,29; then max|h| float vs TPU")
with torch.no_grad():
    for p, t in enumerate(ids):
        h, cos, sin, mask = dec.inputs(int(t))
        ref = []
        for i, layer in enumerate(dec.layers):
            h, k, v = layer(h, cos, sin, dec.k[i], dec.v[i], mask)
            dec.k[i][:, :, dec.t] = k[:, :, 0]
            dec.v[i][:, :, dec.t] = v[:, :, 0]
            ref.append(h.numpy().ravel())
        dec.t += 1
        c = [float(np.dot(ref[i], got[p, i]) / (np.linalg.norm(ref[i]) * np.linalg.norm(got[p, i])))
             for i in range(30)]
        cols = list(range(0, 30, 3)) + [29]
        print(f"pos {p}: " + " ".join(f"{c[i]:.3f}" for i in cols) +
              f"   max|h| {np.abs(ref[29]).max():.1f} vs {np.abs(got[p, 29]).max():.1f}")
