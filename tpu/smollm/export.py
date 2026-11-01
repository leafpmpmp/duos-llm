# SPDX-License-Identifier: MIT
# Export one SmolLM2 decode layer to ONNX, with real calibration inputs.
# Usage: python export.py HF_DIR TEXT LAYER [S] [OUT_DIR] [delta]
import os
import sys

import numpy as np
import torch
from tokenizers import Tokenizer

from model import Decoder

path, text, L = sys.argv[1], sys.argv[2], int(sys.argv[3])
S = int(sys.argv[4]) if len(sys.argv) > 4 else 256
out = sys.argv[5] if len(sys.argv) > 5 else f"layer{L}_s{S}"
delta = len(sys.argv) > 6 and sys.argv[6] == "delta"
steps = min(200, S)
names_in = ["h", "cos", "sin", "k_cache", "v_cache", "mask"]
names_out = ["dh" if delta else "h_out", "k_new", "v_new"]

tok = Tokenizer.from_file(f"{path}/tokenizer.json")
ids = tok.encode(open(text).read()).ids[:steps]
dec = Decoder(path, S)
rec = {L: []}
for t in ids:
    dec.step(t, record=rec)
samples = rec[L]

os.makedirs(f"{out}/calib", exist_ok=True)
layer = dec.layers[L]
layer.delta = delta
# Calibration: 32 positions spread over the run; test: 4 other positions.
calib = np.linspace(4, steps - 1, 32).astype(int)
test = [17, 63, 130, steps - 3]
for i in calib:
    np.savez(f"{out}/calib/pos{i:03d}.npz", **{k: samples[i][k].numpy() for k in names_in})
for i in test:
    x = samples[i]
    with torch.no_grad():
        y = layer(*[x[k] for k in names_in])
    np.savez(f"{out}/test_in_pos{i:03d}.npz", **{k: x[k].numpy() for k in names_in})
    np.savez(f"{out}/test_ref_pos{i:03d}.npz", **{k: v.numpy() for k, v in zip(names_out, y)})

x = samples[test[0]]
torch.onnx.export(layer, tuple(x[k] for k in names_in), f"{out}/layer{L}.onnx",
                  input_names=names_in, output_names=names_out, opset_version=17, dynamo=False)
print(f"layer {L}: {len(calib)} calib + {len(test)} test samples, ONNX in {out}/")
