# SPDX-License-Identifier: MIT
# Export all SmolLM2 decode layers (delta output) and the LM head to ONNX,
# each with calibration and test inputs from one float run, plus the fp16
# embedding table for the CPU-side lookup.
# Usage: [CONV=1] python export_all.py HF_DIR TEXT OUT_DIR [S]
#   CONV=1: linear layers as 1x1 convolutions (per-channel INT8 weights on cv18xx)
import os
import sys

import numpy as np
import torch
from tokenizers import Tokenizer
from torch import nn

from model import Decoder

path, text, out = sys.argv[1], sys.argv[2], sys.argv[3]
S = int(sys.argv[4]) if len(sys.argv) > 4 else 256
steps = min(200, S)
names_in = ["h", "cos", "sin", "k_cache", "v_cache", "mask"]


class Head(nn.Module):
    def __init__(self, dec, conv=False):
        super().__init__()
        self.norm = dec.norm
        self.conv = conv
        if conv:  # one INT8 scale per vocabulary row on cv18xx
            self.w = nn.Parameter(dec.emb.reshape(dec.cfg.vocab, dec.cfg.d, 1, 1).contiguous(), requires_grad=False)
        else:
            self.w = nn.Parameter(dec.emb.t().contiguous(), requires_grad=False)

    def forward(self, h):
        if self.conv:
            y = nn.functional.conv2d(self.norm(h).reshape(1, -1, 1, 1), self.w)
            return y.reshape(1, 1, -1)
        return self.norm(h) @ self.w


tok = Tokenizer.from_file(f"{path}/tokenizer.json")
ids = tok.encode(open(text).read()).ids[:steps]
dec = Decoder(path, S)
L = dec.cfg.layers
rec = {i: [] for i in range(L)}
final = []
with torch.no_grad():
    for t in ids:
        h, cos, sin, mask = dec.inputs(t)
        for i, layer in enumerate(dec.layers):
            rec[i].append(dict(h=h.clone(), cos=cos, sin=sin, k_cache=dec.k[i].clone(),
                               v_cache=dec.v[i].clone(), mask=mask))
            h, k, v = layer(h, cos, sin, dec.k[i], dec.v[i], mask)
            dec.k[i][:, :, dec.t] = k[:, :, 0]
            dec.v[i][:, :, dec.t] = v[:, :, 0]
        final.append(h.clone())
        dec.t += 1

# Calibration: positions 0-3 (the attention sink) + 28 spread; test: 4 others.
calib = sorted(set([0, 1, 2, 3] + list(np.linspace(5, steps - 1, 28).astype(int))))
test = [17, 63, 130, steps - 3]


def dump(d, x, y, out_names):
    os.makedirs(f"{d}/calib", exist_ok=True)
    for i in calib:
        np.savez(f"{d}/calib/pos{i:03d}.npz", **{k: v.numpy() for k, v in x[i].items()})
    for i in test:
        np.savez(f"{d}/test_in_pos{i:03d}.npz", **{k: v.numpy() for k, v in x[i].items()})
        np.savez(f"{d}/test_ref_pos{i:03d}.npz", **{k: v.numpy() for k, v in zip(out_names, y(x[i]))})


with torch.no_grad():
    for i, layer in enumerate(dec.layers):
        layer.delta = True
        layer.conv = os.environ.get("CONV") == "1"
        d = f"{out}/l{i:02d}"
        outs = ["dh", "k_new", "v_new"]
        dump(d, rec[i], lambda x: layer(*[x[k] for k in names_in]), outs)
        x = rec[i][test[0]]
        torch.onnx.export(layer, tuple(x[k] for k in names_in), f"{d}/l{i:02d}.onnx",
                          input_names=names_in, output_names=outs, opset_version=17, dynamo=False)
    head = Head(dec, conv=os.environ.get("CONV") == "1")
    hx = [dict(h=h) for h in final]
    dump(f"{out}/head", hx, lambda x: (head(x["h"]),), ["logits"])
    torch.onnx.export(head, (final[test[0]],), f"{out}/head/head.onnx",
                      input_names=["h"], output_names=["logits"], opset_version=17, dynamo=False)

dec.emb.half().numpy().tofile(f"{out}/embed.f16")
print(f"{L} layers + head in {out}/ ({len(calib)} calib, {len(test)} test samples each), embed.f16")
