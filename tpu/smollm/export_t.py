# SPDX-License-Identifier: MIT
# Export T-token decode layers (DecodeLayerT) and a T-row LM head, with
# calibration and test inputs from one float run. Windows start at position
# 1 or later: position 0 comes from the float prefix (make_prefix.py).
# Writes per model: NAME.onnx, calib/, test_*.npz, shapes, quant_lists.
# Usage: [S=slots] python export_t.py HF_DIR TEXT OUT_DIR T [LAYERS]   (LAYERS: e.g. 15, 0-29 or head; default all + head)
# S: KV cache slots (default 256). The float run uses the first S - 56 tokens of TEXT.
import os
import sys

import numpy as np
import torch
from tokenizers import Tokenizer
from torch import nn

from model import Decoder, DecodeLayerT, multi_inputs

path, text, out, T = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
S = int(os.environ.get("S", 256))
sel = sys.argv[5] if len(sys.argv) > 5 else "all"
# Cache inputs first: TPU-MLIR gives the first five inputs buffers of their own
# (io_mem); the others share private memory with the outputs.
names_in = ["k_cache", "v_cache", "mask", "h", "cos_q", "sin_q", "cos_k", "sin_k"]
names_out = ["dh", "k_new", "v_new"]


class CacheFirst(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, k_cache, v_cache, mask, h, cos_q, sin_q, cos_k, sin_k):
        return self.layer(h, cos_q, sin_q, cos_k, sin_k, k_cache, v_cache, mask)


class Head(nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.norm = dec.norm
        self.w = nn.Parameter(dec.emb.t().contiguous(), requires_grad=False)

    def forward(self, h):
        return self.norm(h) @ self.w


tok = Tokenizer.from_file(f"{path}/tokenizer.json")
ids = tok.encode(open(text).read()).ids[:S - 56]
dec = Decoder(path, S)
L = dec.cfg.layers
layers = (range(L) if sel == "all" else [] if sel == "head" else
          range(int(sel.split("-")[0]), int(sel.split("-")[1]) + 1) if "-" in sel else [int(sel)])
rec = {i: [] for i in range(L)}
final = []
for t in ids:
    h, cos, sin, mask = dec.inputs(t)
    with torch.no_grad():
        for i, layer in enumerate(dec.layers):
            rec[i].append(h.clone())
            h, k, v = layer(h, cos, sin, dec.k[i], dec.v[i], mask)
            dec.k[i][:, :, dec.t] = k[:, :, 0]
            dec.v[i][:, :, dec.t] = v[:, :, 0]
    final.append(h.clone())
    dec.t += 1

starts = list(np.linspace(1, len(ids) - T, 32).astype(int))
test = [17, 63, 130, len(ids) - T - 3]


def cache_at(c, p):  # cache before token p: slots >= p are still empty
    c = c.clone()
    c[:, :, p:] = 0
    return c


def window(i, p):
    cos_q, sin_q, cos_k, sin_k, mask = multi_inputs(dec.cfg, p, T, S)
    return dict(zip(names_in, [cache_at(dec.k[i], p), cache_at(dec.v[i], p), mask,
                               torch.cat([rec[i][p + t] for t in range(T)], dim=1), cos_q, sin_q, cos_k, sin_k]))


def save(d, name, model, xs, onames, shapes, qlists):
    os.makedirs(f"{d}/calib", exist_ok=True)
    for p in starts:
        np.savez(f"{d}/calib/pos{p:03d}.npz", **{k: v.numpy() for k, v in xs(p).items()})
    with torch.no_grad():
        for p in test:
            x = xs(p)
            np.savez(f"{d}/test_in_pos{p:03d}.npz", **{k: v.numpy() for k, v in x.items()})
            np.savez(f"{d}/test_ref_pos{p:03d}.npz", **{k: v.numpy() for k, v in zip(onames, model(*x.values()))})
    x = xs(test[0])
    torch.onnx.export(model, tuple(x.values()), f"{d}/{name}.onnx", input_names=list(x.keys()),
                      output_names=onames, opset_version=17, dynamo=False)
    open(f"{d}/shapes", "w").write(shapes + "\n")
    open(f"{d}/quant_lists", "w").write(qlists + "\n")


g = dec.cfg.nh // dec.cfg.nkv
shapes = (f"[[1,3,{S},64],[1,3,{S},64],[1,3,{T * g},{S + T}],"
          f"[1,{T},576],[1,3,{T * g},64],[1,3,{T * g},64],[1,3,{T},64],[1,3,{T},64]]")
for i in layers:
    dec.layers[i].delta = True
    save(f"{out}/l{i:02d}", f"l{i:02d}", CacheFirst(DecodeLayerT(dec.layers[i], T)), lambda p, i=i: window(i, p),
         names_out, shapes, "--quant_input_list 1,2,3,5,6,7,8 --quant_output_list 2,3")
if sel in ("all", "head"):
    head = Head(dec)
    save(f"{out}/head", "head", head, lambda p: dict(h=torch.cat(final[p:p + T], dim=1)), ["logits"],
         f"[[1,{T},576]]", "")
    dec.emb.half().numpy().tofile(f"{out}/embed.f16")
print(f"T={T}: layers {list(layers)}{' + head' if sel in ('all', 'head') else ''} in {out}/")
