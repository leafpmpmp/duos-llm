"""Export decoder layers for one token per call whose down projection runs on PAD rows.

TPU-MLIR compresses the down projection weight in tiles, and the tiles depend on the row
count. With PAD rows here, as in the PAD-token layers, the weight bytes are the same, so
model_tool --combine puts both programs over one weight section (combine_t1.sh).

FROM: an export_vlm.py build with VLM_T=1 (only its test inputs are used).
CALI: the compiled PAD-token build (lNN/cali_table and lNN/lNN.mlir). The table gets two
lines for the new tensors: the tiled input takes the threshold of the product it copies,
the slice that of the projection.

Usage: python export_pad.py TEXT_DIR FROM CALI OUT [PAD [lNN...]]
"""
import pathlib
import re
import shutil
import sys

import numpy as np
import torch
from torch import nn

text, src, cali, out = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4])
PAD = int(sys.argv[5]) if len(sys.argv) > 5 else 16
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tpu/smollm"))
from model import Decoder, DecodeLayerT  # noqa: E402

NAMES_IN = ["k_cache", "v_cache", "mask", "h", "cos_q", "sin_q", "cos_k", "sin_k"]
NAMES_OUT = ["dh", "k_new", "v_new"]


class PadDown(DecodeLayerT):
    def forward(self, k_cache, v_cache, mask, h, cos_q, sin_q, cos_k, sin_k):
        c, l, T = self.l.cfg, self.l, self.T
        g = c.nh // c.nkv
        x = l.norm1(h)
        q = l.lin(x, l.wq).reshape(1, T, c.nkv, g * c.hd).permute(0, 2, 1, 3).reshape(1, c.nkv, T * g, c.hd)
        k = l.lin(x, l.wk).reshape(1, T, c.nkv, c.hd).permute(0, 2, 1, 3)
        v = l.lin(x, l.wv).reshape(1, T, c.nkv, c.hd).permute(0, 2, 1, 3)
        q = q * cos_q + (q @ l.rot) * sin_q
        k = k * cos_k + (k @ l.rot) * sin_k
        S = k_cache.shape[2]
        s = torch.cat([q @ k_cache.transpose(2, 3), q @ k.transpose(2, 3)], dim=3)
        att = torch.softmax(s + mask, dim=-1)
        o = att[..., :S] @ v_cache + att[..., S:] @ v
        o = o.reshape(1, c.nkv, T, g * c.hd).permute(0, 2, 1, 3).reshape(1, T, c.d)
        o = l.lin(o, l.wo)
        x = l.norm2(h + o)
        a = nn.functional.silu(l.lin(x, l.wg)) * l.lin(x, l.wu)
        m = l.lin(a.expand(1, PAD, a.shape[2]), l.wd)[:, :T]
        return o + m, k, v


class Wrap(nn.Module):  # op names under /layer/, as in export_vlm.py
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, *x):
        return self.layer(*x)


def extend_cali(src, dst):
    """The PAD-token table, plus lines for the tiled input and the slice."""
    lines = open(src).read().splitlines()
    th = {l.split()[0]: l.split(None, 1)[1] for l in lines if l and not l.startswith("#")}
    down = "/layer/MatMul_12_output_0_MatMul"
    mlir = open(src.parent / (src.parent.name + ".mlir")).read()
    loc = dict(re.findall(r'^#(loc\d+) = loc\("([^"]+)"\)', mlir, re.M))
    var = {}
    for m in re.finditer(r'^\s*(%\d+) = "top\.(\w+)"\(([^)]*)\).*?loc\(#(loc\d+)\)\s*$', mlir, re.M):
        var[m.group(1)] = (m.group(2), m.group(3), loc[m.group(4)])
    a = next(v for v in var.values() if v[2] == down)[1].split(",")[0].strip()
    lines += [f"/layer/Expand_output_0_Expand {th[var[a][2]]}", f"/layer/Slice_2_output_0_Slice {th[down]}"]
    open(dst, "w").write("\n".join(lines) + "\n")


dec = Decoder(text, 128)
names = sys.argv[6:] or [f"l{i:02d}" for i in range(len(dec.layers))]
for name in names:
    i = int(name[1:])
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    for f in src.joinpath(name).iterdir():
        if f.name.startswith("test_") or f.name in ("shapes", "quant_lists"):
            shutil.copy(f, d / f.name)
    x = np.load(sorted(d.glob("test_in_pos*.npz"))[0])
    args = tuple(torch.from_numpy(x[k]) for k in NAMES_IN)
    model = Wrap(PadDown(dec.layers[i], args[3].shape[1])).eval()
    with torch.no_grad():
        for t in sorted(d.glob("test_in_pos*.npz")):  # same outputs as the plain layer
            a = np.load(t)
            y = model(*(torch.from_numpy(a[k]) for k in NAMES_IN))
            ref = np.load(str(t).replace("test_in_", "test_ref_"))
            err = max(np.abs(y[j].numpy() - ref[n]).max() for j, n in enumerate(NAMES_OUT))
            assert err < 1e-3, (name, t.name, err)
    torch.onnx.export(model, args, str(d / f"{name}.onnx"), input_names=NAMES_IN, output_names=NAMES_OUT,
                      opset_version=17, dynamo=False)
    extend_cali(cali / name / "cali_table", d / "cali_table")
    print(name, "exported", flush=True)
