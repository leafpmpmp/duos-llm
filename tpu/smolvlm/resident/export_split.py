"""Export a SigLIP block with its few large MLP channels in a branch of their own.

In block 5, 15 of the 3072 fc1 channels pass 8 (the largest 30.8), and one INT8 scale for
all of them leaves the others few levels. So the final build ran its GELU and fc2 in BF16,
at about 360 ms more per image. Here fc1 and fc2 split by channel: the main branch, INT8
like the other blocks, and a branch for the large channels, run in BF16. The sum is the
same MLP. The block also has the grouped attention and the final build's fc2 gains, as
in export_grouped.py.

Usage: python export_split.py D:/duosight-eval LEVEL OLD_DIR OUT_DIR LAYER
  LEVEL: fc1 channels whose largest output on the calibration data passes this go to the
         branch.
"""
import pathlib
import shutil
import sys

import numpy as np
import onnx
import onnx.numpy_helper
import torch
from torch import nn
from transformers import AutoModelForImageTextToText

root, level, old, out, layer_i = (pathlib.Path(sys.argv[1]), float(sys.argv[2]), pathlib.Path(sys.argv[3]),
                                  pathlib.Path(sys.argv[4]), int(sys.argv[5]))
here = pathlib.Path(__file__).resolve().parent
source = (here.parent / "encoder" / "export_block.py").read_text(encoding="utf-8")
ns = {}
exec(source[source.index("def conv1x1"):source.index("model = AutoModelForImageTextToText")],
     {**globals(), "nn": nn, "math": __import__("math"), "torch": torch}, ns)
Block, gelu_tanh = ns["Block"], ns["gelu_tanh"]
Block.__init__.__globals__.update(ns)


class Gelu(nn.Module):
    def forward(self, x):
        return gelu_tanh(x)


def pick(conv, rows=None, cols=None, bias=True):
    w = conv.weight.data
    w = w[rows] if rows is not None else w
    w = w[:, cols] if cols is not None else w
    c = nn.Conv2d(w.shape[1], w.shape[0], 1, bias=bias)
    c.weight.data = w.clone()
    if bias:
        c.bias.data = (conv.bias.data[rows] if rows is not None else conv.bias.data).clone()
    return c


class SplitMlp(nn.Module):
    """fc2(gelu(fc1(x))) as two branches by fc1 channel: main and large."""

    def __init__(self, fc1, fc2, large):
        super().__init__()
        main = [c for c in range(fc1.out_channels) if c not in set(large)]
        self.fc1_main, self.fc1_big = pick(fc1, rows=main), pick(fc1, rows=large)
        self.gelu_main, self.gelu_big = Gelu(), Gelu()
        self.fc2_main, self.fc2_big = pick(fc2, cols=main), pick(fc2, cols=large, bias=False)

    def forward(self, x):
        return self.fc2_main(self.gelu_main(self.fc1_main(x))) + self.fc2_big(self.gelu_big(self.fc1_big(x)))


class SplitBlock(nn.Module):
    def __init__(self, net, large):
        super().__init__()
        self.net = net
        self.mlp = SplitMlp(net.fc1, net.fc2, large)

    def forward(self, x):
        n = self.net
        y = n.norm1(x)
        q, k, v = n.heads(n.q(y)), n.heads(n.k(y)), n.heads(n.v(y))
        k = k.permute(0, 1, 3, 2)
        o = torch.cat([torch.softmax(q[:, i:i + 1] @ k[:, i:i + 1], dim=-1) @ v[:, i:i + 1] for i in range(n.h)], dim=1)
        x = x + n.o(o.permute(0, 1, 3, 2).reshape(1, n.d, 1, n.n))
        return x + n.gain(self.mlp(n.norm2(x)))


model = AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct", dtype=torch.float32, attn_implementation="eager").eval()
src, d = old / f"l{layer_i:02d}", out / f"l{layer_i:02d}"
init = {t.name: onnx.numpy_helper.to_array(t) for t in onnx.load(str(src / "block.onnx")).graph.initializer}
xs = [torch.from_numpy(np.load(f)["x"]) for f in sorted((src / "calib").glob("*.npz"))]
test = torch.from_numpy(np.load(src / "test_in.npz")["x"])
layer = model.model.vision_model.encoder.layers[layer_i]
net = Block(layer, test.shape[-1]).eval()
s = torch.from_numpy(init["gain.weight"]).reshape(-1)
net.fc2.weight.data /= s[:, None, None, None]
net.fc2.bias.data /= s
net.gain = nn.Conv2d(net.d, net.d, 1, groups=net.d, bias=False)
net.gain.weight.data = s[:, None, None, None].clone()

peak = []
hook = net.fc1.register_forward_hook(lambda m, a, o: peak.append(o.detach().abs().amax(dim=(0, 2, 3))))
with torch.no_grad():
    for x in xs:
        net(x)
hook.remove()
peak = torch.stack(peak).amax(dim=0)
large = sorted(int(c) for c in torch.nonzero(peak > level).flatten())
split = SplitBlock(net, large).eval()
with torch.no_grad():
    ref, got = net(test), split(test)
    want = layer(test[0, :, 0, :].T[None], attention_mask=None)
    want = want[0] if isinstance(want, tuple) else want
c = torch.nn.functional.cosine_similarity(got[0, :, 0, :].T.flatten(), want.flatten(), dim=0).item()
print(f"l{layer_i:02d}: {len(large)} channels over {level} in their own branch, main branch peak "
      f"{peak[[i for i in range(len(peak)) if i not in set(large)]].max():.1f}; "
      f"max change {(got - ref).abs().max().item():.2e}, cosine to HF {c:.6f}", flush=True)
if c < 0.9999:
    sys.exit("split block does not match the HF layer")
(d / "calib").mkdir(parents=True, exist_ok=True)
for f in (src / "calib").glob("*.npz"):
    shutil.copy(f, d / "calib" / f.name)
shutil.copy(src / "test_in.npz", d / "test_in.npz")
torch.onnx.export(split, (test,), str(d / "block.onnx"), input_names=["x"], output_names=["y"], opset_version=17, dynamo=False)
print("wrote", d / "block.onnx")
