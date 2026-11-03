"""Export one SigLIP block with MatMul linear layers: the baseline for the 1x1 conv rewrite.

Layout: [N, d], tokens as rows, as in the HF model. Every linear layer stays a MatMul
(ONNX Gemm). The rest follows export_block.py, so only the linear layers differ:
LayerNorm by mean, spread, reciprocal sqrt and scale, tanh GELU, softmax attention, no
fc2 equalization. cv18xx has no BF16 broadcast add, so the LayerNorm bias moves into the
bias of the next linear layers, which is exact.

Usage: python export_block_mm.py D:/duosight-eval BLOCK CONV_DIR OUT_DIR
CONV_DIR holds the conv export (calib/*.npz, test_in.npz as [1, d, 1, N]). The same
samples are written to OUT_DIR as [N, d], with OUT_DIR/block.onnx.
"""
import math
import pathlib
import sys

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForImageTextToText

VLM = "HuggingFaceTB/SmolVLM-256M-Instruct"
root, block, conv_dir, out = pathlib.Path(sys.argv[1]), int(sys.argv[2]), pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4])


class Linear(nn.Module):
    def __init__(self, linear, scale=1.0, shift=None):
        super().__init__()
        w = linear.weight.data * scale
        b = linear.bias.data * scale
        if shift is not None:  # a LayerNorm bias that comes before this layer
            b = b + w @ shift
        self.w = nn.Parameter(w.clone(), requires_grad=False)
        self.b = nn.Parameter(b.clone(), requires_grad=False)

    def forward(self, x):
        return nn.functional.linear(x, self.w, self.b)


class LayerNorm(nn.Module):
    """LayerNorm without its bias, which the next linear layers carry."""

    def __init__(self, ln):
        super().__init__()
        d = ln.weight.numel()
        self.avg = nn.Parameter(torch.full((d, 1), 1.0 / d), requires_grad=False)
        # cv18xx broadcasts only the right operand, so the mean spreads by MatMul.
        self.spread = nn.Parameter(torch.ones(1, d), requires_grad=False)
        self.w = nn.Parameter(ln.weight.data.clone(), requires_grad=False)
        self.eps = ln.eps

    def forward(self, x):
        z = x - (x @ self.avg) @ self.spread
        r = torch.reciprocal(torch.sqrt((z * z) @ self.avg + self.eps))
        return z * r * self.w


class Gelu(nn.Module):
    def forward(self, x):
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x * x * x)))


class Attention(nn.Module):
    def forward(self, q, k, v):
        return torch.softmax(q @ k, dim=-1) @ v


class Block(nn.Module):
    def __init__(self, layer, n):
        super().__init__()
        a = layer.self_attn
        self.d, self.h, self.hd, self.n = a.embed_dim, a.num_heads, a.embed_dim // a.num_heads, n
        self.norm1 = LayerNorm(layer.layer_norm1)
        self.norm2 = LayerNorm(layer.layer_norm2)
        b1, b2 = layer.layer_norm1.bias.data, layer.layer_norm2.bias.data
        # 1/sqrt(hd) folded into q
        self.q = Linear(a.q_proj, 1.0 / math.sqrt(self.hd), b1)
        self.k = Linear(a.k_proj, shift=b1)
        self.v = Linear(a.v_proj, shift=b1)
        self.o = Linear(a.out_proj)
        self.fc1 = Linear(layer.mlp.fc1, shift=b2)
        self.fc2 = Linear(layer.mlp.fc2)
        self.gelu = Gelu()
        self.attn = Attention()

    def heads(self, x):
        return x.reshape(1, self.n, self.h, self.hd).permute(0, 2, 1, 3)  # [1, h, n, hd]

    def forward(self, x):
        y = self.norm1(x)
        q, k, v = self.heads(self.q(y)), self.heads(self.k(y)).permute(0, 1, 3, 2), self.heads(self.v(y))
        o = self.attn(q, k, v).permute(0, 2, 1, 3).reshape(self.n, self.d)
        x = x + self.o(o)
        return x + self.fc2(self.gelu(self.fc1(self.norm2(x))))


to_rows = lambda f: np.load(f)["x"][0, :, 0, :].T.copy()  # [1, d, 1, N] -> [N, d]
calib = sorted((conv_dir / "calib").glob("*.npz"))
test = to_rows(conv_dir / "test_in.npz")
N = test.shape[0]

model = AutoModelForImageTextToText.from_pretrained(VLM, dtype=torch.float32, attn_implementation="eager").eval()
layer = model.model.vision_model.encoder.layers[block]
net = Block(layer, N).eval()
with torch.no_grad():
    worst = 1.0
    for f in calib[:6]:
        h = torch.from_numpy(to_rows(f))
        want = layer(h[None], attention_mask=None)
        want = want[0] if isinstance(want, tuple) else want
        c = torch.nn.functional.cosine_similarity(net(h).flatten(), want.flatten(), dim=0).item()
        worst = min(worst, c)
print(f"MatMul block vs HF layer: worst cosine {worst:.6f}", flush=True)
if worst < 0.9999:
    sys.exit("MatMul block does not match the HF layer")

(out / "calib").mkdir(parents=True, exist_ok=True)
for f in calib:
    np.savez(out / "calib" / f.name, x=to_rows(f))
np.savez(out / "test_in.npz", x=test)
torch.onnx.export(net, (torch.from_numpy(test),), str(out / "block.onnx"), input_names=["x"], output_names=["y"],
                  opset_version=17, dynamo=False)
print("N", N, "calib", len(calib), "wrote", out / "block.onnx")
