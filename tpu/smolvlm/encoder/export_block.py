"""Export one SigLIP block of SmolVLM-256M to ONNX in a form the cv181x TPU likes.

Layout: [1, d, 1, N], tokens as an image. Every linear layer becomes a 1x1 conv,
because cv18xx conv runs near 100 GMAC/s while MatMul runs near 8 GMAC/s (duos-detect
and duos-llm rates). LayerNorm avoids ops without a cv18xx lowering and BF16 broadcast
adds: the scale and offset live in one depthwise conv.

Usage: python export_block.py D:/duosight-eval SIZE BLOCK OUT_DIR
Checks the rewrite against the HF layer, then writes OUT_DIR/block.onnx, calib/*.npz
and test_in.npz.
"""
import math
import pathlib
import sys

import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers import AutoModelForImageTextToText

VLM = "HuggingFaceTB/SmolVLM-256M-Instruct"
root, size, block, out = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), pathlib.Path(sys.argv[4])
N = (size // 16) ** 2
CALIB = 32


def conv1x1(linear, scale=1.0):
    c = nn.Conv2d(linear.in_features, linear.out_features, 1)
    c.weight.data = (linear.weight.data * scale)[:, :, None, None].clone()
    c.bias.data = (linear.bias.data * scale).clone()
    return c


class LayerNorm(nn.Module):
    def __init__(self, ln):
        super().__init__()
        d = ln.weight.numel()
        self.avg = nn.Conv2d(d, 1, 1, bias=False)
        self.avg.weight.data.fill_(1.0 / d)
        # cv18xx broadcasts only the right operand, so the mean spreads by conv.
        self.spread = nn.Conv2d(1, d, 1, bias=False)
        self.spread.weight.data.fill_(1.0)
        self.scale = nn.Conv2d(d, d, 1, groups=d)
        self.scale.weight.data = ln.weight.data[:, None, None, None].clone()
        self.scale.bias.data = ln.bias.data.clone()
        self.eps = ln.eps

    def forward(self, x):
        z = x - self.spread(self.avg(x))
        r = torch.reciprocal(torch.sqrt(self.avg(z * z) + self.eps))
        return self.scale(z * r)


def gelu_tanh(x):
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x * x * x)))


class Block(nn.Module):
    def __init__(self, layer, n):
        super().__init__()
        a = layer.self_attn
        self.d, self.h, self.hd, self.n = a.embed_dim, a.num_heads, a.embed_dim // a.num_heads, n
        self.norm1 = LayerNorm(layer.layer_norm1)
        self.norm2 = LayerNorm(layer.layer_norm2)
        # 1/sqrt(hd) folded into q
        self.q = conv1x1(a.q_proj, 1.0 / math.sqrt(self.hd))
        self.k = conv1x1(a.k_proj)
        self.v = conv1x1(a.v_proj)
        self.o = conv1x1(a.out_proj)
        self.fc1 = conv1x1(layer.mlp.fc1)
        self.fc2 = conv1x1(layer.mlp.fc2)
        self.gain = None
        self.group = self.h  # heads per attention group, see export_grouped.py

    def equalize(self, samples):
        """Give every fc2 output channel a range near 1, and restore it in a depthwise conv.

        One INT8 scale covers the whole tensor, and a few channels reach 50 or more, so
        ordinary channels keep only a few levels. BF16 fc2 fixes that, but costs 370 ms.
        """
        out = []
        hook = self.fc2.register_forward_hook(lambda m, a, o: out.append(o.detach().abs().amax(dim=(0, 2, 3))))
        with torch.no_grad():
            for h in samples:
                self(h)
        hook.remove()
        s = torch.stack(out).amax(dim=0).clamp_min(1e-3)
        self.fc2.weight.data /= s[:, None, None, None]
        self.fc2.bias.data /= s
        self.gain = nn.Conv2d(self.d, self.d, 1, groups=self.d, bias=False)
        self.gain.weight.data = s[:, None, None, None].clone()

    def heads(self, x):
        return x.reshape(1, self.h, self.hd, self.n).permute(0, 1, 3, 2)  # [1, h, n, hd]

    def forward(self, x):
        y = self.norm1(x)
        q, k, v = self.heads(self.q(y)), self.heads(self.k(y)), self.heads(self.v(y))
        # One transpose of k before any grouping keeps the op names of the ungrouped graph.
        k = k.permute(0, 1, 3, 2)
        attend = lambda q, k, v: torch.softmax(q @ k, dim=-1) @ v
        g = self.group
        if g < self.h:
            o = torch.cat([attend(q[:, i:i + g], k[:, i:i + g], v[:, i:i + g]) for i in range(0, self.h, g)], dim=1)
        else:
            o = attend(q, k, v)
        o = o.permute(0, 1, 3, 2).reshape(1, self.d, 1, self.n)
        x = x + self.o(o)
        m = self.fc2(gelu_tanh(self.fc1(self.norm2(x))))
        return x + (self.gain(m) if self.gain is not None else m)


model = AutoModelForImageTextToText.from_pretrained(VLM, dtype=torch.float32, attn_implementation="eager").eval()
vision = model.model.vision_model
layer = vision.encoder.layers[block]

captured = []
hook = layer.register_forward_pre_hook(lambda m, a, k=None: captured.append((a[0] if a else k["hidden_states"]).detach().clone()),
                                       with_kwargs=False)
images = sorted((root / "images").glob("*.jpg"))[:CALIB + 2]
with torch.no_grad():
    for path in images:
        img = Image.open(path).convert("RGB")
        s = size / max(img.size)
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BICUBIC)
        canvas = Image.new("RGB", (size, size), (127, 127, 127))
        canvas.paste(img, (0, 0))
        x = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8).float().view(size, size, 3).permute(2, 0, 1) / 255
        vision(pixel_values=((x - 0.5) / 0.5)[None], patch_attention_mask=torch.ones(1, size // 16, size // 16, dtype=torch.bool))
hook.remove()
print("captured", len(captured), captured[0].shape, flush=True)

net = Block(layer, N).eval()
with torch.no_grad():
    worst = 1.0
    for h in captured[:6]:
        want = layer(h, attention_mask=None)
        want = want[0] if isinstance(want, tuple) else want
        got = net(h.transpose(1, 2)[:, :, None, :])[0, :, 0, :].T
        c = torch.nn.functional.cosine_similarity(got.flatten(), want.flatten(), dim=0).item()
        worst = min(worst, c)
print(f"rewrite vs HF layer: worst cosine {worst:.6f}", flush=True)
if worst < 0.9999:
    sys.exit("rewrite does not match the HF layer")

(out / "calib").mkdir(parents=True, exist_ok=True)
to_nchw = lambda h: h.transpose(1, 2)[:, :, None, :].numpy()
for i, h in enumerate(captured[:CALIB]):
    np.savez(out / "calib" / f"s{i:02d}.npz", x=to_nchw(h))
np.savez(out / "test_in.npz", x=to_nchw(captured[-1]))
sample = torch.from_numpy(to_nchw(captured[-1]))
torch.onnx.export(net, (sample,), str(out / "block.onnx"), input_names=["x"], output_names=["y"], opset_version=17, dynamo=False)
print("macs per block (GMAC):", round((768 * 768 * 3 + 768 * 768 + 768 * 3072 * 2) * N / 1e9 + 12 * N * N * 64 * 2 / 1e9, 2))
print("wrote", out / "block.onnx")
