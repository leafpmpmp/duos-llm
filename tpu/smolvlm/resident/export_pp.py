"""Export patch embedding and the final LayerNorm with the connector, for the TPU.

On the CPU these took 1.28 s and 2.27 s. Both are convolutions:
- patch: the canvas in pixel values 0 to 255, as [1, 3, 512, 512]. A 16x16 conv with
  stride 16 holds the patch weights with the pixel scaling folded in, then the position
  embedding is added. Output [1, 768, 1, 1024], the encoder's input layout.
- post: the encoder output [1, 768, 1, 1024]. LayerNorm as in export_block.py, then the
  pixel shuffle and the projection as one 4x4 conv with stride 4. Output [1, 576, 1, 64]:
  column r is image row r.

Usage: python export_pp.py W_DIR OUT_DIR
W_DIR holds the board's float32 weights and check files (camera/make_weights.py).
"""
import pathlib
import sys

import numpy as np
import torch
from torch import nn

w, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
D, G, P = 768, 32, 16
f32 = lambda name, *shape: torch.from_numpy(np.fromfile(w / name, dtype=np.float32).reshape(shape))


class Patch(nn.Module):
    def __init__(self):
        super().__init__()
        weight, bias = f32("patch.f32", D, 3, P, P), f32("patch_bias.f32", D)
        self.conv = nn.Conv2d(3, D, P, stride=P)
        # (x / 255 - 0.5) / 0.5 = x / 127.5 - 1
        self.conv.weight.data = weight / 127.5
        self.conv.bias.data = bias - weight.sum(dim=(1, 2, 3))
        self.pos = nn.Parameter(f32("pos.f32", G * G, D).T.reshape(1, D, G, G).contiguous(), requires_grad=False)

    def forward(self, x):
        return (self.conv(x) + self.pos).reshape(1, D, 1, G * G)


class LayerNorm(nn.Module):
    """export_block.LayerNorm: means by 1x1 conv, scale and offset in a depthwise conv."""

    def __init__(self, weight, bias, eps):
        super().__init__()
        self.avg = nn.Conv2d(D, 1, 1, bias=False)
        self.avg.weight.data.fill_(1.0 / D)
        self.spread = nn.Conv2d(1, D, 1, bias=False)
        self.spread.weight.data.fill_(1.0)
        self.scale = nn.Conv2d(D, D, 1, groups=D)
        self.scale.weight.data = weight[:, None, None, None].clone()
        self.scale.bias.data = bias.clone()
        self.eps = eps

    def forward(self, x):
        z = x - self.spread(self.avg(x))
        r = torch.reciprocal(torch.sqrt(self.avg(z * z) + self.eps))
        return self.scale(z * r)


class Post(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = LayerNorm(f32("ln_w.f32", D), f32("ln_b.f32", D), 1e-6)
        proj = f32("proj.f32", 576, 4 * 4 * D)
        self.proj = nn.Conv2d(D, 576, 4, stride=4, bias=False)
        # feature (y % 4) * 3072 + (x % 4) * 768 + c of the pixel shuffle
        self.proj.weight.data = proj.reshape(576, 4, 4, D).permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        h = self.norm(x).reshape(1, D, G, G)
        return self.proj(h).reshape(1, 576, 1, 64)


cos = lambda a, b: float((a.ravel() @ b.ravel()) / np.linalg.norm(a) / np.linalg.norm(b))
canvas = np.fromfile(w / "check.rgb", dtype=np.uint8).reshape(512, 512, 3)
x_patch = torch.from_numpy(canvas.transpose(2, 0, 1)[None].astype(np.float32))
x_post = f32("check_last.bin", 1, D, 1, G * G)
for name, net, x, want in (("patch", Patch(), x_patch, np.fromfile(w / "check_in.bin", dtype=np.float32)),
                           ("post", Post(), x_post, np.fromfile(w / "check_rows.bin", dtype=np.float32).reshape(64, 576).T)):
    net.eval()
    with torch.no_grad():
        got = net(x).numpy()
    print(f"{name}: cosine {cos(got, want):.7f}, max difference {np.abs(got.ravel() - want.ravel()).max():.2e}", flush=True)
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "test_in.npz", x=x.numpy())
    torch.onnx.export(net, (x,), str(d / "model.onnx"), input_names=["x"], output_names=["y"], opset_version=17, dynamo=False)
print("wrote", out)
