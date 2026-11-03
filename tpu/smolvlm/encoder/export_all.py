"""Export all 12 SigLIP blocks for one input size, each with its own calibration data.

Usage: python export_all.py D:/duosight-eval SIZE OUT_DIR
Writes OUT_DIR/lNN/{block.onnx,calib/*.npz,test_in.npz}.
"""
import importlib.util
import pathlib
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText

root, size, out = pathlib.Path(sys.argv[1]), int(sys.argv[2]), pathlib.Path(sys.argv[3])
here = pathlib.Path(__file__).parent
source = (here / "export_block.py").read_text(encoding="utf-8")
ns = {}
exec(source[source.index("def conv1x1"):source.index("model = AutoModelForImageTextToText")],
     {**globals(), "nn": torch.nn, "math": __import__("math"), "torch": torch}, ns)
CALIB, N = 32, (size // 16) ** 2
Block = ns["Block"]
globals().update(ns)
Block.__init__.__globals__.update(ns)

model = AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct", dtype=torch.float32, attn_implementation="eager").eval()
vision = model.model.vision_model
layers = vision.encoder.layers
captured = [[] for _ in layers]
for i, layer in enumerate(layers):
    layer.register_forward_pre_hook(lambda m, a, i=i: captured[i].append(a[0].detach().clone()))
with torch.no_grad():
    for path in sorted((root / "images").glob("*.jpg"))[:CALIB + 2]:
        img = Image.open(path).convert("RGB")
        s = size / max(img.size)
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BICUBIC)
        canvas = Image.new("RGB", (size, size), (127, 127, 127))
        canvas.paste(img, (0, 0))
        x = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8).float().view(size, size, 3).permute(2, 0, 1) / 255
        vision(pixel_values=((x - 0.5) / 0.5)[None], patch_attention_mask=torch.ones(1, size // 16, size // 16, dtype=torch.bool))
print("captured", len(captured[0]), "images", flush=True)

to_nchw = lambda h: h.transpose(1, 2)[:, :, None, :].numpy()
for i, layer in enumerate(layers):
    for hook in list(layer._forward_pre_hooks):
        del layer._forward_pre_hooks[hook]
    net = Block(layer, N).eval().cuda()
    net.equalize([h.transpose(1, 2)[:, :, None, :].cuda() for h in captured[i][:CALIB]])
    net = net.cpu()
    with torch.no_grad():
        h = captured[i][0]
        want = layer(h, attention_mask=None)
        want = want[0] if isinstance(want, tuple) else want
        got = net(h.transpose(1, 2)[:, :, None, :])[0, :, 0, :].T
        c = torch.nn.functional.cosine_similarity(got.flatten(), want.flatten(), dim=0).item()
    if c < 0.9999:
        sys.exit(f"block {i}: rewrite cosine {c}")
    d = out / f"l{i:02d}"
    (d / "calib").mkdir(parents=True, exist_ok=True)
    for j, h in enumerate(captured[i][:CALIB]):
        np.savez(d / "calib" / f"s{j:02d}.npz", x=to_nchw(h))
    np.savez(d / "test_in.npz", x=to_nchw(captured[i][-1]))
    sample = torch.from_numpy(to_nchw(captured[i][-1]))
    torch.onnx.export(net, (sample,), str(d / "block.onnx"), input_names=["x"], output_names=["y"], opset_version=17, dynamo=False)
    print(f"l{i:02d} ok, rewrite cosine {c:.6f}", flush=True)
