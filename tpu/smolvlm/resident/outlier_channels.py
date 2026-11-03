"""How many fc1 channels of a SigLIP block carry the large values that INT8 clips.

Runs the final build's block (gains from its ONNX) on its calibration inputs, and prints
how many fc1 output channels and GELU output channels pass each level.

Usage: python outlier_channels.py D:/duosight-eval OLD_DIR LAYER...
"""
import pathlib
import sys

import numpy as np
import onnx
import onnx.numpy_helper
import torch
from transformers import AutoModelForImageTextToText

root, old = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
here = pathlib.Path(__file__).resolve().parent
source = (here.parent / "encoder" / "export_block.py").read_text(encoding="utf-8")
ns = {}
exec(source[source.index("def conv1x1"):source.index("model = AutoModelForImageTextToText")],
     {**globals(), "nn": torch.nn, "math": __import__("math"), "torch": torch}, ns)
Block, gelu = ns["Block"], ns["gelu_tanh"]
Block.__init__.__globals__.update(ns)
model = AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct", dtype=torch.float32, attn_implementation="eager").eval()
for i in map(int, sys.argv[3:]):
    src = old / f"l{i:02d}"
    xs = [torch.from_numpy(np.load(f)["x"]) for f in sorted((src / "calib").glob("*.npz"))]
    net = Block(model.model.vision_model.encoder.layers[i], xs[0].shape[-1]).eval()
    out = []
    net.fc1.register_forward_hook(lambda m, a, o: out.append(o.detach().abs().amax(dim=(0, 2, 3))))
    with torch.no_grad():
        for x in xs:
            net(x)
    m = torch.stack(out).amax(dim=0)
    g = gelu(m)  # GELU is monotonic for positive inputs; negative ones stay small
    print(f"l{i:02d}: fc1 max {m.max():.1f}, median {m.median():.2f}; channels with fc1 over 4/8/16: "
          f"{int((m > 4).sum())}/{int((m > 8).sum())}/{int((m > 16).sum())}; GELU over 4/8: {int((g > 4).sum())}/{int((g > 8).sum())}")
    top = torch.argsort(m, descending=True)[:12]
    print("  largest:", [f"{int(c)}:{m[c]:.1f}" for c in top])
