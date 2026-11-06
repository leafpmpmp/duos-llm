"""Export the equalized SigLIP blocks with attention in groups of heads.

Full attention over 12 heads and 1024 tokens keeps two 24 MiB BF16 tensors, the scores
and the softmax, and they set the 51 MiB shared buffer of every block. Groups of heads
keep only part of them alive at a time. The rest is the final build: the fc2 gains come
from its ONNX files, so its calibration and quantize tables still apply.

Usage: python export_grouped.py D:/duosight-eval HEADS_PER_GROUP OLD_DIR OUT_DIR [LAYER...]
OLD_DIR/lNN holds the final build's block.onnx, calib/ and test_in.npz.
"""
import pathlib
import shutil
import sys

import numpy as np
import onnx
import onnx.numpy_helper
import torch
from transformers import AutoModelForImageTextToText

root, group, old, out = pathlib.Path(sys.argv[1]), int(sys.argv[2]), pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4])
layers = [int(x) for x in sys.argv[5:]] or list(range(12))
here = pathlib.Path(__file__).parent
source = (here / "export_block.py").read_text(encoding="utf-8")
ns = {}
exec(source[source.index("def conv1x1"):source.index("model = AutoModelForImageTextToText")],
     {**globals(), "nn": torch.nn, "math": __import__("math"), "torch": torch}, ns)
Block = ns["Block"]
Block.__init__.__globals__.update(ns)

model = AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct", dtype=torch.float32, attn_implementation="eager").eval()
for i in layers:
    src, d = old / f"l{i:02d}", out / f"l{i:02d}"
    init = {t.name: onnx.numpy_helper.to_array(t) for t in onnx.load(str(src / "block.onnx")).graph.initializer}
    test = torch.from_numpy(np.load(src / "test_in.npz")["x"])
    layer = model.model.vision_model.encoder.layers[i]
    net = Block(layer, test.shape[-1]).eval()
    # Same steps as Block.equalize, with the gains of the final build.
    s = torch.from_numpy(init["gain.weight"]).reshape(-1)
    net.fc2.weight.data /= s[:, None, None, None]
    net.fc2.bias.data /= s
    net.gain = torch.nn.Conv2d(net.d, net.d, 1, groups=net.d, bias=False)
    net.gain.weight.data = s[:, None, None, None].clone()
    with torch.no_grad():
        ref = net(test)
        net.group = group
        got = net(test)
        want = layer(test[0, :, 0, :].T[None], attention_mask=None)
        want = want[0] if isinstance(want, tuple) else want
    same = (got - ref).abs().max().item()
    c = torch.nn.functional.cosine_similarity(got[0, :, 0, :].T.flatten(), want.flatten(), dim=0).item()
    for name in ("fc2.weight", "fc2.bias"):
        assert np.allclose(net.state_dict()[name].numpy(), init[name], rtol=0, atol=1e-6), name
    if c < 0.9999:
        sys.exit(f"l{i:02d}: cosine to the HF layer {c}")
    (d / "calib").mkdir(parents=True, exist_ok=True)
    for f in (src / "calib").glob("*.npz"):
        shutil.copy(f, d / "calib" / f.name)
    shutil.copy(src / "test_in.npz", d / "test_in.npz")
    torch.onnx.export(net, (test,), str(d / "block.onnx"), input_names=["x"], output_names=["y"], opset_version=17, dynamo=False)
    print(f"l{i:02d}: groups of {group} heads, max change {same:.2e}, cosine to HF {c:.6f}", flush=True)
