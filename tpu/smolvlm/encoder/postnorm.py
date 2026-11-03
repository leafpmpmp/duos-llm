"""Compare board features with float after the final LayerNorm, using numpy only.

The decoder sees these values, and raw cosine does not measure them. torch with CUDA
takes about 7 GB of commit on this PC, and this script needs under 2 GB.

Usage: python postnorm.py NAME=FILE [NAME=FILE ...]
Float features come from D:/duosight-eval/tpu/bin_float/out512.bin.
"""
import glob
import json
import sys

import numpy as np

N, D, IMAGES = 1024, 768, 150
files = glob.glob("D:/duosight-eval/hf/hub/models--HuggingFaceTB--SmolVLM-256M-Instruct/snapshots/*/model.safetensors")


def tensor(name):
    """Read one BF16 tensor from the file. numpy has no bfloat16, and BF16 is the top half of fp32."""
    with open(files[0], "rb") as f:
        size = int.from_bytes(f.read(8), "little")
        entry = json.loads(f.read(size))[name]
        start, end = entry["data_offsets"]
        f.seek(8 + size + start)
        raw = f.read(end - start)
    assert entry["dtype"] == "BF16", entry["dtype"]
    return np.frombuffer((np.frombuffer(raw, "<u2").astype(np.uint32) << 16).tobytes(), "<f4")


weight = tensor("model.vision_model.post_layernorm.weight")
bias = tensor("model.vision_model.post_layernorm.bias")


def load(path):
    return np.memmap(path, dtype=np.float32, mode="r", shape=(IMAGES, D, N))


def norm(x):
    x = np.asarray(x).T  # [N, D]
    m = x.mean(1, keepdims=True)
    v = ((x - m) ** 2).mean(1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-6) * weight + bias


def token_cos(a, b):
    return (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)


ref = load("D:/duosight-eval/tpu/bin_float/out512.bin")
print(f"{'build':10} {'after LN: mean':>15} {'p5':>7} {'min':>7} {'image mean':>11}  raw image cos")
for spec in sys.argv[1:]:
    name, path = spec.split("=", 1)
    y = load(path)
    tok, img, raw = [], [], []
    for i in range(IMAGES):
        a, b = norm(y[i]), norm(ref[i])
        tok.append(token_cos(a, b))
        img.append((a * b).sum() / np.linalg.norm(a) / np.linalg.norm(b))
        p, q = np.asarray(y[i], np.float64).ravel(), np.asarray(ref[i], np.float64).ravel()
        raw.append(p @ q / np.linalg.norm(p) / np.linalg.norm(q))
    tok = np.concatenate(tok)
    print(f"{name:10} {tok.mean():15.4f} {np.percentile(tok, 5):7.4f} {tok.min():7.4f} {np.mean(img):11.4f}  {np.mean(raw):.4f}")
