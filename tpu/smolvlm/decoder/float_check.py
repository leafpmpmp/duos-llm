"""Answer the first prompts in float, from the same prompt file and prefix as the board.

A match with the float decoder's answers shows that the prompt file, the prefix and the
layer model are right, so any difference on the board comes from the TPU models.

Usage: python float_check.py ROOT TEXT_DIR PROMPTS.bin PREFIX.kv N
"""
import json
import pathlib
import sys

import numpy as np
import torch
from tokenizers import Tokenizer

from vlm_common import EOS, IMAGE_ID, MAX_NEW

root, text, path, prefix, count = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
sys.path.insert(0, str(root / "duos-llm/tpu/smollm"))
from model import Decoder, load_weights  # noqa: E402

torch.set_num_threads(4)
here = pathlib.Path(__file__).resolve().parent
rows = json.load(open(here.parent / "encoder" / "answers-cpu-final.json"))
tok = Tokenizer.from_file(f"{text}/tokenizer.json")
dec = Decoder(text, 512)
lm = load_weights(text)["lm_head.weight"]
cfg = dec.cfg

raw = open(prefix, "rb").read()
k0, v0 = np.frombuffer(raw[8:], dtype=np.float32).reshape(2, cfg.layers, cfg.nkv, 1, cfg.hd)


def step(h):
    _, cos, sin, mask = dec.inputs(0)
    with torch.no_grad():
        for i, layer in enumerate(dec.layers):
            h, k, v = layer(h, cos, sin, dec.k[i], dec.v[i], mask)
            dec.k[i][:, :, dec.t] = k[:, :, 0]
            dec.v[i][:, :, dec.t] = v[:, :, 0]
        dec.t += 1
        return (dec.norm(h) @ lm.t()).reshape(-1)


same = 0
with open(path, "rb") as f:
    for r in rows[:count]:
        n = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        ids = np.fromfile(f, dtype=np.int32, count=n)
        image = torch.from_numpy(np.fromfile(f, dtype=np.float32, count=64 * 576).reshape(64, 576).copy())
        dec.reset()
        dec.k[:, 0, :, 0] = torch.from_numpy(k0[:, :, 0].copy())
        dec.v[:, 0, :, 0] = torch.from_numpy(v0[:, :, 0].copy())
        dec.t = 1
        row = 0
        for t in ids[1:]:
            if t == IMAGE_ID:
                h = image[row].reshape(1, 1, -1)
                row += 1
            else:
                h = dec.emb[int(t)].reshape(1, 1, -1)
            x = step(h)
        out = []
        while len(out) < MAX_NEW:
            t = int(x.argmax())
            if t == EOS:
                break
            out.append(t)
            x = step(dec.emb[t].reshape(1, 1, -1))
        text_out = tok.decode(out, skip_special_tokens=True).strip()
        same += text_out == r["out"]["tpu512"]
        print(repr(text_out), "| want", repr(r["out"]["tpu512"]), flush=True)
print(f"same as the float decoder: {same} of {count}")
