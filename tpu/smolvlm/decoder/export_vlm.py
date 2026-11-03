"""Export SmolVLM's decoder for the TPU: 30 layers and the head, 4 tokens per call.

This follows export_t.py in duos-llm, with two changes. The float run uses real prompts
with image rows, because every layer sees those rows on the board. The head uses lm_head,
which SmolVLM does not tie to the embedding table.

Calibration uses 32 questions from quality/questions.json, which share no image
with the 150 scored questions. Each gives one 4-token window, spread over the image
rows, the question and the answer.

Usage: python export_vlm.py ROOT TEXT_DIR OUT_DIR
Writes OUT_DIR/lNN and OUT_DIR/head for compile_vlm.sh, plus embed.f16 and prefix-imstart.kv.
  VLM_S, VLM_T: cache slots and tokens per call (default 512 and 4)
  VLM_MIX2=all: calibration data for every layer, not only the q4 MIX2 layers
  VLM_HEAD_T: rows of the head (default T). With 1, the head takes the last row only.
  VLM_TEXT_PREFIX=1: the float prefix covers all the text before the first image row, not
    only position 0. Those tokens are the same in every prompt, and the first of them carry
    activations that INT8 cannot hold, so calibration windows start after them.
"""
import json
import os
import pathlib
import random
import sys

import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer
from torch import nn
from transformers import AutoModelForImageTextToText

from vlm_common import ASK, IMAGE_ID, SNAPSHOT, Connector, prompt

root, text, out = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
sys.path.insert(0, str(root / "duos-llm/tpu/smollm"))
from model import DecodeLayerT, Decoder, load_weights, multi_inputs  # noqa: E402

torch.set_num_threads(4)
S, T, SIZE = int(os.environ.get("VLM_S", 512)), int(os.environ.get("VLM_T", 4)), 512
HEAD_T = int(os.environ.get("VLM_HEAD_T", T))
CALIB, TESTS = 32, 4
MIX2 = set(range(30)) if os.environ.get("VLM_MIX2") == "all" else {0, 6, 15, 16, 18, 20, 22, 24, 25, 26}
NAMES_IN = ["k_cache", "v_cache", "mask", "h", "cos_q", "sin_q", "cos_k", "sin_k"]
NAMES_OUT = ["dh", "k_new", "v_new"]
here = pathlib.Path(__file__).parent


class CacheFirst(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, k_cache, v_cache, mask, h, cos_q, sin_q, cos_k, sin_k):
        return self.layer(h, cos_q, sin_q, cos_k, sin_k, k_cache, v_cache, mask)


class Head(nn.Module):
    def __init__(self, dec, w):
        super().__init__()
        self.norm = dec.norm
        self.w = nn.Parameter(w.t().contiguous(), requires_grad=False)

    def forward(self, h):
        return self.norm(h) @ self.w


def pixels(path):
    img = Image.open(path).convert("RGB")
    s = SIZE / max(img.size)
    img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BICUBIC)
    canvas = Image.new("RGB", (SIZE, SIZE), (127, 127, 127))
    canvas.paste(img, (0, 0))
    x = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8).float().view(SIZE, SIZE, 3).permute(2, 0, 1) / 255
    return ((x - 0.5) / 0.5)[None]


def common(answers):
    return max(set(answers), key=answers.count)


rows = json.load(open(here.parent / "quality" / "questions.json"))
random.Random(5).shuffle(rows)
seen, picked = set(), []
for r in rows:
    if r["image"] not in seen and (root / "images" / r["image"]).exists():
        seen.add(r["image"])
        picked.append(r)
picked = picked[:CALIB + TESTS]

tok = Tokenizer.from_file(f"{text}/tokenizer.json")
model = AutoModelForImageTextToText.from_pretrained(SNAPSHOT, dtype=torch.float32, attn_implementation="eager").eval()
vision = model.model.vision_model
connector = Connector()
samples = []
with torch.no_grad():
    for r in picked:
        y = vision(pixel_values=pixels(root / "images" / r["image"]),
                   patch_attention_mask=torch.ones(1, SIZE // 16, SIZE // 16, dtype=torch.bool)).last_hidden_state
        image = model.model.connector(y)[0]
        ids = tok.encode(prompt(r["question"]) + " " + common(r["answers"]) + "<end_of_utterance>",
                         add_special_tokens=False).ids
        embeds = model.get_input_embeddings()(torch.tensor(ids))
        embeds[torch.tensor(ids) == IMAGE_ID] = image
        samples.append((ids, embeds))
print("built", len(samples), "prompts,", len(samples[0][0]), "tokens in the first", flush=True)

# Check the numpy connector against transformers on a block output.
with torch.no_grad():
    x = torch.randn(1, 1024, 768)
    a = model.model.connector(vision.post_layernorm(x))[0].numpy()
    b = connector(x[0].numpy())
    print(f"numpy connector: max diff {np.abs(a - b).max():.2e}", flush=True)
    assert np.abs(a - b).max() < 1e-3

dec = Decoder(text, S)
lm = load_weights(text)["lm_head.weight"]
cfg = dec.cfg
L = cfg.layers


def run(rows):
    """Float run over one prompt. Returns the h rows into each layer and the final h."""
    dec.reset()
    rec, final = [[] for _ in range(L)], []
    with torch.no_grad():
        for t in range(len(rows)):
            _, cos, sin, mask = dec.inputs(0)
            h = rows[t].reshape(1, 1, -1)
            for i, layer in enumerate(dec.layers):
                rec[i].append(h.clone())
                h, k, v = layer(h, cos, sin, dec.k[i], dec.v[i], mask)
                dec.k[i][:, :, dec.t] = k[:, :, 0]
                dec.v[i][:, :, dec.t] = v[:, :, 0]
            final.append(h.clone())
            dec.t += 1
    return rec, final


# Check the layer model against transformers on the first prompt.
ids, embeds = samples[0]
with torch.no_grad():
    ref = model(inputs_embeds=embeds[None], attention_mask=torch.ones(1, len(ids), dtype=torch.long)).logits[0]
    _, final = run(embeds)
    mine = torch.cat([dec.norm(h) @ lm.t() for h in final], dim=1)[0]
err = (mine - ref).abs().max().item()
same = (mine.argmax(-1) == ref.argmax(-1)).float().mean().item()
print(f"float check: max logit diff {err:.3f}, same top token {same:.3f}", flush=True)
assert same == 1.0 and err < 0.5
del model, vision, ref, mine

qlists = "--quant_input_list 1,2,3,5,6,7,8 --quant_output_list 2,3"
g = cfg.nh // cfg.nkv
shapes = (f"[[1,3,{S},64],[1,3,{S},64],[1,3,{T * g},{S + T}],"
          f"[1,{T},576],[1,3,{T * g},64],[1,3,{T * g},64],[1,3,{T},64],[1,3,{T},64]]")


def cache_at(c, p):
    c = c.clone()
    c[:, :, p:] = 0
    return c


# DecodeLayerT always returns dh. run() needs delta off, so the float stream stays whole.
models = {i: CacheFirst(DecodeLayerT(dec.layers[i], T)) for i in range(L)}
head = Head(dec, lm)
dirs = [out / f"l{i:02d}" for i in range(L)] + [out / "head"]
for d in dirs:
    (d / "calib").mkdir(parents=True, exist_ok=True)

P = samples[0][0].index(IMAGE_ID) if os.environ.get("VLM_TEXT_PREFIX") == "1" else 1
assert all(ids[:P] == samples[0][0][:P] for ids, _ in samples)
first = {}
for j, (ids, embeds) in enumerate(samples):
    rec, final = run(embeds)
    n = len(ids)
    is_test = j >= CALIB
    p = P + int((j * 0.6180339887 % 1.0) * (n - T - P + 1))
    p = min(p, n - T)
    cos_q, sin_q, cos_k, sin_k, mask = multi_inputs(cfg, p, T, S)
    for i in range(L):
        if not is_test and i not in MIX2:
            continue
        h = torch.cat([rec[i][p + t] for t in range(T)], dim=1)
        x = dict(zip(NAMES_IN, [cache_at(dec.k[i], p), cache_at(dec.v[i], p), mask, h, cos_q, sin_q, cos_k, sin_k]))
        arrays = {k: v.numpy() for k, v in x.items()}
        if is_test:
            np.savez(dirs[i] / f"test_in_pos{j:03d}.npz", **arrays)
            with torch.no_grad():
                y = models[i](*x.values())
            np.savez(dirs[i] / f"test_ref_pos{j:03d}.npz", **{k: v.numpy() for k, v in zip(NAMES_OUT, y)})
            first.setdefault(i, x)
        else:
            np.savez(dirs[i] / "calib" / f"pos{j:03d}.npz", **arrays)
    h = torch.cat(final[p + T - HEAD_T:p + T], dim=1)
    if is_test:
        np.savez(dirs[L] / f"test_in_pos{j:03d}.npz", h=h.numpy())
        with torch.no_grad():
            np.savez(dirs[L] / f"test_ref_pos{j:03d}.npz", logits=head(h).numpy())
        first.setdefault(L, dict(h=h))
    else:
        np.savez(dirs[L] / "calib" / f"pos{j:03d}.npz", h=h.numpy())
    print(f"sample {j}: {n} tokens, window at {p}", flush=True)

# The last run leaves the K/V of positions 0..P-1, the same in every prompt, in the caches.
k = dec.k[:, 0, :, :P].numpy().astype(np.float32)
v = dec.v[:, 0, :, :P].numpy().astype(np.float32)
with open(out / "prefix-imstart.kv", "wb") as f:
    np.array([P] + samples[0][0][:P], dtype=np.int32).tofile(f)
    k.tofile(f)
    v.tofile(f)
dec.emb.half().numpy().tofile(out / "embed.f16")

for i in range(L):
    name = f"l{i:02d}"
    x = first[i]
    torch.onnx.export(models[i], tuple(x.values()), str(dirs[i] / f"{name}.onnx"), input_names=list(x.keys()),
                      output_names=NAMES_OUT, opset_version=17, dynamo=False)
    (dirs[i] / "shapes").write_text(shapes + "\n")
    (dirs[i] / "quant_lists").write_text(qlists + "\n")
    print(name, "exported", flush=True)
x = first[L]
torch.onnx.export(head, tuple(x.values()), str(dirs[L] / "head.onnx"), input_names=["h"], output_names=["logits"],
                  opset_version=17, dynamo=False)
(dirs[L] / "shapes").write_text(f"[[1,{HEAD_T},576]]\n")
(dirs[L] / "quant_lists").write_text("\n")
print("done, asked:", ASK)
