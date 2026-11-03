"""Copy SmolVLM's text model into a folder that duos-llm's tpu/smollm/model.py can read.

Only the file layout changes: the weights are copied byte for byte, so this needs no
torch and almost no memory. lm_head stays apart, because SmolVLM does not tie it to the
embedding table.

Usage: python make_text_dir.py SNAPSHOT_DIR OUT_DIR
"""
import json
import pathlib
import shutil
import struct
import sys

src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

f = open(src / "model.safetensors", "rb")
size = struct.unpack("<Q", f.read(8))[0]
header = json.loads(f.read(size))
base = 8 + size

kept = []
for name, info in header.items():
    if name.startswith("model.text_model."):
        kept.append(("model." + name[len("model.text_model."):], info))
    elif name == "lm_head.weight":
        kept.append((name, info))

new, offset = {"__metadata__": {"format": "pt"}}, 0
for name, info in kept:
    n = info["data_offsets"][1] - info["data_offsets"][0]
    new[name] = {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [offset, offset + n]}
    offset += n
text = json.dumps(new, separators=(",", ":")).encode()
text += b" " * (-len(text) % 8)

with open(out / "model.safetensors", "wb") as g:
    g.write(struct.pack("<Q", len(text)))
    g.write(text)
    for _, info in kept:
        a, b = info["data_offsets"]
        f.seek(base + a)
        g.write(f.read(b - a))

config = json.load(open(src / "config.json"))["text_config"]
json.dump(config, open(out / "config.json", "w"), indent=1)
shutil.copy(src / "tokenizer.json", out / "tokenizer.json")
print(f"{len(kept)} tensors, {offset / 2**20:.1f} MiB in {out}")
