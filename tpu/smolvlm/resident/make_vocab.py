"""Write the id-to-bytes table duosight uses to turn answer ids into text.

The same decoding as tok.py: each token's characters map back to bytes through the
byte-level table, and characters outside it are dropped. File: int32 count, then for each
id a uint8 length and the bytes.

Usage: python3 make_vocab.py VOCAB.json OUT.bin   (plain Python)
"""
import json
import pathlib
import struct
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "camera"))
from tok import byte_table  # noqa: E402

vocab = json.load(open(sys.argv[1], encoding="utf-8"))["vocab"]
back = {v: k for k, v in byte_table().items()}
count = max(vocab.values()) + 1
table = [b""] * count
for token, i in vocab.items():
    table[i] = bytes(back[c] for c in token if c in back)
assert max(map(len, table)) < 256
with open(sys.argv[2], "wb") as f:
    f.write(struct.pack("<i", count))
    for b in table:
        f.write(struct.pack("<B", len(b)) + b)
print(count, "ids,", sum(map(len, table)), "bytes")
