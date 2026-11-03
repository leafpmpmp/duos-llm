"""Write the BPE tables duosight uses to turn a question in words into ids (bpe.h).

The same tokenizer as tok.py, over ids instead of strings: the id of each byte's
character (-1 for the few bytes the vocabulary lacks), then every merge as (left id, right id, rank, merged id), sorted by the pair so
duosight can search it. File: int32 byte_ids[256], int32 count, then the merges.

Usage: python3 make_bpe.py VOCAB.json OUT.bin   (plain Python)
"""
import json
import pathlib
import struct
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "camera"))
from tok import byte_table  # noqa: E402

data = json.load(open(sys.argv[1], encoding="utf-8"))
vocab = data["vocab"]
table = byte_table()
byte_ids = [vocab.get(table[b], -1) for b in range(256)]  # -1: a byte with no token, as for tok.py
print(byte_ids.count(-1), "bytes have no token")
merges = []
for rank, m in enumerate(data["merges"]):
    a, b = m.split(" ")
    merges.append((vocab[a], vocab[b], rank, vocab[a + b]))
merges.sort()
assert len({(a, b) for a, b, _, _ in merges}) == len(merges), "a pair merges twice"
with open(sys.argv[2], "wb") as f:
    f.write(struct.pack("<256i", *byte_ids))
    f.write(struct.pack("<i", len(merges)))
    for m in merges:
        f.write(struct.pack("<4i", *m))
print(len(merges), "merges")
