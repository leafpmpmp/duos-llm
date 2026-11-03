"""The SmolVLM tokenizer in plain Python, for a board without numpy or tokenizers.

Usage: tok.py encode VOCAB.json ROWS.bin OUT.bin QUESTION...   (write prompts for vlm)
       tok.py decode VOCAB.json ANSWERS.txt                    (print the answers)
VOCAB.json comes from make_weights.py.
"""
import json
import re
import struct
import sys

ASK = "Answer using a single word or phrase."
SPECIAL = {"<|im_start|>": 1, "<end_of_utterance>": 49279, "<fake_token_around_image>": 49189,
           "<image>": 49190, "<global-img>": 49152}
WORD = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+| ?_+|\s+(?!\S)|\s+")


def byte_table():
    keep = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    table, extra = {}, 0
    for b in range(256):
        if b in keep:
            table[b] = chr(b)
        else:
            table[b] = chr(256 + extra)
            extra += 1
    return table


class Tokenizer:
    def __init__(self, path):
        data = json.load(open(path, encoding="utf-8"))
        self.vocab = data["vocab"]
        self.rank = {m: i for i, m in enumerate(data["merges"])}
        self.bytes = byte_table()
        self.back = {v: k for k, v in self.bytes.items()}
        self.ids = {v: k for k, v in self.vocab.items()}

    def bpe(self, word):
        parts = list(word)
        while len(parts) > 1:
            pairs = {f"{a} {b}" for a, b in zip(parts, parts[1:])}
            best = min(pairs, key=lambda p: self.rank.get(p, 1 << 30))
            if best not in self.rank:
                break
            merged, i = [], 0
            while i < len(parts):
                if i + 1 < len(parts) and f"{parts[i]} {parts[i + 1]}" == best:
                    merged.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = merged
        return parts

    def text(self, s):
        out = []
        for piece in re.split(r"(\d)", s):  # each digit stands alone
            for word in WORD.findall(piece):
                mapped = "".join(self.bytes[b] for b in word.encode("utf-8"))
                out += [self.vocab[t] for t in self.bpe(mapped)]
        return out

    def prompt(self, question):
        return ([SPECIAL["<|im_start|>"]] + self.text("User:") + [SPECIAL["<fake_token_around_image>"], SPECIAL["<global-img>"]]
                + [SPECIAL["<image>"]] * 64 + [SPECIAL["<fake_token_around_image>"]]
                + self.text(f"{question} {ASK}") + [SPECIAL["<end_of_utterance>"]] + self.text("\nAssistant:"))

    def decode(self, ids):
        text = "".join(self.ids.get(i, "") for i in ids)
        return bytes(self.back[c] for c in text if c in self.back).decode("utf-8", "replace").strip()


if __name__ == "__main__":
    tok = Tokenizer(sys.argv[2])
    if sys.argv[1] == "encode":
        rows = open(sys.argv[3], "rb").read()
        with open(sys.argv[4], "wb") as f:
            for question in sys.argv[5:]:
                ids = tok.prompt(question)
                f.write(struct.pack(f"<i{len(ids)}i", len(ids), *ids))
                f.write(rows)
    else:
        for line in open(sys.argv[3]):
            print(tok.decode([int(x) for x in line.split()[3:]]))
