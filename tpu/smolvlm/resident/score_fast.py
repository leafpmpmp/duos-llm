"""Score duosight runs on the 150 unseen "other" questions, paired against earlier builds.

Each run is a duosight reply file: "ans PREFILL DECODE REUSED ID..." lines in question order
("image" lines give the encoder steps). The answers are compared with:
  q4: both TPU parts, q4 decoder (decoder, 47.1)
  float: float encoder and float decoder (50.9)

Usage: python score_fast.py TOKENIZER.json OUT.json NAME=FILE...
OUT.json gets every answer under its NAME.
"""
import json
import pathlib
import random
import re
import statistics
import sys

from tokenizers import Tokenizer

here = pathlib.Path(__file__).resolve().parent
rows = json.load(open(here.parent / "decoder" / "answers-tpu-decoder.json"))
tok = Tokenizer.from_file(sys.argv[1])
WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
         "seven": "7", "eight": "8", "nine": "9", "ten": "10"}


def norm(text):
    text = text.lower().strip().split("\n")[0]
    text = re.sub(r"[.,;:!?\"'()]", "", text)
    return " ".join(WORDS.get(w, w) for w in text.split() if w not in ("a", "an", "the"))


def vqa(pred, humans):
    pred, humans = norm(pred), [norm(h) for h in humans]
    score = 0.0
    for i in range(len(humans)):
        rest = humans[:i] + humans[i + 1:]
        score += min(sum(bool(h) and f" {h} " in f" {pred} " for h in rest) / 3, 1)
    return score / len(humans)


def mean(xs):
    return 100 * sum(xs) / len(xs)


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    rng = random.Random(7)
    boots = sorted(mean(rng.choices(d, k=len(d))) for _ in range(10000))
    return f"{mean(d):+.1f} [{boots[250]:.1f}, {boots[9750]:.1f}]"


refs = {"q4": "tpu_decoder", "float": "native512"}
for spec in sys.argv[3:]:
    name, path = spec.split("=", 1)
    lines = [line.split() for line in open(path)]
    ans = [p for p in lines if p[0] == "ans"]
    img = [p for p in lines if p[0] == "image"]
    assert len(ans) == len(rows), (path, len(ans))
    for r, p in zip(rows, ans):
        r["out"][name] = tok.decode([int(x) for x in p[4:]], skip_special_tokens=True).strip()
    s = [vqa(r["out"][name], r["answers"]) for r in rows]
    print(f"{name}: {mean(s):.1f}", end="")
    for label, key in refs.items():
        t = [vqa(r["out"][key], r["answers"]) for r in rows]
        same = sum(r["out"][name] == r["out"][key] for r in rows)
        print(f" | vs {label} ({mean(t):.1f}) {paired(s, t)}, same {same}", end="")
    pre = [float(p[1]) for p in ans]
    dec = [float(p[2]) for p in ans]
    gen = sum(len(p) - 4 for p in ans)
    print(f"\n  prefill median {statistics.median(pre):.0f} ms, decode median {statistics.median(dec):.0f} ms, "
          f"{1e3 * gen / sum(dec):.1f} tok/s over {gen} tokens", end="")
    if img:
        steps = list(zip(*[[float(x) for x in p[1:5]] for p in img]))
        print(", image " + " / ".join(f"{statistics.median(x):.0f}" for x in steps) + " ms (canvas / patch / blocks / connector)", end="")
    print()
json.dump(rows, open(sys.argv[2], "w"), indent=1)
