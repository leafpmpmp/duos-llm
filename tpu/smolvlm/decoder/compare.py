"""Paired comparisons for the TPU decoder on the 150 unseen questions.

Usage: python compare.py
"""
import json
import pathlib
import random
import re

here = pathlib.Path(__file__).resolve().parent
rows = json.load(open(here / "answers-tpu-decoder.json"))
fl = json.load(open(here.parent / "encoder" / "answers-cpu-float.json"))
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


def paired(name, a, b):
    d = [x - y for x, y in zip(a, b)]
    rng = random.Random(7)
    boots = sorted(mean(rng.choices(d, k=len(d))) for _ in range(10000))
    print(f"{name}: {mean(a):.1f} vs {mean(b):.1f}, difference {mean(d):.1f}, 95% CI [{boots[250]:.1f}, {boots[9750]:.1f}]")


tpu = [vqa(r["out"]["tpu_decoder"], r["answers"]) for r in rows]
paired("TPU decoder vs float decoder, TPU features", tpu, [vqa(r["out"]["tpu512"], r["answers"]) for r in rows])
paired("TPU encoder and decoder vs all float", tpu, [vqa(r["out"]["tpu512"], r["answers"]) for r in fl])
paired("TPU encoder and decoder vs JSON + LLM", tpu, [vqa(r["out"]["json_llm"], r["answers"]) for r in rows])
print("same as all float:", sum(a["out"]["tpu_decoder"] == b["out"]["tpu512"] for a, b in zip(rows, fl)), "of", len(rows))
for r in rows:
    if vqa(r["out"]["tpu_decoder"], r["answers"]) + 0.3 < vqa(r["out"]["tpu512"], r["answers"]):
        print(f"  lost: {r['question'][:46]!r} tpu {r['out']['tpu_decoder']!r} float {r['out']['tpu512']!r}")
