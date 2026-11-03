"""Score vlm's answers against the same features with the float decoder.

Only the decoder differs between the two conditions: both use the TPU encoder's output.

Usage: python score_vlm.py ANSWERS.txt TOKENIZER.json OUT.json
"""
import json
import pathlib
import random
import re
import sys

from tokenizers import Tokenizer

here = pathlib.Path(__file__).resolve().parent
rows = json.load(open(here.parent / "encoder" / "answers-cpu-final.json"))
tok = Tokenizer.from_file(sys.argv[2])
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


lines = [line.split() for line in open(sys.argv[1])]
for r, p in zip(rows, lines):
    r["out"]["tpu_decoder"] = tok.decode([int(x) for x in p[3:]], skip_special_tokens=True).strip()
rows = rows[:len(lines)]
json.dump(rows, open(sys.argv[3], "w"), indent=1)

a = [vqa(r["out"]["tpu_decoder"], r["answers"]) for r in rows]
b = [vqa(r["out"]["tpu512"], r["answers"]) for r in rows]
d = [x - y for x, y in zip(a, b)]
rng = random.Random(7)
boots = sorted(mean(rng.choices(d, k=len(d))) for _ in range(10000))
same = sum(r["out"]["tpu_decoder"] == r["out"]["tpu512"] for r in rows)
tp = sorted(int(p[1]) for p in lines)
td = sum(int(p[2]) for p in lines)
gen = sum(len(p) - 3 for p in lines)
print(f"questions {len(rows)}")
print(f"TPU decoder {mean(a):.1f}, float decoder {mean(b):.1f}, same answer {same}")
print(f"TPU minus float {mean(d):.1f}, 95% CI [{boots[250]:.1f}, {boots[9750]:.1f}]")
print(f"prefill median {tp[len(tp) // 2]} ms, decode {1e3 * gen / max(td, 1):.1f} tok/s over {gen} tokens")
