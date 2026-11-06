"""Score answers.json with the VQA metric and apply the rule in PREDICTIONS.md.

Usage: python score.py [answers.json]
"""
import json
import pathlib
import random
import re
import sys

here = pathlib.Path(__file__).parent
rows = json.load(open(sys.argv[1] if len(sys.argv) > 1 else here / "answers.json"))
NAMES = ["json_llm", "full256", "crop256", "full512", "crop512"]
WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
         "seven": "7", "eight": "8", "nine": "9", "ten": "10"}


def norm(text):
    text = text.lower().strip().split("\n")[0]
    text = re.sub(r"[.,;:!?\"'()]", "", text)
    words = [WORDS.get(w, w) for w in text.split() if w not in ("a", "an", "the")]
    return " ".join(words)


def exact(pred, human):
    return pred == human


def lenient(pred, human):
    return bool(human) and f" {human} " in f" {pred} "


def vqa(pred, humans, match):
    pred = norm(pred)
    humans = [norm(h) for h in humans]
    score = 0.0
    for i in range(len(humans)):
        rest = humans[:i] + humans[i + 1:]
        score += min(sum(match(pred, h) for h in rest) / 3, 1)
    return score / len(humans)


for r in rows:
    r["exact"] = {n: vqa(r["out"][n], r["answers"], exact) for n in NAMES}
    r["lenient"] = {n: vqa(r["out"][n], r["answers"], lenient) for n in NAMES}


def mean(xs):
    return 100 * sum(xs) / len(xs)


for key in ("lenient", "exact"):
    print(f"--- {key}")
    print(f"{'type':8} {'n':>4} " + " ".join(f"{n:>9}" for n in NAMES))
    for t in ("other", "yes/no", "number", "all"):
        sub = [r for r in rows if t in ("all", r["answer_type"])]
        print(f"{t:8} {len(sub):4d} " + " ".join(f"{mean([r[key][n] for r in sub]):9.1f}" for n in NAMES))


def paired(a, b, sub, key="lenient", n=10000):
    d = [r[key][a] - r[key][b] for r in sub]
    rng = random.Random(7)
    boots = sorted(mean(rng.choices(d, k=len(d))) for _ in range(n))
    return mean(d), boots[int(0.025 * n)], boots[int(0.975 * n)]


other = [r for r in rows if r["answer_type"] == "other"]
gain, low, high = paired("crop256", "json_llm", other)
loss, llow, lhigh = paired("full512", "crop256", other)
print(f"\nOn lenient 'other' scores:")
print(f"crop256 - json_llm: {gain:.1f} points, 95% CI [{low:.1f}, {high:.1f}]")
print(f"full512 - crop256: {loss:.1f} points, 95% CI [{llow:.1f}, {lhigh:.1f}]")
for a, b in (("crop256", "full256"), ("crop512", "full512"), ("full512", "full256")):
    d = paired(a, b, other)
    print(f"{a} - {b}: {d[0]:.1f} points, 95% CI [{d[1]:.1f}, {d[2]:.1f}]")
print("rule 1 (gain >= 10 and CI above 0):", "pass" if gain >= 10 and low > 0 else "FAIL")
print("rule 2 (full512 - crop256 <= 10):", "pass" if loss <= 10 else "FAIL")
