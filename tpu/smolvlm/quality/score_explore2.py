"""Score answers-explore2.json. Exploratory.

Usage: python score_explore2.py [file]
"""
import json
import pathlib
import random
import re
import sys

here = pathlib.Path(__file__).parent
rows = json.load(open(sys.argv[1] if len(sys.argv) > 1 else here / "answers-explore2.json"))
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
    return 100 * sum(xs) / len(xs) if xs else float("nan")


def s(r, n):
    return vqa(r["out"][n], r["answers"])


have = rows[0]["out"].keys()
other = [r for r in rows if r["answer_type"] == "other"]
number = [r for r in rows if r["answer_type"] == "number"]
names = [n for n in ("json_llm", "native512", "masked512", "up384_bilinear", "up320_bilinear", "up256_bilinear",
                     "up256_nearest", "masked256", "pixels256_tokens64") if n in have]
print(f"{'condition':22} {'other':>6} {'number':>7}")
for n in names:
    print(f"{n:22} {mean([s(r, n) for r in other]):6.1f} {mean([s(r, n) for r in number]):7.1f}")


def paired(a, b, sub=other, k=10000):
    d = [s(r, a) - s(r, b) for r in sub]
    rng = random.Random(7)
    boots = sorted(mean(rng.choices(d, k=len(d))) for _ in range(k))
    return f"{a} - {b}: {mean(d):.1f} [{boots[int(.025 * k)]:.1f}, {boots[int(.975 * k)]:.1f}]"


print()
for a, b in (("native512", "up384_bilinear"), ("native512", "up320_bilinear"), ("native512", "up256_bilinear"),
             ("up256_bilinear", "up256_nearest"), ("up256_bilinear", "masked256"), ("up256_bilinear", "json_llm"),
             ("up320_bilinear", "json_llm"), ("up384_bilinear", "json_llm")):
    if a in have and b in have:
        print(paired(a, b))


def frac(r):
    x1, y1, x2, y2 = max(r["boxes"], key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    return (x2 - x1) * (y2 - y1) / (r["width"] * r["height"])


if "maskcrop512" in have:
    print("\nCrop versus full frame on 'other', by the share of the frame the largest person fills")
    print(f"{'share':10} {'n':>4} {'full512':>8} {'crop512':>8} {'full384':>8} {'crop384':>8}")
    for lo, hi in ((0, .1), (.1, .3), (.3, 1.01)):
        sub = [r for r in other if lo <= frac(r) < hi]
        print(f"{lo:.1f}-{hi:.1f}    {len(sub):4d} {mean([s(r, 'masked512') for r in sub]):8.1f} {mean([s(r, 'maskcrop512') for r in sub]):8.1f}"
              f" {mean([s(r, 'masked384') for r in sub]):8.1f} {mean([s(r, 'maskcrop384') for r in sub]):8.1f}")
