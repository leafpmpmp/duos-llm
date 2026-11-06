"""Score answers-explore.json. Exploratory: this does not change the rule in PREDICTIONS.md.

Usage: python score_explore.py
"""
import json
import pathlib
import random
import re

here = pathlib.Path(__file__).parent
rows = json.load(open(here / "answers-explore.json"))
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


def score(r, name):
    return vqa(r["out"][name], r["answers"])


NAMES = ["json_llm", "official512", "full512", "masked512", "masked448", "masked384", "masked320", "masked256",
         "pixels256_tokens64", "maskcrop512", "maskcrop384"]
print(f"{'condition':20} {'other':>6} {'number':>7} {'tokens':>7} {'patches':>8}")
tokens = {"512": 64, "448": 49, "384": 36, "320": 25, "256": 16}
for n in NAMES:
    other = mean([score(r, n) for r in rows if r["answer_type"] == "other"])
    number = mean([score(r, n) for r in rows if r["answer_type"] == "number"])
    t = next((v for k, v in tokens.items() if n.endswith(k)), 64 if "tokens64" in n else 0)
    print(f"{n:20} {other:6.1f} {number:7.1f} {t:7d} {t * 16:8d}")


def paired(a, b, n=10000):
    sub = [r for r in rows if r["answer_type"] == "other"]
    d = [score(r, a) - score(r, b) for r in sub]
    rng = random.Random(7)
    boots = sorted(mean(rng.choices(d, k=len(d))) for _ in range(n))
    return f"{a} - {b}: {mean(d):.1f} [{boots[int(.025 * n)]:.1f}, {boots[int(.975 * n)]:.1f}]"


print()
for a, b in (("official512", "masked512"), ("masked512", "full512"), ("masked512", "masked384"),
             ("masked512", "masked320"), ("masked512", "masked256"), ("masked384", "json_llm"),
             ("masked320", "json_llm"), ("maskcrop384", "masked384"), ("maskcrop512", "masked512"),
             ("pixels256_tokens64", "masked256"), ("masked512", "pixels256_tokens64")):
    print(paired(a, b))

print("\nyes/no with the prompt 'Answer yes or no.'")
yn = [r for r in rows if r["answer_type"] == "yes/no"]
for n in ("yesno256", "yesno384", "yesno512"):
    print(f"{n}: {mean([score(r, n) for r in yn]):.1f}")
print("always 'no' on these questions:", f"{mean([vqa('no', r['answers']) for r in yn]):.1f}")
print("always 'yes' on these questions:", f"{mean([vqa('yes', r['answers']) for r in yn]):.1f}")
