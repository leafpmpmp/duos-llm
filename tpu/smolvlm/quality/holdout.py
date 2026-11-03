"""Sample 150 unseen "other" questions, seed 43, and answer them with SmolLM2 from JSON.

The input sizes were chosen on the first 150 questions, so their scores run high.
These questions took no part in that choice.

Usage: python holdout.py D:/duosight-eval
Writes answers-holdout.json next to this script.
"""
import json
import pathlib
import random
import re
import sys
import urllib.request

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ultralytics import YOLO

LLM = "HuggingFaceTB/SmolLM2-135M-Instruct"
ASK = "Answer using a single word or phrase."
N = 150
root = pathlib.Path(sys.argv[1])
here = pathlib.Path(__file__).parent
PERSON = re.compile(r"\b(man|men|woman|women|person|people|boy|girl|child|children|kid|kids|guy|guys|lady|player|skier|surfer)\b", re.I)

used = {r["image"] for r in json.load(open(here / "questions.json"))}
questions = json.load(open(root / "data/v2_OpenEnded_mscoco_val2014_questions.json"))["questions"]
answers = {a["question_id"]: a for a in json.load(open(root / "data/v2_mscoco_val2014_annotations.json"))["annotations"]}
pool = [q for q in questions if PERSON.search(q["question"])]
random.Random(43).shuffle(pool)

detector = YOLO(str(root / "yolo11n.pt"))
rows, seen = [], set()
for q in pool:
    if len(rows) == N:
        break
    a = answers[q["question_id"]]
    name = f"COCO_val2014_{q['image_id']:012d}.jpg"
    if a["answer_type"] != "other" or name in used or name in seen:
        continue
    path = root / "images" / name
    if not path.exists():
        urllib.request.urlretrieve(f"http://images.cocodataset.org/val2014/{name}", path)
    r = detector.predict(str(path), imgsz=448, conf=0.4, classes=[0], verbose=False)[0]
    boxes = r.boxes.xyxy.cpu().tolist()
    if not boxes:
        continue
    seen.add(name)
    h, w = r.orig_shape
    rows.append({"question_id": q["question_id"], "image": name, "question": q["question"], "answer_type": "other",
                 "answers": [x["answer"] for x in a["answers"]], "width": w, "height": h, "boxes": boxes,
                 "scores": r.boxes.conf.cpu().tolist(), "out": {}})

tok = AutoTokenizer.from_pretrained(LLM)
model = AutoModelForCausalLM.from_pretrained(LLM, dtype=torch.bfloat16).to("cuda").eval()
with torch.no_grad():
    for r in rows:
        people = [{"box": [round(v / (r["width"] if i % 2 == 0 else r["height"]), 2) for i, v in enumerate(b)],
                   "score": round(s, 2)} for b, s in zip(r["boxes"], r["scores"])]
        msgs = [{"role": "system", "content": "You answer questions about a camera image. You cannot see it. "
                                              "A person detector reported these boxes, as fractions of the frame: "
                                              + json.dumps({"people": people})},
                {"role": "user", "content": f"{r['question']} {ASK}"}]
        ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), return_tensors="pt").input_ids.to("cuda")
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=24, do_sample=False)
        r["out"]["json_llm"] = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
json.dump(rows, open(here / "answers-holdout.json", "w"), indent=1)
print(f"wrote {len(rows)} unseen questions")
