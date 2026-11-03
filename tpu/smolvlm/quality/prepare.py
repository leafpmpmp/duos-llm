"""Sample VQA v2 person questions and run YOLO11n on their COCO images.

Usage: python prepare.py D:/duosight-eval [--smoke]
Writes questions.json next to this script.
"""
import json
import pathlib
import random
import re
import sys
import urllib.request

from ultralytics import YOLO

root = pathlib.Path(sys.argv[1])
smoke = "--smoke" in sys.argv
here = pathlib.Path(__file__).parent
QUOTA = {"other": 150, "yes/no": 100, "number": 50}
if smoke:
    QUOTA = {"other": 8, "yes/no": 4, "number": 2}
PERSON = re.compile(r"\b(man|men|woman|women|person|people|boy|girl|child|children|kid|kids|guy|guys|lady|player|skier|surfer)\b", re.I)

questions = json.load(open(root / "data/v2_OpenEnded_mscoco_val2014_questions.json"))["questions"]
answers = {a["question_id"]: a for a in json.load(open(root / "data/v2_mscoco_val2014_annotations.json"))["annotations"]}

pool = [q for q in questions if PERSON.search(q["question"])]
print(f"{len(questions)} questions, {len(pool)} name a person")
random.Random(42).shuffle(pool)

(root / "images").mkdir(exist_ok=True)
detector = YOLO(str(root / "yolo11n.pt"))
seen, out, left = set(), [], dict(QUOTA)
for q in pool:
    if not any(left.values()):
        break
    a = answers[q["question_id"]]
    t = a["answer_type"]
    if left.get(t, 0) == 0 or q["image_id"] in seen:
        continue
    name = f"COCO_val2014_{q['image_id']:012d}.jpg"
    path = root / "images" / name
    if not path.exists():
        urllib.request.urlretrieve(f"http://images.cocodataset.org/val2014/{name}", path)
    r = detector.predict(str(path), imgsz=448, conf=0.4, classes=[0], verbose=False)[0]
    boxes = r.boxes.xyxy.cpu().tolist()
    scores = r.boxes.conf.cpu().tolist()
    if not boxes:
        continue
    seen.add(q["image_id"])
    left[t] -= 1
    h, w = r.orig_shape
    out.append({
        "question_id": q["question_id"], "image": name, "question": q["question"],
        "answer_type": t, "question_type": a["question_type"],
        "answers": [x["answer"] for x in a["answers"]],
        "width": w, "height": h, "boxes": boxes, "scores": scores,
    })
    if len(out) % 25 == 0:
        print(len(out), left, flush=True)

json.dump(out, open(here / ("questions-smoke.json" if smoke else "questions.json"), "w"), indent=1)
print(f"wrote {len(out)} questions, left {left}")
