"""Exploratory follow-up to run.py. Not part of the rule in PREDICTIONS.md.

Why it exists: the pre-registered test failed at 256 px, and the yes/no scores were
unusable because the prompt made the model echo keywords. This run asks:
  1. Does my preprocessing match the official processor at 512 px?
  2. Is the 256 px loss from too few pixels, or from too few image tokens?
  3. Which input size is the smallest that stays close to 512 px?

Usage: python explore.py D:/duosight-eval
Writes answers-explore.json next to this script.
"""
import json
import pathlib
import random
import sys
import time

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

VLM = "HuggingFaceTB/SmolVLM-256M-Instruct"
ASK = "Answer using a single word or phrase."
YES_NO = "Answer yes or no."
MARGIN = 0.10

root = pathlib.Path(sys.argv[1])
here = pathlib.Path(__file__).parent
rows = json.load(open(here / "answers.json"))
device, dtype = "cuda", torch.bfloat16


def fit(img, size):
    """Scale so the longest edge is size. Pad right and bottom, and mask the padding."""
    s = size / max(img.size)
    w, h = max(1, round(img.width * s)), max(1, round(img.height * s))
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(img.resize((w, h), Image.BICUBIC), (0, 0))
    x = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8).float()
    x = (x.view(size, size, 3).permute(2, 0, 1) / 255 - 0.5) / 0.5
    mask = torch.zeros(size, size, dtype=torch.long)
    mask[:h, :w] = 1
    x = x * mask
    return x[None, None].to(device, dtype), mask[None, None].to(device)


def prompt(question, tokens, ask):
    image = "<fake_token_around_image><global-img>" + "<image>" * tokens + "<fake_token_around_image>"
    return f"<|im_start|>User:{image}{question} {ask}<end_of_utterance>\nAssistant:"


def crop_largest(img, boxes):
    x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    mx, my = (x2 - x1) * MARGIN, (y2 - y1) * MARGIN
    return img.crop((max(0, x1 - mx), max(0, y1 - my), min(img.width, x2 + mx), min(img.height, y2 + my)))


@torch.no_grad()
def answer(model, tok, img, question, size, ask=ASK, content=None):
    """size sets the encoder input. content first shrinks the image, to separate pixels from tokens."""
    if content:
        s = content / max(img.size)
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BICUBIC)
    pixels, mask = fit(img, size)
    ids = tok(prompt(question, (size // 64) ** 2, ask), return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pixels,
                         pixel_attention_mask=mask, max_new_tokens=24, do_sample=False)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def official(model, proc, img, question):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"{question} {ASK}"}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = proc(text=text, images=[[img]], do_image_splitting=False, return_tensors="pt").to(device, dtype)
    out = model.generate(**inputs, max_new_tokens=24, do_sample=False)
    return proc.tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


proc = AutoProcessor.from_pretrained(VLM)
tok = proc.tokenizer
model = AutoModelForImageTextToText.from_pretrained(VLM, dtype=dtype).to(device).eval()
image = lambda r: Image.open(root / "images" / r["image"]).convert("RGB")

runs = {
    "official512": lambda r: official(model, proc, image(r), r["question"]),
    "masked256": lambda r: answer(model, tok, image(r), r["question"], 256),
    "masked320": lambda r: answer(model, tok, image(r), r["question"], 320),
    "masked384": lambda r: answer(model, tok, image(r), r["question"], 384),
    "masked448": lambda r: answer(model, tok, image(r), r["question"], 448),
    "masked512": lambda r: answer(model, tok, image(r), r["question"], 512),
    "pixels256_tokens64": lambda r: answer(model, tok, image(r), r["question"], 512, content=256),
    "maskcrop384": lambda r: answer(model, tok, crop_largest(image(r), r["boxes"]), r["question"], 384),
    "maskcrop512": lambda r: answer(model, tok, crop_largest(image(r), r["boxes"]), r["question"], 512),
}
for name, fn in runs.items():
    t0 = time.time()
    for r in rows:
        r["out"][name] = fn(r)
    print(f"{name}: {time.time() - t0:.0f} s", flush=True)

for size in (256, 384, 512):
    name = f"yesno{size}"
    for r in rows:
        if r["answer_type"] == "yes/no":
            r["out"][name] = answer(model, tok, image(r), r["question"], size, ask=YES_NO)
    print(name, "done", flush=True)

json.dump(rows, open(here / "answers-explore.json", "w"), indent=1)
