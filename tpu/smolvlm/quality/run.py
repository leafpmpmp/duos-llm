"""Answer the sampled questions under each condition in PREDICTIONS.md.

Usage: python run.py D:/duosight-eval [--smoke]
Writes answers.json next to this script.
"""
import json
import pathlib
import sys
import time

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

VLM = "HuggingFaceTB/SmolVLM-256M-Instruct"
LLM = "HuggingFaceTB/SmolLM2-135M-Instruct"
ASK = "Answer using a single word or phrase."
MARGIN = 0.10
TOKENS = {256: 16, 512: 64}

root = pathlib.Path(sys.argv[1])
smoke = "--smoke" in sys.argv
here = pathlib.Path(__file__).parent
qs = json.load(open(here / ("questions-smoke.json" if smoke else "questions.json")))
device = "cuda"
dtype = torch.bfloat16


def letterbox(img, size):
    """Fit img in a size x size square with grey bars, as the board would."""
    s = size / max(img.size)
    w, h = max(1, round(img.width * s)), max(1, round(img.height * s))
    canvas = Image.new("RGB", (size, size), (127, 127, 127))
    canvas.paste(img.resize((w, h), Image.BICUBIC), ((size - w) // 2, (size - h) // 2))
    return canvas


def to_tensor(img):
    x = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).float()
    x = x.view(img.height, img.width, 3).permute(2, 0, 1) / 255
    return (x - 0.5) / 0.5


def largest_crop(img, boxes):
    x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    mx, my = (x2 - x1) * MARGIN, (y2 - y1) * MARGIN
    return img.crop((max(0, x1 - mx), max(0, y1 - my), min(img.width, x2 + mx), min(img.height, y2 + my)))


def vlm_prompt(question, size):
    image = "<fake_token_around_image><global-img>" + "<image>" * TOKENS[size] + "<fake_token_around_image>"
    return f"<|im_start|>User:{image}{question} {ASK}<end_of_utterance>\nAssistant:"


@torch.no_grad()
def vlm_answer(model, tok, img, question, size):
    ids = tok(vlm_prompt(question, size), return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    pixels = to_tensor(letterbox(img, size))[None, None].to(device, dtype)
    mask = torch.ones(1, 1, size, size, dtype=torch.long, device=device)
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pixels,
                         pixel_attention_mask=mask, max_new_tokens=24, do_sample=False)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def detections_json(q):
    people = [{"box": [round(v / (q["width"] if i % 2 == 0 else q["height"]), 2) for i, v in enumerate(b)], "score": round(s, 2)}
              for b, s in zip(q["boxes"], q["scores"])]
    return json.dumps({"people": people})


@torch.no_grad()
def llm_answer(model, tok, q):
    msgs = [{"role": "system", "content": "You answer questions about a camera image. You cannot see it. "
                                          "A person detector reported these boxes, as fractions of the frame: "
                                          + detections_json(q)},
            {"role": "user", "content": f"{q['question']} {ASK}"}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=24, do_sample=False)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def check_prompt(tok, proc):
    """My prompt must tokenize exactly as the processor does for one 512 px image."""
    img = Image.new("RGB", (512, 512), (127, 127, 127))
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"Q? {ASK}"}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    want = proc(text=text, images=[[img]], do_image_splitting=False, return_tensors="pt").input_ids[0].tolist()
    got = tok(vlm_prompt("Q?", 512), add_special_tokens=False).input_ids
    if want != got:
        drop = want[1:] if want[1:] == got else None
        print("prompt mismatch", len(want), len(got), proc.tokenizer.decode(want[:12]), "|", proc.tokenizer.decode(got[:12]))
        if drop is None:
            sys.exit(1)
    print("prompt matches the processor", len(got), "tokens")


answers = {q["question_id"]: {} for q in qs}

proc = AutoProcessor.from_pretrained(VLM)
tok = proc.tokenizer
check_prompt(tok, proc)
model = AutoModelForImageTextToText.from_pretrained(VLM, dtype=dtype).to(device).eval()
for name, crop, size in (("full256", False, 256), ("crop256", True, 256), ("full512", False, 512), ("crop512", True, 512)):
    t0 = time.time()
    for q in qs:
        img = Image.open(root / "images" / q["image"]).convert("RGB")
        if crop:
            img = largest_crop(img, q["boxes"])
        answers[q["question_id"]][name] = vlm_answer(model, tok, img, q["question"], size)
    print(f"{name}: {time.time() - t0:.0f} s", flush=True)
del model
torch.cuda.empty_cache()

tok = AutoTokenizer.from_pretrained(LLM)
model = AutoModelForCausalLM.from_pretrained(LLM, dtype=dtype).to(device).eval()
t0 = time.time()
for q in qs:
    answers[q["question_id"]]["json_llm"] = llm_answer(model, tok, q)
print(f"json_llm: {time.time() - t0:.0f} s", flush=True)

json.dump([{**q, "out": answers[q["question_id"]]} for q in qs],
          open(here / ("answers-smoke.json" if smoke else "answers.json"), "w"), indent=1)
