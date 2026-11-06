"""Second exploratory run. Not part of the rule in PREDICTIONS.md.

The first run showed that 16 image tokens lose far more than 256 px of detail does.
This run keeps the small, cheap encoder input and resamples its patch features to
the 32 x 32 grid the connector was trained on, so the decoder still sees 64 tokens.

Usage: python explore2.py D:/duosight-eval [questions.json-like input]
Writes answers-explore2.json next to this script, or NAME-up.json for another input.
"""
import json
import pathlib
import sys
import time

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

VLM = "HuggingFaceTB/SmolVLM-256M-Instruct"
ASK = "Answer using a single word or phrase."

root = pathlib.Path(sys.argv[1])
here = pathlib.Path(__file__).parent
source = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else here / "answers-explore.json"
target = here / "answers-explore2.json" if len(sys.argv) <= 2 else source.with_name(source.stem + "-up.json")
rows = json.load(open(source))
device, dtype = "cuda", torch.bfloat16
MODE = {"name": None}


def resample(module, args):
    """Connector pre-hook: bring the patch grid to 32 x 32 before pixel shuffle."""
    x = args[0]
    if MODE["name"] is None or x.shape[1] == 1024:
        return None
    b, n, c = x.shape
    g = int(n ** 0.5)
    y = x.float().transpose(1, 2).reshape(b, c, g, g)
    kw = {"align_corners": False} if MODE["name"] == "bilinear" else {}
    y = F.interpolate(y, size=(32, 32), mode=MODE["name"], **kw)
    return (y.flatten(2).transpose(1, 2).contiguous().to(x.dtype),) + tuple(args[1:])


def fit(img, size):
    s = size / max(img.size)
    w, h = max(1, round(img.width * s)), max(1, round(img.height * s))
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(img.resize((w, h), Image.BICUBIC), (0, 0))
    x = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8).float()
    x = (x.view(size, size, 3).permute(2, 0, 1) / 255 - 0.5) / 0.5
    return x[None, None].to(device, dtype), torch.ones(1, 1, size, size, dtype=torch.long, device=device)


@torch.no_grad()
def answer(model, tok, img, question, size):
    pixels, mask = fit(img, size)
    image = "<fake_token_around_image><global-img>" + "<image>" * 64 + "<fake_token_around_image>"
    text = f"<|im_start|>User:{image}{question} {ASK}<end_of_utterance>\nAssistant:"
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pixels,
                         pixel_attention_mask=mask, max_new_tokens=24, do_sample=False)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


proc = AutoProcessor.from_pretrained(VLM)
tok = proc.tokenizer
model = AutoModelForImageTextToText.from_pretrained(VLM, dtype=dtype).to(device).eval()
connector = model.model.connector
connector.register_forward_pre_hook(resample)

sizes = [("up256_bilinear", 256, "bilinear"), ("up256_nearest", 256, "nearest"),
         ("up320_bilinear", 320, "bilinear"), ("up384_bilinear", 384, "bilinear"), ("native512", 512, None)]
for name, size, mode in sizes:
    MODE["name"] = mode
    t0 = time.time()
    for r in rows:
        img = Image.open(root / "images" / r["image"]).convert("RGB")
        r["out"][name] = answer(model, tok, img, r["question"], size)
    print(f"{name}: {time.time() - t0:.0f} s", flush=True)

json.dump(rows, open(target, "w"), indent=1)
