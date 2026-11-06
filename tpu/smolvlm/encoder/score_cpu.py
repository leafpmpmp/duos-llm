"""Answer the 150 unseen questions from board features, on the CPU.

Uses about 3 GB of commit, against 7 GB or more for torch with CUDA. The vision model is
skipped: the features go straight in as image_hidden_states, after the float LayerNorm
and connector.

Usage: python score_cpu.py FEATURES.bin OUT.json [THREADS] [LIMIT]
"""
import glob
import json
import pathlib
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

VLM = "HuggingFaceTB/SmolVLM-256M-Instruct"
ASK = "Answer using a single word or phrase."
N, D = 1024, 768
path, target = sys.argv[1], sys.argv[2]
torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 4)
limit = int(sys.argv[4]) if len(sys.argv) > 4 else None
here = pathlib.Path(__file__).parent
rows = json.load(open(here.parent / "quality" / "answers-holdout-up.json"))
snapshot = glob.glob("D:/duosight-eval/hf/hub/models--HuggingFaceTB--SmolVLM-256M-Instruct/snapshots/*")[0]

proc = AutoProcessor.from_pretrained(snapshot)
tok = proc.tokenizer
model = AutoModelForImageTextToText.from_pretrained(snapshot, dtype=torch.float32).eval()
features = np.memmap(path, dtype=np.float32, mode="r", shape=(len(rows), D, N))
image = "<fake_token_around_image><global-img>" + "<image>" * 64 + "<fake_token_around_image>"
image_id = tok.convert_tokens_to_ids("<image>")
t0 = time.time()
with torch.no_grad():
    for i, r in enumerate(rows[:limit]):
        h = torch.from_numpy(np.array(features[i])).T[None]
        h = model.model.connector(model.model.vision_model.post_layernorm(h))
        ids = tok(f"<|im_start|>User:{image}{r['question']} {ASK}<end_of_utterance>\nAssistant:",
                  return_tensors="pt", add_special_tokens=False).input_ids
        embeds = model.get_input_embeddings()(ids)
        embeds[ids == image_id] = h.reshape(-1, h.shape[-1])
        out = model.generate(inputs_embeds=embeds, attention_mask=torch.ones_like(ids),
                             max_new_tokens=24, do_sample=False)
        r["out"]["tpu512"] = tok.decode(out[0], skip_special_tokens=True).strip()
        if i % 25 == 24:
            print(i + 1, f"{time.time() - t0:.0f} s", flush=True)
json.dump(rows[:limit], open(target, "w"), indent=1)
print("answered", len(rows[:limit]), f"in {time.time() - t0:.0f} s")
