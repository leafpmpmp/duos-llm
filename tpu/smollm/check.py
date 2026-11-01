# SPDX-License-Identifier: MIT
# Check model.py against transformers' LlamaForCausalLM, then generate greedily.
# Usage: python check.py HF_DIR
import sys

import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

from model import Decoder

path = sys.argv[1]
tok = Tokenizer.from_file(f"{path}/tokenizer.json")
ids = tok.encode("The capital of France is").ids

dec = Decoder(path, S=256)
mine = torch.stack([dec.step(t) for t in ids])

ref = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
with torch.no_grad():
    theirs = ref(torch.tensor([ids])).logits[0]
err = (mine - theirs).abs().max().item()
cos = torch.nn.functional.cosine_similarity(mine, theirs, dim=-1)
print(f"tokens {len(ids)}: max |logit diff| {err:.2e}, min cos {cos.min().item():.6f}")

out = list(ids)
nxt = int(mine[-1].argmax())
for _ in range(24):
    out.append(nxt)
    nxt = int(dec.step(nxt).argmax())
print(repr(tok.decode(out)))
