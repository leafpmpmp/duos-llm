"""Answer questions about one camera canvas in full float, as an upper bound for the board.

Usage: python float_frame.py CANVAS.rgb QUESTION...
"""
import pathlib
import sys

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "decoder"))
from vlm_common import IMAGE_ID, MAX_NEW, SNAPSHOT, prompt  # noqa: E402

torch.set_num_threads(4)
SIZE = 512
proc = AutoProcessor.from_pretrained(SNAPSHOT)
tok = proc.tokenizer
model = AutoModelForImageTextToText.from_pretrained(SNAPSHOT, dtype=torch.float32, attn_implementation="eager").eval()
x = np.fromfile(sys.argv[1], dtype=np.uint8).reshape(SIZE, SIZE, 3).astype(np.float32)
x = torch.from_numpy(x).permute(2, 0, 1)[None] / 255
with torch.no_grad():
    y = model.model.vision_model(pixel_values=(x - 0.5) / 0.5,
                                 patch_attention_mask=torch.ones(1, SIZE // 16, SIZE // 16, dtype=torch.bool)).last_hidden_state
    image = model.model.connector(y)
    for q in sys.argv[2:]:
        ids = tok(prompt(q), return_tensors="pt", add_special_tokens=False).input_ids
        embeds = model.get_input_embeddings()(ids)
        embeds[ids == IMAGE_ID] = image.reshape(-1, image.shape[-1])
        out = model.generate(inputs_embeds=embeds, attention_mask=torch.ones_like(ids), max_new_tokens=MAX_NEW, do_sample=False)
        print(f"{q} -> {tok.decode(out[0], skip_special_tokens=True).strip()}", flush=True)
