"""Turn the encoder's board output into prompts for vlm.

Only numpy and tokenizers are needed, so this stays small in memory.

Usage: python make_prompts.py FEATURES.bin TOKENIZER.json OUT.bin
"""
import json
import pathlib
import sys

import numpy as np
from tokenizers import Tokenizer

from vlm_common import IMAGE_ID, Connector, prompt

here = pathlib.Path(__file__).resolve().parent
rows = json.load(open(here.parent / "encoder" / "answers-cpu-final.json"))
features = np.memmap(sys.argv[1], dtype=np.float32, mode="r", shape=(len(rows), 768, 1024))
tok = Tokenizer.from_file(sys.argv[2])
connector = Connector()
longest = 0
with open(sys.argv[3], "wb") as f:
    for i, r in enumerate(rows):
        ids = np.array(tok.encode(prompt(r["question"]), add_special_tokens=False).ids, dtype=np.int32)
        if (ids == IMAGE_ID).sum() != 64 or ids[0] != 1:
            sys.exit(f"bad prompt {i}")
        longest = max(longest, len(ids))
        np.array([len(ids)], dtype=np.int32).tofile(f)
        ids.tofile(f)
        connector(np.array(features[i]).T).astype(np.float32).tofile(f)
print(f"{len(rows)} prompts, longest {longest} tokens")
