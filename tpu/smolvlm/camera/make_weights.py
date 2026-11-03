"""Write the weights and tokenizer files for the board's CPU parts, and test data for them.

Usage: python make_weights.py OUT_DIR TOKENIZER.json IN512.bin OUT512.bin
Writes float32 weights and vocab.json. Then it checks tok.py against tokenizers on the
question files, and writes check_* files that compare the board's pre and post with numpy.
"""
import json
import pathlib
import sys

import numpy as np
from PIL import Image
from tokenizers import Tokenizer as Reference

here = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "decoder"))
from vlm_common import Connector, load_bf16, prompt  # noqa: E402

sys.path.insert(0, str(here))
from tok import Tokenizer  # noqa: E402

out, tokenizer, in512, out512 = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
out.mkdir(parents=True, exist_ok=True)
SIZE, GRID, D = 512, 32, 768

patch, bias, pos, lw, lb, proj = load_bf16(["model.vision_model.embeddings.patch_embedding.weight",
                                            "model.vision_model.embeddings.patch_embedding.bias",
                                            "model.vision_model.embeddings.position_embedding.weight",
                                            "model.vision_model.post_layernorm.weight",
                                            "model.vision_model.post_layernorm.bias",
                                            "model.connector.modality_projection.proj.weight"])
for name, x in (("patch", patch), ("patch_bias", bias), ("pos", pos), ("ln_w", lw), ("ln_b", lb), ("proj", proj)):
    x.astype(np.float32).tofile(out / f"{name}.f32")

raw = json.load(open(tokenizer, encoding="utf-8"))["model"]
json.dump({"vocab": raw["vocab"], "merges": [" ".join(m) for m in raw["merges"]]},
          open(out / "vocab.json", "w", encoding="utf-8"), ensure_ascii=False)

# tok.py against tokenizers.
mine, ref = Tokenizer(out / "vocab.json"), Reference.from_file(tokenizer)
root = here.parent
questions = [r["question"] for r in json.load(open(root / "quality" / "questions.json"))]
questions += [r["question"] for r in json.load(open(root / "encoder" / "answers-cpu-final.json"))]
questions += ["What's on the shelf?", "Is there a 2nd door, or only 1?", "How many people are in the room?", "what_is that?"]
bad = [q for q in questions if mine.prompt(q) != ref.encode(prompt(q), add_special_tokens=False).ids]
print(f"tok.py: {len(questions) - len(bad)} of {len(questions)} prompts match", bad[:3])
ids = ref.encode("Baseball stadium, 2 men's hats.", add_special_tokens=False).ids
print("decode:", repr(mine.decode(ids)))

# Test data: the first scored image, through the same canvas as the float tests.
rows = json.load(open(root / "encoder" / "answers-cpu-final.json"))
img = Image.open(pathlib.Path("D:/duosight-eval/images") / rows[0]["image"]).convert("RGB")
s = SIZE / max(img.size)
img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BICUBIC)
canvas = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
canvas.paste(img, (0, 0))
canvas.save(out / "check.png")
open(out / "check.rgb", "wb").write(canvas.tobytes())
x = (np.asarray(canvas, dtype=np.float32) / 255 - 0.5) / 0.5
blocks = x.reshape(GRID, 16, GRID, 16, 3).transpose(0, 2, 4, 1, 3).reshape(1024, -1)
want = (blocks @ patch.reshape(D, -1).T + bias + pos).T.astype(np.float32)
want.tofile(out / "check_in.bin")
board = np.memmap(in512, dtype=np.float32, mode="r", shape=(150, D, 1024))
print(f"numpy embedding vs in512.bin: cosine {np.dot(want.ravel(), board[0].ravel()) / np.linalg.norm(want) / np.linalg.norm(board[0]):.5f}")
last = np.array(np.memmap(out512, dtype=np.float32, mode="r", shape=(150, D, 1024))[0])
last.tofile(out / "check_last.bin")
Connector()(last.T).astype(np.float32).tofile(out / "check_rows.bin")
print("wrote", out)
