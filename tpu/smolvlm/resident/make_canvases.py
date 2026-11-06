"""Write the 150 scored images as 512x512 canvases, and duosight commands for them.

Each canvas is the scored tests' input: long side to 512 (bicubic), black padding. The
commands put "rgb FILE" before each prompt, so the board runs every step itself. The
prompt ids come from a commands file of feed.py, in the same order.

Usage: python make_canvases.py OUT_DIR FEED_COMMANDS BOARD_DIR > commands.txt
"""
import json
import pathlib
import sys

from PIL import Image

here = pathlib.Path(__file__).resolve().parent
out, feed, board = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
out.mkdir(parents=True, exist_ok=True)
rows = json.load(open(here.parent / "encoder" / "answers-cpu-final.json"))
asks = [line.strip() for line in open(feed) if line.startswith("ask ")]
assert len(asks) == len(rows), (len(asks), len(rows))
for i, (r, ask) in enumerate(zip(rows, asks)):
    img = Image.open(pathlib.Path("D:/duosight-eval/images") / r["image"]).convert("RGB")
    s = 512 / max(img.size)
    img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BICUBIC)
    canvas = Image.new("RGB", (512, 512), (0, 0, 0))
    canvas.paste(img, (0, 0))
    (out / f"{i:03d}.rgb").write_bytes(canvas.tobytes())
    print(f"rgb {board}/{i:03d}.rgb")
    print(ask)
