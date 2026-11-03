"""Turn a vlm prompts.bin into duosight commands, for the scored questions.

prompts.bin, for each prompt: int32 n, int32 ids[n], float32 rows[m][576] for the m ids
equal to 49190. Each prompt's rows go to ROWS_DIR/NNN.bin, and the commands go to stdout:
"rows FILE" then "ask N ID...".

Usage: python3 feed.py prompts.bin ROWS_DIR > commands.txt   (plain Python, for the board)
"""
import os
import struct
import sys

src, rows_dir = sys.argv[1], sys.argv[2]
os.makedirs(rows_dir, exist_ok=True)
with open(src, "rb") as f:
    i = 0
    while True:
        head = f.read(4)
        if len(head) < 4:
            break
        n = struct.unpack("<i", head)[0]
        ids = struct.unpack(f"<{n}i", f.read(4 * n))
        m = ids.count(49190)
        path = os.path.join(rows_dir, f"{i:03d}.bin")
        with open(path, "wb") as out:
            out.write(f.read(4 * 576 * m))
        print(f"rows {path}")
        print(f"ask {n} " + " ".join(map(str, ids)))
        i += 1
