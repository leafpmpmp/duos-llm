"""Run the scored questions through duosight's bg path and compare with rgb + ask.

Each "rgb FILE" line of the command file becomes "bg rgb FILE", and its question goes only
after the "bg" reply, so it runs from the image part in the cache. The answer ids must be
the same as in REFERENCE, a reply file of the same commands sent as they are.

Usage: python3 bg_score.py DUOSIGHT COMMANDS REFERENCE OUT   (plain Python, on the board)
OUT gets the replies, for score_fast.py.
"""
import os
import subprocess
import sys

duosight, commands, reference, out = sys.argv[1:5]
env = dict(os.environ, LD_LIBRARY_PATH="/root/tpu/lib", DUOSIGHT_TIDY="1", DUOSIGHT_STOP="30")
vlm = subprocess.Popen([duosight, "/root/fast/enc", "/root/fast/decp", "/root/fast/decp/prefix-imstart.kv"],
                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env, bufsize=1)
ref = [line.split()[4:] for line in open(reference) if line.startswith("ans")]
got = []
with open(out, "w") as f:
    for line in open(commands):
        word = line.split()[0]
        vlm.stdin.write(("bg " + line) if word == "rgb" else line)
        vlm.stdin.flush()
        reply = vlm.stdout.readline()
        f.write(reply)
        if reply.split()[0] not in ("bg", "ans"):
            raise SystemExit(reply)
        if word == "ask":
            got.append(reply.split())
vlm.stdin.close()
vlm.wait(timeout=30)
same = sum(g[4:] == r for g, r in zip(got, ref))
reused = sorted(set(g[3] for g in got))
print(f"{len(got)} answers, {same} the same ids as {reference}; prompt tokens kept: {' '.join(reused)}")
for i, (g, r) in enumerate(zip(got, ref)):
    if g[4:] != r:
        print(f"  {i:03d}: {' '.join(g[4:])} against {' '.join(r)}")
