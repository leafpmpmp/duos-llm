"""Check duosight's bg command: questions between the steps of a frame.

1. Image rows of frame A encoded at once (rgb) and in steps with questions in between (bg)
   must be the same bytes.
2. Questions during the steps get the last frame, B, and every answer is the one of a
   prefill from the prefix (rgb, then the question).
3. The time from a question to its answer, during the steps and after them.

Usage: python3 bg_check.py DUOSIGHT A.rgb B.rgb [QUESTION]   (plain Python, on the board)
"""
import os
import queue
import statistics
import subprocess
import sys
import threading
import time

DUOSIGHT, A, B = sys.argv[1:4]
QUESTION = sys.argv[4] if len(sys.argv) > 4 else "What is the person doing?"
env = dict(os.environ, LD_LIBRARY_PATH="/root/tpu/lib", DUOSIGHT_TIDY="1", DUOSIGHT_VOCAB="/root/fast/vocab.bin",
           DUOSIGHT_STOP="30", DUOSIGHT_BPE="/root/fast/bpe.bin")
vlm = subprocess.Popen([DUOSIGHT, "/root/fast/enc", "/root/fast/decp", "/root/fast/decp/prefix-imstart.kv"],
                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env, bufsize=1)
replies = queue.Queue()
threading.Thread(target=lambda: [replies.put((time.monotonic(), line.rstrip("\n"))) for line in vlm.stdout],
                 daemon=True).start()


def send(line):
    vlm.stdin.write(line + "\n")
    vlm.stdin.flush()
    return time.monotonic()


def reply(word):
    t, line = replies.get(timeout=60)
    assert line.split()[0] == word, (word, line)
    return t, line


def say():
    """Ask and wait: (seconds, answer text, ans line)."""
    t0 = send("say " + QUESTION)
    _, ans = reply("ans")
    t, txt = reply("txt")
    return t - t0, txt[4:], ans


send("rgb /dev/null")
reply("err")
send(f"rgb {A}")
reply("image")
send("dump /tmp/bg_a.bin")
reply("dumped")
send(f"rgb {B}")
reply("image")
_, ref_b, line = say()
print(f"B at once: {ref_b!r} ({line.split()[1]} ms prefill)")
_, again_b, _ = say()
print(f"B again (K/V kept): {again_b!r}")

# A in steps, a question every 0.25 s until the frame is ready.
t0 = send(f"bg rgb {A}")
during, done = [], None
while done is None:
    t1 = send("say " + QUESTION)
    while True:
        t, line = replies.get(timeout=60)
        word = line.split()[0]
        if word == "bg":
            done = (t - t0, line)
        elif word == "txt":
            during.append((t - t1, line[4:], done is not None))  # after: answered once A was ready
            break
    if done is None:
        time.sleep(0.25)
print(f"bg reply after {done[0]:.2f} s: {done[1]}")
before = [a for _, a, after in during if not after]
print(f"{len(during)} questions during the steps, {len(before)} answered before A was ready: answers "
      f"{sorted(set(before))}, same as B at once: {all(a == ref_b for a in before)}")
print("  seconds to answer: " + " ".join(f"{s:.2f}{'*' if after else ''}" for s, _, after in during)
      + "   (* after A was ready)")
send("dump /tmp/bg_b.bin")
reply("dumped")
same = open("/tmp/bg_a.bin", "rb").read() == open("/tmp/bg_b.bin", "rb").read()
print(f"image rows of A, at once and in steps: {'same bytes' if same else 'DIFFERENT'}")

after = [say() for _ in range(5)]
print(f"A from the cache: {after[0][1]!r}; seconds to answer: " + " ".join(f"{s:.2f}" for s, _, _ in after))
print("  ans lines: " + " | ".join(a for _, _, a in after[:2]))
send(f"rgb {A}")
reply("image")
_, ref_a, line = say()
print(f"A at once: {ref_a!r} ({line.split()[1]} ms prefill), same as from the cache: "
      f"{all(a == ref_a for _, a, _ in after)}")
times = [s for s, _, _ in during] + [s for s, _, _ in after]
print(f"all {len(times)} questions: median {statistics.median(times):.2f} s, max {max(times):.2f} s")
vlm.stdin.close()
vlm.wait(timeout=30)
