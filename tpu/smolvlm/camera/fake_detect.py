"""Stand in for the detector, to test when watch.py fires without a person in the room.

Prints a fixed script of frames with and without a person. SIGUSR1 writes a frame file.
The expected result is in the comments. Run it with:
  DETECT=fake_detect.py ASK=fake_ask.sh python3 watch.py --frames 3 --seconds 8
"""
import json
import signal
import sys
import time

BOX = [[100.0, 100.0, 300.0, 600.0, 0.8]]
SCRIPT = [(0, 20),    # empty room
          (1, 1),     # one stray frame: must not fire
          (0, 20),
          (1, 30),    # a person stays: fires once, on the third frame
          (0, 5),     # a short gap
          (1, 20),    # the same person: must not fire again
          (0, 20),    # the person leaves
          (1, 10),    # someone new: fires a second time
          (0, 30)]


def save(sig, frame):
    open("/tmp/ask/f.raw", "wb").write(b"x")


signal.signal(signal.SIGUSR1, save)
signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
seq = 0
for person, count in SCRIPT:
    for _ in range(count):
        print(json.dumps({"seq": seq, "latency_ms": 1.0, "frame": [1728, 1296], "boxes": BOX if person else []}), flush=True)
        seq += 1
        time.sleep(0.05)
