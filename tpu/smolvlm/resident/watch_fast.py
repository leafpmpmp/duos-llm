"""YOLO watches, and the resident VLM answers when a person appears.

Like camera/watch.py, but the VLM is one resident process (duosight) that loads every
model once, so an event pays no load time. A short-lived process tokenizes the questions
at start, and duosight turns the answers into text: the tokenizer would hold 23 MB here,
and memory is what limits the board.
The detector runs all the time and holds the camera; after a person stays for a few
frames, it saves the current raw frame and duosight answers about it. Each event is one
JSON line in events.jsonl.

Usage: python3 watch_fast.py [--fps 5] [--gain 940] [--frames 3] [--seconds 0]
                             [--force-after S] [--once]
  --force-after S  fire one event after S seconds, without a person (a test of the path)
  --once           stop after the first event
"""
import argparse
import json
import os
import signal
import subprocess
import time

DETECT = os.environ.get("DETECT", "/root/vlm/duos-detect-tap")
MODEL = "/root/tpu/models/yolo11n-silu-448x320.cvimodel"
SENSOR = "/dev/v4l-subdev2"
RAW = "/tmp/ask/f.raw"
VLM = ["/root/fast/duosight", "/root/fast/enc", "/root/fast/decp", "/root/fast/decp/prefix-imstart.kv"]
QUESTIONS = ["What is the person doing?", "What is the person wearing?"]
CLEAR = 15  # frames without a person before the next event can fire

ap = argparse.ArgumentParser()
ap.add_argument("--fps", type=float, default=5)
ap.add_argument("--gain", type=int, default=940)
ap.add_argument("--exposure", type=int, default=0, help="exposure in lines (0: the longest the frame rate allows)")
ap.add_argument("--focus", type=int, default=480)
ap.add_argument("--conf", type=float, default=0.6)
ap.add_argument("--frames", type=int, default=3)
ap.add_argument("--seconds", type=float, default=0, help="stop after this long (0: never)")
ap.add_argument("--force-after", type=float, default=0)
ap.add_argument("--once", action="store_true")
args = ap.parse_args()

os.makedirs("/tmp/ask", exist_ok=True)
EVENTS = "/root/fast/events"  # the canvas of each event, to check what the detector saw
os.makedirs(EVENTS, exist_ok=True)
# 3 camera buffers instead of 4 leave 3 MiB more CMA for the VLM.
env = dict(os.environ, LD_LIBRARY_PATH="/root/tpu/lib", DUOSIGHT_TIDY="1", DUOSIGHT_VOCAB="/root/fast/vocab.bin",
           DUOSIGHT_STOP="30", DETECT_BUFFERS="3")
TOKENIZE = ("import json, sys; sys.path.insert(0, '/root/vlm'); from tok import Tokenizer; "
            "t = Tokenizer('/root/vlm/w/vocab.json'); print(json.dumps([t.prompt(q) for q in sys.argv[1:]]))")
prompts = json.loads(subprocess.run(["python3", "-c", TOKENIZE] + QUESTIONS, capture_output=True, text=True, check=True).stdout)

# The detector first: it needs 14 MiB of CMA, and the VLM leaves only a few. The VLM
# then tidies memory before each model, so the detector's buffers are no obstacle.
log = open("/tmp/ask/detect.log", "w")
det = subprocess.Popen([DETECT, "-m", MODEL, "-r", str(args.fps), "-p", "-c", str(args.conf), "-f", str(args.focus)],
                       stdout=subprocess.PIPE, stderr=log, text=True, env=env, bufsize=1)
time.sleep(3)
t0 = time.monotonic()
vlm = subprocess.Popen(VLM, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open("/tmp/ask/duosight.log", "w"),
                       text=True, env=env, bufsize=1)
vlm.stdin.write("rgb /dev/null\n")  # answers "err" once the models are loaded
vlm.stdin.flush()
if not vlm.stdout.readline():
    det.send_signal(signal.SIGINT)
    det.wait(timeout=10)
    raise SystemExit("the VLM did not load: see /tmp/ask/duosight.log")
print(f"VLM loaded in {time.monotonic() - t0:.1f} s", flush=True)


def control(**values):
    pairs = ",".join(f"{k}={v}" for k, v in values.items())
    subprocess.run(["v4l2-ctl", "-d", SENSOR, "-c", pairs], check=True)


def longest_exposure():
    text = subprocess.run(["v4l2-ctl", "-d", SENSOR, "-l"], capture_output=True, text=True).stdout
    line = next(x for x in text.splitlines() if " exposure " in x)
    return int(line.split("max=")[1].split()[0])


def ask(event):
    """Encode the saved frame and answer every question; fills event."""
    t = time.monotonic()
    event["canvas"] = f"{EVENTS}/{event['time'].replace(':', '')}.rgb"
    vlm.stdin.write(f"raw {RAW} 1728 1296 {event['canvas']}\n")
    vlm.stdin.flush()
    reply = vlm.stdout.readline().split()
    if reply[0] != "image":
        raise RuntimeError(" ".join(reply))
    event["image_ms"] = dict(zip(("canvas", "patch", "blocks", "connector"), map(float, reply[1:])))
    event["answers"], event["answer_ms"] = [], []
    for ids in prompts:
        vlm.stdin.write(f"ask {len(ids)} " + " ".join(map(str, ids)) + "\n")
        vlm.stdin.flush()
        reply = vlm.stdout.readline().split()
        if reply[0] != "ans":
            raise RuntimeError(" ".join(reply))
        event["answers"].append(vlm.stdout.readline().rstrip("\n")[4:])  # the "txt" line
        event["answer_ms"].append({"prefill": float(reply[1]), "decode": float(reply[2]), "reused": int(reply[3])})
        if len(event["answers"]) == 1:
            event["seconds_to_first_answer"] = round(time.monotonic() - event["t0"], 2)
    event["vlm_s"] = round(time.monotonic() - t, 2)


streak = clear = frames = 0
armed = True
forced = False
start = time.monotonic()
first_seen = None
events = 0
try:
    for line in det.stdout:
        now = time.monotonic()
        if frames == 0:
            control(exposure=args.exposure or longest_exposure(), analogue_gain=args.gain)
            print("detector running", flush=True)
        frames += 1
        boxes = json.loads(line)["boxes"]
        if boxes:
            if streak == 0:
                first_seen = now
            streak += 1
            clear = 0
        else:
            streak = 0
            clear += 1
            armed = armed or clear >= CLEAR

        force = args.force_after and not forced and now - start >= args.force_after
        if (armed and streak >= args.frames) or force:
            forced = forced or bool(force)
            armed = False
            seen = first_seen if streak else now
            if os.path.exists(RAW):
                os.unlink(RAW)
            det.send_signal(signal.SIGUSR1)
            while not os.path.exists(RAW):  # the detector renames it into place when complete
                time.sleep(0.005)
            event = {"time": time.strftime("%H:%M:%S"), "forced": bool(force), "people": len(boxes), "boxes": boxes,
                     "frame_wait_s": round(time.monotonic() - now, 3), "t0": now}
            ask(event)
            event["seconds_to_answers"] = round(time.monotonic() - now, 2)
            event["seconds_seen_to_answers"] = round(time.monotonic() - seen, 2)
            del event["t0"]
            with open("/root/vlm/events.jsonl", "a") as f:
                f.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)
            events += 1
            if args.once:
                break
        if args.seconds and now - start >= args.seconds:
            break
finally:
    det.send_signal(signal.SIGINT)
    det.wait(timeout=10)
    vlm.stdin.close()
    vlm.wait(timeout=10)
    control(vertical_blanking=40, analogue_gain=0, exposure=1288)
    print(f"stopped after {frames} frames, {events} events", flush=True)
