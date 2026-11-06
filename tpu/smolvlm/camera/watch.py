"""YOLO watches, and the VLM answers when a person appears.

The detector runs all the time and holds the camera. After a person stays for a few
frames, this asks it for the current raw frame and runs ask.sh on that frame. The
detector keeps running during the answer. Each event is one JSON line in events.jsonl.

Usage: python3 watch.py [--fps 5] [--gain 940] [--frames 3] [--seconds 0]
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
ASK = os.environ.get("ASK", "/root/vlm/ask.sh")
MODEL ="/root/tpu/models/yolo11n-silu-448x320.cvimodel"
SENSOR = "/dev/v4l-subdev2"
RAW = "/tmp/ask/f.raw"
QUESTIONS = ["What is the person doing?", "What is the person wearing?"]
CLEAR = 15  # frames without a person before the next event can fire

ap = argparse.ArgumentParser()
ap.add_argument("--fps", type=float, default=5)
ap.add_argument("--gain", type=int, default=940)
ap.add_argument("--focus", type=int, default=480)
ap.add_argument("--conf", type=float, default=0.6)
ap.add_argument("--frames", type=int, default=3)
ap.add_argument("--seconds", type=float, default=0, help="stop after this long (0: never)")
ap.add_argument("--force-after", type=float, default=0)
ap.add_argument("--once", action="store_true")
args = ap.parse_args()

os.makedirs("/tmp/ask", exist_ok=True)
env = dict(os.environ, LD_LIBRARY_PATH="/root/tpu/lib")
log = open("/tmp/ask/detect.log", "w")
det = subprocess.Popen([DETECT, "-m", MODEL, "-r", str(args.fps), "-p", "-c", str(args.conf), "-f", str(args.focus)],
                       stdout=subprocess.PIPE, stderr=log, text=True, env=env, bufsize=1)


def control(**values):
    pairs = ",".join(f"{k}={v}" for k, v in values.items())
    subprocess.run(["v4l2-ctl", "-d", SENSOR, "-c", pairs], check=True)


def longest_exposure():
    text = subprocess.run(["v4l2-ctl", "-d", SENSOR, "-l"], capture_output=True, text=True).stdout
    line = next(x for x in text.splitlines() if " exposure " in x)
    return int(line.split("max=")[1].split()[0])


def event_line(event):
    with open("/root/vlm/events.jsonl", "a") as f:
        f.write(json.dumps(event) + "\n")
    print(json.dumps(event), flush=True)


job = event = None
streak = clear = frames = 0
armed = True
forced = False
start = time.monotonic()
first_seen = None
try:
    for line in det.stdout:
        now = time.monotonic()
        if frames == 0:
            control(exposure=longest_exposure(), analogue_gain=args.gain)
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

        if job and job.poll() is not None:
            out = job.stdout.read()
            event["seconds_to_answer"] = round(time.monotonic() - event["t0"], 2)
            event["seconds_seen_to_answer"] = round(time.monotonic() - event["seen"], 2)
            event["answers"] = [x[3:].rsplit("   (", 1)[0] for x in out.splitlines() if x.startswith("A: ")]
            event["steps"] = [x for x in out.splitlines() if " s" in x and not x.startswith(("Q:", "A:", "loaded", "1 images"))]
            del event["t0"], event["seen"]
            event_line(event)
            job = None
            if args.once:
                break

        force = args.force_after and not forced and now - start >= args.force_after
        if not job and ((armed and streak >= args.frames) or force):
            forced = forced or bool(force)
            armed = False
            seen = first_seen if streak else now
            if os.path.exists(RAW):
                os.unlink(RAW)
            det.send_signal(signal.SIGUSR1)
            while not os.path.exists(RAW):
                time.sleep(0.005)
            event = {"time": time.strftime("%H:%M:%S"), "forced": bool(force), "people": len(boxes), "boxes": boxes,
                     "frame_wait_s": round(time.monotonic() - now, 3), "t0": now, "seen": seen}
            job = subprocess.Popen(["bash", ASK] + QUESTIONS, stdout=subprocess.PIPE, text=True,
                                   env=dict(os.environ, RAW=RAW))
        if args.seconds and now - start >= args.seconds and not job:
            break
finally:
    det.send_signal(signal.SIGINT)
    det.wait(timeout=10)
    control(vertical_blanking=40, analogue_gain=0, exposure=1288)
    print(f"stopped after {frames} frames", flush=True)
