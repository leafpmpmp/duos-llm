"""DuoSight live: YOLO watches, the VLM keeps the newest frame ready, and a question from a
browser gets its answer in under a second.

The detector runs all the time and holds the camera. live.py asks it for the current raw
frame, and duosight encodes the frame in steps (bg), answering questions about the last
frame between the steps, then keeps the image part of the prompt in its K/V cache. A
question then runs only its own tokens. The answer is about the newest frame duosight has
finished, so the page says how old that frame is.

A new frame costs the detector: with frames one after another (--refresh always) it kept
9.6 of 15 fps. So by default YOLO decides when the VLM needs a new frame (--refresh
change): at once when the person count, stable over --frames detector frames, differs
from the frame's, or a person moved by more than --move of the frame width; otherwise when
the frame is older than --max-age with a person in view, or --idle-age without one.

A page on --port shows that frame with the detector's boxes, the frame's age and a
question box. Each question and each frame is one JSON line in /root/fast/live.jsonl.

Usage: python3 live.py [--port 8080] [--fps 15] [--gain 940] [--exposure 0]
                       [--refresh change|always] [--max-age 10] [--idle-age 60]
                       [--seconds 0] [--bench N]
  --bench N    ask N questions at random times (0.5 to 3 s apart), print the times, stop
"""
import argparse
import json
import math
import os
import random
import signal
import statistics
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DETECT = os.environ.get("DETECT", "/root/vlm/duos-detect-tap")
MODEL = "/root/tpu/models/yolo11n-silu-448x320.cvimodel"
SENSOR = "/dev/v4l-subdev2"
RAW = "/tmp/ask/f.raw"
W, H, SIZE = 1728, 1296, 512
ROWS = round(H * SIZE / W)  # canvas rows above the black padding
VLM = [os.environ.get("DUOSIGHT", "/root/fast/duosight"), "/root/fast/enc", "/root/fast/decp",
       "/root/fast/decp/prefix-imstart.kv"]
LOG = "/root/fast/live.jsonl"
QUESTIONS = ["What is the person doing?", "What is the person wearing?", "What is the person holding?",
             "Is there a person?", "Where is this?"]

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--fps", type=float, default=15)
ap.add_argument("--gain", type=int, default=940)
ap.add_argument("--exposure", type=int, default=0, help="exposure in lines (0: the longest the frame rate allows)")
ap.add_argument("--focus", type=int, default=480)
ap.add_argument("--conf", type=float, default=0.6)
ap.add_argument("--refresh", choices=("change", "always"), default="change")
ap.add_argument("--frames", type=int, default=3)
ap.add_argument("--move", type=float, default=0.15)
ap.add_argument("--max-age", type=float, default=10)
ap.add_argument("--idle-age", type=float, default=60)
ap.add_argument("--seconds", type=float, default=0, help="stop after this long (0: never)")
ap.add_argument("--bench", type=int, default=0)
args = ap.parse_args()

os.makedirs("/tmp/ask", exist_ok=True)
os.makedirs("/tmp/live", exist_ok=True)  # tmpfs: the canvas of the newest frames
env = dict(os.environ, LD_LIBRARY_PATH="/root/tpu/lib", DUOSIGHT_TIDY="1", DUOSIGHT_VOCAB="/root/fast/vocab.bin",
           DUOSIGHT_BPE="/root/fast/bpe.bin", DUOSIGHT_STOP="30", DETECT_BUFFERS="3")

lock = threading.Lock()  # duosight's stdin and the list of questions in flight
pending = []             # questions sent to duosight, oldest first
cur = {"frame": None, "next": None, "busy": False}  # the frame in duosight's cache, the one in steps
det = {"frames": 0, "boxes": [], "history": [], "times": [], "worst_ms": 0.0}
ready = threading.Event()  # duosight finished a frame
stop = threading.Event()
reasons = {}  # frames by what started them


def log(record):
    with open(LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


def control(**values):
    pairs = ",".join(f"{k}={v}" for k, v in values.items())
    subprocess.run(["v4l2-ctl", "-d", SENSOR, "-c", pairs], check=True)


def longest_exposure():
    text = subprocess.run(["v4l2-ctl", "-d", SENSOR, "-l"], capture_output=True, text=True).stdout
    line = next(x for x in text.splitlines() if " exposure " in x)
    return int(line.split("max=")[1].split()[0])


def read_detector():
    for line in detector.stdout:
        d = json.loads(line)
        now = time.monotonic()
        if det["frames"] == 0:
            control(exposure=args.exposure or longest_exposure(), analogue_gain=args.gain)
        det["frames"] += 1
        det["boxes"] = d["boxes"]
        det["history"] = det["history"][-(args.frames - 1):] + [d["boxes"]] if args.frames > 1 else [d["boxes"]]
        if det["frames"] > 10:  # after the start-up frames
            det["worst_ms"] = max(det["worst_ms"], d.get("latency_ms", 0))
        det["times"] = [t for t in det["times"] if now - t < 2] + [now]


def read_vlm():
    """Replies come in the order of the commands, so the frame in the cache when an "ans"
    line comes is the frame that answer is about."""
    for line in vlm.stdout:
        line = line.rstrip("\n")
        word = line.split(" ", 1)[0]
        now = time.monotonic()
        if word == "bg":
            f = cur["next"]
            f["ms"] = dict(zip(("canvas", "patch", "blocks", "connector", "prefill"), map(float, line.split()[1:])))
            f["ready"] = now
            cur["frame"], cur["busy"] = f, False
            ready.set()
        elif word == "err" and line.startswith("err cannot read"):  # the frame, not a question
            cur["busy"] = False
            ready.set()
        elif word == "ans":
            with lock:
                pending[0]["ans"], pending[0]["frame"] = line.split(), cur["frame"]
        elif word in ("txt", "err"):
            with lock:
                q = pending.pop(0)
            q["text" if word == "txt" else "error"] = line[4:]
            q["done"] = now
            q["event"].set()


def scene_changed(f):
    """YOLO's view against frame f: the person count, stable over the last --frames
    detector frames, differs, or a person is more than --move of the width from every
    person in f."""
    hist = det["history"]
    if len(hist) < args.frames or len({len(b) for b in hist}) > 1:
        return False
    if len(hist[-1]) != len(f["boxes"]):
        return True

    def centers(boxes):
        return [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]

    old = centers(f["boxes"])
    return any(min(math.dist(c, o) for o in old) > args.move * W for c in centers(hist[-1]))


def refresh_reason():
    f = cur["frame"]
    if f is None:
        return "first"
    if args.refresh == "always":
        return "always"
    if scene_changed(f):
        return "change"
    if time.monotonic() - f["captured"] > (args.max_age if f["boxes"] or det["boxes"] else args.idle_age):
        return "age"
    return None


def refresh():
    n = 0
    while not stop.is_set():
        reason = refresh_reason()
        if not reason:
            stop.wait(0.1)
            continue
        if os.path.exists(RAW):
            os.unlink(RAW)
        detector.send_signal(signal.SIGUSR1)
        while not os.path.exists(RAW):  # the detector renames it into place when complete
            if stop.wait(0.005):
                return
        n += 1
        cur["next"] = {"id": n, "captured": time.monotonic(), "boxes": det["boxes"], "canvas": f"/tmp/live/c{n % 2}.rgb"}
        ready.clear()
        with lock:
            cur["busy"] = True
            vlm.stdin.write(f"bg raw {RAW} {W} {H} {cur['next']['canvas']}\n")
            vlm.stdin.flush()
        ready.wait()
        f = cur["frame"]
        if f and f["id"] == n:
            reasons[reason] = reasons.get(reason, 0) + 1
            log({"type": "frame", "time": time.strftime("%H:%M:%S"), "id": n, "reason": reason,
                 "people": len(f["boxes"]), "seconds": round(f["ready"] - f["captured"], 2), "ms": f["ms"]})


def ask(question):
    q = {"event": threading.Event(), "t0": time.monotonic()}
    with lock:
        during = cur["busy"]
        pending.append(q)
        vlm.stdin.write("say " + question + "\n")
        vlm.stdin.flush()
    q["event"].wait(60)
    f = q.get("frame")
    r = {"type": "question", "time": time.strftime("%H:%M:%S"), "question": question,
         "answer": q.get("text"), "error": q.get("error"), "seconds": round(q.get("done", time.monotonic()) - q["t0"], 3),
         "during_refresh": during}
    if f:
        a = q["ans"]
        r.update(frame=f["id"], frame_age_s=round(q["done"] - f["captured"], 2), people_in_frame=len(f["boxes"]),
                 prefill_ms=float(a[1]), decode_ms=float(a[2]), kept=int(a[3]))
    log(r)
    return r


def state():
    f, now = cur["frame"], time.monotonic()
    s = SIZE / W
    return {"frame": f and f["id"], "age_s": f and round(now - f["captured"], 1),
            "boxes": f and [[round(v * s, 1) for v in b[:4]] + [b[4]] for b in f["boxes"]],
            "people_now": len(det["boxes"]), "detector_fps": len(det["times"]) / 2, "refreshing": cur["busy"],
            "changed": bool(f and scene_changed(f))}


class Page(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, code, body, kind="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self.reply(200, PAGE.encode(), "text/html; charset=utf-8")
        elif path == "/state":
            self.reply(200, json.dumps(state()).encode())
        elif path == "/frame" and cur["frame"]:
            with open(cur["frame"]["canvas"], "rb") as f:
                self.reply(200, f.read(SIZE * ROWS * 3), "application/octet-stream")
        else:
            self.reply(404, b"{}")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            question = " ".join(json.loads(self.rfile.read(n))["question"].split())[:200] if n < 4096 else ""
        except (ValueError, KeyError, TypeError, AttributeError):
            question = ""
        if self.path != "/ask" or not question:
            self.reply(400, b'{"error": "no question"}')
        else:
            self.reply(200, json.dumps(ask(question)).encode())


PAGE = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DuoSight</title>
<style>
:root { --bg: #fff; --fg: #15171a; --muted: #5f6670; --line: #e3e5e8; --accent: #c2410c; --chip: #f1f2f4; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #111315; --fg: #eceef0; --muted: #9aa1a9; --line: #2a2e33; --accent: #fb923c; --chip: #1d2024; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.45 system-ui, sans-serif; }
main { max-width: 560px; margin: 0 auto; padding: 16px; }
h1 { font-size: 18px; margin: 0 0 10px; }
canvas { display: block; width: 100%; height: auto; background: #000; border-radius: 8px; }
.meta { display: flex; flex-wrap: wrap; gap: 4px 14px; margin: 8px 0 14px; color: var(--muted); font-size: 14px;
        font-variant-numeric: tabular-nums; }
form { display: flex; gap: 8px; }
input { flex: 1; min-width: 0; padding: 10px 12px; font: inherit; color: var(--fg); background: var(--bg);
        border: 1px solid var(--line); border-radius: 8px; }
button { padding: 10px 16px; font: inherit; color: #fff; background: var(--accent); border: 0; border-radius: 8px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 4px; }
.chips button { padding: 6px 10px; font-size: 14px; color: var(--fg); background: var(--chip); }
ol { list-style: none; margin: 8px 0; padding: 0; }
li { padding: 10px 0; border-top: 1px solid var(--line); }
.q { color: var(--muted); font-size: 14px; }
.a { font-size: 20px; font-weight: 600; }
.t { color: var(--muted); font-size: 13px; font-variant-numeric: tabular-nums; }
</style>
<main>
<h1>DuoSight: ask the camera</h1>
<canvas id="view" width="512" height="ROWS"></canvas>
<div class="meta"><span id="age">waiting for the first frame</span><span id="people"></span><span id="det"></span></div>
<form id="ask"><input id="q" maxlength="200" autocomplete="off" placeholder="Ask about what the camera sees">
<button>Ask</button></form>
<div class="chips" id="chips"></div>
<ol id="log"></ol>
</main>
<script>
const QUESTIONS = QLIST;
const view = document.getElementById("view"), ctx = view.getContext("2d");
let shown = null, boxes = [];
function draw(rgb) {
  const img = ctx.createImageData(512, ROWS);
  for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) {
    img.data[j] = rgb[i]; img.data[j + 1] = rgb[i + 1]; img.data[j + 2] = rgb[i + 2]; img.data[j + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  ctx.strokeStyle = "#fb923c"; ctx.lineWidth = 3;
  for (const b of boxes) ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
}
async function poll() {
  try {
    const s = await (await fetch("/state")).json();
    document.getElementById("people").textContent = "YOLO now: " + s.people_now + (s.people_now == 1 ? " person" : " people");
    document.getElementById("det").textContent = "detector " + s.detector_fps.toFixed(1) + " fps";
    if (s.frame) {
      document.getElementById("age").textContent = "frame " + s.frame + ", " + s.age_s.toFixed(1) + " s old"
        + (s.refreshing ? ", a newer one on the way" : s.changed ? "" : ", YOLO sees no change since");
      if (s.frame !== shown) {
        const rgb = new Uint8Array(await (await fetch("/frame")).arrayBuffer());
        shown = s.frame; boxes = s.boxes; draw(rgb);
      }
    }
  } catch (e) {}
  setTimeout(poll, 500);
}
async function ask(question) {
  const li = document.createElement("li");
  li.innerHTML = '<div class="q"></div><div class="a">…</div><div class="t"></div>';
  li.querySelector(".q").textContent = question;
  document.getElementById("log").prepend(li);
  const t0 = performance.now();
  try {
    const r = await (await fetch("/ask", {method: "POST", body: JSON.stringify({question})})).json();
    const ms = performance.now() - t0;
    li.querySelector(".a").textContent = r.answer ?? ("error: " + (r.error || "no answer"));
    li.querySelector(".t").textContent = r.frame
      ? (ms / 1000).toFixed(2) + " s here, " + r.seconds.toFixed(2) + " s on the board, about frame " + r.frame
        + " (" + r.frame_age_s.toFixed(1) + " s old)"
      : (ms / 1000).toFixed(2) + " s";
  } catch (e) { li.querySelector(".a").textContent = "error: " + e; }
}
document.getElementById("ask").onsubmit = e => {
  e.preventDefault();
  const q = document.getElementById("q");
  if (q.value.trim()) ask(q.value.trim());
  q.value = "";
};
for (const text of QUESTIONS) {
  const b = document.createElement("button"); b.type = "button"; b.textContent = text; b.onclick = () => ask(text);
  document.getElementById("chips").append(b);
}
poll();
</script>
</html>
""".replace("ROWS", str(ROWS)).replace("QLIST", json.dumps(QUESTIONS))


def bench(count):
    rng = random.Random(1)
    rs = []
    for _ in range(count):
        stop.wait(rng.uniform(0.5, 3))
        if stop.is_set():
            break
        r = ask(rng.choice(QUESTIONS))
        rs.append(r)
        print(json.dumps(r), flush=True)
    ok = [r for r in rs if r.get("answer") is not None]
    if not ok:
        return
    sec = sorted(r["seconds"] for r in ok)
    age = sorted(r["frame_age_s"] for r in ok)
    busy = [r["seconds"] for r in ok if r["during_refresh"]]
    print(f"{len(ok)} of {len(rs)} answered: seconds median {statistics.median(sec):.2f}, "
          f"90% {sec[int(0.9 * (len(sec) - 1))]:.2f}, max {sec[-1]:.2f}; "
          f"{len(busy)} asked during a frame (median {statistics.median(busy) if busy else 0:.2f} s); "
          f"frame age median {statistics.median(age):.1f} s, max {age[-1]:.1f} s", flush=True)


detlog = open("/tmp/live/detect.log", "w")
detector = subprocess.Popen([DETECT, "-m", MODEL, "-r", str(args.fps), "-p", "-c", str(args.conf), "-f", str(args.focus)],
                            stdout=subprocess.PIPE, stderr=detlog, text=True, env=env, bufsize=1)
time.sleep(3)  # the detector first: it needs 14 MiB of CMA, and the VLM leaves only a few
t0 = time.monotonic()
vlm = subprocess.Popen(VLM, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open("/tmp/live/duosight.log", "w"),
                       text=True, env=env, bufsize=1)
vlm.stdin.write("rgb /dev/null\n")  # answers "err" once the models are loaded
vlm.stdin.flush()
if not vlm.stdout.readline():
    detector.send_signal(signal.SIGINT)
    detector.wait(timeout=10)
    raise SystemExit("the VLM did not load: see /tmp/live/duosight.log")
print(f"VLM loaded in {time.monotonic() - t0:.1f} s", flush=True)
signal.signal(signal.SIGTERM, lambda *a: stop.set())
start = time.monotonic()
for target in (read_detector, read_vlm, refresh):
    threading.Thread(target=target, daemon=True).start()
server = ThreadingHTTPServer(("0.0.0.0", args.port), Page)
server.daemon_threads = True
threading.Thread(target=server.serve_forever, daemon=True).start()
print(f"page on port {args.port}", flush=True)
try:
    while not ready.wait(1) and not stop.is_set():
        pass
    print("first frame ready", flush=True)
    if args.bench:
        bench(args.bench)
    else:
        stop.wait(args.seconds or None)
except KeyboardInterrupt:
    pass
finally:
    stop.set()
    server.shutdown()
    detector.send_signal(signal.SIGINT)
    detector.wait(timeout=10)
    vlm.stdin.close()
    vlm.wait(timeout=30)
    control(vertical_blanking=40, analogue_gain=0, exposure=1288)
    detlog.close()
    total = [x for x in open("/tmp/live/detect.log") if x.startswith("total")]
    print(f"frames by reason: {reasons}", flush=True)
    print(f"detector over {time.monotonic() - start:.0f} s: {total[-1].strip() if total else 'no total line'}", flush=True)
