"""Measures a VLM server the way our plugins use it. Usage: python3 tools/vlm_bench.py URL MODEL
A: fire-and-smoke check (4 frames) x5 in a row; B: worker-at-height PPE check on worker photos;
C: 4 fire checks at the same time (capacity)."""
import glob, os, statistics, sys, time
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vp.plugins.fire_smoke import PROMPT as FIRE_PROMPT
from vp.plugins.worker_at_height import PPE_PROMPT
from vp.vlm import VLMClient

url, model = sys.argv[1], sys.argv[2]
cap, frames = cv2.VideoCapture(os.path.expanduser("~/vision-pipeline/tools/test_clip.mp4")), []
for n in range(4):
    cap.set(cv2.CAP_PROP_POS_FRAMES, n * 30)
    ok, f = cap.read()
    frames.append(cv2.cvtColor(cv2.resize(f, (512, 288)), cv2.COLOR_BGR2RGB))

one = VLMClient(url, model, timeout_s=120, max_in_flight=1)
times = []
print("A. fire and smoke, 4 frames, one at a time:")
for i in range(5):
    r = one.ask(frames, FIRE_PROMPT).result()
    times.append(r["seconds"])
    print(f"   {r['seconds']:5.2f} s  answer {r['answer']}  raw {r['raw'][:60]!r}")
print(f"   median {statistics.median(times):.2f} s per fire check")

print("B. worker at height, PPE prompt, one worker photo per request:")
paths = sorted(p for p in glob.glob(os.path.expanduser("~/vision-trial/ppe/test_images/*"))
               if p.lower().endswith((".jpg", ".jpeg", ".png")))
for path in paths[:2]:
    raw = cv2.imread(path)
    if raw is None:
        print(f"   skipped (unreadable): {os.path.basename(path)}")
        continue
    img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    img = cv2.resize(img, (768, int(768 * h / w)))
    r = one.ask([img], PPE_PROMPT, max_tokens=120).result()
    print(f"   {os.path.basename(path)[:40]:40s} {r['seconds']:5.2f} s  answer {r['answer']}")

print("C. 4 fire checks at the same time:")
many = VLMClient(url, model, timeout_s=120, max_in_flight=4)
t0 = time.time()
results = [f.result() for f in [many.ask(frames, FIRE_PROMPT) for _ in range(4)]]
total = time.time() - t0
print(f"   all 4 done in {total:.2f} s -> {4 / total:.2f} fire checks per second "
      f"(single-request median {statistics.median(times):.2f} s)")
