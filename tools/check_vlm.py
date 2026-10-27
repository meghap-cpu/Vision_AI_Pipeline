"""Sends 4 frames from a clip to a VLM server with the fire-and-smoke question, the way the plugin will.
Usage: python3 tools/check_vlm.py URL MODEL [CLIP]"""
import os, sys
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vp.vlm import VLMClient

url, model = sys.argv[1], sys.argv[2]
clip = sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser("~/vision-pipeline/tools/test_clip.mp4")
cap, frames = cv2.VideoCapture(clip), []
for n in range(4):
    cap.set(cv2.CAP_PROP_POS_FRAMES, n * 30)              # one frame per second, like the repo
    ok, f = cap.read()
    h, w = f.shape[:2]
    frames.append(cv2.cvtColor(cv2.resize(f, (512, int(512 * h / w))), cv2.COLOR_BGR2RGB))
prompt = ('Analyze ALL these frames. Reply ONLY with this JSON, nothing else: '
          '{"fire": true or false, "smoke": true or false}')
result = VLMClient(url, model, timeout_s=60).ask(frames, prompt).result()
print(f"answer: {result['answer']}  |  took {result['seconds']} s  |  raw reply: {result['raw']!r}")
