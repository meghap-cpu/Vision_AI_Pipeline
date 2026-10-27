#!/usr/bin/env bash
# Start mediamtx and N looping RTSP test cameras (cam0..camN-1), skipping any already running.
N=${1:-16}
CLIP=${2:-$HOME/vision-pipeline/tools/test_clip.mp4}
pgrep -x mediamtx >/dev/null || (cd ~/mediamtx && nohup ./mediamtx >/tmp/mediamtx.log 2>&1 &)
sleep 1
for i in $(seq 0 $((N-1))); do
  pgrep -f "8554/cam$i\$" >/dev/null && continue
  nohup ffmpeg -nostdin -loglevel error -re -stream_loop -1 -i "$CLIP" -an -c copy \
    -f rtsp -rtsp_transport tcp "rtsp://127.0.0.1:8554/cam$i" >/dev/null 2>&1 &
done
sleep 2; echo "test cameras running: $(pgrep -fc 'rtsp://127.0.0.1:8554/cam')"
