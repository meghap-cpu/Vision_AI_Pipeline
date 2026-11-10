#!/usr/bin/env bash
# Simulated network cable drill for the pilot, where hub, test cameras and VLM all run on this Jetson (127.0.0.1):
# drops traffic on the hub (8100), test cameras (8554) and stand-in VLM (8001) for N seconds, then restores it.
# The block is always removed on exit, even on Ctrl+C. Usage: tools/net_drill.sh [seconds, default 120]
SECS=${1:-120}
cleanup() {
  sudo iptables -D OUTPUT -o lo -j VP_DRILL 2>/dev/null
  sudo iptables -F VP_DRILL 2>/dev/null; sudo iptables -X VP_DRILL 2>/dev/null
  echo "network: restored at $(date +%T)"
  pkill -f "rtsp://127.0.0.1:8554/cam"; sleep 1   # test bench only: ffmpeg loops wedge during the cut
  "$(dirname "$0")/start_test_cameras.sh" 16
}
trap cleanup EXIT
trap 'exit 130' INT TERM
sudo iptables -N VP_DRILL || exit 1
sudo iptables -A VP_DRILL -p tcp -m multiport --dports 8100,8554,8001 -j DROP
sudo iptables -A VP_DRILL -p tcp -m multiport --sports 8100,8554,8001 -j DROP
sudo iptables -I OUTPUT -o lo -j VP_DRILL
echo "network: cut at $(date +%T) for $SECS s"
sleep 5
curl -s -m 3 -o /dev/null http://127.0.0.1:8100/health && echo "WARNING: hub still reachable - drill is not cutting the hub" \
  || echo "check: hub unreachable from the node side (as intended)"
sleep $((SECS - 8))
