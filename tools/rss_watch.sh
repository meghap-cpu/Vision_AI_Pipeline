#!/usr/bin/env bash
# Logs the node's main process memory every ${1:-120} s. Matches the node itself, not wrappers like timeout.
while sleep "${1:-120}"; do
  pid=$(pgrep -f "^/usr/bin/python3 -u -m vp.main" | head -1)
  [ -z "$pid" ] && { echo "$(date +%H:%M) no node"; continue; }
  echo "$(date +%H:%M) pid $pid $(awk '/VmRSS/{printf "%.1f MB", $2/1024}' /proc/$pid/status)"
done
