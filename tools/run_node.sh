#!/usr/bin/env bash
# Runs the node and restarts it: at once when it exits to apply new settings (code 3), after 5 s on a crash.
# Ctrl+C (or a clean stop, code 0) ends the loop.
cd "$(dirname "$0")/.."
while true; do
  /usr/bin/python3 -u -m vp.main "${1:-config/node.yaml}"
  code=$?
  if [ $code -eq 3 ]; then echo "run_node: applying new settings"; continue; fi
  if [ $code -eq 0 ] || [ $code -eq 130 ]; then echo "run_node: stopped"; break; fi
  echo "run_node: exited with code $code; restarting in 5 s"; sleep 5
done
