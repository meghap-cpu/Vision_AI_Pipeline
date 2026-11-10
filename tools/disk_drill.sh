#!/usr/bin/env bash
# Disk nearly full drill: fills the disk that holds data/ until only LEAVE_GB is free, holds it, then frees it.
# The filler is always removed on exit, even on Ctrl+C. Usage: tools/disk_drill.sh [leave_gb, default 1.5] [hold_s, default 330]
LEAVE_GB=${1:-1.5}; HOLD=${2:-330}
cd "$(dirname "$0")/.." || exit 1
FILL=data/disk_drill.fill
free_gb() { df -k --output=avail data | tail -1 | awk '{printf "%.1f GB", $1/1048576}'; }
cleanup() { rm -f "$FILL"; echo "disk: freed at $(date +%T), free now $(free_gb)"; }
trap cleanup EXIT
trap 'exit 130' INT TERM
echo "disk: free before $(free_gb)"
fill_kb=$(df -k --output=avail data | tail -1 | awk -v l="$LEAVE_GB" '{printf "%d", $1 - l*1048576}')
if [ "$fill_kb" -le 0 ]; then echo "disk: already under $LEAVE_GB GB free, nothing to fill"; exit 0; fi
fallocate -l "${fill_kb}K" "$FILL" || exit 1
echo "disk: filled at $(date +%T), free now $(free_gb); holding $HOLD s (retention checks every 300 s)"
sleep "$HOLD"
