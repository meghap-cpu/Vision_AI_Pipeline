#!/usr/bin/env bash
# Memory (RSS) of every process in the soak, saved with a timestamp: run twice and compare.
out=/tmp/soak/rss_$(date +%H%M).txt
ps -eo pid,ppid,rss,etimes,args --sort=-rss | grep -E "vp.main|multiprocessing|standin|mediamtx|ffmpeg" | grep -v grep \
  | awk '{n=$1; m=$3/1024; $1=$2=$3=$4=""; printf "%7d %8.0f MB %s\n", n, m, substr($0, 1, 90)}' > "$out"
awk '{s+=$2} END {printf "total %.0f MB in %d processes\n", s, NR}' "$out"; echo "saved $out"
