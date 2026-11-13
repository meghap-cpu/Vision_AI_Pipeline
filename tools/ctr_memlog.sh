#!/usr/bin/env bash
# Logs the node container's memory every 10 min, for soak tests. Usage: ctr_memlog.sh LOGFILE
# Writes "container_not_running" when the container is missing, so a gap is visible, never silent.
LOG=${1:-$HOME/vision-pipeline/data/soak_memory.log}
while true; do
  line=$(docker top docker-vision-node-1 -o pid,rss 2>/dev/null | awk 'NR==2{m=$2} NR>1{s+=$2} END{if (NR>1) printf "main_MB=%d all_MB=%d", m/1024, s/1024}')
  echo "$(date +%F_%T) ${line:-container_not_running}" >> "$LOG"
  sleep 600
done
