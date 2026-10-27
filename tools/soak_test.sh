#!/usr/bin/env bash
# M1 soak test: 10 minutes, cam5 dropped at minute 3 and restored at minute 5, memory every minute.
cd ~/vision-pipeline
rm -rf data soak.log; tools/start_test_cameras.sh 16
python3 -u -m vp.main config/node.yaml > soak.log 2>&1 &
PID=$!
for m in $(seq 1 10); do
  sleep 60
  echo "minute $m: node memory $(( $(ps -o rss= -p $PID) / 1024 )) MB"
  [ $m -eq 3 ] && { echo ">>> stopping cam5's stream"; pkill -f "8554/cam5\$"; }
  [ $m -eq 5 ] && { echo ">>> restoring cam5's stream"; tools/start_test_cameras.sh 16 >/dev/null; }
done
echo ">>> stopping the node"; T0=$(date +%s); kill -INT $PID
for i in $(seq 1 20); do kill -0 $PID 2>/dev/null || break; sleep 1; done
kill -0 $PID 2>/dev/null && { echo "shutdown hung after 20 s, forcing"; kill -9 $PID; }
echo "shutdown took $(( $(date +%s) - T0 )) s"
echo "--- errors (should mention only src-cam05, if anything):"
grep -E "camera error|pipeline error|worker|Traceback" soak.log | head -5
echo "--- one fps line per minute:"
grep "fps" soak.log | awk 'NR%6==0' | cut -c1-120
python3 -c "import sqlite3; db=sqlite3.connect('data/outbox.db'); print('events:', db.execute('select count(*) from events').fetchone()[0], '| without snapshot:', db.execute('select count(*) from events where snapshot is null').fetchone()[0])"
