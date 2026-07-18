#!/bin/bash
# Launch eval_paper1_v2.py with an 18 GiB RSS watchdog on the REAL python pid.
set -u
cd /home/lars/trading-bot
LOG="logs/eval_paper1_v2_$(date +%Y%m%d_%H%M%S).log"
echo "$LOG" > /tmp/paper1_log.txt
# Start python first so we know its pid
.venv/bin/python eval_paper1_v2.py > "$LOG" 2>&1 &
PY=$!
echo "$PY" > /tmp/paper1_py.pid
# Watch the real python pid (not this shell)
/tmp/rss_watchdog.sh "$PY" 19327352832 > /tmp/watchdog_paper1.log 2>&1 &
echo "python pid=$PY  log=$LOG"
wait "$PY"
echo "PY_EXIT=$?" >> "$LOG"
