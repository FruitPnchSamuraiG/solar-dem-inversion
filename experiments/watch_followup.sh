#!/bin/bash
# Poll Torch every 5 minutes and exit as soon as a new follow-up evaluation
# result appears, or a follow-up job fails in a way not seen before. Running it
# in the background turns "a result landed" into a notification.
#   bash experiments/watch_followup.sh
list_results() {
  timeout 60 ssh -o BatchMode=yes torch \
    'ls ~/projects/dem/output/experiments/diagnostics/ | grep -E "_fu_.*\.json$" | sort' 2>/dev/null
}
list_failed() {
  timeout 60 ssh -o BatchMode=yes torch \
    'sacct -u $USER -S 2026-10-06T16:00 -n -X -o JobID,JobName%22,State%14 | grep -E " f[udae]_" | grep -E "FAILED|TIMEOUT|OUT_OF_ME|NODE_FAIL" | sort' 2>/dev/null
}
known=$(list_results); known_failed=$(list_failed)
while true; do
  sleep 300
  now=$(list_results) || continue
  if [ -n "$now" ] && [ "$now" != "$known" ]; then
    echo "NEW RESULTS:"; comm -13 <(echo "$known") <(echo "$now"); exit 0
  fi
  failed=$(list_failed) || continue
  if [ "$failed" != "$known_failed" ]; then
    echo "NEW FAILURES:"; comm -13 <(echo "$known_failed") <(echo "$failed"); exit 0
  fi
done
