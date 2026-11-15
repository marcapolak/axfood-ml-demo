#!/usr/bin/env bash
# Upload a notebook and run it once on serverless compute. Usage: scripts/run_notebook.sh <file.py> [deps...]
# Env: PROFILE (default fevm-agentbricks-demo), WS_DIR
set -euo pipefail
PROFILE="${PROFILE:-fevm-agentbricks-demo}"
ME=$(databricks current-user me --profile "$PROFILE" | python3 -c "import sys,json;print(json.load(sys.stdin)['userName'])")
WS_DIR="${WS_DIR:-/Workspace/Users/$ME/axfood_demo}"
F="$1"; shift || true
NAME=$(basename "$F" .py)
databricks workspace mkdirs "$WS_DIR" --profile "$PROFILE"
databricks workspace import "$WS_DIR/$NAME" --file "$F" --format SOURCE --language PYTHON --overwrite --profile "$PROFILE"
DEPS=$(python3 -c "import json,sys;print(json.dumps(sys.argv[1:]))" "$@")
RUN_ID=$(databricks jobs submit --no-wait --profile "$PROFILE" --json "{
  \"run_name\": \"axfood-$NAME\",
  \"tasks\": [{\"task_key\": \"$NAME\", \"notebook_task\": {\"notebook_path\": \"$WS_DIR/$NAME\"}, \"environment_key\": \"env\"}],
  \"environments\": [{\"environment_key\": \"env\", \"spec\": {\"client\": \"4\", \"dependencies\": $DEPS}}]
}" | python3 -c "import sys,json;print(json.load(sys.stdin)['run_id'])")
echo "run_id=$RUN_ID"
while true; do
  S=$(databricks jobs get-run "$RUN_ID" --profile "$PROFILE" -o json)
  LC=$(echo "$S" | python3 -c "import sys,json;print(json.load(sys.stdin)['state']['life_cycle_state'])")
  case "$LC" in TERMINATED|SKIPPED|INTERNAL_ERROR) break;; esac
  sleep 15
done
RES=$(echo "$S" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['state'].get('result_state','?'))")
TASK=$(echo "$S" | python3 -c "import sys,json;print(json.load(sys.stdin)['tasks'][0]['run_id'])")
echo "result_state=$RES"
databricks jobs get-run-output "$TASK" --profile "$PROFILE" -o json | python3 -c "
import sys,json
d=json.load(sys.stdin)
print('output:', (d.get('notebook_output') or {}).get('result'))
if d.get('error'): print('error:', d['error']); print((d.get('error_trace') or '')[-2500:])
"
[ "$RES" = "SUCCESS" ]
