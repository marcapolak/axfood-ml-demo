#!/usr/bin/env bash
# Remove the billable pieces created by this demo. Only touches resources named in this repo. Usage: PROFILE=<profile> scripts/teardown.sh
set -euo pipefail
PROFILE="${PROFILE:?set PROFILE}"
echo "Serving endpoint (scales to zero, but delete when done):"
databricks serving-endpoints delete axfood-demand-forecast --profile "$PROFILE" || true
echo "Online store (runs continuously, this is the main cost):"
python3 - <<PY || true
from databricks.feature_engineering import FeatureEngineeringClient
FeatureEngineeringClient().delete_online_store("axfood-demo-online-store")
PY
echo "Unpause or leave the scheduled job paused. Bundle resources: databricks bundle destroy --profile $PROFILE"
echo "Tables stay in agentbricks_demo_catalog.axfood_demo (cheap). Drop with: DROP SCHEMA agentbricks_demo_catalog.axfood_demo CASCADE"
