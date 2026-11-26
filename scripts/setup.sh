#!/usr/bin/env bash
# Rebuild the whole Axfood demo in a workspace. Idempotent. Usage: PROFILE=<profile> scripts/setup.sh
# Prereqs: databricks CLI (this repo uses .bin/databricks 1.26.0 for `air` and the AI Function task), python3.
set -euo pipefail
cd "$(dirname "$0")/.."
export PROFILE="${PROFILE:?set PROFILE to a databricks CLI profile}"
export PATH="$PWD/.bin:$PATH"                     # newer CLI first, needed for `databricks air`
step() { echo; echo "=== $*"; }

step "1/9 synthetic data";            scripts/run_notebook.sh notebooks/01_generate_data.py
step "2/9 AI Functions: ai_forecast baseline, reviews, ai_classify/ai_extract"
python3 scripts/run_sql.py sql/01_ai_forecast_baseline.sql
python3 scripts/run_sql.py sql/02_reviews_data.sql
python3 scripts/run_sql.py sql/03_ai_reviews_enrich.sql
step "3/9 features in Unity Catalog";  scripts/run_notebook.sh notebooks/02_build_features.py
step "4/9 train, track, register";     scripts/run_notebook.sh notebooks/03_train_register.py
step "5/9 batch scoring";              scripts/run_notebook.sh notebooks/04_batch_score.py
step "6/9 baseline vs custom model";   scripts/run_notebook.sh notebooks/05_compare_baseline.py
step "7/9 bundle: pipeline job + scheduled scoring job"
databricks bundle deploy -t dev --profile "$PROFILE"
step "8/9 serving endpoint (online store takes about 10 minutes)"; scripts/run_notebook.sh notebooks/06_serve_endpoint.py
step "9/9 Genie space";                python3 genie/build_space.py --profile "$PROFILE"
echo; echo "Optional: GPU step"
echo "  scripts/run_notebook.sh notebooks/07_prepare_gpu_data.py"
echo "  (cd gpu && databricks air run --file workload.yaml --watch --profile $PROFILE)"
