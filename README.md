# Axfood demo: from notebook to production with no Docker and no DAG

A 15 minute live demo for a data science team that today uses custom Docker images, SageMaker and Airflow DAGs.
Use case: weekly grocery demand forecast per store and product group. **All data is synthetic.**

**The demo is one notebook**: `notebooks/Axfood_Demo.py`, imported at
`/Workspace/Users/marcelina.polak@databricks.com/axfood_demo/Axfood_Demo`. Run it top to bottom on serverless.
Everything else in this repo (bundle, scripts, SQL, GPU yaml) is the rebuildable backup and the "how it is productionised" story.

- Workspace used for the build: `fevm-agentbricks-demo` (AWS), `https://fevm-agentbricks-demo.cloud.databricks.com`
- All objects live in `agentbricks_demo_catalog.axfood_demo` (CREATE CATALOG is blocked on this shared metastore)
- The workspace expires 2026-11-27. Rebuild anywhere with `scripts/setup.sh`.

## Cell markers in the notebook

| Marker | Meaning |
|--------|---------|
| LIVE | runs in seconds in front of the audience |
| PRECOMPUTED | the slow part was run before (online store, endpoint warm-up, GPU run, full review classification); the cell reads the result |
| SHOW ONLY | code or YAML to read, not to run |

## Presenter script: 15 minutes, one notebook

Before you start: run the notebook once end to end (about 6 minutes), hit the endpoint once so it is warm (it scales to zero),
open the Genie space in a second tab. Attach the notebook to **Serverless**.

| Min | Section | Say | Marker |
|-----|---------|-----|--------|
| 0:00 | Intro | "Same work you do today. No image, no job definition, no DAG file." | |
| 0:45 | 1. Data | Data is in Unity Catalog, governed like any table. | LIVE |
| 1:30 | 2a. `ai_forecast` | "Before we train anything, one SQL statement gives a forecast with zero training." Runs on the SQL warehouse, about 10 s. | LIVE |
| 2:30 | 2b. `ai_classify`, `ai_extract` | Reviews classified and extracted with SQL. The full table is precomputed. About 2 min total for section 2. | LIVE + PRECOMPUTED |
| 3:30 | 3. Feature Store | Features are governed tables, `as_of_ts` gives point in time correctness, same definition for training, batch and serving. | LIVE |
| 5:30 | 4. Train and compare | Three runs, one line of autologging, compare in MLflow. About a minute of training: talk while it runs. | LIVE |
| 7:30 | 5. Register | New model version, `@champion` alias, lineage in Catalog Explorer. | LIVE |
| 8:30 | 6. Batch score | What a scheduled Job runs every morning. Then the comparison table: zero training vs your own model. | LIVE + PRECOMPUTED |
| 10:00 | 7. Serving | Serverless endpoint, scale to zero, you send only keys. Falls back to notebook scoring if the endpoint is not ready. | LIVE (fallback built in) |
| 11:00 | 8. GPU | "A10 with no cluster to request." Show the MLflow table of the GPU run. Mention H100, 8xH100 on demand, 8xB300 waitlist. Show the UCDataset code. | PRECOMPUTED + SHOW ONLY |
| 12:30 | 9. The Job | The bundle YAML replaces the Airflow DAG. Open the Jobs page for the deployed job. | SHOW ONLY |
| 13:30 | Genie | In the Genie tab: "Which stores are forecast to be short next week?" | LIVE |
| 14:30 | Close | "One platform, one governance model, for data, features, models, GPUs and agents. Next step: a 2 hour hands-on with one of your own use cases." | |

## Notebook-only fallback

The notebook has no hard dependency on the endpoint or the GPU:

- **Endpoint not ready:** section 7 checks the endpoint state and, if it is not `READY`, scores the same two rows with the registered model through the same feature lookup and prints `FALLBACK`.
- **GPU not available:** section 8 only reads the MLflow result of an earlier run. Nothing waits for GPU capacity. If you never ran the GPU step it prints a note instead of failing.
- **`ai_forecast` slow or blocked:** the table `ai_forecast_baseline` already exists, query it.
- **Genie slow:** open `gold_store_stock_risk` in a SQL editor with `WHERE risk_flag = 'SHORT_RISK'`.
- **Workspace reaped:** rebuild with `scripts/setup.sh` on another AWS workspace. Record a backup video of a full run before the session.

## Rebuild runbook

Prerequisites: a Databricks CLI profile and python3. The GPU step needs CLI 1.24.0 or newer: the repo expects a local copy at `.bin/databricks`
(v1.26.0 was used, not committed, download from `github.com/databricks/cli/releases`). Your global CLI is not touched.

```bash
PROFILE=fevm-agentbricks-demo scripts/setup.sh        # about 25 minutes, the online store is the slow part
scripts/run_notebook.sh notebooks/07_prepare_gpu_data.py
(cd gpu && ../.bin/databricks air run --file workload.yaml --watch --profile fevm-agentbricks-demo)   # A10, about 1 minute
scripts/run_notebook.sh notebooks/Axfood_Demo.py       # the presenter notebook, run once before the session
```

Teardown: `PROFILE=... scripts/teardown.sh`. The online store (Lakebase) is the only thing that keeps costing money while idle.

## Repo map

```
notebooks/Axfood_Demo.py            THE presenter notebook (9 sections)
notebooks/01..07_*.py               the same steps as separate notebooks, used by the bundle and setup.sh
sql/01..03_*.sql                    AI Functions as SQL files (ai_forecast v1 and v2, reviews, ai_classify/ai_extract)
databricks.yml, resources/*.yml     bundle: pipeline job + scheduled daily scoring job (PAUSED)
optional/review_classification.job.yml.disabled   native AI Function task (Beta), needs a preview, see below
gpu/workload.yaml, gpu/train_nn.py  AI Runtime GPU job (databricks air run)
genie/build_space.py                idempotent Genie space builder
scripts/                            setup.sh, teardown.sh, run_notebook.sh, run_sql.py
```

## Honest notes (say these out loud if asked)

- **`ai_forecast` is shown with version 1.** Version 2 (Beta, foundation model, holidays) needs the Predictive AI Functions preview, which is off in this workspace. `sql/01_ai_forecast_baseline_v2.sql` is ready but was never run. In a serverless notebook `ai_forecast` is blocked even for version 1, so the notebook sends it to the SQL warehouse.
- **The zero training vs custom model comparison is not like for like.** `ai_forecast` looks 10 weeks ahead from one cut-off, the custom model forecasts the next week with fresh features. We also show week 1 only (14.1 percent vs 3.6 percent WAPE). The point is a governed baseline in one query versus an owned model, not a benchmark.
- **UCDataset (Beta) did not run here.** The A10 job started, but UCDataset needs direct file access and the demo workspace's storage role denied `s3:ListBucket`. The GPU job therefore reads a Parquet copy from a Unity Catalog volume (`DATA_SOURCE=volume`). Set `DATA_SOURCE=ucdataset` in `gpu/workload.yaml` in a workspace where the role allows it.
- **The native Lakeflow AI Function task did not deploy.** `bundle validate` passes with CLI 1.26, but `bundle deploy` returns 403 "Lakeflow AI Function Task is not enabled for this workspace". A workspace admin must turn on "Lakeflow AI Function Task" on the Previews page, then move `optional/review_classification.job.yml.disabled` back to `resources/`. The plain SQL `ai_classify` and `ai_extract` step is the shipped version.
- The endpoint serves model version 1. The presenter notebook registers a new version each run and moves `@champion`, so batch scoring uses the newest version while the endpoint stays on version 1 until updated. Same model code, fine for the demo.
- SageMaker has managed MLflow too. The argument is one governed place for data, features, models, GPUs and agents, not MLflow alone.
- UCDataset and the AI Function task are Beta. 8xB300 is a Public Preview waitlist. GPU capacity is on demand and can be constrained.
- Synthetic data only. Do not present the numbers as Axfood results.
