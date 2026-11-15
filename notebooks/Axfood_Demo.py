# Databricks notebook source
# MAGIC %md
# MAGIC # From notebook to production: no Docker image, no DAG
# MAGIC **Demand forecast per store and product group.** All data in this notebook is synthetic.
# MAGIC
# MAGIC Run top to bottom on **serverless** compute. Each section is marked:
# MAGIC * **LIVE** runs here in seconds, safe to run in front of an audience
# MAGIC * **PRECOMPUTED** the slow part was run before, this cell reads the result (online store, endpoint warm-up, GPU)
# MAGIC * **SHOW ONLY** code to read, not to run
# MAGIC
# MAGIC Sections: 1 data, 2 AI Functions, 3 Feature Store, 4 train and compare, 5 register, 6 batch score, 7 serving, 8 GPU, 9 the Job that replaces the DAG.

# COMMAND ----------

# MAGIC %pip install "databricks-feature-engineering>=0.16.0" "mlflow>=3.1" lightgbm scikit-learn "pyarrow>=16" --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "agentbricks_demo_catalog")
dbutils.widgets.text("schema", "axfood_demo")
dbutils.widgets.text("warehouse_id", "7046f7a2659b62ba")
CATALOG = dbutils.widgets.get("catalog"); SCHEMA = dbutils.widgets.get("schema"); WAREHOUSE = dbutils.widgets.get("warehouse_id")
T = lambda n: f"{CATALOG}.{SCHEMA}.{n}"
MODEL = T("demand_forecast")
ME = spark.sql("SELECT current_user()").first()[0]
import mlflow, pandas as pd, numpy as np
from pyspark.sql import functions as F
mlflow.set_registry_uri("databricks-uc")
print("catalog.schema:", f"{CATALOG}.{SCHEMA}", "| user:", ME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Data in Unity Catalog &nbsp; `LIVE`
# MAGIC Weekly units per store and product group. Governed like every other table: permissions, lineage, audit.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT count(*) AS rows, count(DISTINCT store_id) AS stores, count(DISTINCT product_group) AS product_groups, min(week_start) AS first_week, max(week_start) AS last_week
# MAGIC FROM IDENTIFIER(:catalog || '.' || :schema || '.weekly_sales')

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT week_start, units_sold, promo_flag, avg_temp_c, holiday_flag
# MAGIC FROM IDENTIFIER(:catalog || '.' || :schema || '.weekly_sales')
# MAGIC WHERE store_id = 101 AND product_group = 'Beverages' ORDER BY week_start DESC LIMIT 8

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. AI Functions: forecast and understand text with SQL, zero training
# MAGIC ### 2a. `ai_forecast` baseline &nbsp; `LIVE` (runs on the SQL warehouse, about 10 seconds)
# MAGIC One SQL statement, no model, no features, no endpoint. 20 series (5 stores x 4 product groups), the last 10 weeks held out.
# MAGIC `ai_forecast` is a SQL warehouse function, so the cell sends it to the serverless warehouse.
# MAGIC Version 1 is used here. Version 2 (foundation model, holidays) needs the Predictive AI Functions preview, which is off in this workspace.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
AI_FORECAST_SQL = f"""
CREATE OR REPLACE TABLE {T('ai_forecast_baseline')} AS
SELECT CAST(split(series, '[|]')[0] AS BIGINT) AS store_id, split(series, '[|]')[1] AS product_group, week_start,
       units_sold_forecast AS ai_forecast_units, units_sold_lower AS ai_forecast_lower, units_sold_upper AS ai_forecast_upper
FROM ai_forecast(
  observed => TABLE(SELECT week_start, units_sold, concat(store_id, '|', product_group) AS series FROM {T('weekly_sales')}
                    WHERE store_id IN (101,102,103,104,105) AND product_group IN ('Dairy','Beverages','Fish','Ready meals') AND week_start < DATE'2026-09-01'),
  horizon => '2026-11-09', time_col => 'week_start', value_col => 'units_sold', group_col => 'series', version => '1')
"""
r = w.statement_execution.execute_statement(warehouse_id=WAREHOUSE, statement=AI_FORECAST_SQL, wait_timeout="50s")
print("ai_forecast:", r.status.state.value)
assert r.status.state.value == "SUCCEEDED", r.status

# COMMAND ----------

display(spark.sql(f"""
SELECT a.store_id, a.product_group, a.week_start, s.units_sold AS actual, round(a.ai_forecast_units) AS ai_forecast, round(a.ai_forecast_lower) AS lower, round(a.ai_forecast_upper) AS upper
FROM {T('ai_forecast_baseline')} a JOIN {T('weekly_sales')} s USING (store_id, product_group, week_start)
WHERE a.store_id = 101 AND a.product_group = 'Beverages' ORDER BY a.week_start"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2b. `ai_classify`, `ai_analyze_sentiment` and `ai_extract` on free text &nbsp; `LIVE`
# MAGIC Customer reviews and store notes (49 synthetic rows). Plain SQL, results persisted once to a Delta table.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT review_id, store_id, left(review_text, 80) AS review,
# MAGIC        ai_classify(review_text, '["out_of_stock","quality","price","delivery","other"]', map('version','2.0')):response[0]::STRING AS topic,
# MAGIC        ai_analyze_sentiment(review_text) AS sentiment,
# MAGIC        ai_extract(review_text, '{"product":{"type":"string"},"price_sek":{"type":"number"}}', map('version','2.0')):response AS extracted
# MAGIC FROM IDENTIFIER(:catalog || '.' || :schema || '.customer_reviews') LIMIT 6

# COMMAND ----------

# MAGIC %md
# MAGIC `PRECOMPUTED` The full table (all 49 reviews) was classified once. Query the cheap Delta table, not the LLM:

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT topic, count(*) AS reviews, round(avg(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) * 100) AS pct_negative
# MAGIC FROM IDENTIFIER(:catalog || '.' || :schema || '.customer_reviews_enriched') GROUP BY topic ORDER BY reviews DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Feature Store: features as governed tables &nbsp; `LIVE`
# MAGIC Defined once, used for training, batch scoring and online serving. `as_of_ts` is when the value became knowable,
# MAGIC so the training set never sees the future (point in time join).

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
fe = FeatureEngineeringClient()
display(spark.sql(f"SELECT * FROM {T('store_product_features')} WHERE store_id = 101 AND product_group = 'Dairy' ORDER BY as_of_ts DESC LIMIT 4"))

# COMMAND ----------

labels = (spark.table(T("weekly_sales"))
          .select("store_id", "product_group", "week_start", "units_sold", "promo_flag", "holiday_flag", "avg_temp_c", "week_of_year")
          .withColumn("as_of_ts", F.col("week_start").cast("timestamp")))
lookups = [FeatureLookup(table_name=T("store_features"), lookup_key="store_id"),
           FeatureLookup(table_name=T("store_product_features"), lookup_key=["store_id", "product_group"], timestamp_lookup_key="as_of_ts")]
CUTOFF = "2026-09-01"
mk = lambda df: fe.create_training_set(df=df, feature_lookups=lookups, label="units_sold", exclude_columns=["week_start", "as_of_ts", "store_id"])
ts_train, ts_test = mk(labels.where(f"week_start < '{CUTOFF}'")), mk(labels.where(f"week_start >= '{CUTOFF}'"))
train_pd, test_pd = ts_train.load_df().toPandas(), ts_test.load_df().toPandas()
X_train, y_train = train_pd.drop(columns=["units_sold"]), train_pd["units_sold"]
X_test, y_test = test_pd.drop(columns=["units_sold"]), test_pd["units_sold"]
print("train", X_train.shape, "| test", X_test.shape)
display(X_train.head(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Train three runs, compare in MLflow &nbsp; `LIVE` (about a minute)
# MAGIC Autologging records parameters, metrics and code. No image, no job definition.

# COMMAND ----------

import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
os_dir = f"/Users/{ME}/axfood_demo"; import os; os.makedirs(f"/Workspace{os_dir}", exist_ok=True)
mlflow.set_experiment(f"{os_dir}/presenter_demand_forecast")
CAT = ["product_group", "region", "store_format"]
def build(p):
    prep = ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT)], remainder="passthrough", verbose_feature_names_out=False)
    return Pipeline([("prep", prep), ("model", lgb.LGBMRegressor(verbose=-1, random_state=7, **p))])
wape = lambda y, p: float(np.abs(y - p).sum() / np.abs(y).sum())
CONFIGS = {"baseline_small": dict(n_estimators=150, learning_rate=0.10, num_leaves=15),
           "balanced": dict(n_estimators=400, learning_rate=0.05, num_leaves=31),
           "deeper": dict(n_estimators=700, learning_rate=0.03, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)}
naive = wape(y_test.values, X_test["sales_lag_1w"].fillna(y_train.mean()).values)
mlflow.sklearn.autolog(log_models=False, silent=True)
runs = {}
for name, p in CONFIGS.items():
    with mlflow.start_run(run_name=name) as run:
        pipe = build(p).fit(X_train, y_train); pred = pipe.predict(X_test)
        mlflow.log_metrics({"test_wape": wape(y_test.values, pred), "naive_last_week_wape": naive})
        runs[name] = (run.info.run_id, wape(y_test.values, pred), pipe)
best = min(runs, key=lambda k: runs[k][1])

# COMMAND ----------

cmp = pd.DataFrame([{"run": k, "test_wape": round(v[1], 4)} for k, v in runs.items()] + [{"run": "naive: same as last week", "test_wape": round(naive, 4)}]).sort_values("test_wape")
display(cmp)
print("best:", best, "(lower WAPE is better). Open the experiment to compare the runs side by side.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Register to Unity Catalog, set `@champion` &nbsp; `LIVE`
# MAGIC `fe.log_model` stores which feature tables the model needs, so scoring and serving look them up automatically.
# MAGIC Re-running adds a new model version and moves the alias, so it is safe to repeat.

# COMMAND ----------

from mlflow.tracking import MlflowClient
with mlflow.start_run(run_name=f"register_{best}"):
    mlflow.log_params({"chosen_config": best, **CONFIGS[best]})
    mlflow.log_metrics({"test_wape": runs[best][1]})
    fe.log_model(model=runs[best][2], artifact_path="model", flavor=mlflow.sklearn, training_set=ts_train, registered_model_name=MODEL)
client = MlflowClient(registry_uri="databricks-uc")
latest = max(client.search_model_versions(f"name='{MODEL}'"), key=lambda v: int(v.version))
client.set_registered_model_alias(MODEL, "champion", latest.version)
print(f"{MODEL} version {latest.version} is now @champion")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Batch scoring into a gold table &nbsp; `LIVE`
# MAGIC This is the task a scheduled Job runs every morning. Only keys and planned inputs go in; features are looked up.

# COMMAND ----------

last_week = spark.table(T("weekly_sales")).agg(F.max("week_start")).first()[0]
target_week = spark.sql(f"SELECT date_add(DATE'{last_week}', 7)").first()[0]
keys = (spark.table(T("weekly_sales")).where(F.col("week_start") == F.lit(last_week))
        .select("store_id", "product_group", "promo_flag", "avg_temp_c").withColumn("week_start", F.lit(target_week).cast("date"))
        .withColumn("week_of_year", F.weekofyear("week_start")).withColumn("holiday_flag", F.lit(0)).withColumn("as_of_ts", F.col("week_start").cast("timestamp")))
scored = fe.score_batch(model_uri=f"models:/{MODEL}@champion", df=keys)
out = (scored.select("store_id", "product_group", "week_start", "promo_flag", "avg_temp_c", "prediction")
       .join(spark.table(T("stores")).select("store_id", "store_name", "region", "store_format"), "store_id")
       .withColumn("forecast_units", F.round("prediction").cast("int")).drop("prediction").withColumn("scored_at", F.current_timestamp()))
out.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T("gold_weekly_forecast"))
spark.sql(f"""CREATE OR REPLACE TABLE {T('gold_store_stock_risk')} AS
SELECT f.store_id, f.store_name, f.region, f.store_format, f.product_group, f.week_start AS forecast_week, f.forecast_units,
       round(h.avg_4w, 0) AS avg_units_last_4_weeks, round((f.forecast_units - h.avg_4w) / h.avg_4w * 100, 1) AS pct_above_recent_average,
       CASE WHEN f.forecast_units > 1.15 * h.avg_4w THEN 'SHORT_RISK' ELSE 'OK' END AS risk_flag, f.promo_flag
FROM {T('gold_weekly_forecast')} f
JOIN (SELECT store_id, product_group, avg(units_sold) AS avg_4w FROM {T('weekly_sales')} WHERE week_start > date_sub(DATE'{last_week}', 28) GROUP BY store_id, product_group) h USING (store_id, product_group)""")
display(spark.sql(f"SELECT store_name, product_group, forecast_units, avg_units_last_4_weeks, pct_above_recent_average FROM {T('gold_store_stock_risk')} WHERE risk_flag = 'SHORT_RISK' ORDER BY pct_above_recent_average DESC LIMIT 8"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Zero training vs your own model, both governed &nbsp; `PRECOMPUTED`
# MAGIC Not like for like: `ai_forecast` looks 10 weeks ahead from one cut-off, the custom model forecasts the next week with fresh features. We also show week 1 only.

# COMMAND ----------

display(spark.table(T("forecast_comparison_summary")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Real-time serving &nbsp; `LIVE` if the endpoint is warm, otherwise the notebook-only fallback runs
# MAGIC Serverless endpoint, scales to zero, automatic feature lookup. The caller sends only the store, product group and planned inputs.
# MAGIC The online store and endpoint were created before the session (the slow part).

# COMMAND ----------

ENDPOINT = "axfood-demand-forecast"
request = [{"store_id": 101, "product_group": "Beverages", "promo_flag": 1, "holiday_flag": 0, "avg_temp_c": 4.0, "week_of_year": 47},
           {"store_id": 104, "product_group": "Dairy", "promo_flag": 0, "holiday_flag": 0, "avg_temp_c": 4.0, "week_of_year": 47}]
served = False
try:
    ep = w.serving_endpoints.get(ENDPOINT)
    print("endpoint state:", ep.state.ready.value if ep.state and ep.state.ready else None)
    if ep.state and ep.state.ready and ep.state.ready.value == "READY":
        resp = w.serving_endpoints.query(ENDPOINT, dataframe_records=request)
        print("LIVE endpoint response:", resp.predictions); served = True
except Exception as e:
    print("endpoint not usable:", str(e)[:200])
if not served:
    print("FALLBACK (notebook only): same rows scored with the registered model, same feature lookup")
    rows = spark.createDataFrame(pd.DataFrame(request)).withColumn("as_of_ts", F.current_timestamp())
    display(fe.score_batch(model_uri=f"models:/{MODEL}@champion", df=rows).select("store_id", "product_group", "prediction"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. GPUs on AI Runtime (serverless GPU) &nbsp; `PRECOMPUTED` result, `SHOW ONLY` code
# MAGIC A small neural forecaster trained on a serverless **A10** with no cluster to request. The run below already happened
# MAGIC (from a notebook attached to Serverless GPU, or from the CLI as a backup), so nothing here waits for GPU capacity.
# MAGIC H100 and 8xH100 are on demand too; 8xB300 is a Public Preview waitlist.

# COMMAND ----------

exps = [e for e in mlflow.search_experiments(max_results=500) if "axfood-gpu-forecaster" in e.name]
if exps:
    gpu = mlflow.search_runs(experiment_ids=[e.experiment_id for e in exps], order_by=["start_time DESC"], max_results=5)
    keep = [c for c in ["start_time", "params.gpu", "params.train_rows", "params.data_source", "metrics.test_wape", "metrics.train_seconds"] if c in gpu.columns]
    display(gpu[keep].dropna(subset=["metrics.test_wape"]) if "metrics.test_wape" in gpu.columns else gpu[keep])
else:
    print("No GPU run found in MLflow yet. Run the GPU step once before the session (see README).")

# COMMAND ----------

# MAGIC %md
# MAGIC `SHOW ONLY` Reading the Unity Catalog table straight into GPU training with **UCDataset** (Beta). No export, no `toPandas()`, no Spark session on the GPU node.
# MAGIC ```python
# MAGIC # on a notebook attached to Serverless GPU:  %pip install databricks-uc-dataset==0.0.1b1
# MAGIC from databricks.uc_dataset.beta import UCDataset
# MAGIC from databricks.uc_dataset.beta.dataframe import table, col
# MAGIC df = table("agentbricks_demo_catalog.axfood_demo.gpu_training_data").select("sales_lag_1w", "sales_avg_4w", "label").filter(col("is_test") == 0.0)
# MAGIC loader = torch.utils.data.DataLoader(UCDataset(df=df, batch_size=1024), batch_size=None, num_workers=2)
# MAGIC for batch in loader:
# MAGIC     x, y = batch["sales_lag_1w"], batch["label"]       # tensors, ready for the GPU
# MAGIC ```
# MAGIC Honest note: UCDataset is Beta, and in this demo workspace the storage role did not allow the direct file access it needs
# MAGIC (`s3:ListBucket`), so the training run reads a Parquet copy from a Unity Catalog volume. In a workspace where the role allows it, set `DATA_SOURCE=ucdataset`.
# MAGIC
# MAGIC `SHOW ONLY` Backup from a laptop, same GPU: `databricks air run --file gpu/workload.yaml --watch` (CLI 1.24.0 or newer).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. The Job that replaces the Airflow DAG &nbsp; `SHOW ONLY`
# MAGIC The whole pipeline is a Declarative Automation Bundle: files in Git, deployed with `databricks bundle deploy`. Retries and failure alerts are settings, not code.
# MAGIC The daily scoring job ships **paused** so it costs nothing until unpaused.
# MAGIC
# MAGIC ```yaml
# MAGIC # databricks.yml
# MAGIC bundle: {name: axfood_demo}
# MAGIC variables:
# MAGIC   catalog: {default: agentbricks_demo_catalog}
# MAGIC   schema:  {default: axfood_demo}
# MAGIC include: [resources/*.yml]
# MAGIC targets:
# MAGIC   dev: {default: true, mode: development, workspace: {host: https://<workspace>.cloud.databricks.com}}
# MAGIC
# MAGIC # resources/daily_scoring.job.yml
# MAGIC resources:
# MAGIC   jobs:
# MAGIC     daily_scoring:
# MAGIC       name: axfood-daily-batch-scoring
# MAGIC       schedule: {quartz_cron_expression: "0 0 6 * * ?", timezone_id: Europe/Stockholm, pause_status: PAUSED}
# MAGIC       email_notifications: {on_failure: ["${workspace.current_user.userName}"]}
# MAGIC       environments: [{environment_key: serverless_env, spec: {client: "4"}}]
# MAGIC       tasks:
# MAGIC         - task_key: batch_score
# MAGIC           environment_key: serverless_env
# MAGIC           max_retries: 2
# MAGIC           notebook_task: {notebook_path: ../notebooks/04_batch_score.py}
# MAGIC ```
# MAGIC In Airflow this is a DAG file, an operator, a Docker image and a deployment. Here it is the notebook you just ran, plus 15 lines of YAML.
# MAGIC
# MAGIC **What we showed:** data, AI Functions, features, three tracked runs, a governed model with an alias, batch scoring, serving, a GPU run, and the Job.
# MAGIC One platform, one governance model.

# COMMAND ----------

import json
dbutils.notebook.exit(json.dumps({"best": best, "test_wape": round(runs[best][1], 4), "naive_wape": round(naive, 4), "model_version": latest.version,
                                  "short_risk_rows": spark.table(T("gold_store_stock_risk")).where("risk_flag='SHORT_RISK'").count(), "endpoint_live": served}))
