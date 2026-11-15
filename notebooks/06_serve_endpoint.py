# Databricks notebook source
# MAGIC %md
# MAGIC # 06. Real-time serving with automatic feature lookup
# MAGIC * publish the two feature tables to an online store (Lakebase) for low-latency lookups
# MAGIC * deploy the `@champion` model to a **serverless Model Serving endpoint that scales to zero**
# MAGIC * call it with only store, product group and the planned inputs. The endpoint fetches the rolling features itself,
# MAGIC   exactly as they were defined for training (no training/serving skew).
# MAGIC
# MAGIC Cost note: the online store runs continuously, tear it down after the demo (see README).

# COMMAND ----------

# MAGIC %pip install "databricks-feature-engineering>=0.16.0" "mlflow>=3.1" "pyarrow>=16"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "agentbricks_demo_catalog")
dbutils.widgets.text("schema", "axfood_demo")
CATALOG = dbutils.widgets.get("catalog"); SCHEMA = dbutils.widgets.get("schema")
T = lambda n: f"{CATALOG}.{SCHEMA}.{n}"
MODEL = T("demand_forecast")
ENDPOINT = "axfood-demand-forecast"
ONLINE_STORE = "axfood-demo-online-store"        # DNS compliant: hyphens only
import time, json, mlflow
from mlflow.tracking import MlflowClient
from databricks.feature_engineering import FeatureEngineeringClient
mlflow.set_registry_uri("databricks-uc")
fe = FeatureEngineeringClient()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Online store and publish feature tables

# COMMAND ----------

store = fe.get_online_store(name=ONLINE_STORE)          # returns None when it does not exist
if store is None:
    fe.create_online_store(name=ONLINE_STORE, capacity="CU_1")
    for _ in range(40):
        time.sleep(30)
        store = fe.get_online_store(name=ONLINE_STORE)
        print("online store state:", getattr(store, "state", store))
        if store is not None and "AVAILABLE" in str(getattr(store, "state", "")).upper(): break
print(store)

# COMMAND ----------

for name in ["store_features", "store_product_features"]:
    spark.sql(f"ALTER TABLE {T(name)} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    fe.publish_table(online_store=store, source_table_name=T(name), online_table_name=T(name + "_online"), publish_mode="SNAPSHOT")
    print("published", name)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Wait for the online tables to finish syncing
# MAGIC Creating the endpoint right after `publish_table` fails with "Online feature store setup failed" because the SNAPSHOT sync is still running.
# MAGIC Wait until both online tables hold the expected rows.

# COMMAND ----------

expected = {name: spark.table(T(name)).select("store_id", "product_group").distinct().count() if name == "store_product_features" else spark.table(T(name)).count()
            for name in ["store_features", "store_product_features"]}
for _ in range(40):
    try:
        got = {n: spark.table(T(n + "_online")).count() for n in expected}
    except Exception as e:
        got = {n: -1 for n in expected}
    print(time.strftime("%H:%M:%S"), "online rows", got, "expected", expected)
    if all(got[n] >= expected[n] for n in expected): break
    time.sleep(30)
assert all(got[n] >= expected[n] for n in expected), "online tables not synced"

# COMMAND ----------
# MAGIC %md
# MAGIC ## Serverless endpoint, scale to zero

# COMMAND ----------

from mlflow.deployments import get_deploy_client
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
client = MlflowClient(registry_uri="databricks-uc")
champ = client.get_model_version_by_alias(MODEL, "champion")
dc = get_deploy_client("databricks")
cfg = {"served_entities": [{"name": "demand-forecast", "entity_name": MODEL, "entity_version": champ.version,
                            "workload_size": "Small", "scale_to_zero_enabled": True}]}
def endpoint_state():
    try:
        ep = w.serving_endpoints.get(ENDPOINT); return str(ep.state.ready), str(ep.state.config_update)
    except Exception:
        return None, None
ready, upd = endpoint_state()
if ready is not None and "UPDATE_FAILED" in upd:           # a previous attempt failed: our own endpoint, recreate it
    w.serving_endpoints.delete(ENDPOINT); time.sleep(10); ready = None
if ready is None:
    dc.create_endpoint(name=ENDPOINT, config=cfg); print("creating", ENDPOINT)
else:
    dc.update_endpoint_config(endpoint=ENDPOINT, config=cfg); print("updating", ENDPOINT)

# COMMAND ----------

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
for _ in range(60):
    ep = w.serving_endpoints.get(ENDPOINT)
    ready, upd = str(ep.state.ready), str(ep.state.config_update)
    print(time.strftime("%H:%M:%S"), ready, upd)
    if "READY" in ready and "NOT_UPDATING" in upd: break
    time.sleep(30)
assert "READY" in ready, "endpoint not ready"

# COMMAND ----------
# MAGIC %md
# MAGIC ## Call it: only keys and planned inputs, features are looked up automatically

# COMMAND ----------

req = {"dataframe_records": [
    {"store_id": 101, "product_group": "Beverages", "promo_flag": 1, "holiday_flag": 0, "avg_temp_c": 4.0, "week_of_year": 47},
    {"store_id": 104, "product_group": "Dairy", "promo_flag": 0, "holiday_flag": 0, "avg_temp_c": 4.0, "week_of_year": 47}]}
resp = dc.predict(endpoint=ENDPOINT, inputs=req)
print(resp)

# COMMAND ----------

dbutils.notebook.exit(json.dumps({"endpoint": ENDPOINT, "model_version": champ.version, "predictions": resp.get("predictions")}))
