# Databricks notebook source
# MAGIC %md
# MAGIC # 03. Train, track and register the demand model
# MAGIC * point-in-time training set from the Unity Catalog feature tables
# MAGIC * three LightGBM runs, tracked and comparable in MLflow (autologging)
# MAGIC * best run registered as a governed model `demand_forecast` with the `@champion` alias
# MAGIC
# MAGIC No Docker image, no SageMaker job, no DAG.

# COMMAND ----------

# MAGIC %pip install "databricks-feature-engineering>=0.16.0" "mlflow>=3.1" lightgbm scikit-learn "pyarrow>=16"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "agentbricks_demo_catalog")
dbutils.widgets.text("schema", "axfood_demo")
CATALOG = dbutils.widgets.get("catalog"); SCHEMA = dbutils.widgets.get("schema")
T = lambda n: f"{CATALOG}.{SCHEMA}.{n}"
MODEL = T("demand_forecast")
ME = spark.sql("SELECT current_user()").first()[0]
EXP_DIR = f"/Users/{ME}/axfood_demo"
import os; os.makedirs(f"/Workspace{EXP_DIR}", exist_ok=True)

import mlflow, numpy as np, pandas as pd, lightgbm as lgb
from mlflow.tracking import MlflowClient
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"{EXP_DIR}/demand_forecast_experiment")
fe = FeatureEngineeringClient()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Point-in-time training set
# MAGIC The label table holds the target week and the inputs known in advance (planned promotion, holiday, expected temperature).
# MAGIC Feature values are joined as of the start of the target week.

# COMMAND ----------

from pyspark.sql import functions as F
labels = (spark.table(T("weekly_sales"))
          .select("store_id", "product_group", "week_start", "units_sold", "promo_flag", "holiday_flag", "avg_temp_c", "week_of_year")
          .withColumn("as_of_ts", F.col("week_start").cast("timestamp")))

lookups = [
    FeatureLookup(table_name=T("store_features"), lookup_key="store_id"),
    FeatureLookup(table_name=T("store_product_features"), lookup_key=["store_id", "product_group"],
                  timestamp_lookup_key="as_of_ts"),
]
EXCLUDE = ["week_start", "as_of_ts", "store_id"]
CUTOFF = "2026-09-01"          # train before, test after (the most recent ~10 weeks)

def training_set(df):
    return fe.create_training_set(df=df, feature_lookups=lookups, label="units_sold", exclude_columns=EXCLUDE)

ts_train = training_set(labels.where(f"week_start < '{CUTOFF}'"))
ts_test = training_set(labels.where(f"week_start >= '{CUTOFF}'"))
train_pd, test_pd = ts_train.load_df().toPandas(), ts_test.load_df().toPandas()
X_train, y_train = train_pd.drop(columns=["units_sold"]), train_pd["units_sold"]
X_test, y_test = test_pd.drop(columns=["units_sold"]), test_pd["units_sold"]
print(X_train.shape, X_test.shape); print(list(X_train.columns))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Three runs, compared in MLflow

# COMMAND ----------

CAT = ["product_group", "region", "store_format"]
def build(params):
    prep = ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT)],
                             remainder="passthrough", verbose_feature_names_out=False)
    return Pipeline([("prep", prep), ("model", lgb.LGBMRegressor(verbose=-1, random_state=7, **params))])

def wape(y, p): return float(np.abs(y - p).sum() / np.abs(y).sum())

CONFIGS = {
    "baseline_small":  dict(n_estimators=150, learning_rate=0.10, num_leaves=15),
    "balanced":        dict(n_estimators=400, learning_rate=0.05, num_leaves=31),
    "deeper":          dict(n_estimators=700, learning_rate=0.03, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.8),
}
naive = float(wape(y_test.values, X_test["sales_lag_1w"].fillna(y_train.mean()).values))   # "same as last week" benchmark
print("naive last-week WAPE:", round(naive, 4))

mlflow.sklearn.autolog(log_models=False, silent=True)    # params, metrics and artifacts logged automatically
results = {}
for name, params in CONFIGS.items():
    with mlflow.start_run(run_name=name) as run:
        pipe = build(params).fit(X_train, y_train)
        pred = pipe.predict(X_test)
        m = {"test_wape": wape(y_test.values, pred), "test_rmse": float(np.sqrt(np.mean((y_test.values - pred) ** 2))),
             "naive_last_week_wape": naive}
        mlflow.log_metrics(m)
        results[name] = (run.info.run_id, m["test_wape"], pipe)
        print(name, {k: round(v, 4) for k, v in m.items()})

best = min(results, key=lambda k: results[k][1])
print("best run:", best, round(results[best][1], 4))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Register the winner to Unity Catalog with feature lineage
# MAGIC `fe.log_model` stores which feature tables the model needs, so batch scoring and the serving endpoint look them up automatically.

# COMMAND ----------

with mlflow.start_run(run_name=f"register_{best}") as run:
    mlflow.log_metrics({"test_wape": results[best][1], "naive_last_week_wape": naive})
    mlflow.log_params({"chosen_config": best, **CONFIGS[best]})
    fe.log_model(model=results[best][2], artifact_path="model", flavor=mlflow.sklearn,
                 training_set=ts_train, registered_model_name=MODEL)

client = MlflowClient(registry_uri="databricks-uc")
latest = max(client.search_model_versions(f"name='{MODEL}'"), key=lambda v: int(v.version))
client.set_registered_model_alias(MODEL, "champion", latest.version)
client.update_registered_model(MODEL, description="Weekly grocery demand forecast per store and product group. Synthetic demo data.")
print("registered", MODEL, "version", latest.version, "alias @champion")

# COMMAND ----------

import json
dbutils.notebook.exit(json.dumps({"model": MODEL, "version": latest.version, "chosen": best, "test_wape": round(results[best][1], 4), "naive_wape": round(naive, 4)}))
