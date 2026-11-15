# Databricks notebook source
# MAGIC %md
# MAGIC # 02. Features in Unity Catalog (Feature Store)
# MAGIC Two governed feature tables, defined once and reused for training, batch scoring and online serving:
# MAGIC * `store_features`: static store attributes, keyed by `store_id`
# MAGIC * `store_product_features`: rolling demand features, keyed by `store_id, product_group` with a timestamp for **point-in-time correct** joins
# MAGIC
# MAGIC Re-running refreshes the tables (idempotent).

# COMMAND ----------

# MAGIC %pip install "databricks-feature-engineering>=0.16.0" "mlflow>=3.1"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "agentbricks_demo_catalog")
dbutils.widgets.text("schema", "axfood_demo")
CATALOG = dbutils.widgets.get("catalog"); SCHEMA = dbutils.widgets.get("schema")
from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import functions as F, Window
fe = FeatureEngineeringClient()
T = lambda n: f"{CATALOG}.{SCHEMA}.{n}"

# COMMAND ----------
# MAGIC %md
# MAGIC ## Static store features

# COMMAND ----------

store_df = spark.table(T("stores")).select("store_id", "region", "store_format", "sales_area_sqm", "has_bakery", "opened_year")

def upsert_feature_table(name, df, pks, ts=None, description=""):
    full = T(name)
    if spark.catalog.tableExists(full):
        spark.sql(f"DROP TABLE {full}")      # demo tables only: recreate so the notebook is re-runnable
    kwargs = dict(name=full, primary_keys=pks, df=df, description=description)
    if ts: kwargs["timeseries_column"] = ts
    fe.create_table(**kwargs)
    # online serving later needs change data feed
    spark.sql(f"ALTER TABLE {full} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

upsert_feature_table("store_features", store_df, ["store_id"],
                     description="Static store attributes (synthetic demo data)")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Rolling demand features (point in time)
# MAGIC `as_of_ts` is the moment the feature row became knowable: the end of the week it describes.
# MAGIC The training set joins each target week to the latest row with `as_of_ts <= target week start`,
# MAGIC so the model never sees the future.

# COMMAND ----------

sales = spark.table(T("weekly_sales")).select("store_id", "product_group", "week_start", "units_sold")
w = Window.partitionBy("store_id", "product_group").orderBy("week_start")
w4, w13 = w.rowsBetween(-3, 0), w.rowsBetween(-12, 0)

feat = (sales
    .withColumn("sales_lag_1w", F.col("units_sold").cast("double"))
    .withColumn("sales_avg_4w", F.avg("units_sold").over(w4))
    .withColumn("sales_avg_13w", F.avg("units_sold").over(w13))
    .withColumn("sales_std_4w", F.coalesce(F.stddev("units_sold").over(w4), F.lit(0.0)))
    # the target week is week_start + 1 week, so the same week last year is 51 rows back from this row
    .withColumn("sales_same_week_last_year", F.lag("units_sold", 51).over(w).cast("double"))
    .withColumn("trend_4w_vs_13w", F.col("sales_avg_4w") / F.col("sales_avg_13w"))
    .withColumn("as_of_ts", (F.col("week_start").cast("timestamp") + F.expr("INTERVAL 7 DAYS")))
    .select("store_id", "product_group", "as_of_ts", "sales_lag_1w", "sales_avg_4w", "sales_avg_13w",
            "sales_std_4w", "sales_same_week_last_year", "trend_4w_vs_13w"))

upsert_feature_table("store_product_features", feat, ["store_id", "product_group", "as_of_ts"], ts="as_of_ts",
                     description="Rolling weekly demand features per store and product group (synthetic demo data)")

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {T('store_product_features')} WHERE store_id = 101 AND product_group = 'Dairy' ORDER BY as_of_ts DESC LIMIT 5"))

# COMMAND ----------

import json
n = spark.table(T("store_product_features")).count()
dbutils.notebook.exit(json.dumps({"store_product_features_rows": n, "store_features_rows": spark.table(T("store_features")).count()}))
