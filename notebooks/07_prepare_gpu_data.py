# Databricks notebook source
# MAGIC %md
# MAGIC # 07. One governed table feeds GPU training too
# MAGIC Materialise the same point-in-time feature join used for the LightGBM model into one numeric Delta table.
# MAGIC The GPU job reads it **directly from Unity Catalog** (UCDataset, Beta): no Parquet export, no toPandas, no Spark session on the GPU node.

# COMMAND ----------

# MAGIC %pip install "databricks-feature-engineering>=0.16.0" "mlflow>=3.1" "pyarrow>=16"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "agentbricks_demo_catalog")
dbutils.widgets.text("schema", "axfood_demo")
CATALOG = dbutils.widgets.get("catalog"); SCHEMA = dbutils.widgets.get("schema")
T = lambda n: f"{CATALOG}.{SCHEMA}.{n}"
from pyspark.sql import functions as F
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
fe = FeatureEngineeringClient()

labels = (spark.table(T("weekly_sales"))
          .select("store_id", "product_group", "week_start", "units_sold", "promo_flag", "holiday_flag", "avg_temp_c", "week_of_year")
          .withColumn("as_of_ts", F.col("week_start").cast("timestamp")))
ts = fe.create_training_set(df=labels, label="units_sold", exclude_columns=["as_of_ts"], feature_lookups=[
    FeatureLookup(table_name=T("store_features"), lookup_key="store_id"),
    FeatureLookup(table_name=T("store_product_features"), lookup_key=["store_id", "product_group"], timestamp_lookup_key="as_of_ts")])
df = ts.load_df()

groups = [r[0] for r in spark.table(T("products")).orderBy("product_group").select("product_group").collect()]
regions = [r[0] for r in spark.table(T("stores")).select("region").distinct().orderBy("region").collect()]
formats = ["Convenience", "Supermarket", "Hypermarket"]
def code(col, values):
    expr = F.lit(-1.0)
    for i, v in reversed(list(enumerate(values))): expr = F.when(F.col(col) == v, F.lit(float(i))).otherwise(expr)
    return expr

num = lambda c: F.coalesce(F.col(c).cast("double"), F.lit(0.0)).alias(c)
out = df.select(
    F.col("units_sold").cast("double").alias("label"),
    code("product_group", groups).alias("product_group_code"), code("region", regions).alias("region_code"), code("store_format", formats).alias("format_code"),
    num("promo_flag"), num("holiday_flag"), num("avg_temp_c"), num("week_of_year"), num("sales_area_sqm"), num("has_bakery"), num("opened_year"),
    num("sales_lag_1w"), num("sales_avg_4w"), num("sales_avg_13w"), num("sales_std_4w"), num("sales_same_week_last_year"), num("trend_4w_vs_13w"),
    F.col("sales_same_week_last_year").isNull().cast("double").alias("no_last_year"),
    (F.col("week_start") >= F.lit("2026-09-01").cast("date")).cast("double").alias("is_test"))
out.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T("gpu_training_data"))
spark.sql(f"COMMENT ON TABLE {T('gpu_training_data')} IS 'Numeric point-in-time training table for the GPU neural forecaster (read with UCDataset). is_test=1 for weeks from 2026-09-01. Synthetic demo data.'")
n = spark.table(T("gpu_training_data")).count()
print(n)

# Also write a Parquet copy to a Unity Catalog volume. This is the default data path of the GPU job because
# UCDataset (Beta) needs direct storage access that is not allowed by the storage role in the demo workspace.
import os
vol_dir = f"/Volumes/{CATALOG}/{SCHEMA}/gpu_data"
os.makedirs(vol_dir, exist_ok=True)
pdf = spark.table(T("gpu_training_data")).toPandas()
pdf.attrs = {}      # Spark Connect adds a metrics object to attrs that pyarrow cannot serialise
pdf.to_parquet(f"{vol_dir}/gpu_training_data.parquet", index=False)
print("wrote", f"{vol_dir}/gpu_training_data.parquet")

# COMMAND ----------

import json
dbutils.notebook.exit(json.dumps({"gpu_training_rows": n}))
