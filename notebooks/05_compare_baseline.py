# Databricks notebook source
# MAGIC %md
# MAGIC # 05. Zero-training baseline vs your own model
# MAGIC Both live in Unity Catalog and are governed the same way.
# MAGIC * `ai_forecast`: one SQL statement, no training, 10 week horizon from one cut-off date (2026-09-01)
# MAGIC * `demand_forecast` (LightGBM + features): next-week forecast that uses the latest actual sales as features
# MAGIC
# MAGIC **Read the numbers fairly:** the two do not do the same job. `ai_forecast` has to look 10 weeks ahead with no new data,
# MAGIC the custom model looks one week ahead with fresh features and planned promotions. We therefore also report week 1 only.

# COMMAND ----------

# MAGIC %pip install "databricks-feature-engineering>=0.16.0" "mlflow>=3.1" lightgbm scikit-learn "pyarrow>=16"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "agentbricks_demo_catalog")
dbutils.widgets.text("schema", "axfood_demo")
CATALOG = dbutils.widgets.get("catalog"); SCHEMA = dbutils.widgets.get("schema")
T = lambda n: f"{CATALOG}.{SCHEMA}.{n}"
import mlflow
from pyspark.sql import functions as F
from databricks.feature_engineering import FeatureEngineeringClient
mlflow.set_registry_uri("databricks-uc")
fe = FeatureEngineeringClient()

# COMMAND ----------

base = spark.table(T("ai_forecast_baseline"))                      # 20 series x 10 weeks
actual = spark.table(T("weekly_sales")).select("store_id", "product_group", "week_start", "units_sold", "promo_flag", "holiday_flag", "avg_temp_c", "week_of_year")
keys = (base.select("store_id", "product_group", "week_start").join(actual, ["store_id", "product_group", "week_start"])
        .withColumn("as_of_ts", F.col("week_start").cast("timestamp")).drop("units_sold"))
custom = (fe.score_batch(model_uri=f"models:/{T('demand_forecast')}@champion", df=keys)
          .select("store_id", "product_group", "week_start", F.col("prediction").alias("custom_model_units")))

cmp = (base.join(actual.select("store_id", "product_group", "week_start", "units_sold"), ["store_id", "product_group", "week_start"])
       .join(custom, ["store_id", "product_group", "week_start"])
       .select("store_id", "product_group", "week_start", "units_sold", "ai_forecast_units", "ai_forecast_lower", "ai_forecast_upper", "custom_model_units"))
cmp.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T("forecast_comparison"))
spark.sql(f"COMMENT ON TABLE {T('forecast_comparison')} IS 'Actual vs ai_forecast (zero training, 10 week horizon) vs custom LightGBM model (next week, with features), 20 series, last 10 weeks. Synthetic demo data.'")

# COMMAND ----------

def wape(df, col): 
    r = df.agg((F.sum(F.abs(F.col("units_sold") - F.col(col))) / F.sum("units_sold")).alias("w")).first()["w"]
    return round(float(r), 4)

c = spark.table(T("forecast_comparison"))
first = c.where(F.col("week_start") == F.lit("2026-09-07").cast("date"))
rows = [
    ("ai_forecast (zero training, no features)", "all 10 weeks, 10 week horizon", wape(c, "ai_forecast_units")),
    ("ai_forecast (zero training, no features)", "first week only", wape(first, "ai_forecast_units")),
    ("demand_forecast (LightGBM + UC features)", "all 10 weeks, next-week forecast", wape(c, "custom_model_units")),
    ("demand_forecast (LightGBM + UC features)", "first week only", wape(first, "custom_model_units")),
]
res = spark.createDataFrame(rows, "approach string, scope string, wape double")
res.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T("forecast_comparison_summary"))
spark.sql(f"COMMENT ON TABLE {T('forecast_comparison_summary')} IS 'WAPE (lower is better) of ai_forecast vs the custom model on the same 20 series and weeks. The horizons differ, see notebook 05. Synthetic demo data.'")
display(res)

# COMMAND ----------

import json
dbutils.notebook.exit(json.dumps({r[0].split(" ")[0] + "|" + r[1]: r[2] for r in rows}))
