# Databricks notebook source
# MAGIC %md
# MAGIC # 04. Daily batch scoring
# MAGIC Scores next week's demand for every store and product group with the `@champion` model.
# MAGIC Features are looked up automatically from the feature tables the model was trained with.
# MAGIC This notebook is the task in the scheduled Lakeflow Job (the replacement for an Airflow DAG).

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
# MAGIC %md
# MAGIC ## Inputs for the coming week
# MAGIC In production the planned promotions, holidays and weather forecast come from other tables. Here we derive a simple plan from the calendar.

# COMMAND ----------

last_week = spark.table(T("weekly_sales")).agg(F.max("week_start")).first()[0]
target_week = spark.sql(f"SELECT date_add(DATE'{last_week}', 7)").first()[0]
print("last observed week:", last_week, "| forecasting week starting:", target_week)

keys = (spark.table(T("weekly_sales")).where(F.col("week_start") == F.lit(last_week))
        .select("store_id", "product_group", "promo_flag", "avg_temp_c")
        .withColumn("week_start", F.lit(target_week).cast("date"))
        .withColumn("week_of_year", F.weekofyear("week_start"))
        .withColumn("holiday_flag", F.lit(0))
        .withColumn("as_of_ts", F.col("week_start").cast("timestamp")))

# COMMAND ----------

scored = fe.score_batch(model_uri=f"models:/{T('demand_forecast')}@champion", df=keys)
out = (scored.select("store_id", "product_group", "week_start", "promo_flag", "avg_temp_c", "prediction")   # drop looked-up feature columns
       .join(spark.table(T("stores")).select("store_id", "store_name", "region", "store_format"), "store_id")
       .withColumn("forecast_units", F.round("prediction").cast("int"))
       .select("store_id", "store_name", "region", "store_format", "product_group", "week_start", "forecast_units", "promo_flag", "avg_temp_c")
       .withColumn("scored_at", F.current_timestamp()))
out.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T("gold_weekly_forecast"))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Compare with recent demand
# MAGIC "Short" means the forecast is more than 15% above the 4 week average, so the store may need more stock than the standard replenishment.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {T('gold_store_stock_risk')}
COMMENT 'Next week forecast vs recent 4 week average per store and product group. risk_flag = forecast more than 15 percent above recent average. Synthetic demo data.'
AS
SELECT f.store_id, f.store_name, f.region, f.store_format, f.product_group, f.week_start AS forecast_week,
       f.forecast_units,
       ROUND(h.avg_4w, 0) AS avg_units_last_4_weeks,
       ROUND((f.forecast_units - h.avg_4w) / h.avg_4w * 100, 1) AS pct_above_recent_average,
       CASE WHEN f.forecast_units > 1.15 * h.avg_4w THEN 'SHORT_RISK' ELSE 'OK' END AS risk_flag,
       f.promo_flag
FROM {T('gold_weekly_forecast')} f
JOIN (SELECT store_id, product_group, AVG(units_sold) AS avg_4w
      FROM {T('weekly_sales')}
      WHERE week_start > date_sub(DATE'{last_week}', 28) GROUP BY store_id, product_group) h
  USING (store_id, product_group)
""")
n_rows = spark.table(T("gold_weekly_forecast")).count()
n_risk = spark.table(T("gold_store_stock_risk")).where("risk_flag = 'SHORT_RISK'").count()
display(spark.table(T("gold_store_stock_risk")).orderBy(F.desc("pct_above_recent_average")).limit(10))

# COMMAND ----------

import json
dbutils.notebook.exit(json.dumps({"forecast_rows": n_rows, "short_risk_rows": n_risk, "forecast_week": str(target_week)}))
