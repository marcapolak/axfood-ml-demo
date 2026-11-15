# Databricks notebook source
# MAGIC %md
# MAGIC # 01. Synthetic grocery data
# MAGIC Weekly unit sales per store and product group, plus promotions, holidays and weather.
# MAGIC **All data is synthetic.** No real customer data is used anywhere in this demo.
# MAGIC Re-running this notebook rebuilds the tables (idempotent).

# COMMAND ----------

dbutils.widgets.text("catalog", "agentbricks_demo_catalog")
dbutils.widgets.text("schema", "axfood_demo")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.gpu_data")

# COMMAND ----------

import numpy as np, pandas as pd

rng = np.random.default_rng(42)

N_STORES = 40
GROUPS = {  # base weekly units for an average store, seasonality profile, promo uplift, temperature sensitivity
    "Dairy":               (900, "flat",      0.18,  0.000),
    "Bread and bakery":    (700, "flat",      0.15,  0.000),
    "Fruit and vegetables":(1100,"summer",    0.22,  0.012),
    "Meat":                (800, "grill",     0.25,  0.010),
    "Fish":                (300, "christmas", 0.20,  0.000),
    "Frozen":              (500, "flat",      0.20, -0.003),
    "Beverages":           (1300,"summer",    0.28,  0.020),
    "Snacks and candy":    (650, "christmas", 0.30,  0.002),
    "Household":           (450, "flat",      0.12,  0.000),
    "Personal care":       (350, "flat",      0.14,  0.000),
    "Pet food":            (260, "flat",      0.10,  0.000),
    "Ready meals":         (420, "winter",    0.24, -0.004),
}
REGIONS = {"Stockholm": 6.5, "Gothenburg": 7.5, "Malmo": 9.0, "Uppsala": 5.5, "Orebro": 6.0, "Norrland": 2.0}
FORMATS = {"Convenience": 0.45, "Supermarket": 1.0, "Hypermarket": 1.9}

# COMMAND ----------
# MAGIC %md
# MAGIC ## Stores and product groups

# COMMAND ----------

region_names = list(REGIONS)
fmt_names = list(FORMATS)
stores = pd.DataFrame({
    "store_id": np.arange(101, 101 + N_STORES),
    "region": rng.choice(region_names, N_STORES, p=[.2, .18, .15, .12, .15, .2]),
    "store_format": rng.choice(fmt_names, N_STORES, p=[.3, .5, .2]),
})
stores["store_name"] = "Store " + stores["store_id"].astype(str)
stores["sales_area_sqm"] = [int(rng.normal(400, 60) * FORMATS[f] * 2) for f in stores["store_format"]]
stores["has_bakery"] = rng.random(N_STORES) < 0.6
stores["opened_year"] = rng.integers(1995, 2024, N_STORES)
stores["size_index"] = stores["store_format"].map(FORMATS) * rng.normal(1, 0.12, N_STORES)

products = pd.DataFrame({"product_group": list(GROUPS)})
products["base_units"] = [g[0] for g in GROUPS.values()]
products["category"] = ["Chilled", "Fresh", "Fresh", "Fresh", "Fresh", "Frozen", "Ambient", "Ambient", "Non-food", "Non-food", "Non-food", "Chilled"]

# COMMAND ----------
# MAGIC %md
# MAGIC ## Calendar, holidays and weather

# COMMAND ----------

weeks = pd.date_range("2024-01-01", "2026-11-09", freq="W-MON")  # Mondays
cal = pd.DataFrame({"week_start": weeks})
cal["week_of_year"] = cal["week_start"].dt.isocalendar().week.astype(int)
cal["year"] = cal["week_start"].dt.year

def holiday(ws):
    we = ws + pd.Timedelta(days=6)
    m, d = ws.month, ws.day
    flags = []
    if (m == 12 and d >= 18) or (m == 12 and we.day <= 31 and ws.day >= 18): flags.append("christmas")
    if m == 6 and 17 <= d <= 24: flags.append("midsummer")
    if (m == 3 and d >= 25) or (m == 4 and d <= 21 and d >= 1 and ws.isocalendar().week in (13, 14, 15, 16, 17)):
        if ws.isocalendar().week in (14, 15, 16): flags.append("easter")
    return ",".join(flags)

cal["holiday"] = cal["week_start"].map(holiday)
cal["holiday_flag"] = (cal["holiday"] != "").astype(int)

# COMMAND ----------

# Sales simulation
rows = []
doy = cal["week_start"].dt.dayofyear.values
for _, s in stores.iterrows():
    base_temp = REGIONS[s["region"]]
    temp = base_temp + 11 * np.sin((doy - 110) / 365 * 2 * np.pi) + rng.normal(0, 2.2, len(cal))
    for g, (base, prof, uplift, tsens) in GROUPS.items():
        season = np.ones(len(cal))
        woy = cal["week_of_year"].values
        if prof == "summer":    season = 1 + 0.25 * np.exp(-((woy - 28) ** 2) / 90)
        if prof == "winter":    season = 1 + 0.18 * np.exp(-((np.minimum(woy, 53 - woy)) ** 2) / 60)
        if prof == "christmas": season = 1 + 0.55 * (np.exp(-((woy - 51) ** 2) / 4) + np.exp(-((woy + 1) ** 2) / 4))
        if prof == "grill":     season = 1 + 0.30 * np.exp(-((woy - 26) ** 2) / 50)
        hol = cal["holiday"].values
        hol_mult = np.where(np.char.find(hol.astype(str), "christmas") >= 0, 1.25, 1.0) * np.where(np.char.find(hol.astype(str), "midsummer") >= 0, 1.2, 1.0) * np.where(np.char.find(hol.astype(str), "easter") >= 0, 1.12, 1.0)
        promo = (rng.random(len(cal)) < 0.18).astype(int)
        trend = 1 + 0.0007 * np.arange(len(cal))
        mu = base * s["size_index"] * season * hol_mult * trend * (1 + uplift * promo) * (1 + tsens * (temp - 8))
        units = np.maximum(0, rng.normal(mu, 0.07 * mu)).round().astype(int)
        rows.append(pd.DataFrame({
            "store_id": s["store_id"], "product_group": g, "week_start": cal["week_start"].values,
            "units_sold": units, "promo_flag": promo, "avg_temp_c": temp.round(1),
            "holiday_flag": cal["holiday_flag"].values, "week_of_year": cal["week_of_year"].values,
        }))
sales = pd.concat(rows, ignore_index=True)
print(sales.shape, sales["week_start"].min(), sales["week_start"].max())

# COMMAND ----------
# MAGIC %md
# MAGIC ## Write to Unity Catalog

# COMMAND ----------

def write(df, name, **kw):
    spark.createDataFrame(df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.{name}")

write(stores.drop(columns=["size_index"]).assign(has_bakery=lambda d: d["has_bakery"].astype(int)), "stores")
write(products, "products")
write(cal.assign(week_start=cal["week_start"].dt.date), "calendar")
sdf = spark.createDataFrame(sales).withColumn("week_start", __import__("pyspark.sql.functions", fromlist=["to_date"]).to_date("week_start"))
sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.weekly_sales")

spark.sql(f"COMMENT ON TABLE {CATALOG}.{SCHEMA}.weekly_sales IS 'Synthetic weekly unit sales per store and product group, with promotion, holiday and temperature columns. Demo data only.'")
spark.sql(f"COMMENT ON TABLE {CATALOG}.{SCHEMA}.stores IS 'Synthetic store master data. Demo data only.'")
spark.sql(f"COMMENT ON TABLE {CATALOG}.{SCHEMA}.products IS 'Synthetic product groups. Demo data only.'")

display(spark.sql(f"SELECT count(*) AS rows, min(week_start) AS first_week, max(week_start) AS last_week FROM {CATALOG}.{SCHEMA}.weekly_sales"))

# COMMAND ----------

import json
dbutils.notebook.exit(json.dumps({"rows": int(sales.shape[0]), "stores": int(len(stores)), "groups": len(GROUPS), "last_week": str(sales["week_start"].max().date())}))
