#!/usr/bin/env python3
"""Create or update the Axfood demo Genie space. Usage: build_space.py [--profile P] [--catalog C] [--schema S] [--warehouse W]
Idempotent: if a space with the same title exists in the parent folder it is updated, otherwise created."""
import argparse, json, subprocess, hashlib
ap = argparse.ArgumentParser()
ap.add_argument("--profile", default="fevm-agentbricks-demo"); ap.add_argument("--catalog", default="agentbricks_demo_catalog")
ap.add_argument("--schema", default="axfood_demo"); ap.add_argument("--warehouse", default="7046f7a2659b62ba")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
TITLE = "Axfood demo: demand forecast and store feedback"
P = f"{a.catalog}.{a.schema}"
hid = lambda s: hashlib.md5(s.encode()).hexdigest()           # stable 32 hex ids
def cli(*args):
    r = subprocess.run(["databricks", *args, "--profile", a.profile], capture_output=True, text=True)
    if r.returncode: raise SystemExit(r.stderr or r.stdout)
    return r.stdout

samples = ["Which stores are forecast to be short next week?", "Which product groups have the biggest forecast jump over the recent average?",
           "How did the zero training ai_forecast baseline compare with our custom model?", "What are customers complaining about most, by topic?"]
examples = [
 ("Which stores are forecast to be short next week?",
  f"SELECT gold_store_stock_risk.store_name, gold_store_stock_risk.product_group, gold_store_stock_risk.forecast_units, gold_store_stock_risk.avg_units_last_4_weeks, gold_store_stock_risk.pct_above_recent_average FROM {P}.gold_store_stock_risk WHERE gold_store_stock_risk.risk_flag = 'SHORT_RISK' ORDER BY gold_store_stock_risk.pct_above_recent_average DESC LIMIT 20"),
 ("Which product groups have the most stores at risk next week?",
  f"SELECT gold_store_stock_risk.product_group, COUNT(*) AS stores_at_risk FROM {P}.gold_store_stock_risk WHERE gold_store_stock_risk.risk_flag = 'SHORT_RISK' GROUP BY gold_store_stock_risk.product_group ORDER BY stores_at_risk DESC"),
 ("What are customers complaining about most?",
  f"SELECT customer_reviews_enriched.topic, COUNT(*) AS reviews FROM {P}.customer_reviews_enriched WHERE customer_reviews_enriched.sentiment = 'negative' GROUP BY customer_reviews_enriched.topic ORDER BY reviews DESC"),
]
def cc(name, **kw): return {"column_name": name, **kw}
tables = {
 "gold_store_stock_risk": [cc("forecast_units", description=["Forecast units for the coming week from the champion demand model"]),
                           cc("pct_above_recent_average", description=["Forecast versus the average of the last 4 weeks, in percent"], synonyms=["jump", "increase", "uplift"]),
                           cc("risk_flag", description=["SHORT_RISK when the forecast is more than 15 percent above the recent 4 week average, else OK"], synonyms=["short", "stock risk", "shortage"], enable_format_assistance=True, enable_entity_matching=True),
                           cc("product_group", enable_format_assistance=True, enable_entity_matching=True), cc("region", enable_format_assistance=True, enable_entity_matching=True),
                           cc("store_format", enable_format_assistance=True, enable_entity_matching=True)],
 "forecast_comparison": [cc("units_sold", description=["Actual units sold in the week"]), cc("ai_forecast_units", description=["Zero training ai_forecast prediction, 10 week horizon"]),
                         cc("custom_model_units", description=["Custom LightGBM model with Unity Catalog features, next week forecast"])],
 "forecast_comparison_summary": [cc("wape", description=["Weighted absolute percentage error, lower is better"], synonyms=["error", "accuracy"])],
 "customer_reviews_enriched": [cc("topic", description=["Topic from ai_classify: out_of_stock, quality, price, delivery or other"], enable_format_assistance=True, enable_entity_matching=True),
                               cc("sentiment", enable_format_assistance=True, enable_entity_matching=True), cc("extract_error", exclude=True)],
}
space = {"version": 2,
 "config": {"sample_questions": sorted([{"id": hid("s" + q), "question": [q]} for q in samples], key=lambda x: x["id"])},
 "data_sources": {"tables": sorted([{"identifier": f"{P}.{t}", "description": [d], "column_configs": sorted(cols, key=lambda c: c["column_name"])}
    for t, cols, d in [
      ("gold_store_stock_risk", tables["gold_store_stock_risk"], "Next week forecast per store and product group versus the recent 4 week average, with a stock risk flag. One row per store and product group. Synthetic demo data."),
      ("forecast_comparison", tables["forecast_comparison"], "Actual units versus ai_forecast and the custom model for 20 store and product group series over the last 10 weeks. Synthetic demo data."),
      ("forecast_comparison_summary", tables["forecast_comparison_summary"], "WAPE of ai_forecast versus the custom model. The horizons differ: ai_forecast looks 10 weeks ahead, the custom model one week ahead. Synthetic demo data."),
      ("customer_reviews_enriched", tables["customer_reviews_enriched"], "Synthetic customer reviews classified with ai_classify, with sentiment and extracted fields. One row per review.")]],
    key=lambda x: x["identifier"])},
 "instructions": {"example_question_sqls": sorted([{"id": hid("e" + q), "question": [q], "sql": [sql]} for q, sql in examples], key=lambda x: x["id"])}}
payload = json.dumps(space, separators=(",", ":"))
if a.dry_run: print(json.dumps(space, indent=1)); raise SystemExit
me = json.loads(cli("current-user", "me"))["userName"]; parent = f"/Workspace/Users/{me}/genie_spaces"
cli("workspace", "mkdirs", parent)
existing = [s for s in json.loads(cli("genie", "list-spaces", "-o", "json") or "{}").get("spaces", []) if s.get("title") == TITLE]
desc = "Ask which stores are forecast to be short next week, how the zero training baseline compares with the custom model, and what customers complain about. Synthetic data."
if existing:
    sid = existing[0]["space_id"]
    cli("genie", "update-space", sid, "--json", json.dumps({"title": TITLE, "description": desc, "warehouse_id": a.warehouse, "serialized_space": payload})); print("updated", sid)
else:
    out = json.loads(cli("genie", "create-space", "--json", json.dumps({"warehouse_id": a.warehouse, "title": TITLE, "description": desc, "parent_path": parent, "serialized_space": payload})))
    sid = out.get("space_id"); print("created", sid)
print(f"https://fevm-agentbricks-demo.cloud.databricks.com/genie/rooms/{sid}")
