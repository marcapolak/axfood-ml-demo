-- Lakeflow Spark Declarative Pipeline: bronze -> silver -> gold.
-- Source is the synthetic weekly_sales table, treated as the "raw" feed. All data is synthetic.
-- Declare the result you want; the pipeline handles ordering, incremental refresh, retries and lineage.

CREATE OR REFRESH MATERIALIZED VIEW pl_bronze_sales
COMMENT 'Bronze: raw weekly sales as received. Synthetic demo data.'
AS SELECT * FROM ${catalog}.${schema}.weekly_sales;

CREATE OR REFRESH MATERIALIZED VIEW pl_silver_sales (
  CONSTRAINT units_not_negative EXPECT (units_sold >= 0) ON VIOLATION DROP ROW,
  CONSTRAINT store_known EXPECT (store_name IS NOT NULL)
)
COMMENT 'Silver: validated rows joined with store and product master data.'
AS SELECT s.store_id, st.store_name, st.region, st.store_format,
          s.product_group, p.category, s.week_start,
          s.units_sold, s.promo_flag, s.avg_temp_c, s.holiday_flag
   FROM pl_bronze_sales s
   LEFT JOIN ${catalog}.${schema}.stores st USING (store_id)
   LEFT JOIN ${catalog}.${schema}.products p USING (product_group);

CREATE OR REFRESH MATERIALIZED VIEW pl_gold_store_week
COMMENT 'Gold: units per store and week by category, ready for features, BI and Genie.'
AS SELECT store_id, store_name, region, category, week_start,
          sum(units_sold) AS units_sold, max(promo_flag) AS any_promo
   FROM pl_silver_sales
   GROUP BY store_id, store_name, region, category, week_start;
