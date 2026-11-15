-- Zero-training baseline with ai_forecast. No model, no features, no endpoint: one SQL statement.
-- 20 series (5 stores x 4 product groups with different seasonality). The last 10 weeks are held out,
-- the same test window used for the custom LightGBM model, so the two can be compared fairly.
-- Version 1 (Public Preview) is used because Predictive AI Functions (needed for version 2) is not enabled here.
-- When it is enabled run sql/01_ai_forecast_baseline_v2.sql instead (holiday support, covariates).
CREATE OR REPLACE TABLE ${catalog}.${schema}.ai_forecast_baseline
COMMENT 'ai_forecast (zero training) next 10 weeks per store and product group, trained on data before 2026-09-01. Synthetic demo data.'
AS
SELECT CAST(split(series, '[|]')[0] AS BIGINT) AS store_id,
       split(series, '[|]')[1]              AS product_group,
       week_start,
       units_sold_forecast                  AS ai_forecast_units,
       units_sold_lower                     AS ai_forecast_lower,
       units_sold_upper                     AS ai_forecast_upper
FROM ai_forecast(
  observed => TABLE(
    SELECT week_start, units_sold, concat(store_id, '|', product_group) AS series
    FROM ${catalog}.${schema}.weekly_sales
    WHERE store_id IN (101, 102, 103, 104, 105)
      AND product_group IN ('Dairy', 'Beverages', 'Fish', 'Ready meals')
      AND week_start < DATE'2026-09-01'),
  horizon => '2026-11-09', time_col => 'week_start', value_col => 'units_sold', group_col => 'series', version => '1');
