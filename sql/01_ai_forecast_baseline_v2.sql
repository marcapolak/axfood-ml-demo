-- Version 2 (Beta): time series foundation model, built in holiday calendar, non-negative forecasts.
-- Needs the Predictive AI Functions preview enabled by a workspace admin (Manage Databricks previews).
-- NOT run in the current workspace (preview disabled). Same table name, so everything downstream keeps working.
CREATE OR REPLACE TABLE ${catalog}.${schema}.ai_forecast_baseline
COMMENT 'ai_forecast version 2 (zero training) next 10 weeks per store and product group. Synthetic demo data.'
AS
SELECT CAST(split(series, '[|]')[0] AS BIGINT) AS store_id, split(series, '[|]')[1] AS product_group, week_start,
       units_sold_forecast AS ai_forecast_units, units_sold_lower AS ai_forecast_lower, units_sold_upper AS ai_forecast_upper
FROM ai_forecast(
  observed => TABLE(
    SELECT week_start, units_sold, concat(store_id, '|', product_group) AS series
    FROM ${catalog}.${schema}.weekly_sales
    WHERE store_id IN (101, 102, 103, 104, 105)
      AND product_group IN ('Dairy', 'Beverages', 'Fish', 'Ready meals')
      AND week_start < DATE'2026-09-01'),
  horizon => '2026-11-09', time_col => 'week_start', value_col => 'units_sold', group_col => 'series',
  holiday_region => 'SE', positive_only => true, version => '2');
