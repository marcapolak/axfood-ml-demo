-- ai_classify + ai_analyze_sentiment + ai_extract on free text, straight from SQL. Run once, persist to Delta.
-- Requires a Pro or Serverless SQL warehouse (not Classic).
CREATE OR REPLACE TABLE ${catalog}.${schema}.customer_reviews_enriched
COMMENT 'Customer reviews enriched with AI Functions: topic (ai_classify), sentiment, extracted fields (ai_extract). Synthetic demo data.'
AS
WITH scored AS (
  SELECT *,
    ai_classify(review_text, '["out_of_stock","quality","price","delivery","other"]', map('version','2.0')):response[0]::STRING AS topic,
    ai_analyze_sentiment(review_text) AS sentiment,
    ai_extract(review_text,
      '{"product":{"type":"string","description":"the grocery product mentioned"},
        "store_name":{"type":"string","description":"store the customer mentions"},
        "days_or_hours":{"type":"number","description":"number of days or hours mentioned"},
        "price_sek":{"type":"number","description":"first price in SEK or kr mentioned"}}',
      map('version','2.0')) AS ex
  FROM ${catalog}.${schema}.customer_reviews)
SELECT review_id, store_id, product_group, review_date, review_text, topic, sentiment,
       ex:response:product::STRING AS product, ex:response:store_name::STRING AS store_name,
       ex:response:days_or_hours::DOUBLE AS days_or_hours, ex:response:price_sek::DOUBLE AS price_sek,
       ex:error_message::STRING AS extract_error
FROM scored;
