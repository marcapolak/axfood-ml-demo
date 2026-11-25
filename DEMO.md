# Axfood demo: from notebook to production, no Docker, no DAG

## Audience and goal
Axfood data science team (new to Databricks). Today: custom Docker images + SageMaker + Airflow DAGs, even for simple projects.
Goal: show the same lifecycle on one platform, then prove how simple the "run" part is with a real deployed Job and a real pipeline.
All data is synthetic. Do not present numbers as Axfood results.

## Use case
Weekly grocery demand forecast per store and product group.

## Architecture (target, shown in the demo and the deck)
Unity Catalog governs everything.
1. Data: raw weekly sales files land in a UC volume. A Lakeflow Spark Declarative Pipeline ingests and transforms them: bronze (raw) -> silver (clean) -> gold (`weekly_sales`).
2. Features: Feature Engineering in UC (offline Delta, online store on Lakebase).
3. Train and track: serverless notebooks (CPU) and AI Runtime serverless GPU; MLflow runs.
4. Register: UC model, `@champion` alias.
5. Run: Lakeflow Jobs (pipeline job, daily batch scoring) and a Model Serving endpoint.
6. Consume: gold tables to Genie, BI, apps. AI Functions (`ai_forecast`, `ai_classify`, `ai_extract`) give zero-training results in SQL.
7. Git + Declarative Automation Bundles deploy dev/staging/prod.

Diagram: built in Solution Builder (go/lakedraw), exported as PNG, used in the notebook section 0 and the deck (after slide 19).

## Components
- `notebooks/Axfood_Demo.py`: the presenter notebook (section 0 = architecture + deployed jobs).
- `resources/pipeline.job.yml`: end-to-end pipeline job (4 tasks).
- `resources/daily_scoring.job.yml`: scheduled batch scoring job (PAUSED).
- `resources/axfood_sales.pipeline.yml` (new): Spark Declarative Pipeline bronze/silver/gold.
- `genie/build_space.py`: Genie space.

## Workspace
`fevm-agentbricks-demo` (AWS), catalog `agentbricks_demo_catalog`, schema `axfood_demo`. Expires 2026-11-27, rebuild with `scripts/setup.sh`.

## Style
Databricks 2024 brand: dark teal + coral. No emojis, no em dashes in text.
