"""Small neural demand forecaster on AI Runtime (serverless GPU). Tracks to MLflow.
Data path (DATA_SOURCE env var):
  volume (default): read a Parquet copy of the table from a Unity Catalog volume (written by notebook 07). Fine for small tables.
  ucdataset: stream the table straight from storage with UCDataset (Beta). Needs the workspace storage role to allow
             s3:ListBucket on the catalog bucket. Launch with torchrun (see workload.yaml)."""
import os, time
import numpy as np, torch, torch.nn as nn
import mlflow

CATALOG = os.environ.get("CATALOG", "agentbricks_demo_catalog"); SCHEMA = os.environ.get("SCHEMA", "axfood_demo")
EPOCHS = int(os.environ.get("EPOCHS", "12")); DATA_SOURCE = os.environ.get("DATA_SOURCE", "volume")
TABLE = f"{CATALOG}.{SCHEMA}.gpu_training_data"
FEATURES = ["product_group_code", "region_code", "format_code", "promo_flag", "holiday_flag", "avg_temp_c", "week_of_year", "sales_area_sqm",
            "has_bakery", "opened_year", "sales_lag_1w", "sales_avg_4w", "sales_avg_13w", "sales_std_4w", "sales_same_week_last_year",
            "trend_4w_vs_13w", "no_last_year"]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", dev, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

def batches_from_arrays(X, y, bs, shuffle):
    idx = torch.randperm(len(X)) if shuffle else torch.arange(len(X))
    for i in range(0, len(X), bs):
        j = idx[i:i + bs]; yield X[j], y[j]

if DATA_SOURCE == "ucdataset":
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        torch.distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    from torch.utils.data import DataLoader
    from databricks.uc_dataset.beta import UCDataset
    from databricks.uc_dataset.beta.dataframe import table, col
    def stream(is_test, bs):
        df = table(TABLE).select(*FEATURES, "label", "is_test").filter(col("is_test") == (1.0 if is_test else 0.0))
        for b in DataLoader(UCDataset(df=df, batch_size=bs), batch_size=None, num_workers=2):
            yield torch.stack([torch.as_tensor(b[f], dtype=torch.float32) for f in FEATURES], 1), torch.as_tensor(b["label"], dtype=torch.float32)
else:
    import pandas as pd
    pdf = pd.read_parquet(f"/Volumes/{CATALOG}/{SCHEMA}/gpu_data/gpu_training_data.parquet")
    def split(flag):
        d = pdf[pdf["is_test"] == flag]
        return torch.tensor(d[FEATURES].values, dtype=torch.float32), torch.tensor(d["label"].values, dtype=torch.float32)
    TR, TE = split(0.0), split(1.0)
    def stream(is_test, bs):
        X, y = TE if is_test else TR
        yield from batches_from_arrays(X, y, bs, shuffle=not is_test)

n, s, ss = 0, torch.zeros(len(FEATURES)), torch.zeros(len(FEATURES))
for x, _ in stream(False, 4096):
    n += x.shape[0]; s += x.sum(0); ss += (x ** 2).sum(0)
mean = s / n; std = ((ss / n) - mean ** 2).clamp_min(1e-6).sqrt(); mean, std = mean.to(dev), std.to(dev)
print("train rows:", n, "| data source:", DATA_SOURCE)

model = nn.Sequential(nn.Linear(len(FEATURES), 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1)).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4); loss_fn = nn.HuberLoss(delta=0.5)

# AI Runtime attaches this run to the experiment named in workload.yaml, no set_experiment needed.
with mlflow.start_run(run_name="neural_forecaster_gpu"):
    mlflow.log_params({"epochs": EPOCHS, "device": str(dev), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
                       "features": len(FEATURES), "data_source": f"{DATA_SOURCE}:{TABLE}", "train_rows": n})
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); tot, cnt = 0.0, 0
        for x, y in stream(False, 1024):
            x, y = (x.to(dev) - mean) / std, torch.log1p(y.to(dev))
            opt.zero_grad(); loss = loss_fn(model(x).squeeze(1), y); loss.backward(); opt.step()
            tot += loss.item() * x.shape[0]; cnt += x.shape[0]
        mlflow.log_metric("train_loss", tot / cnt, step=ep); print(f"epoch {ep} loss {tot/cnt:.4f}")
    model.eval(); abs_err, tot_y = 0.0, 0.0
    with torch.no_grad():
        for x, y in stream(True, 4096):
            p = torch.expm1(model((x.to(dev) - mean) / std).squeeze(1)).clamp_min(0)
            abs_err += (p - y.to(dev)).abs().sum().item(); tot_y += y.sum().item()
    wape = abs_err / tot_y
    mlflow.log_metrics({"test_wape": wape, "train_seconds": time.time() - t0})
    print("test WAPE:", round(wape, 4), "train seconds:", round(time.time() - t0, 1))
