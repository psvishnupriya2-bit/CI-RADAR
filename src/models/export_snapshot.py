import json

import mlflow
import mlflow.lightgbm

from src.models.anomaly import find_slow_builds

mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.lightgbm.load_model("models:/ci-radar-risk@production")
model.save_model("app/model.txt")
print("Saved app/model.txt")

df = find_slow_builds()
top = df[df["slow_alert"]].sort_values("ratio", ascending=False).head(15)
top = top[["repo", "started_at", "duration_s", "median_prior", "ratio"]].copy()
top["started_at"] = top["started_at"].astype(str)
rows = top.round(1).to_dict(orient="records")
with open("app/alerts.json", "w") as f:
    json.dump(rows, f)
print("Saved app/alerts.json with", len(rows), "rows")
