import mlflow
import mlflow.lightgbm
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

mlflow.set_tracking_uri("sqlite:///mlflow.db")
MODEL = mlflow.lightgbm.load_model("models:/ci-radar-risk@production")

REPOS = {
    "numpy/numpy": 0,
    "pallets/flask": 1,
    "pandas-dev/pandas": 2,
    "scikit-learn/scikit-learn": 3,
}
FEATURES = [
    "is_pr", "hour_of_day", "day_of_week", "fail_rate_last10",
    "fail_rate_last50", "prev_failed", "builds_in_window", "repo_code",
]

app = FastAPI(title="CI Radar")


class Build(BaseModel):
    repo: str = "pandas-dev/pandas"
    is_pr: int = 0
    hour_of_day: int = 12
    day_of_week: int = 2
    fail_rate_last10: float = 0.1
    fail_rate_last50: float = 0.1
    prev_failed: int = 0
    builds_in_window: int = 10


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(b: Build):
    row = b.model_dump()
    row["repo_code"] = REPOS.get(row.pop("repo"), -1)
    risk = float(MODEL.predict(pd.DataFrame([row])[FEATURES])[0])
    label = "high" if risk >= 0.3 else "medium" if risk >= 0.15 else "low"
    return {"risk": round(risk, 3), "label": label}
