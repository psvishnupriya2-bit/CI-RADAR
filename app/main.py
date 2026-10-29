import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

HERE = Path(__file__).parent
MODEL = lgb.Booster(model_file=str(HERE / "model.txt"))
ALERTS = json.loads((HERE / "alerts.json").read_text())

REPOS = {"numpy/numpy": 0, "pallets/flask": 1, "pandas-dev/pandas": 2, "scikit-learn/scikit-learn": 3}
FEATURES = ["is_pr", "hour_of_day", "day_of_week", "fail_rate_last10",
            "fail_rate_last50", "prev_failed", "builds_in_window", "repo_code"]

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


@app.get("/alerts")
def alerts():
    return ALERTS


@app.get("/", response_class=HTMLResponse)
def home():
    return (HERE / "index.html").read_text(encoding="utf-8")
