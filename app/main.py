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
STATS = json.loads((HERE / "stats.json").read_text())
REPOS = STATS["repos"]
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


@app.get("/stats")
def stats():
    return STATS


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
    html = (HERE / "index.html").read_text(encoding="utf-8")
    s = STATS
    options = "".join(f"<option>{r}</option>" for r in REPOS)
    html = html.replace(
        "<option>pandas-dev/pandas</option><option>numpy/numpy</option>"
        "<option>pallets/flask</option><option>scikit-learn/scikit-learn</option>",
        options,
    )
    html = html.replace("13,886", f"{s['builds']:,}")
    html = html.replace("4 repos, Mar to Aug 2026", f"{len(REPOS)} repos, updated {s['updated']}")
    html = html.replace("10.1%", f"{s['fail_rate'] * 100:.1f}%")
    html = html.replace("0.555", f"{s['pr_auc']}")
    html = html.replace("baseline 0.435, random 0.104", f"random {s['random']}, test set")
    html = html.replace(
        "Low below 15%, medium 15 to 30%, high above 30%.",
        f"Low below 15%, medium 15 to 30%, high above 30%. "
        f"Test precision {s['precision']}, recall {s['recall']}, F1 {s['f1']}.",
    )
    return html
