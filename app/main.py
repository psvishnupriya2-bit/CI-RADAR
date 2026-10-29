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


PAGE = """
<html><head><title>CI Radar</title>
<style>body{font-family:sans-serif;max-width:800px;margin:30px auto}
td,th{border-bottom:1px solid #ccc;padding:4px 8px;text-align:left}</style></head>
<body><h1>CI Radar</h1><h2>Failure risk</h2>
Repo <select id="repo"><option>pandas-dev/pandas</option><option>numpy/numpy</option>
<option>pallets/flask</option><option>scikit-learn/scikit-learn</option></select>
Recent failure rate (0 to 1) <input id="rate" value="0.1" size="4">
Pull request? <select id="pr"><option value="0">no</option><option value="1">yes</option></select>
<button onclick="go()">Predict</button><p id="out"></p>
<h2>Slow-build alerts</h2><table id="tbl"></table>
<script>
async function go(){
  const r=parseFloat(document.getElementById('rate').value);
  const body={repo:document.getElementById('repo').value,fail_rate_last10:r,
    fail_rate_last50:r,is_pr:parseInt(document.getElementById('pr').value)};
  const res=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const d=await res.json();
  document.getElementById('out').innerText='Risk: '+d.risk+' ('+d.label+')';
}
async function load(){
  const res=await fetch('/alerts'); const rows=await res.json();
  let h='<tr><th>Repo</th><th>Started</th><th>Seconds</th><th>Usual</th><th>Ratio</th></tr>';
  for(const r of rows){h+='<tr><td>'+r.repo+'</td><td>'+r.started_at+'</td><td>'+r.duration_s+'</td><td>'+r.median_prior+'</td><td>'+r.ratio+'</td></tr>';}
  document.getElementById('tbl').innerHTML=h;
}
load();
</script></body></html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE
