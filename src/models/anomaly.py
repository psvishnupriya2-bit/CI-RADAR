import os

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest

load_dotenv()

MIN_BUILD_S = 60
MIN_ALERT_S = 300

SQL = f"""
SELECT b.build_id, r.full_name AS repo, b.started_at, b.duration_s, b.run_attempt
FROM builds b JOIN repos r USING (repo_id)
WHERE b.duration_s >= {MIN_BUILD_S}
ORDER BY r.full_name, b.started_at
"""


def find_slow_builds(contamination=0.02):
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    df = pd.read_sql(SQL, conn)
    conn.close()

    df["median_prior"] = df.groupby("repo")["duration_s"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=5).median()
    )
    df = df.dropna(subset=["median_prior"]).copy()
    df["ratio"] = df["duration_s"] / df["median_prior"]
    df["log_ratio"] = np.log(df["ratio"].clip(lower=1e-3))
    df["log_duration"] = np.log1p(df["duration_s"])
    df["hour"] = pd.to_datetime(df["started_at"]).dt.hour
    df["run_attempt"] = df["run_attempt"].fillna(1)

    X = df[["log_ratio", "log_duration", "hour", "run_attempt"]]
    model = IsolationForest(contamination=contamination, random_state=42)
    df["outlier"] = model.fit_predict(X) == -1
    df["slow_alert"] = df["outlier"] & (df["ratio"] > 1) & (df["duration_s"] > MIN_ALERT_S)
    return df


if __name__ == "__main__":
    df = find_slow_builds()
    print("Builds checked:", len(df))
    print("Outliers:", int(df["outlier"].sum()), " Slow alerts:", int(df["slow_alert"].sum()))
    print()
    print(df[df["slow_alert"]].groupby("repo").size())
    print()
    top = df[df["slow_alert"]].sort_values("ratio", ascending=False).head(8).copy()
    top["duration_s"] = top["duration_s"].round(0)
    top["median_prior"] = top["median_prior"].round(0)
    top["ratio"] = top["ratio"].round(1)
    print(top[["repo", "started_at", "duration_s", "median_prior", "ratio"]].to_string(index=False))
