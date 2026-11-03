import json
from datetime import date

from sklearn.metrics import average_precision_score, f1_score
from sklearn.metrics import precision_score, recall_score

from src.features.load_features import load_features
from src.models.anomaly import find_slow_builds
from src.models.baseline import time_split
from src.models.evaluate import best_threshold, fit
from src.models.train import FEATURES


def main():
    df = load_features()
    repos = sorted(df["repo"].unique())
    codes = {r: i for i, r in enumerate(repos)}
    df["repo_code"] = df["repo"].map(codes)
    tr, va, te = time_split(df)

    model = fit(tr, va)
    t = best_threshold(va["failed"], model.predict(va[FEATURES]))
    pt = model.predict(te[FEATURES])
    y = te["failed"]
    model.save_model("app/model.txt")

    stats = {
        "builds": int(len(df)),
        "fail_rate": round(float(df["failed"].mean()), 3),
        "pr_auc": round(float(average_precision_score(y, pt)), 3),
        "precision": round(float(precision_score(y, pt >= t, zero_division=0)), 3),
        "recall": round(float(recall_score(y, pt >= t)), 3),
        "f1": round(float(f1_score(y, pt >= t)), 3),
        "random": round(float(y.mean()), 3),
        "repos": codes,
        "updated": str(date.today()),
    }
    with open("app/stats.json", "w") as f:
        json.dump(stats, f)

    al = find_slow_builds()
    top = al[al["slow_alert"]].sort_values("ratio", ascending=False).head(15)
    top = top[["repo", "started_at", "duration_s", "median_prior", "ratio"]].copy()
    top["started_at"] = top["started_at"].astype(str)
    with open("app/alerts.json", "w") as f:
        json.dump(top.round(1).to_dict(orient="records"), f)

    print(stats)
    print("Saved app/model.txt, app/stats.json, app/alerts.json")


if __name__ == "__main__":
    main()
