import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, f1_score
from sklearn.metrics import precision_recall_curve, precision_score, recall_score

from src.features.load_features import load_features
from src.models.baseline import time_split
from src.models.train import FEATURES, PARAMS


def fit(tr, va, weight=None):
    params = dict(PARAMS)
    if weight:
        params["scale_pos_weight"] = weight
    dtrain = lgb.Dataset(tr[FEATURES], tr["failed"])
    dval = lgb.Dataset(va[FEATURES], va["failed"], reference=dtrain)
    return lgb.train(params, dtrain, num_boost_round=500, valid_sets=[dval],
                     callbacks=[lgb.early_stopping(30, verbose=False)])


def best_threshold(y, p):
    best_t, best_f = 0.5, -1
    for t in np.arange(0.05, 0.95, 0.01):
        f = f1_score(y, p >= t, zero_division=0)
        if f > best_f:
            best_t, best_f = t, f
    return best_t


def report(name, model, va, te):
    pv = model.predict(va[FEATURES])
    pt = model.predict(te[FEATURES])
    t = best_threshold(va["failed"], pv)
    y = te["failed"]
    print(f"{name}: threshold {t:.2f} (chosen on validation)")
    print("  PR-AUC   ", round(average_precision_score(y, pt), 3))
    print("  precision", round(precision_score(y, pt >= t, zero_division=0), 3))
    print("  recall   ", round(recall_score(y, pt >= t), 3))
    print("  F1       ", round(f1_score(y, pt >= t), 3))
    return pt


def main():
    df = load_features()
    df["repo_code"] = df["repo"].astype("category").cat.codes
    tr, va, te = time_split(df)
    rate = tr["failed"].mean()

    m1 = fit(tr, va)
    p1 = report("LightGBM (no weights)", m1, va, te)
    m2 = fit(tr, va, weight=(1 - rate) / rate)
    report("LightGBM (class weights)", m2, va, te)

    print()
    print("PR-AUC per repo (no weights):")
    for repo, g in te.assign(p=p1).groupby("repo"):
        if g["failed"].sum() > 0:
            print(" ", repo, round(average_precision_score(g["failed"], g["p"]), 3),
                  "failures:", int(g["failed"].sum()))

    prec, rec, _ = precision_recall_curve(te["failed"], p1)
    plt.figure(figsize=(6, 4))
    plt.plot(rec, prec, label="LightGBM")
    plt.axhline(te["failed"].mean(), color="grey", linestyle="--", label="random")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curve (test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("notebooks/pr_curve.png", dpi=120)
    print("Saved notebooks/pr_curve.png")


if __name__ == "__main__":
    main()
