import numpy as np
from sklearn.metrics import average_precision_score

from src.features.load_features import load_features


def time_split(df, train=0.70, val=0.15):
    df = df.sort_values("started_at").reset_index(drop=True)
    n = len(df)
    i = int(n * train)
    j = int(n * (train + val))
    return df.iloc[:i], df.iloc[i:j], df.iloc[j:]


if __name__ == "__main__":
    df = load_features()
    tr, va, te = time_split(df)

    print("rows  train/val/test:", len(tr), len(va), len(te))
    print("failure rate  train/val/test:",
          round(tr["failed"].mean(), 3),
          round(va["failed"].mean(), 3),
          round(te["failed"].mean(), 3))

    # Baseline score: fail_rate_last10 (empty for each repo's first build -> use 0)
    score = te["fail_rate_last10"].fillna(0)
    pr_auc = average_precision_score(te["failed"], score)

    print()
    print("Random model PR-AUC (= failure rate):", round(te["failed"].mean(), 3))
    print("Baseline PR-AUC (fail_rate_last10):  ", round(pr_auc, 3))