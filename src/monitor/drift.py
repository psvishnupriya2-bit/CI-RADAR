import numpy as np

from src.features.load_features import load_features
from src.models.baseline import time_split

CHECK = [
    "is_pr",
    "hour_of_day",
    "day_of_week",
    "fail_rate_last10",
    "fail_rate_last50",
    "prev_failed",
]


def psi(reference, recent, bins=10):
    reference = reference.dropna()
    recent = recent.dropna()
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.array([reference.min() - 1, reference.mean(), reference.max() + 1])
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, edges)[0] / len(reference)
    new_pct = np.histogram(recent, edges)[0] / len(recent)
    ref_pct = np.clip(ref_pct, 1e-4, None)
    new_pct = np.clip(new_pct, 1e-4, None)
    return float(np.sum((new_pct - ref_pct) * np.log(new_pct / ref_pct)))


def main():
    df = load_features()
    tr, va, te = time_split(df)
    print("Drift: training data vs newest 15% of builds")
    print("PSI above 0.2 = drift\n")
    for col in CHECK:
        value = psi(tr[col], te[col])
        flag = "DRIFT" if value > 0.2 else "ok"
        print(f"{col:20s} PSI {value:6.3f}  {flag}")
    print()
    print("Failure rate  train:", round(tr["failed"].mean(), 3), " newest:", round(te["failed"].mean(), 3))


if __name__ == "__main__":
    main()
