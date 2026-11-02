import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.features.load_features import load_features
from src.models.baseline import time_split
from src.models.evaluate import fit
from src.models.train import FEATURES


def main():
    df = load_features()
    df["repo_code"] = df["repo"].astype("category").cat.codes
    tr, va, te = time_split(df)
    model = fit(tr, va)

    sample = te.sample(min(800, len(te)), random_state=0)
    X = sample[FEATURES]
    values = shap.TreeExplainer(model).shap_values(X)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)

    mean_abs = pd.Series(np.abs(values).mean(axis=0), index=FEATURES).sort_values()
    print("Average importance (SHAP), biggest last:")
    print(mean_abs.round(3).to_string())

    plt.figure(figsize=(6, 4))
    mean_abs.plot(kind="barh")
    plt.xlabel("Average impact on predicted failure (log-odds)")
    plt.title("What drives the failure risk (SHAP)")
    plt.tight_layout()
    plt.savefig("notebooks/shap_importance.png", dpi=120)
    print("Saved notebooks/shap_importance.png")

    top = int(np.argmax(model.predict(X)))
    row = X.iloc[top]
    print()
    print("Highest-risk build in the sample, risk:", round(float(model.predict(X)[top]), 3))
    contrib = pd.Series(values[top], index=FEATURES).sort_values(key=abs, ascending=False)
    for name, c in contrib.head(4).items():
        direction = "raises" if c > 0 else "lowers"
        print(f"  {name} = {row[name]:.2f} {direction} the risk ({c:+.2f})")


if __name__ == "__main__":
    main()
