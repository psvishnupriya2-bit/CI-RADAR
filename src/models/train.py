import lightgbm as lgb
import mlflow
import pandas as pd
from sklearn.metrics import average_precision_score

from src.features.load_features import load_features
from src.models.baseline import time_split

FEATURES = [
    "is_pr",
    "hour_of_day",
    "day_of_week",
    "fail_rate_last10",
    "fail_rate_last50",
    "prev_failed",
    "builds_in_window",
    "repo_code",
]

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "seed": 42,
    "verbose": -1,
}


if __name__ == "__main__":
    df = load_features()
    df["repo_code"] = df["repo"].astype("category").cat.codes
    tr, va, te = time_split(df)

    dtrain = lgb.Dataset(tr[FEATURES], tr["failed"])
    dval = lgb.Dataset(va[FEATURES], va["failed"])

    mlflow.set_experiment("ci-radar")
    with mlflow.start_run(run_name="lightgbm_v1"):
        model = lgb.train(
            PARAMS,
            dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        val_pr = average_precision_score(va["failed"], model.predict(va[FEATURES]))
        test_pr = average_precision_score(te["failed"], model.predict(te[FEATURES]))

        mlflow.log_params(PARAMS)
        mlflow.log_param("features", ",".join(FEATURES))
        mlflow.log_metric("val_pr_auc", val_pr)
        mlflow.log_metric("test_pr_auc", test_pr)
        mlflow.log_metric("test_positive_rate", te["failed"].mean())

        print("Random PR-AUC (test):  ", round(te["failed"].mean(), 3))
        print("Baseline PR-AUC (test): 0.435")
        print("LightGBM val PR-AUC:   ", round(val_pr, 3))
        print("LightGBM test PR-AUC:  ", round(test_pr, 3))
        print()
        imp = pd.Series(model.feature_importance("gain"), index=FEATURES)
        print(imp.sort_values(ascending=False).round(0))