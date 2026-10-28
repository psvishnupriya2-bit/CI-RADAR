import lightgbm as lgb
import mlflow
import optuna
from sklearn.metrics import average_precision_score

from src.features.load_features import load_features
from src.models.baseline import time_split
from src.models.train import FEATURES

N_TRIALS = 30


def fit(params, tr, va):
    params = dict(params)
    params["feature_pre_filter"] = False
    dtrain = lgb.Dataset(tr[FEATURES], tr["failed"], params=params, free_raw_data=False)
    dval = lgb.Dataset(va[FEATURES], va["failed"], reference=dtrain, params=params, free_raw_data=False)
    return lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )


def main():
    df = load_features()
    df["repo_code"] = df["repo"].astype("category").cat.codes
    tr, va, te = time_split(df)

    mlflow.set_experiment("ci-radar")

    def objective(trial):
        params = {
            "objective": "binary",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 64),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
            "seed": 42,
            "verbose": -1,
        }
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            model = fit(params, tr, va)
            val_pr = average_precision_score(va["failed"], model.predict(va[FEATURES]))
            mlflow.log_params(params)
            mlflow.log_metric("val_pr_auc", val_pr)
        return val_pr

    with mlflow.start_run(run_name="optuna_tuning"):
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(),
        )
        study.optimize(objective, n_trials=N_TRIALS)

        best = study.best_params
        best.update({"objective": "binary", "seed": 42, "verbose": -1})
        model = fit(best, tr, va)
        val_pr = average_precision_score(va["failed"], model.predict(va[FEATURES]))
        test_pr = average_precision_score(te["failed"], model.predict(te[FEATURES]))
        mlflow.log_params({f"best_{k}": v for k, v in best.items()})
        mlflow.log_metric("best_val_pr_auc", val_pr)
        mlflow.log_metric("test_pr_auc", test_pr)

        print("Best val PR-AUC:   ", round(val_pr, 3))
        print("Tuned test PR-AUC: ", round(test_pr, 3))
        print("LightGBM v1 test:   0.555")
        print("Baseline test:      0.435")


if __name__ == "__main__":
    main()
