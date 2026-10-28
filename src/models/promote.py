import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from mlflow.tracking import MlflowClient
from sklearn.metrics import average_precision_score

from src.features.load_features import load_features
from src.models.baseline import time_split
from src.models.train import FEATURES

MODEL_NAME = "ci-radar-risk"
MARGIN = 0.01
PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "seed": 42,
    "verbose": -1,
}


def main():
    df = load_features()
    df["repo_code"] = df["repo"].astype("category").cat.codes
    tr, va, te = time_split(df)

    dtrain = lgb.Dataset(tr[FEATURES], tr["failed"])
    dval = lgb.Dataset(va[FEATURES], va["failed"], reference=dtrain)
    model = lgb.train(
        PARAMS,
        dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    val_pr = average_precision_score(va["failed"], model.predict(va[FEATURES]))
    test_pr = average_precision_score(te["failed"], model.predict(te[FEATURES]))

    mlflow.set_experiment("ci-radar")
    client = MlflowClient()

    with mlflow.start_run(run_name="register_candidate") as run:
        mlflow.log_params(PARAMS)
        mlflow.log_metric("val_pr_auc", val_pr)
        mlflow.log_metric("test_pr_auc", test_pr)
        mlflow.lightgbm.log_model(model, name="model")
        run_id = run.info.run_id

    version = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    client.set_model_version_tag(MODEL_NAME, version.version, "val_pr_auc", str(val_pr))

    try:
        current = client.get_model_version_by_alias(MODEL_NAME, "production")
        current_val = float(current.tags.get("val_pr_auc", 0))
    except Exception:
        current = None
        current_val = None

    print("Candidate version:", version.version, "val PR-AUC:", round(val_pr, 3))
    if current is None:
        client.set_registered_model_alias(MODEL_NAME, "production", version.version)
        print("No production model yet -> promoted version", version.version)
    elif val_pr >= current_val + MARGIN:
        client.set_registered_model_alias(MODEL_NAME, "production", version.version)
        print("Better than production", round(current_val, 3), "-> promoted")
    else:
        print("Not better than production", round(current_val, 3), "by", MARGIN, "-> NOT promoted")


if __name__ == "__main__":
    main()
