# CI Radar

Predicts failing builds and flags unusually slow builds for public GitHub projects (pandas, numpy, scikit-learn, flask).

Live demo: https://ci-radar.onrender.com
## Data
- Source: GitHub Actions API, March to August 2026, 4 repos
- 13,886 builds, 1,409 failures (10.1%), stored in Postgres
- SQL features use CTEs, window functions and joins. Each build only sees earlier builds, so there is no leakage.

## Results (time-based split: 70% train, 15% validation, 15% test)
| Model | Test PR-AUC |
|---|---|
| Random guessing | 0.104 |
| Baseline (fail_rate_last10) | 0.435 |
| LightGBM | 0.555 |
| LightGBM tuned with Optuna (30 trials) | 0.516 |

Tuning raised validation PR-AUC (0.612 to 0.624) but not test PR-AUC. The test set has about 217 failures, so scores are noisy.
All runs are tracked in MLflow. The model is registered as ci-radar-risk and is promoted only if it beats production by 0.01 validation PR-AUC.
