import os
import warnings

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

load_dotenv()
warnings.filterwarnings("ignore")

REPO = "pandas-dev/pandas"
SQL = """
SELECT DATE(b.started_at) AS day, PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY b.duration_s) AS median_s
FROM builds b JOIN repos r USING (repo_id)
WHERE r.full_name = %s AND b.duration_s >= 60
GROUP BY 1 ORDER BY 1
"""


def main():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    df = pd.read_sql(SQL, conn, params=(REPO,))
    conn.close()

    s = df.set_index(pd.to_datetime(df["day"]))["median_s"].asfreq("D").interpolate().bfill()
    print("Repo:", REPO, " days:", len(s))
    print("ADF p-value:", round(adfuller(s)[1], 3), "(below 0.05 = stationary)")

    sarima_err, naive_err = [], []
    start = len(s) - 6 * 7
    for k in range(6):
        cut = start + k * 7
        train, test = s.iloc[:cut], s.iloc[cut:cut + 7]
        fit = SARIMAX(train, order=(1, 0, 1), seasonal_order=(1, 0, 1, 7)).fit(disp=False)
        pred = fit.forecast(7)
        sarima_err.append(np.mean(np.abs(test.values - pred.values)))
        naive_err.append(np.mean(np.abs(test.values - train.iloc[-7:].values)))

    print()
    print("Backtest MAE (seconds), 6 weeks:")
    print("  SARIMA :", round(float(np.mean(sarima_err)), 1))
    print("  Naive  :", round(float(np.mean(naive_err)), 1))


if __name__ == "__main__":
    main()
