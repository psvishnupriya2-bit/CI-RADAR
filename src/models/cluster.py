import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

load_dotenv()

SQL = """
SELECT r.full_name AS repo,
       DATE_TRUNC('month', b.started_at)::date AS month,
       COUNT(*) AS builds,
       AVG((b.conclusion = 'failure')::int) AS fail_rate,
       AVG((b.event = 'pull_request')::int) AS pr_share,
       AVG((COALESCE(b.run_attempt, 1) > 1)::int) AS retry_rate,
       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY b.duration_s)
           FILTER (WHERE b.duration_s >= 60) AS median_s,
       PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY b.duration_s)
           FILTER (WHERE b.duration_s >= 60)
     - PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY b.duration_s)
           FILTER (WHERE b.duration_s >= 60) AS iqr_s
FROM builds b JOIN repos r USING (repo_id)
GROUP BY 1, 2
HAVING COUNT(*) >= 20
ORDER BY 1, 2
"""
COLS = ["fail_rate", "pr_share", "retry_rate", "log_median", "log_iqr", "log_builds"]


def main():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    df = pd.read_sql(SQL, conn)
    conn.close()

    df[["median_s", "iqr_s"]] = df[["median_s", "iqr_s"]].fillna(0)
    df["log_median"] = np.log1p(df["median_s"])
    df["log_iqr"] = np.log1p(df["iqr_s"])
    df["log_builds"] = np.log1p(df["builds"])
    X = StandardScaler().fit_transform(df[COLS])

    print("Repo-months:", len(df))
    best_k, best_s = 2, -1
    for k in range(2, 6):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        s = silhouette_score(X, labels)
        print(f"k={k}  silhouette={s:.3f}")
        if s > best_s:
            best_k, best_s = k, s
    print("Chosen k:", best_k)

    df["cluster"] = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit_predict(X)
    print()
    print(df.groupby("cluster")[["fail_rate", "pr_share", "median_s", "builds"]].mean().round(2))
    print()
    print(pd.crosstab(df["repo"], df["cluster"]))

    pca = PCA(n_components=2)
    pts = pca.fit_transform(X)
    print()
    print("PCA explained variance:", np.round(pca.explained_variance_ratio_, 3))

    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {r: m for r, m in zip(sorted(df["repo"].unique()), "osD^")}
    for repo, g in df.groupby("repo"):
        idx = g.index
        ax.scatter(pts[idx, 0], pts[idx, 1], c=g["cluster"], cmap="viridis",
                   vmin=0, vmax=best_k - 1, marker=markers[repo], s=70, label=repo)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Repo-months by build behaviour (colour = cluster)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs("notebooks", exist_ok=True)
    fig.savefig("notebooks/clusters.png", dpi=120)
    print("Saved notebooks/clusters.png")


if __name__ == "__main__":
    main()
