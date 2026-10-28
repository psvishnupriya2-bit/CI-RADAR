import os
import sys
from datetime import datetime

import psycopg2
from dotenv import load_dotenv

from src.ingest.github_client import get, runs_in_window

load_dotenv()

REPOS = ["pandas-dev/pandas", "scikit-learn/scikit-learn", "numpy/numpy"]
START = "2025-09-01"
END = "2025-09-30"



def connect():
    return psycopg2.connect(os.environ["DATABASE_URL"])


def parse_time(value):
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def save_repo(cur, repo):
    info = get(f"https://api.github.com/repos/{repo}")
    cur.execute(
        """
        INSERT INTO repos (repo_id, full_name, language, stars)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (repo_id) DO NOTHING
        """,
        (info["id"], info["full_name"], info["language"], info["stargazers_count"]),
    )
    return info["id"]


def save_builds(cur, repo, repo_id):
    count = 0
    for run in runs_in_window(repo, START, END):
        if run["conclusion"] not in ("success", "failure"):
            continue
        started = parse_time(run.get("run_started_at"))
        updated = parse_time(run.get("updated_at"))
        if started is None or updated is None:
            continue
        duration = (updated - started).total_seconds()
        author = (run.get("actor") or {}).get("login")
        cur.execute(
            """
            INSERT INTO builds (build_id, repo_id, commit_sha, branch, event,
                                created_at, started_at, updated_at, duration_s,
                                conclusion, run_attempt, author)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (build_id) DO NOTHING
            """,
            (
                run["id"],
                repo_id,
                run["head_sha"],
                run["head_branch"],
                run["event"],
                parse_time(run["created_at"]),
                started,
                updated,
                duration,
                run["conclusion"],
                run.get("run_attempt"),
                author,
            ),
        )
        count += 1
    return count


def main():
    conn = connect()
    cur = conn.cursor()
    for repo in REPOS:
        print(f"Loading {repo} ...")
        repo_id = save_repo(cur, repo)
        n = save_builds(cur, repo, repo_id)
        conn.commit()
        print(f"  done: {n} builds processed")
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()