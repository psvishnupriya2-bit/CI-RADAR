import os

import psycopg2
from dotenv import load_dotenv

from src.ingest.github_client import get, runs_in_window
from src.ingest.load_builds import MONTHS, REPOS, month_window

load_dotenv()


def main():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()
    for repo in REPOS:
        for year, month in MONTHS:
            start, end = month_window(year, month)
            n = 0
            for run in runs_in_window(repo, start, end):
                cur.execute(
                    "UPDATE builds SET workflow_name = %s WHERE build_id = %s",
                    (run.get("name"), run["id"]),
                )
                n += 1
            conn.commit()
            print(f"{repo} {start[:7]}: {n} runs updated", flush=True)
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
