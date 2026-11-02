from src.ingest.github_client import runs_in_window
from src.ingest.load_builds import MONTHS, connect, month_window, parse_time, save_repo

NEW_REPOS = ["django/django", "psf/requests", "fastapi/fastapi"]


def main():
    conn = connect()
    cur = conn.cursor()
    for repo in NEW_REPOS:
        repo_id = save_repo(cur, repo)
        conn.commit()
        for year, month in MONTHS:
            start, end = month_window(year, month)
            n = 0
            for run in runs_in_window(repo, start, end):
                if run["conclusion"] not in ("success", "failure"):
                    continue
                started = parse_time(run.get("run_started_at"))
                updated = parse_time(run.get("updated_at"))
                if started is None or updated is None:
                    continue
                cur.execute(
                    """
                    INSERT INTO builds (build_id, repo_id, commit_sha, branch, event,
                        created_at, started_at, updated_at, duration_s, conclusion,
                        run_attempt, author, workflow_name)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (build_id) DO NOTHING
                    """,
                    (run["id"], repo_id, run["head_sha"], run["head_branch"], run["event"],
                     parse_time(run["created_at"]), started, updated,
                     (updated - started).total_seconds(), run["conclusion"],
                     run.get("run_attempt"), (run.get("actor") or {}).get("login"),
                     run.get("name")),
                )
                n += 1
            conn.commit()
            print(f"{repo} {start[:7]}: {n} builds", flush=True)
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
