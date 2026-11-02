import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()

SQL = """
SELECT r.full_name, b.workflow_name, COUNT(*) AS n
FROM (
  SELECT repo_id, workflow_name, commit_sha
  FROM builds
  WHERE workflow_name IS NOT NULL
  GROUP BY repo_id, workflow_name, commit_sha
  HAVING SUM((conclusion = 'failure')::int) > 0
     AND SUM((conclusion = 'success')::int) > 0
) b JOIN repos r USING (repo_id)
GROUP BY r.full_name, b.workflow_name
ORDER BY n DESC
"""


def main():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()
    cur.execute(SQL)
    rows = cur.fetchall()
    conn.close()

    per_repo = {}
    for repo, wf, n in rows:
        per_repo[repo] = per_repo.get(repo, 0) + n

    print("Commits where the SAME workflow both passed and failed:")
    for repo, n in sorted(per_repo.items(), key=lambda x: -x[1]):
        print(" ", repo, n)
    print("Top workflows:")
    for repo, wf, n in rows[:8]:
        print(" ", repo, "|", wf, "|", n)

    lines = ["", "## Flaky candidates by workflow (sql/flaky_by_workflow.sql)",
             "- Same workflow, same commit, both a pass and a fail (builds Mar to Aug 2026):"]
    for repo, n in sorted(per_repo.items(), key=lambda x: -x[1]):
        lines.append(f"  - {repo}: {n} commits")
    lines.append("- Still not per-test flakiness: a re-run after an infrastructure error also counts.")
    with open("README.md", "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("Added to README.md")


if __name__ == "__main__":
    main()
