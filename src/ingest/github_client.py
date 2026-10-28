import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

HEADERS = {
    "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
    "Accept": "application/vnd.github+json",
}


def get(url, params=None):
    while True:
        r = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if r.status_code == 403 and r.headers.get("X-RateLimit-Remaining") == "0":
            wait = int(r.headers["X-RateLimit-Reset"]) - time.time() + 5
            time.sleep(max(wait, 1))
            continue
        r.raise_for_status()
        return r.json()


def runs_in_window(repo, start, end):
    page = 1
    while True:
        data = get(
            f"https://api.github.com/repos/{repo}/actions/runs",
            {
                "per_page": 100,
                "page": page,
                "status": "completed",
                "created": f"{start}..{end}",
            },
        )
        runs = data["workflow_runs"]
        if not runs:
            break
        yield from runs
        page += 1