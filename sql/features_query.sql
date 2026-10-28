WITH base AS (
  SELECT
    b.build_id,
    b.repo_id,
    b.started_at,
    b.duration_s,
    b.event,
    b.branch,
    b.run_attempt,
    (b.conclusion = 'failure')::int AS failed,
    (b.event = 'pull_request')::int AS is_pr,
    EXTRACT(HOUR FROM b.started_at) AS hour_of_day,
    EXTRACT(DOW FROM b.started_at) AS day_of_week
  FROM builds b
  WHERE b.conclusion IN ('success', 'failure')
    AND b.started_at IS NOT NULL
),
windowed AS (
  SELECT
    base.*,
    AVG(failed) OVER w10 AS fail_rate_last10,
    AVG(failed) OVER w50 AS fail_rate_last50,
    LAG(failed) OVER w AS prev_failed,
    AVG(duration_s) OVER w10 AS avg_duration_last10,
    COUNT(*) OVER w10 AS builds_in_window
  FROM base
  WINDOW
    w   AS (PARTITION BY repo_id ORDER BY started_at, build_id),
    w10 AS (PARTITION BY repo_id ORDER BY started_at, build_id
            ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING),
    w50 AS (PARTITION BY repo_id ORDER BY started_at, build_id
            ROWS BETWEEN 50 PRECEDING AND 1 PRECEDING)
)
SELECT
  w.build_id,
  r.full_name AS repo,
  w.started_at,
  w.failed,
  w.is_pr,
  w.hour_of_day,
  w.day_of_week,
  w.fail_rate_last10,
  w.fail_rate_last50,
  w.prev_failed,
  w.avg_duration_last10,
  w.builds_in_window
FROM windowed w
JOIN repos r ON r.repo_id = w.repo_id
ORDER BY w.repo_id, w.started_at, w.build_id;
