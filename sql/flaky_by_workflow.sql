SELECT r.full_name AS repo,
       b.workflow_name,
       COUNT(*) AS commits_with_both
FROM (
  SELECT repo_id, workflow_name, commit_sha
  FROM builds
  WHERE workflow_name IS NOT NULL
  GROUP BY repo_id, workflow_name, commit_sha
  HAVING SUM((conclusion = 'failure')::int) > 0
     AND SUM((conclusion = 'success')::int) > 0
) b JOIN repos r USING (repo_id)
GROUP BY r.full_name, b.workflow_name
ORDER BY commits_with_both DESC
LIMIT 15;
