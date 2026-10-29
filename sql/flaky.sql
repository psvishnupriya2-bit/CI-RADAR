SELECT r.full_name AS repo,
       LEFT(b.commit_sha, 8) AS commit,
       COUNT(*) AS runs,
       SUM((b.conclusion = 'failure')::int) AS failures,
       SUM((b.conclusion = 'success')::int) AS passes
FROM builds b JOIN repos r USING (repo_id)
GROUP BY r.full_name, b.commit_sha
HAVING SUM((b.conclusion = 'failure')::int) > 0
   AND SUM((b.conclusion = 'success')::int) > 0
ORDER BY runs DESC
LIMIT 10;
