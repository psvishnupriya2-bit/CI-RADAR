CREATE TABLE IF NOT EXISTS repos (
  repo_id BIGINT PRIMARY KEY,
  full_name TEXT UNIQUE NOT NULL,
  language TEXT,
  stars INT
);

CREATE TABLE IF NOT EXISTS builds (
  build_id BIGINT PRIMARY KEY,
  repo_id BIGINT REFERENCES repos(repo_id),
  commit_sha TEXT,
  branch TEXT,
  event TEXT,
  created_at TIMESTAMPTZ,
  started_at TIMESTAMPTZ,
  updated_at TIMESTAMPTZ,
  duration_s DOUBLE PRECISION,
  conclusion TEXT,
  run_attempt INT,
  author TEXT
);
CREATE INDEX IF NOT EXISTS idx_builds_repo_time ON builds(repo_id, started_at);

CREATE TABLE IF NOT EXISTS jobs (
  job_id BIGINT PRIMARY KEY,
  build_id BIGINT REFERENCES builds(build_id),
  name TEXT,
  runner_os TEXT,
  duration_s DOUBLE PRECISION,
  conclusion TEXT
);

CREATE TABLE IF NOT EXISTS test_results (
  build_id BIGINT,
  job_id BIGINT,
  test_name TEXT,
  status TEXT,
  duration_s DOUBLE PRECISION
);