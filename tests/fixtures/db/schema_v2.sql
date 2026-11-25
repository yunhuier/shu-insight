
CREATE TABLE IF NOT EXISTS schema_meta (
  version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  base_url TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS source_status (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
  state TEXT NOT NULL,
  checked_at TEXT NOT NULL,
  message TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS notices (
  id TEXT PRIMARY KEY,
  canonical_url TEXT NOT NULL UNIQUE,
  source_id TEXT NOT NULL REFERENCES sources(id),
  title TEXT NOT NULL,
  published_at TEXT,
  deadline_at TEXT,
  deadline_confidence TEXT NOT NULL,
  summary TEXT NOT NULL,
  location TEXT NOT NULL DEFAULT '',
  department TEXT NOT NULL DEFAULT '',
  column_name TEXT NOT NULL DEFAULT '',
  education_levels_json TEXT NOT NULL DEFAULT '[]',
  grades_json TEXT NOT NULL DEFAULT '[]',
  colleges_json TEXT NOT NULL DEFAULT '[]',
  majors_json TEXT NOT NULL DEFAULT '[]',
  categories_json TEXT NOT NULL DEFAULT '[]',
  topic_tags_json TEXT NOT NULL DEFAULT '[]',
  content_fingerprint TEXT NOT NULL,
  summary_simhash INTEGER NOT NULL,
  parse_state TEXT NOT NULL,
  fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS notice_versions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  notice_id TEXT NOT NULL REFERENCES notices(id) ON DELETE CASCADE,
  content_fingerprint TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS duplicate_relations (
  left_id TEXT NOT NULL REFERENCES notices(id) ON DELETE CASCADE,
  right_id TEXT NOT NULL REFERENCES notices(id) ON DELETE CASCADE,
  left_fingerprint TEXT NOT NULL,
  right_fingerprint TEXT NOT NULL,
  evidence_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY(left_id, right_id, left_fingerprint, right_fingerprint)
);

CREATE TABLE IF NOT EXISTS duplicate_clusters (
  id TEXT PRIMARY KEY,
  confidence REAL NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS duplicate_cluster_members (
  cluster_id TEXT NOT NULL REFERENCES duplicate_clusters(id) ON DELETE CASCADE,
  notice_id TEXT NOT NULL REFERENCES notices(id) ON DELETE CASCADE,
  relation TEXT NOT NULL,
  PRIMARY KEY (cluster_id, notice_id)
);

CREATE TABLE IF NOT EXISTS personas (
  id TEXT PRIMARY KEY,
  label TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS annotations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  notice_id TEXT NOT NULL REFERENCES notices(id) ON DELETE CASCADE,
  payload_json TEXT NOT NULL,
  reviewed_by TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recommendation_runs (
  id TEXT PRIMARY KEY,
  persona_id TEXT REFERENCES personas(id) ON DELETE SET NULL,
  parameters_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recommendation_items (
  run_id TEXT NOT NULL REFERENCES recommendation_runs(id) ON DELETE CASCADE,
  notice_id TEXT NOT NULL REFERENCES notices(id) ON DELETE CASCADE,
  rank INTEGER NOT NULL,
  score REAL NOT NULL,
  breakdown_json TEXT NOT NULL,
  PRIMARY KEY (run_id, notice_id)
);
