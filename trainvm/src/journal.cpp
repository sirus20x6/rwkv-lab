#include "trainvm/journal.hpp"

#include "trainvm/document.hpp"
#include "trainvm/reflection_json.hpp"

#include <sqlite3.h>

#include <openssl/rand.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cerrno>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <tuple>
#include <unistd.h>
#include <utility>

namespace trainvm {
namespace {

constexpr std::string_view kConnectionPragmas = R"sql(
PRAGMA synchronous=FULL;
PRAGMA foreign_keys=ON;
PRAGMA busy_timeout=5000;
PRAGMA trusted_schema=OFF;
)sql";

constexpr std::string_view kWalPragma = "PRAGMA journal_mode=WAL;";

constexpr std::string_view kSchema = R"sql(
CREATE TABLE IF NOT EXISTS journal_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
) WITHOUT ROWID;

INSERT INTO journal_meta(key, value) VALUES('schema_version', '7')
ON CONFLICT(key) DO NOTHING;

INSERT INTO journal_meta(key, value) VALUES(
  'chain_head',
  '0000000000000000000000000000000000000000000000000000000000000000'
) ON CONFLICT(key) DO NOTHING;

CREATE TABLE IF NOT EXISTS events (
  journal_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id TEXT NOT NULL UNIQUE,
  run_id TEXT NOT NULL,
  run_revision INTEGER NOT NULL,
  plan_revision INTEGER NOT NULL,
  node_id TEXT NOT NULL,
  attempt_id TEXT NOT NULL,
  worker_sequence INTEGER NOT NULL,
  event_type TEXT NOT NULL,
  event_version INTEGER NOT NULL,
  wall_time_ns INTEGER NOT NULL,
  monotonic_time_ns INTEGER NOT NULL,
  optimizer_step INTEGER,
  payload_json TEXT NOT NULL,
  previous_hash TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  chain_hash TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_run_sequence ON events(run_id, journal_sequence);
CREATE INDEX IF NOT EXISTS idx_events_attempt_worker_sequence
  ON events(run_id, node_id, attempt_id, worker_sequence)
  WHERE worker_sequence > 0;

CREATE TABLE IF NOT EXISTS run_projection (
  run_id TEXT PRIMARY KEY,
  experiment_name TEXT NOT NULL,
  plan_hash TEXT NOT NULL,
  desired_state TEXT NOT NULL,
  observed_state TEXT NOT NULL,
  current_node_id TEXT NOT NULL,
  current_attempt_id TEXT NOT NULL,
  run_revision INTEGER NOT NULL,
  optimizer_step INTEGER NOT NULL,
  last_heartbeat_ns INTEGER NOT NULL,
  last_event_sequence INTEGER NOT NULL,
  failure_summary TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS compiled_plans (
  plan_hash TEXT PRIMARY KEY,
  experiment_name TEXT NOT NULL,
  canonical_plan_json TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS resource_leases (
  concurrency_key TEXT PRIMARY KEY,
  owner_run_id TEXT NOT NULL,
  lease_id TEXT NOT NULL,
  fencing_token INTEGER NOT NULL,
  clock_domain TEXT NOT NULL CHECK(clock_domain IN ('boottime/v1','legacy-wall/v1')),
  boot_id TEXT,
  acquired_boottime_ns INTEGER,
  expires_boottime_ns INTEGER,
  acquired_wall_time_ns INTEGER NOT NULL,
  expires_wall_time_ns INTEGER NOT NULL,
  released_wall_time_ns INTEGER,
  CHECK(
    (clock_domain='boottime/v1' AND boot_id IS NOT NULL AND
     acquired_boottime_ns IS NOT NULL AND expires_boottime_ns IS NOT NULL) OR
    (clock_domain='legacy-wall/v1' AND boot_id IS NULL AND
     acquired_boottime_ns IS NULL AND expires_boottime_ns IS NULL)
  )
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS resource_lease_releases (
  concurrency_key TEXT NOT NULL,
  owner_run_id TEXT NOT NULL,
  lease_id TEXT NOT NULL,
  fencing_token INTEGER NOT NULL,
  clock_domain TEXT NOT NULL CHECK(clock_domain IN ('boottime/v1','legacy-wall/v1')),
  boot_id TEXT,
  released_wall_time_ns INTEGER NOT NULL,
  CHECK(
    (clock_domain='boottime/v1' AND boot_id IS NOT NULL) OR
    (clock_domain='legacy-wall/v1' AND boot_id IS NULL)
  ),
  PRIMARY KEY(concurrency_key, lease_id, fencing_token)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS resource_lease_renewals (
  concurrency_key TEXT NOT NULL,
  owner_run_id TEXT NOT NULL,
  lease_id TEXT NOT NULL,
  fencing_token INTEGER NOT NULL,
  clock_domain TEXT NOT NULL CHECK(clock_domain='boottime/v1'),
  boot_id TEXT NOT NULL,
  acquired_boottime_ns INTEGER NOT NULL,
  acquired_wall_time_ns INTEGER NOT NULL,
  prior_expires_boottime_ns INTEGER NOT NULL,
  new_expires_boottime_ns INTEGER NOT NULL,
  prior_expires_wall_time_ns INTEGER NOT NULL,
  new_expires_wall_time_ns INTEGER NOT NULL,
  renewed_boottime_ns INTEGER NOT NULL,
  renewed_wall_time_ns INTEGER NOT NULL,
  CHECK(acquired_boottime_ns >= 0 AND acquired_wall_time_ns >= 0),
  CHECK(prior_expires_boottime_ns > acquired_boottime_ns),
  CHECK(new_expires_boottime_ns > prior_expires_boottime_ns),
  CHECK(renewed_boottime_ns >= acquired_boottime_ns AND
        renewed_boottime_ns < prior_expires_boottime_ns),
  CHECK(prior_expires_wall_time_ns >= 0 AND
        new_expires_wall_time_ns >= renewed_wall_time_ns AND
        renewed_wall_time_ns >= 0),
  CHECK(new_expires_boottime_ns - renewed_boottime_ns =
        new_expires_wall_time_ns - renewed_wall_time_ns),
  PRIMARY KEY(concurrency_key, lease_id, fencing_token,
              prior_expires_boottime_ns),
  UNIQUE(concurrency_key, lease_id, fencing_token,
         new_expires_boottime_ns)
) WITHOUT ROWID;

CREATE TRIGGER resource_lease_renewals_no_conflicting_insert
BEFORE INSERT ON resource_lease_renewals
WHEN EXISTS(
  SELECT 1 FROM resource_lease_renewals
  WHERE (concurrency_key=NEW.concurrency_key AND lease_id=NEW.lease_id AND
         fencing_token=NEW.fencing_token AND
         prior_expires_boottime_ns=NEW.prior_expires_boottime_ns)
     OR (concurrency_key=NEW.concurrency_key AND lease_id=NEW.lease_id AND
         fencing_token=NEW.fencing_token AND
         new_expires_boottime_ns=NEW.new_expires_boottime_ns)
)
BEGIN
  SELECT RAISE(ABORT, 'resource lease renewal receipt already exists');
END;

CREATE TRIGGER resource_lease_renewals_no_update
BEFORE UPDATE ON resource_lease_renewals
BEGIN
  SELECT RAISE(ABORT, 'resource lease renewal receipts are immutable');
END;

CREATE TRIGGER resource_lease_renewals_no_delete
BEFORE DELETE ON resource_lease_renewals
BEGIN
  SELECT RAISE(ABORT, 'resource lease renewal receipts are immutable');
END;

CREATE TABLE IF NOT EXISTS node_dispatches (
  dispatch_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  run_revision INTEGER NOT NULL,
  plan_revision INTEGER NOT NULL,
  node_id TEXT NOT NULL,
  attempt_id TEXT NOT NULL,
  component TEXT NOT NULL,
  operation TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('prepared', 'completed')),
  result_event_id TEXT,
  UNIQUE(run_id, node_id, attempt_id)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS control_commands (
  command_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  idempotency_key TEXT NOT NULL,
  expected_run_revision INTEGER NOT NULL,
  expected_control_revision INTEGER NOT NULL,
  control_revision INTEGER NOT NULL,
  plan_revision INTEGER NOT NULL,
  apply_point TEXT NOT NULL,
  requires_pause INTEGER NOT NULL,
  assignments_json TEXT NOT NULL,
  author TEXT NOT NULL,
  reason TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('requested','applied','rejected','restart_required')),
  effective_step INTEGER,
  effective_values_json TEXT NOT NULL,
  diagnostics_json TEXT NOT NULL,
  ack_concurrency_key TEXT,
  ack_lease_id TEXT,
  ack_fencing_token INTEGER,
  ack_node_id TEXT,
  ack_attempt_id TEXT,
  ack_worker_sequence INTEGER,
  acknowledged_at_ns INTEGER,
  UNIQUE(run_id, idempotency_key),
  UNIQUE(run_id, control_revision)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS host_resource_requests (
  request_id TEXT PRIMARY KEY,
  journal_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  concurrency_key TEXT NOT NULL,
  logical_lease_id TEXT NOT NULL,
  logical_fencing_token INTEGER NOT NULL CHECK(logical_fencing_token > 0),
  request_digest TEXT NOT NULL UNIQUE,
  canonical_request_json TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS host_resource_grants (
  request_id TEXT PRIMARY KEY REFERENCES host_resource_requests(request_id),
  allocation_id TEXT NOT NULL UNIQUE,
  request_digest TEXT NOT NULL,
  grant_digest TEXT NOT NULL UNIQUE,
  canonical_grant_json TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS host_resource_release_intents (
  release_request_id TEXT PRIMARY KEY,
  request_id TEXT NOT NULL UNIQUE REFERENCES host_resource_requests(request_id),
  allocation_id TEXT NOT NULL UNIQUE,
  grant_digest TEXT NOT NULL,
  release_request_digest TEXT NOT NULL UNIQUE,
  canonical_release_request_json TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS host_resource_release_receipts (
  release_request_id TEXT PRIMARY KEY
    REFERENCES host_resource_release_intents(release_request_id),
  request_id TEXT NOT NULL UNIQUE REFERENCES host_resource_requests(request_id),
  release_receipt_digest TEXT NOT NULL UNIQUE,
  canonical_release_receipt_json TEXT NOT NULL
) WITHOUT ROWID;

CREATE TRIGGER host_resource_requests_no_update BEFORE UPDATE ON host_resource_requests
BEGIN SELECT RAISE(ABORT, 'host resource requests are immutable'); END;
CREATE TRIGGER host_resource_requests_no_delete BEFORE DELETE ON host_resource_requests
BEGIN SELECT RAISE(ABORT, 'host resource requests are immutable'); END;
CREATE TRIGGER host_resource_grants_no_update BEFORE UPDATE ON host_resource_grants
BEGIN SELECT RAISE(ABORT, 'host resource grants are immutable'); END;
CREATE TRIGGER host_resource_grants_no_delete BEFORE DELETE ON host_resource_grants
BEGIN SELECT RAISE(ABORT, 'host resource grants are immutable'); END;
CREATE TRIGGER host_resource_release_intents_no_update
BEFORE UPDATE ON host_resource_release_intents
BEGIN SELECT RAISE(ABORT, 'host resource release intents are immutable'); END;
CREATE TRIGGER host_resource_release_intents_no_delete
BEFORE DELETE ON host_resource_release_intents
BEGIN SELECT RAISE(ABORT, 'host resource release intents are immutable'); END;
CREATE TRIGGER host_resource_release_receipts_no_update
BEFORE UPDATE ON host_resource_release_receipts
BEGIN SELECT RAISE(ABORT, 'host resource release receipts are immutable'); END;
CREATE TRIGGER host_resource_release_receipts_no_delete
BEFORE DELETE ON host_resource_release_receipts
BEGIN SELECT RAISE(ABORT, 'host resource release receipts are immutable'); END;
)sql";

constexpr std::string_view kSchemaV6Migration = R"sql(
CREATE TABLE resource_lease_renewals (
  concurrency_key TEXT NOT NULL,
  owner_run_id TEXT NOT NULL,
  lease_id TEXT NOT NULL,
  fencing_token INTEGER NOT NULL,
  clock_domain TEXT NOT NULL CHECK(clock_domain='boottime/v1'),
  boot_id TEXT NOT NULL,
  acquired_boottime_ns INTEGER NOT NULL,
  acquired_wall_time_ns INTEGER NOT NULL,
  prior_expires_boottime_ns INTEGER NOT NULL,
  new_expires_boottime_ns INTEGER NOT NULL,
  prior_expires_wall_time_ns INTEGER NOT NULL,
  new_expires_wall_time_ns INTEGER NOT NULL,
  renewed_boottime_ns INTEGER NOT NULL,
  renewed_wall_time_ns INTEGER NOT NULL,
  CHECK(acquired_boottime_ns >= 0 AND acquired_wall_time_ns >= 0),
  CHECK(prior_expires_boottime_ns > acquired_boottime_ns),
  CHECK(new_expires_boottime_ns > prior_expires_boottime_ns),
  CHECK(renewed_boottime_ns >= acquired_boottime_ns AND
        renewed_boottime_ns < prior_expires_boottime_ns),
  CHECK(prior_expires_wall_time_ns >= 0 AND
        new_expires_wall_time_ns >= renewed_wall_time_ns AND
        renewed_wall_time_ns >= 0),
  CHECK(new_expires_boottime_ns - renewed_boottime_ns =
        new_expires_wall_time_ns - renewed_wall_time_ns),
  PRIMARY KEY(concurrency_key, lease_id, fencing_token,
              prior_expires_boottime_ns),
  UNIQUE(concurrency_key, lease_id, fencing_token,
         new_expires_boottime_ns)
) WITHOUT ROWID;

CREATE TRIGGER resource_lease_renewals_no_conflicting_insert
BEFORE INSERT ON resource_lease_renewals
WHEN EXISTS(
  SELECT 1 FROM resource_lease_renewals
  WHERE (concurrency_key=NEW.concurrency_key AND lease_id=NEW.lease_id AND
         fencing_token=NEW.fencing_token AND
         prior_expires_boottime_ns=NEW.prior_expires_boottime_ns)
     OR (concurrency_key=NEW.concurrency_key AND lease_id=NEW.lease_id AND
         fencing_token=NEW.fencing_token AND
         new_expires_boottime_ns=NEW.new_expires_boottime_ns)
)
BEGIN
  SELECT RAISE(ABORT, 'resource lease renewal receipt already exists');
END;

CREATE TRIGGER resource_lease_renewals_no_update
BEFORE UPDATE ON resource_lease_renewals
BEGIN
  SELECT RAISE(ABORT, 'resource lease renewal receipts are immutable');
END;

CREATE TRIGGER resource_lease_renewals_no_delete
BEFORE DELETE ON resource_lease_renewals
BEGIN
  SELECT RAISE(ABORT, 'resource lease renewal receipts are immutable');
END;

UPDATE journal_meta SET value='6' WHERE key='schema_version' AND value='5';
)sql";

constexpr std::string_view kSchemaV7Migration = R"sql(
CREATE TABLE host_resource_requests (
  request_id TEXT PRIMARY KEY,
  journal_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  concurrency_key TEXT NOT NULL,
  logical_lease_id TEXT NOT NULL,
  logical_fencing_token INTEGER NOT NULL CHECK(logical_fencing_token > 0),
  request_digest TEXT NOT NULL UNIQUE,
  canonical_request_json TEXT NOT NULL
) WITHOUT ROWID;
CREATE TABLE host_resource_grants (
  request_id TEXT PRIMARY KEY REFERENCES host_resource_requests(request_id),
  allocation_id TEXT NOT NULL UNIQUE,
  request_digest TEXT NOT NULL,
  grant_digest TEXT NOT NULL UNIQUE,
  canonical_grant_json TEXT NOT NULL
) WITHOUT ROWID;
CREATE TABLE host_resource_release_intents (
  release_request_id TEXT PRIMARY KEY,
  request_id TEXT NOT NULL UNIQUE REFERENCES host_resource_requests(request_id),
  allocation_id TEXT NOT NULL UNIQUE,
  grant_digest TEXT NOT NULL,
  release_request_digest TEXT NOT NULL UNIQUE,
  canonical_release_request_json TEXT NOT NULL
) WITHOUT ROWID;
CREATE TABLE host_resource_release_receipts (
  release_request_id TEXT PRIMARY KEY
    REFERENCES host_resource_release_intents(release_request_id),
  request_id TEXT NOT NULL UNIQUE REFERENCES host_resource_requests(request_id),
  release_receipt_digest TEXT NOT NULL UNIQUE,
  canonical_release_receipt_json TEXT NOT NULL
) WITHOUT ROWID;
CREATE TRIGGER host_resource_requests_no_update BEFORE UPDATE ON host_resource_requests
BEGIN SELECT RAISE(ABORT, 'host resource requests are immutable'); END;
CREATE TRIGGER host_resource_requests_no_delete BEFORE DELETE ON host_resource_requests
BEGIN SELECT RAISE(ABORT, 'host resource requests are immutable'); END;
CREATE TRIGGER host_resource_grants_no_update BEFORE UPDATE ON host_resource_grants
BEGIN SELECT RAISE(ABORT, 'host resource grants are immutable'); END;
CREATE TRIGGER host_resource_grants_no_delete BEFORE DELETE ON host_resource_grants
BEGIN SELECT RAISE(ABORT, 'host resource grants are immutable'); END;
CREATE TRIGGER host_resource_release_intents_no_update
BEFORE UPDATE ON host_resource_release_intents
BEGIN SELECT RAISE(ABORT, 'host resource release intents are immutable'); END;
CREATE TRIGGER host_resource_release_intents_no_delete
BEFORE DELETE ON host_resource_release_intents
BEGIN SELECT RAISE(ABORT, 'host resource release intents are immutable'); END;
CREATE TRIGGER host_resource_release_receipts_no_update
BEFORE UPDATE ON host_resource_release_receipts
BEGIN SELECT RAISE(ABORT, 'host resource release receipts are immutable'); END;
CREATE TRIGGER host_resource_release_receipts_no_delete
BEFORE DELETE ON host_resource_release_receipts
BEGIN SELECT RAISE(ABORT, 'host resource release receipts are immutable'); END;
UPDATE journal_meta SET value='7' WHERE key='schema_version' AND value='6';
)sql";

constexpr std::string_view kSchemaV4 = R"sql(
CREATE TABLE IF NOT EXISTS journal_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS events (
  journal_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id TEXT NOT NULL UNIQUE,
  run_id TEXT NOT NULL,
  run_revision INTEGER NOT NULL,
  plan_revision INTEGER NOT NULL,
  node_id TEXT NOT NULL,
  attempt_id TEXT NOT NULL,
  worker_sequence INTEGER NOT NULL,
  event_type TEXT NOT NULL,
  event_version INTEGER NOT NULL,
  wall_time_ns INTEGER NOT NULL,
  monotonic_time_ns INTEGER NOT NULL,
  optimizer_step INTEGER,
  payload_json TEXT NOT NULL,
  previous_hash TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  chain_hash TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_run_sequence ON events(run_id, journal_sequence);
CREATE INDEX IF NOT EXISTS idx_events_attempt_worker_sequence
  ON events(run_id, node_id, attempt_id, worker_sequence)
  WHERE worker_sequence > 0;

CREATE TABLE IF NOT EXISTS run_projection (
  run_id TEXT PRIMARY KEY,
  experiment_name TEXT NOT NULL,
  plan_hash TEXT NOT NULL,
  desired_state TEXT NOT NULL,
  observed_state TEXT NOT NULL,
  current_node_id TEXT NOT NULL,
  current_attempt_id TEXT NOT NULL,
  run_revision INTEGER NOT NULL,
  optimizer_step INTEGER NOT NULL,
  last_heartbeat_ns INTEGER NOT NULL,
  last_event_sequence INTEGER NOT NULL,
  failure_summary TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS compiled_plans (
  plan_hash TEXT PRIMARY KEY,
  experiment_name TEXT NOT NULL,
  canonical_plan_json TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS resource_leases (
  concurrency_key TEXT PRIMARY KEY,
  owner_run_id TEXT NOT NULL,
  lease_id TEXT NOT NULL,
  fencing_token INTEGER NOT NULL,
  acquired_at_ns INTEGER NOT NULL,
  expires_at_ns INTEGER NOT NULL,
  released_at_ns INTEGER
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS resource_lease_releases (
  concurrency_key TEXT NOT NULL,
  owner_run_id TEXT NOT NULL,
  lease_id TEXT NOT NULL,
  fencing_token INTEGER NOT NULL,
  released_at_ns INTEGER NOT NULL,
  PRIMARY KEY(concurrency_key, lease_id, fencing_token)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS node_dispatches (
  dispatch_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  run_revision INTEGER NOT NULL,
  plan_revision INTEGER NOT NULL,
  node_id TEXT NOT NULL,
  attempt_id TEXT NOT NULL,
  component TEXT NOT NULL,
  operation TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('prepared', 'completed')),
  result_event_id TEXT,
  UNIQUE(run_id, node_id, attempt_id)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS control_commands (
  command_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  idempotency_key TEXT NOT NULL,
  expected_run_revision INTEGER NOT NULL,
  expected_control_revision INTEGER NOT NULL,
  control_revision INTEGER NOT NULL,
  plan_revision INTEGER NOT NULL,
  apply_point TEXT NOT NULL,
  requires_pause INTEGER NOT NULL,
  assignments_json TEXT NOT NULL,
  author TEXT NOT NULL,
  reason TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('requested','applied','rejected','restart_required')),
  effective_step INTEGER,
  effective_values_json TEXT NOT NULL,
  diagnostics_json TEXT NOT NULL,
  ack_concurrency_key TEXT,
  ack_lease_id TEXT,
  ack_fencing_token INTEGER,
  ack_node_id TEXT,
  ack_attempt_id TEXT,
  ack_worker_sequence INTEGER,
  acknowledged_at_ns INTEGER,
  UNIQUE(run_id, idempotency_key),
  UNIQUE(run_id, control_revision)
) WITHOUT ROWID;
)sql";

class Statement {
 public:
  Statement(sqlite3* database, std::string_view sql) : database_(database) {
    const std::string owned(sql);
    if (sqlite3_prepare_v2(database_, owned.c_str(), -1, &statement_, nullptr) != SQLITE_OK) {
      throw std::runtime_error("sqlite prepare failed: " + std::string(sqlite3_errmsg(database_)));
    }
  }

  ~Statement() { sqlite3_finalize(statement_); }
  Statement(const Statement&) = delete;
  Statement& operator=(const Statement&) = delete;

  sqlite3_stmt* get() const { return statement_; }

 private:
  sqlite3* database_{};
  sqlite3_stmt* statement_{};
};

using SchemaSnapshot = std::map<std::string, std::string>;

std::string normalized_schema_sql(std::string_view sql) {
  enum class LexicalState {
    normal_sql,
    single_quoted_literal,
    double_quoted_identifier,
    backtick_quoted_identifier,
    bracket_quoted_identifier,
    line_comment,
    block_comment,
  };

  std::string normalized;
  normalized.reserve(sql.size());
  LexicalState state = LexicalState::normal_sql;
  for (std::size_t index = 0; index < sql.size(); ++index) {
    const char character = sql[index];
    const char next = index + 1U < sql.size() ? sql[index + 1U] : '\0';
    if (state == LexicalState::normal_sql) {
      if (character == '\'') {
        state = LexicalState::single_quoted_literal;
      } else if (character == '"') {
        state = LexicalState::double_quoted_identifier;
      } else if (character == '`') {
        state = LexicalState::backtick_quoted_identifier;
      } else if (character == '[') {
        state = LexicalState::bracket_quoted_identifier;
      } else if (character == '-' && next == '-') {
        state = LexicalState::line_comment;
      } else if (character == '/' && next == '*') {
        state = LexicalState::block_comment;
      } else if (character == ' ' || character == '\t' || character == '\r' ||
                 character == '\n' || character == '\f' || character == '\v') {
        continue;
      }
      normalized.push_back(character);
      continue;
    }

    normalized.push_back(character);
    if (state == LexicalState::single_quoted_literal && character == '\'') {
      if (next == '\'') {
        normalized.push_back(next);
        ++index;
      } else {
        state = LexicalState::normal_sql;
      }
    } else if (state == LexicalState::double_quoted_identifier && character == '"') {
      if (next == '"') {
        normalized.push_back(next);
        ++index;
      } else {
        state = LexicalState::normal_sql;
      }
    } else if (state == LexicalState::backtick_quoted_identifier && character == '`') {
      if (next == '`') {
        normalized.push_back(next);
        ++index;
      } else {
        state = LexicalState::normal_sql;
      }
    } else if (state == LexicalState::bracket_quoted_identifier && character == ']') {
      state = LexicalState::normal_sql;
    } else if (state == LexicalState::line_comment &&
               (character == '\n' || character == '\r')) {
      state = LexicalState::normal_sql;
    } else if (state == LexicalState::block_comment && character == '*' && next == '/') {
      normalized.push_back(next);
      ++index;
      state = LexicalState::normal_sql;
    }
  }
  return normalized;
}

SchemaSnapshot schema_snapshot(sqlite3* database) {
  SchemaSnapshot snapshot;
  const auto text_at = [](sqlite3_stmt* statement, int index) {
    const auto* value = sqlite3_column_text(statement, index);
    return value ? std::string(reinterpret_cast<const char*>(value))
                 : std::string{};
  };
  Statement query(database, R"sql(
    SELECT type, name, sql FROM sqlite_master
    WHERE name NOT LIKE 'sqlite_%' AND sql IS NOT NULL
    ORDER BY type, name
  )sql");
  int status = SQLITE_ROW;
  while ((status = sqlite3_step(query.get())) == SQLITE_ROW) {
    const std::string type = text_at(query.get(), 0);
    const std::string name = text_at(query.get(), 1);
    const auto [position, inserted] = snapshot.emplace(
        type + "\n" + name,
        normalized_schema_sql(text_at(query.get(), 2)));
    (void)position;
    if (!inserted) {
      throw std::runtime_error("journal schema contains a duplicate object identity");
    }
  }
  if (status != SQLITE_DONE) {
    throw std::runtime_error("could not inspect complete journal schema: " +
                             std::string(sqlite3_errmsg(database)));
  }
  return snapshot;
}

SchemaSnapshot canonical_schema(std::string_view schema) {
  sqlite3* database = nullptr;
  if (sqlite3_open_v2(":memory:", &database,
                      SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr) !=
      SQLITE_OK) {
    const std::string message =
        database ? sqlite3_errmsg(database) : "unknown error";
    if (database != nullptr) sqlite3_close(database);
    throw std::runtime_error("could not create canonical schema database: " +
                             message);
  }
  struct CloseDatabase final {
    sqlite3* database;
    ~CloseDatabase() { sqlite3_close(database); }
  } close{database};
  char* error_message = nullptr;
  const std::string owned(schema);
  if (sqlite3_exec(database, owned.c_str(), nullptr, nullptr, &error_message) !=
      SQLITE_OK) {
    const std::string message =
        error_message ? error_message : sqlite3_errmsg(database);
    sqlite3_free(error_message);
    throw std::runtime_error("could not create canonical journal schema: " +
                             message);
  }
  return schema_snapshot(database);
}

const SchemaSnapshot& canonical_schema_v4() {
  static const SchemaSnapshot schema = canonical_schema(kSchemaV4);
  return schema;
}

const SchemaSnapshot& canonical_schema_v5() {
  static const SchemaSnapshot schema = [] {
    SchemaSnapshot result = canonical_schema(kSchema);
    for (const std::string_view object : {
             "table\nhost_resource_requests",
             "table\nhost_resource_grants",
             "table\nhost_resource_release_intents",
             "table\nhost_resource_release_receipts",
             "trigger\nhost_resource_requests_no_update",
             "trigger\nhost_resource_requests_no_delete",
             "trigger\nhost_resource_grants_no_update",
             "trigger\nhost_resource_grants_no_delete",
             "trigger\nhost_resource_release_intents_no_update",
             "trigger\nhost_resource_release_intents_no_delete",
             "trigger\nhost_resource_release_receipts_no_update",
             "trigger\nhost_resource_release_receipts_no_delete"}) {
      result.erase(std::string(object));
    }
    result.erase("table\nresource_lease_renewals");
    result.erase("trigger\nresource_lease_renewals_no_conflicting_insert");
    result.erase("trigger\nresource_lease_renewals_no_update");
    result.erase("trigger\nresource_lease_renewals_no_delete");
    return result;
  }();
  return schema;
}

const SchemaSnapshot& canonical_schema_v6() {
  static const SchemaSnapshot schema = [] {
    SchemaSnapshot result = canonical_schema(kSchema);
    for (const std::string_view object : {
             "table\nhost_resource_requests",
             "table\nhost_resource_grants",
             "table\nhost_resource_release_intents",
             "table\nhost_resource_release_receipts",
             "trigger\nhost_resource_requests_no_update",
             "trigger\nhost_resource_requests_no_delete",
             "trigger\nhost_resource_grants_no_update",
             "trigger\nhost_resource_grants_no_delete",
             "trigger\nhost_resource_release_intents_no_update",
             "trigger\nhost_resource_release_intents_no_delete",
             "trigger\nhost_resource_release_receipts_no_update",
             "trigger\nhost_resource_release_receipts_no_delete"}) {
      result.erase(std::string(object));
    }
    return result;
  }();
  return schema;
}

const SchemaSnapshot& canonical_schema_v7() {
  static const SchemaSnapshot schema = canonical_schema(kSchema);
  return schema;
}

void require_exact_schema(sqlite3* database, const SchemaSnapshot& expected,
                          std::string_view version) {
  if (schema_snapshot(database) != expected) {
    throw std::runtime_error("journal schema " + std::string(version) +
                             " does not exactly match its authoritative schema");
  }
}

class Transaction {
 public:
  explicit Transaction(sqlite3* database) : database_(database) {
    char* error_message = nullptr;
    if (sqlite3_exec(database_, "BEGIN IMMEDIATE", nullptr, nullptr, &error_message) != SQLITE_OK) {
      const std::string message = error_message ? error_message : sqlite3_errmsg(database_);
      sqlite3_free(error_message);
      throw std::runtime_error("sqlite begin failed: " + message);
    }
  }

  ~Transaction() {
    if (!committed_) {
      sqlite3_exec(database_, "ROLLBACK", nullptr, nullptr, nullptr);
    }
  }

  void commit() {
    char* error_message = nullptr;
    if (sqlite3_exec(database_, "COMMIT", nullptr, nullptr, &error_message) != SQLITE_OK) {
      const std::string message = error_message ? error_message : sqlite3_errmsg(database_);
      sqlite3_free(error_message);
      throw std::runtime_error("sqlite commit failed: " + message);
    }
    committed_ = true;
  }

 private:
  sqlite3* database_{};
  bool committed_{};
};

std::int64_t checked_integer(std::uint64_t value, std::string_view field) {
  if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument(std::string(field) + " exceeds SQLite INTEGER range");
  }
  return static_cast<std::int64_t>(value);
}

std::string random_journal_id() {
  std::array<unsigned char, 16> bytes{};
  if (RAND_bytes(bytes.data(), static_cast<int>(bytes.size())) != 1) {
    throw std::runtime_error("could not generate journal identity");
  }
  constexpr char hexadecimal[] = "0123456789abcdef";
  std::string output;
  output.reserve(bytes.size() * 2U);
  for (const unsigned char byte : bytes) {
    output.push_back(hexadecimal[byte >> 4U]);
    output.push_back(hexadecimal[byte & 0x0fU]);
  }
  return output;
}

bool valid_journal_id(std::string_view identity) {
  return identity.size() == 32U &&
         std::all_of(identity.begin(), identity.end(), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

void bind_text(sqlite3_stmt* statement, int index, const std::string& value) {
  if (sqlite3_bind_text(statement, index, value.c_str(), static_cast<int>(value.size()), SQLITE_TRANSIENT) !=
      SQLITE_OK) {
    throw std::runtime_error("sqlite text bind failed");
  }
}

void bind_integer(sqlite3_stmt* statement, int index, std::int64_t value) {
  if (sqlite3_bind_int64(statement, index, value) != SQLITE_OK) {
    throw std::runtime_error("sqlite integer bind failed");
  }
}

std::string column_text(sqlite3_stmt* statement, int index) {
  const auto* value = sqlite3_column_text(statement, index);
  return value ? reinterpret_cast<const char*>(value) : "";
}

void require_done(sqlite3* database, sqlite3_stmt* statement, std::string_view action) {
  const int result = sqlite3_step(statement);
  if (result != SQLITE_DONE) {
    throw std::runtime_error(std::string(action) + " failed: " + sqlite3_errmsg(database));
  }
}

std::string content_hash(const Event& event) {
  return sha256_hex(event_json(event).dump());
}

constexpr std::string_view kLeaseAuthorityMetadataPrefix =
    "lease_authority_head:";
constexpr std::string_view kLegacyControllerAuthorityMetadataKey =
    "hostd_controller_head";
constexpr std::string_view kControllerAuthorityMetadataPrefix =
    "hostd_controller_head:";
constexpr std::string_view kControllerIdentityMetadataPrefix =
    "hostd_controller_id:";

bool valid_hash_hex(std::string_view value) {
  return value.size() == 64U &&
         std::ranges::all_of(value, [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

std::string lease_authority_identity(std::string_view concurrency_key,
                                     std::uint64_t fencing_token) {
  std::string framed("trainvm.lease-authority-identity/v1");
  framed.push_back('\0');
  framed.append(concurrency_key);
  framed.push_back('\0');
  framed.append(std::to_string(fencing_token));
  return sha256_hex(framed);
}

std::string lease_authority_metadata_key(std::string_view concurrency_key,
                                         std::uint64_t fencing_token) {
  return std::string(kLeaseAuthorityMetadataPrefix) +
         lease_authority_identity(concurrency_key, fencing_token);
}

std::string lease_authority_event_id(std::string_view concurrency_key,
                                     std::uint64_t fencing_token,
                                     std::uint64_t revision) {
  return "journal-lease-authority-" +
         lease_authority_identity(concurrency_key, fencing_token) + "-" +
         std::to_string(revision);
}

std::string controller_scope_identity(std::string_view concurrency_key) {
  std::string framed("trainvm.hostd-controller-scope/v1");
  framed.push_back('\0');
  framed.append(std::to_string(concurrency_key.size()));
  framed.push_back('\0');
  framed.append(concurrency_key);
  return sha256_hex(framed);
}

std::string controller_authority_metadata_key(
    std::string_view concurrency_key) {
  return std::string(kControllerAuthorityMetadataPrefix) +
         controller_scope_identity(concurrency_key);
}

std::string controller_event_id(std::string_view concurrency_key,
                                std::uint64_t generation) {
  return "journal-hostd-controller-" +
         controller_scope_identity(concurrency_key) + "-" +
         std::to_string(generation);
}

std::string controller_identity_metadata_key(
    std::string_view concurrency_key, std::string_view controller_id) {
  std::string framed("trainvm.hostd-controller-identity/v1");
  framed.push_back('\0');
  framed.append(std::to_string(concurrency_key.size()));
  framed.push_back('\0');
  framed.append(concurrency_key);
  framed.push_back('\0');
  framed.append(std::to_string(controller_id.size()));
  framed.push_back('\0');
  framed.append(controller_id);
  return std::string(kControllerIdentityMetadataPrefix) + sha256_hex(framed);
}

bool verify_controller_scope_metadata(sqlite3* database,
                                      std::string* reason = nullptr) {
  const auto fail = [&](std::string_view detail) {
    if (reason != nullptr) *reason = std::string(detail);
    return false;
  };
  Statement query(database,
                  "SELECT key, value FROM journal_meta ORDER BY key");
  int status = SQLITE_ROW;
  while ((status = sqlite3_step(query.get())) == SQLITE_ROW) {
    const std::string key = column_text(query.get(), 0);
    if (key == kLegacyControllerAuthorityMetadataKey)
      return fail("legacy global hostd controller authority is present");
    if (!key.starts_with(kControllerAuthorityMetadataPrefix)) continue;
    const std::string encoded = column_text(query.get(), 1);
    try {
      const nlohmann::json head = nlohmann::json::parse(encoded);
      const bool exact_shape =
          head.is_object() && head.size() == 9U &&
          head.contains("broker_epoch") &&
          head.contains("concurrency_key") &&
          head.contains("controller_generation") &&
          head.contains("controller_id") && head.contains("event_sequence") &&
          head.contains("event_hash") &&
          head.contains("logical_fencing_token") &&
          head.contains("logical_lease_id") && head.contains("run_id");
      if (!exact_shape || head.dump() != encoded ||
          !head.at("broker_epoch").is_string() ||
          !head.at("concurrency_key").is_string() ||
          !head.at("controller_generation").is_number_unsigned() ||
          !head.at("controller_id").is_string() ||
          !head.at("event_sequence").is_number_unsigned() ||
          !head.at("event_hash").is_string() ||
          !head.at("logical_fencing_token").is_number_unsigned() ||
          !head.at("logical_lease_id").is_string() ||
          !head.at("run_id").is_string() ||
          !valid_hash_hex(head.at("event_hash").get<std::string>()))
        return fail("hostd controller scope head is noncanonical");
      const std::string concurrency_key =
          head.at("concurrency_key").get<std::string>();
      if (concurrency_key.empty() ||
          controller_authority_metadata_key(concurrency_key) != key)
        return fail("hostd controller scope head key does not match its scope");
    } catch (...) {
      return fail("hostd controller scope head is malformed");
    }
  }
  if (status != SQLITE_DONE)
    return fail("hostd controller scope metadata is unreadable");
  return true;
}

struct LeaseAuthorityHead final {
  std::string concurrency_key;
  std::string owner_run_id;
  std::string lease_id;
  std::uint64_t fencing_token{};
  std::uint64_t authority_revision{};
  std::uint64_t head_event_sequence{};
  std::string head_event_hash;
  bool released{};
};

nlohmann::json lease_authority_head_json(const LeaseAuthorityHead& head) {
  return {{"authority_revision", head.authority_revision},
          {"concurrency_key", head.concurrency_key},
          {"fencing_token", head.fencing_token},
          {"head_event_sequence", head.head_event_sequence},
          {"head_event_hash", head.head_event_hash},
          {"lease_id", head.lease_id},
          {"owner_run_id", head.owner_run_id},
          {"released", head.released}};
}

LeaseAuthorityHead parse_lease_authority_head(std::string_view encoded) {
  try {
    const nlohmann::json value = nlohmann::json::parse(encoded);
    if (!value.is_object() || value.size() != 8U ||
        !value.contains("authority_revision") ||
        !value.contains("concurrency_key") || !value.contains("fencing_token") ||
        !value.contains("head_event_hash") ||
        !value.contains("head_event_sequence") ||
        !value.contains("lease_id") ||
        !value.contains("owner_run_id") || !value.contains("released") ||
        !value.at("authority_revision").is_number_unsigned() ||
        !value.at("concurrency_key").is_string() ||
        !value.at("fencing_token").is_number_unsigned() ||
        !value.at("head_event_sequence").is_number_unsigned() ||
        !value.at("head_event_hash").is_string() ||
        !value.at("lease_id").is_string() ||
        !value.at("owner_run_id").is_string() ||
        !value.at("released").is_boolean() || value.dump() != encoded) {
      throw std::runtime_error("lease authority head is noncanonical");
    }
    LeaseAuthorityHead head{
        .concurrency_key = value.at("concurrency_key").get<std::string>(),
        .owner_run_id = value.at("owner_run_id").get<std::string>(),
        .lease_id = value.at("lease_id").get<std::string>(),
        .fencing_token = value.at("fencing_token").get<std::uint64_t>(),
        .authority_revision =
            value.at("authority_revision").get<std::uint64_t>(),
        .head_event_sequence =
            value.at("head_event_sequence").get<std::uint64_t>(),
        .head_event_hash = value.at("head_event_hash").get<std::string>(),
        .released = value.at("released").get<bool>()};
    if (head.concurrency_key.empty() || head.owner_run_id.empty() ||
        head.lease_id.empty() || head.fencing_token == 0U ||
        head.head_event_sequence == 0U ||
        !valid_hash_hex(head.head_event_hash)) {
      throw std::runtime_error("lease authority head fields are invalid");
    }
    return head;
  } catch (const nlohmann::json::exception&) {
    throw std::runtime_error("lease authority head is malformed");
  }
}

void create_projection(sqlite3* database, const Event& event, std::uint64_t journal_sequence) {
  if (!event.payload.is_object()) {
    throw std::invalid_argument("run.created payload must be an object");
  }
  const auto required_string = [&](std::string_view name) -> std::string {
    const auto iterator = event.payload.find(name);
    if (iterator == event.payload.end() || !iterator->is_string() || iterator->get<std::string>().empty()) {
      throw std::invalid_argument("run.created requires string payload field " + std::string(name));
    }
    return iterator->get<std::string>();
  };
  Statement insert(database, R"sql(
    INSERT INTO run_projection(
      run_id, experiment_name, plan_hash, desired_state, observed_state,
      current_node_id, current_attempt_id, run_revision, optimizer_step,
      last_heartbeat_ns, last_event_sequence, failure_summary
    ) VALUES(?, ?, ?, ?, ?, '', '', ?, ?, 0, ?, '')
  )sql");
  bind_text(insert.get(), 1, event.run_id);
  bind_text(insert.get(), 2, required_string("experiment_name"));
  bind_text(insert.get(), 3, required_string("plan_hash"));
  bind_text(insert.get(), 4, required_string("desired_state"));
  bind_text(insert.get(), 5, required_string("observed_state"));
  bind_integer(insert.get(), 6, checked_integer(event.run_revision, "run_revision"));
  bind_integer(insert.get(), 7,
               event.optimizer_step ? checked_integer(*event.optimizer_step, "optimizer_step") : 0);
  bind_integer(insert.get(), 8, checked_integer(journal_sequence, "journal_sequence"));
  require_done(database, insert.get(), "create projection");
}

void ensure_projection_exists(sqlite3* database, const std::string& run_id) {
  Statement query(database, "SELECT 1 FROM run_projection WHERE run_id=?");
  bind_text(query.get(), 1, run_id);
  if (sqlite3_step(query.get()) != SQLITE_ROW) {
    throw std::invalid_argument("run event precedes run.created for " + run_id);
  }
}

void update_projection(sqlite3* database, const Event& event, std::uint64_t journal_sequence) {
  if (event.event_type == "run.created") {
    create_projection(database, event, journal_sequence);
    return;
  }
  ensure_projection_exists(database, event.run_id);

  std::string sql = "UPDATE run_projection SET run_revision=?, last_event_sequence=?";
  enum class Extra { none, desired, observed, node, heartbeat, failure };
  Extra extra = Extra::none;
  if (event.event_type == "run.desired_state_changed") {
    sql += ", desired_state=?";
    extra = Extra::desired;
  } else if (event.event_type == "run.observed_state_changed") {
    sql += R"sql(, observed_state=?,
      current_node_id=CASE WHEN ? IN ('acquiring','completed','failed','cancelled') THEN '' ELSE current_node_id END,
      current_attempt_id=CASE WHEN ? IN ('acquiring','completed','failed','cancelled') THEN '' ELSE current_attempt_id END
    )sql";
    extra = Extra::observed;
  } else if (event.event_type == "node.entered") {
    sql += ", current_node_id=?, current_attempt_id=?";
    extra = Extra::node;
  } else if (event.event_type == "worker.heartbeat") {
    sql += ", optimizer_step=?, last_heartbeat_ns=?";
    extra = Extra::heartbeat;
  } else if (event.event_type == "run.failed") {
    sql += ", observed_state='failed', failure_summary=?";
    extra = Extra::failure;
  } else if (event.optimizer_step) {
    sql += ", optimizer_step=MAX(optimizer_step, ?)";
    extra = Extra::heartbeat;
  }
  sql += " WHERE run_id=?";

  Statement update(database, sql);
  int bind_index = 1;
  bind_integer(update.get(), bind_index++, checked_integer(event.run_revision, "run_revision"));
  bind_integer(update.get(), bind_index++, checked_integer(journal_sequence, "journal_sequence"));
  const auto payload_string = [&](std::string_view name) -> std::string {
    const auto iterator = event.payload.find(name);
    if (iterator == event.payload.end() || !iterator->is_string()) {
      throw std::invalid_argument(event.event_type + " requires string payload field " + std::string(name));
    }
    return iterator->get<std::string>();
  };
  switch (extra) {
    case Extra::desired:
      bind_text(update.get(), bind_index++, payload_string("state"));
      break;
    case Extra::observed: {
      const std::string state = payload_string("state");
      bind_text(update.get(), bind_index++, state);
      bind_text(update.get(), bind_index++, state);
      bind_text(update.get(), bind_index++, state);
      break;
    }
    case Extra::node:
      bind_text(update.get(), bind_index++, event.node_id);
      bind_text(update.get(), bind_index++, event.attempt_id);
      break;
    case Extra::heartbeat:
      bind_integer(update.get(), bind_index++,
                   event.optimizer_step ? checked_integer(*event.optimizer_step, "optimizer_step") : 0);
      if (event.event_type == "worker.heartbeat") {
        bind_integer(update.get(), bind_index++, event.wall_time_ns);
      }
      break;
    case Extra::failure:
      bind_text(update.get(), bind_index++, payload_string("summary"));
      break;
    case Extra::none:
      break;
  }
  bind_text(update.get(), bind_index, event.run_id);
  require_done(database, update.get(), "update projection");
  if (sqlite3_changes(database) != 1) {
    throw std::runtime_error("projection update affected an unexpected number of rows");
  }
}

void update_control_projection(sqlite3* database, const Event& event) {
  if (!event.event_type.starts_with("control.")) {
    return;
  }
  const auto required_string = [&](std::string_view name) {
    const auto found = event.payload.find(name);
    if (found == event.payload.end() || !found->is_string() || found->get<std::string>().empty()) {
      throw std::invalid_argument(event.event_type + " requires string payload field " +
                                  std::string(name));
    }
    return found->get<std::string>();
  };
  const auto required_unsigned = [&](std::string_view name) {
    const auto found = event.payload.find(name);
    if (found == event.payload.end() || !found->is_number_unsigned()) {
      throw std::invalid_argument(event.event_type + " requires unsigned payload field " +
                                  std::string(name));
    }
    return found->get<std::uint64_t>();
  };
  if (event.event_type == "control.requested") {
    const auto assignments = event.payload.find("assignments");
    const auto requires_pause = event.payload.find("requires_pause");
    if (assignments == event.payload.end() || !assignments->is_object() || assignments->empty() ||
        requires_pause == event.payload.end() || !requires_pause->is_boolean()) {
      throw std::invalid_argument("control.requested has invalid assignments or pause requirement");
    }
    Statement insert(database, R"sql(
      INSERT INTO control_commands(
        command_id, run_id, idempotency_key, expected_run_revision,
        expected_control_revision, control_revision, plan_revision, apply_point,
        requires_pause, assignments_json, author, reason, status, effective_step,
        effective_values_json, diagnostics_json, ack_concurrency_key, ack_lease_id,
        ack_fencing_token, ack_node_id, ack_attempt_id, ack_worker_sequence,
        acknowledged_at_ns
      ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'requested', NULL, '{}', '[]',
               NULL, NULL, NULL, NULL, NULL, NULL, NULL)
    )sql");
    bind_text(insert.get(), 1, required_string("command_id"));
    bind_text(insert.get(), 2, event.run_id);
    bind_text(insert.get(), 3, required_string("idempotency_key"));
    bind_integer(insert.get(), 4,
                 checked_integer(required_unsigned("expected_run_revision"),
                                 "expected_run_revision"));
    bind_integer(insert.get(), 5,
                 checked_integer(required_unsigned("expected_control_revision"),
                                 "expected_control_revision"));
    bind_integer(insert.get(), 6,
                 checked_integer(required_unsigned("control_revision"), "control_revision"));
    bind_integer(insert.get(), 7,
                 checked_integer(required_unsigned("plan_revision"), "plan_revision"));
    bind_text(insert.get(), 8, required_string("apply_point"));
    bind_integer(insert.get(), 9, requires_pause->get<bool>() ? 1 : 0);
    bind_text(insert.get(), 10, assignments->dump());
    bind_text(insert.get(), 11, required_string("author"));
    bind_text(insert.get(), 12, required_string("reason"));
    require_done(database, insert.get(), "rebuild control request projection");
    return;
  }

  const std::string status = event.event_type.substr(std::string("control.").size());
  if (status != "applied" && status != "rejected" && status != "restart_required") {
    throw std::invalid_argument("unsupported control projection event " + event.event_type);
  }
  const auto effective_values = event.payload.find("effective_values");
  const auto diagnostics = event.payload.find("diagnostics");
  if (effective_values == event.payload.end() || !effective_values->is_object() ||
      diagnostics == event.payload.end() || !diagnostics->is_array()) {
    throw std::invalid_argument("control acknowledgement has invalid effective values or diagnostics");
  }
  Statement update(database, R"sql(
    UPDATE control_commands
    SET status=?, effective_step=?, effective_values_json=?, diagnostics_json=?,
        ack_concurrency_key=?, ack_lease_id=?, ack_fencing_token=?, ack_node_id=?,
        ack_attempt_id=?, ack_worker_sequence=?, acknowledged_at_ns=?
    WHERE command_id=? AND run_id=? AND control_revision=? AND status='requested'
  )sql");
  bind_text(update.get(), 1, status);
  if (event.optimizer_step) {
    bind_integer(update.get(), 2, checked_integer(*event.optimizer_step, "effective_step"));
  } else if (sqlite3_bind_null(update.get(), 2) != SQLITE_OK) {
    throw std::runtime_error("sqlite null bind failed");
  }
  bind_text(update.get(), 3, effective_values->dump());
  bind_text(update.get(), 4, diagnostics->dump());
  bind_text(update.get(), 5, required_string("concurrency_key"));
  bind_text(update.get(), 6, required_string("lease_id"));
  bind_integer(update.get(), 7,
               checked_integer(required_unsigned("fencing_token"), "fencing_token"));
  bind_text(update.get(), 8, event.node_id);
  bind_text(update.get(), 9, event.attempt_id);
  bind_integer(update.get(), 10, checked_integer(event.worker_sequence, "worker_sequence"));
  bind_integer(update.get(), 11, event.wall_time_ns);
  bind_text(update.get(), 12, required_string("command_id"));
  bind_text(update.get(), 13, event.run_id);
  bind_integer(update.get(), 14,
               checked_integer(required_unsigned("control_revision"), "control_revision"));
  require_done(database, update.get(), "rebuild control acknowledgement projection");
  if (sqlite3_changes(database) != 1) {
    throw std::runtime_error("control acknowledgement replay affected an unexpected command");
  }
}

Event event_from_row(sqlite3_stmt* statement) {
  Event event;
  event.event_id = column_text(statement, 1);
  event.run_id = column_text(statement, 2);
  event.run_revision = static_cast<std::uint64_t>(sqlite3_column_int64(statement, 3));
  event.plan_revision = static_cast<std::uint64_t>(sqlite3_column_int64(statement, 4));
  event.node_id = column_text(statement, 5);
  event.attempt_id = column_text(statement, 6);
  event.worker_sequence = static_cast<std::uint64_t>(sqlite3_column_int64(statement, 7));
  event.event_type = column_text(statement, 8);
  event.event_version = static_cast<std::uint32_t>(sqlite3_column_int(statement, 9));
  event.wall_time_ns = sqlite3_column_int64(statement, 10);
  event.monotonic_time_ns = static_cast<std::uint64_t>(sqlite3_column_int64(statement, 11));
  if (sqlite3_column_type(statement, 12) != SQLITE_NULL) {
    event.optimizer_step = static_cast<std::uint64_t>(sqlite3_column_int64(statement, 12));
  }
  event.payload = nlohmann::json::parse(column_text(statement, 13));
  return event;
}

RunProjection run_projection_from_row(sqlite3_stmt* statement) {
  return {
      .run_id = column_text(statement, 0),
      .experiment_name = column_text(statement, 1),
      .plan_hash = column_text(statement, 2),
      .desired_state = column_text(statement, 3),
      .observed_state = column_text(statement, 4),
      .current_node_id = column_text(statement, 5),
      .current_attempt_id = column_text(statement, 6),
      .run_revision = static_cast<std::uint64_t>(
          sqlite3_column_int64(statement, 7)),
      .optimizer_step = static_cast<std::uint64_t>(
          sqlite3_column_int64(statement, 8)),
      .last_heartbeat_ns = sqlite3_column_int64(statement, 9),
      .last_event_sequence = static_cast<std::uint64_t>(
          sqlite3_column_int64(statement, 10)),
      .failure_summary = column_text(statement, 11),
  };
}

bool canonical_boot_id(std::string_view value) {
  if (value.size() != 36U || value[8U] != '-' || value[13U] != '-' ||
      value[18U] != '-' || value[23U] != '-') {
    return false;
  }
  for (std::size_t index = 0; index < value.size(); ++index) {
    if (index == 8U || index == 13U || index == 18U || index == 23U) continue;
    const char character = value[index];
    if (!((character >= '0' && character <= '9') ||
          (character >= 'a' && character <= 'f'))) {
      return false;
    }
  }
  return true;
}

void require_authority_time(const AuthorityTimeSample& now) {
  if (now.wall.nanoseconds < 0 || now.boot.nanoseconds < 0 ||
      !canonical_boot_id(now.boot_id)) {
    throw std::invalid_argument("lease authority time is malformed");
  }
}

std::int64_t lease_expiration(std::int64_t now_ns, std::int64_t timeout_ns) {
  if (timeout_ns <= 0) {
    throw std::invalid_argument("lease timeout must be positive");
  }
  if (now_ns > std::numeric_limits<std::int64_t>::max() - timeout_ns) {
    throw std::invalid_argument("lease expiration exceeds the signed nanosecond range");
  }
  return now_ns + timeout_ns;
}

void require_lease_identity(const std::string& concurrency_key, const std::string& owner_run_id,
                            const std::string& lease_id) {
  if (concurrency_key.empty() || owner_run_id.empty() || lease_id.empty()) {
    throw std::invalid_argument("lease concurrency_key, owner_run_id, and lease_id must not be empty");
  }
}

ResourceLease lease_from_row(sqlite3_stmt* statement) {
  return ResourceLease{
      .concurrency_key = column_text(statement, 0),
      .owner_run_id = column_text(statement, 1),
      .lease_id = column_text(statement, 2),
      .fencing_token = static_cast<std::uint64_t>(sqlite3_column_int64(statement, 3)),
      .clock_domain = column_text(statement, 4),
      .boot_id = sqlite3_column_type(statement, 5) == SQLITE_NULL
                     ? std::string{}
                     : column_text(statement, 5),
      .acquired_boottime_ns = sqlite3_column_type(statement, 6) == SQLITE_NULL
                            ? std::int64_t{}
                            : sqlite3_column_int64(statement, 6),
      .expires_boottime_ns = sqlite3_column_type(statement, 7) == SQLITE_NULL
                           ? std::int64_t{}
                           : sqlite3_column_int64(statement, 7),
      .acquired_wall_time_ns = sqlite3_column_int64(statement, 8),
      .expires_wall_time_ns = sqlite3_column_int64(statement, 9),
  };
}

std::optional<HostGrantSagaSnapshot> load_host_grant_saga(
    sqlite3* database, const std::string& request_id) {
  if (request_id.empty()) {
    throw std::invalid_argument("host resource request_id must not be empty");
  }
  Statement query(database, R"sql(
    SELECT request.canonical_request_json,
           grant.canonical_grant_json,
           release_intent.canonical_release_request_json,
           release_receipt.canonical_release_receipt_json
    FROM host_resource_requests AS request
    LEFT JOIN host_resource_grants AS grant
      ON grant.request_id=request.request_id
    LEFT JOIN host_resource_release_intents AS release_intent
      ON release_intent.request_id=request.request_id
    LEFT JOIN host_resource_release_receipts AS release_receipt
      ON release_receipt.request_id=request.request_id
    WHERE request.request_id=?
  )sql");
  bind_text(query.get(), 1, request_id);
  const int status = sqlite3_step(query.get());
  if (status == SQLITE_DONE) return std::nullopt;
  if (status != SQLITE_ROW) {
    throw std::runtime_error("could not read host grant saga");
  }
  HostGrantSagaSnapshot result{
      .request = resource_request_from_json(
          nlohmann::json::parse(column_text(query.get(), 0))),
      .busy_outcome_digest = std::nullopt,
      .grant = std::nullopt,
      .release_intent = std::nullopt,
      .release_receipt = std::nullopt,
  };
  if (sqlite3_column_type(query.get(), 1) != SQLITE_NULL) {
    result.grant = resource_bundle_grant_from_json(
        nlohmann::json::parse(column_text(query.get(), 1)));
  }
  if (sqlite3_column_type(query.get(), 2) != SQLITE_NULL) {
    result.release_intent = resource_release_request_from_json(
        nlohmann::json::parse(column_text(query.get(), 2)));
  }
  if (sqlite3_column_type(query.get(), 3) != SQLITE_NULL) {
    result.release_receipt = resource_release_receipt_from_json(
        nlohmann::json::parse(column_text(query.get(), 3)));
  }
  if (sqlite3_step(query.get()) != SQLITE_DONE) {
    throw std::runtime_error("host grant saga identity is not unique");
  }
  Statement busy(database, R"sql(
    SELECT run_id, event_type, payload_json FROM events WHERE event_id=?
  )sql");
  bind_text(busy.get(), 1, "host-resource-busy:" + request_id);
  const int busy_status = sqlite3_step(busy.get());
  if (busy_status == SQLITE_ROW) {
    const auto payload = nlohmann::json::parse(column_text(busy.get(), 2));
    if (column_text(busy.get(), 0) != result.request.run_id ||
        column_text(busy.get(), 1) != "host.resource_busy_recorded" ||
        !payload.is_object() || payload.size() != 3U ||
        payload.value("request_id", std::string{}) != request_id ||
        payload.value("request_digest", std::string{}) !=
            result.request.canonical_request_digest ||
        !payload.contains("outcome_digest") ||
        !payload.at("outcome_digest").is_string()) {
      throw std::runtime_error("host busy outcome event is malformed");
    }
    result.busy_outcome_digest =
        payload.at("outcome_digest").get<std::string>();
    if (sqlite3_step(busy.get()) != SQLITE_DONE) {
      throw std::runtime_error("host busy outcome identity is not unique");
    }
  } else if (busy_status != SQLITE_DONE) {
    throw std::runtime_error("could not read host busy outcome");
  }
  return result;
}

Event host_saga_event(sqlite3* database, std::string event_id,
                      std::string event_type, std::string run_id,
                      std::int64_t wall_time_ns,
                      std::uint64_t monotonic_time_ns,
                      nlohmann::json payload) {
  Statement projection(database, R"sql(
    SELECT run_revision FROM run_projection WHERE run_id=?
  )sql");
  bind_text(projection.get(), 1, run_id);
  if (sqlite3_step(projection.get()) != SQLITE_ROW) {
    throw OperationPreconditionError(
        "host grant saga requires a durable run projection");
  }
  const auto run_revision =
      static_cast<std::uint64_t>(sqlite3_column_int64(projection.get(), 0));
  Statement plan(database,
                 "SELECT COALESCE(MAX(plan_revision),0) FROM events WHERE run_id=?");
  bind_text(plan.get(), 1, run_id);
  if (sqlite3_step(plan.get()) != SQLITE_ROW) {
    throw std::runtime_error("could not bind host saga plan revision");
  }
  return {.event_id = std::move(event_id),
          .run_id = std::move(run_id),
          .run_revision = run_revision,
          .plan_revision =
              static_cast<std::uint64_t>(sqlite3_column_int64(plan.get(), 0)),
          .node_id = {},
          .attempt_id = {},
          .worker_sequence = 0,
          .event_type = std::move(event_type),
          .event_version = 1,
          .wall_time_ns = wall_time_ns,
          .monotonic_time_ns = monotonic_time_ns,
          .optimizer_step = std::nullopt,
          .payload = std::move(payload)};
}

bool verify_host_saga_projection(sqlite3* database, std::string* reason) {
  const auto fail = [&](std::string message) {
    if (reason != nullptr) *reason = std::move(message);
    return false;
  };
  try {
    const auto require_event = [&](const std::string& event_id,
                                   std::string_view event_type,
                                   const std::string& run_id,
                                   const nlohmann::json& payload) {
      Statement event(database, R"sql(
        SELECT run_id, event_type, payload_json FROM events WHERE event_id=?
      )sql");
      bind_text(event.get(), 1, event_id);
      if (sqlite3_step(event.get()) != SQLITE_ROW ||
          column_text(event.get(), 0) != run_id ||
          column_text(event.get(), 1) != event_type ||
          nlohmann::json::parse(column_text(event.get(), 2)) != payload ||
          sqlite3_step(event.get()) != SQLITE_DONE) {
        throw std::runtime_error("host saga projection has no exact chained event " +
                                 event_id);
      }
    };
    std::uint64_t requests = 0;
    std::uint64_t busy_outcomes = 0;
    std::uint64_t grants = 0;
    std::uint64_t release_intents = 0;
    std::uint64_t release_receipts = 0;
    Statement rows(database, R"sql(
      SELECT request_id, journal_id, run_id, concurrency_key,
             logical_lease_id, logical_fencing_token, request_digest,
             canonical_request_json
      FROM host_resource_requests ORDER BY request_id
    )sql");
    int status = SQLITE_OK;
    while ((status = sqlite3_step(rows.get())) == SQLITE_ROW) {
      ++requests;
      const std::string request_id = column_text(rows.get(), 0);
      const std::string concurrency_key = column_text(rows.get(), 3);
      const auto saga = load_host_grant_saga(database, request_id);
      if (!saga || saga->request.journal_id != column_text(rows.get(), 1) ||
          saga->request.run_id != column_text(rows.get(), 2) ||
          saga->request.logical_lease_id != column_text(rows.get(), 4) ||
          saga->request.logical_fencing_token !=
              static_cast<std::uint64_t>(sqlite3_column_int64(rows.get(), 5)) ||
          saga->request.canonical_request_digest != column_text(rows.get(), 6) ||
          resource_request_json(saga->request).dump() != column_text(rows.get(), 7)) {
        return fail("host resource request projection diverges from canonical JSON");
      }
      Statement lease(database, R"sql(
        SELECT owner_run_id, lease_id, fencing_token
        FROM resource_leases WHERE concurrency_key=?
      )sql");
      bind_text(lease.get(), 1, concurrency_key);
      if (sqlite3_step(lease.get()) != SQLITE_ROW) {
        return fail("host resource request has no logical lease lineage");
      }
      const auto current_token =
          static_cast<std::uint64_t>(sqlite3_column_int64(lease.get(), 2));
      if (current_token < saga->request.logical_fencing_token ||
          (current_token == saga->request.logical_fencing_token &&
           (column_text(lease.get(), 0) != saga->request.run_id ||
            column_text(lease.get(), 1) != saga->request.logical_lease_id))) {
        return fail("host resource request disagrees with logical lease lineage");
      }
      require_event("host-resource-request:" + request_id,
                    "host.resource_request_recorded", saga->request.run_id,
                    {{"concurrency_key", concurrency_key},
                     {"request", resource_request_json(saga->request)}});
      if (saga->busy_outcome_digest) {
        ++busy_outcomes;
        if (saga->grant || saga->release_intent || saga->release_receipt) {
          return fail("host busy outcome conflicts with a grant saga");
        }
        require_event("host-resource-busy:" + request_id,
                      "host.resource_busy_recorded", saga->request.run_id,
                      {{"request_id", request_id},
                       {"request_digest",
                        saga->request.canonical_request_digest},
                       {"outcome_digest", *saga->busy_outcome_digest}});
      }
      if (saga->grant) {
        ++grants;
        if (saga->grant->request_id != request_id ||
            saga->grant->request_digest !=
                saga->request.canonical_request_digest ||
            saga->grant->journal_id != saga->request.journal_id ||
            saga->grant->run_id != saga->request.run_id ||
            saga->grant->logical_lease_id != saga->request.logical_lease_id ||
            saga->grant->logical_fencing_token !=
                saga->request.logical_fencing_token) {
          return fail("host grant projection is not closed over its request");
        }
        Statement grant(database, R"sql(
          SELECT allocation_id, request_digest, grant_digest,
                 canonical_grant_json FROM host_resource_grants WHERE request_id=?
        )sql");
        bind_text(grant.get(), 1, request_id);
        if (sqlite3_step(grant.get()) != SQLITE_ROW ||
            column_text(grant.get(), 0) != saga->grant->allocation_id ||
            column_text(grant.get(), 1) != saga->grant->request_digest ||
            column_text(grant.get(), 2) != saga->grant->receipt_digest ||
            column_text(grant.get(), 3) !=
                resource_bundle_grant_json(*saga->grant).dump()) {
          return fail("host grant scalar projection diverges from its receipt");
        }
        require_event("host-resource-grant:" + request_id,
                      "host.resource_grant_recorded", saga->request.run_id,
                      {{"request_id", request_id},
                       {"grant", resource_bundle_grant_json(*saga->grant)}});
      }
      if (saga->release_intent) {
        ++release_intents;
        if (!saga->grant ||
            saga->release_intent->allocation_id != saga->grant->allocation_id ||
            saga->release_intent->grant_digest != saga->grant->receipt_digest ||
            saga->release_intent->journal_id != saga->request.journal_id ||
            saga->release_intent->run_id != saga->request.run_id ||
            saga->release_intent->logical_lease_id !=
                saga->request.logical_lease_id ||
            saga->release_intent->logical_fencing_token !=
                saga->request.logical_fencing_token) {
          return fail("host release intent is not closed over its grant");
        }
        Statement intent(database, R"sql(
          SELECT release_request_id, allocation_id, grant_digest,
                 release_request_digest, canonical_release_request_json
          FROM host_resource_release_intents WHERE request_id=?
        )sql");
        bind_text(intent.get(), 1, request_id);
        if (sqlite3_step(intent.get()) != SQLITE_ROW ||
            column_text(intent.get(), 0) !=
                saga->release_intent->release_request_id ||
            column_text(intent.get(), 1) != saga->release_intent->allocation_id ||
            column_text(intent.get(), 2) != saga->release_intent->grant_digest ||
            column_text(intent.get(), 3) !=
                saga->release_intent->canonical_request_digest ||
            column_text(intent.get(), 4) !=
                resource_release_request_json(*saga->release_intent).dump()) {
          return fail("host release intent scalar projection diverges");
        }
        require_event("host-resource-release-intent:" + request_id,
                      "host.resource_release_intent_recorded",
                      saga->request.run_id,
                      {{"request_id", request_id},
                       {"release", resource_release_request_json(
                                       *saga->release_intent)}});
      }
      if (saga->release_receipt) {
        ++release_receipts;
        if (!saga->grant || !saga->release_intent ||
            saga->release_receipt->release_request_id !=
                saga->release_intent->release_request_id ||
            saga->release_receipt->release_request_digest !=
                saga->release_intent->canonical_request_digest ||
            saga->release_receipt->allocation_id != saga->grant->allocation_id ||
            saga->release_receipt->grant_digest != saga->grant->receipt_digest ||
            saga->release_receipt->host_id != saga->grant->host_id ||
            saga->release_receipt->boot_id != saga->grant->boot_id ||
            saga->release_receipt->broker_epoch != saga->grant->broker_epoch) {
          return fail("host release receipt is not closed over its intent");
        }
        Statement receipt(database, R"sql(
          SELECT release_request_id, release_receipt_digest,
                 canonical_release_receipt_json
          FROM host_resource_release_receipts WHERE request_id=?
        )sql");
        bind_text(receipt.get(), 1, request_id);
        if (sqlite3_step(receipt.get()) != SQLITE_ROW ||
            column_text(receipt.get(), 0) !=
                saga->release_receipt->release_request_id ||
            column_text(receipt.get(), 1) !=
                saga->release_receipt->receipt_digest ||
            column_text(receipt.get(), 2) !=
                resource_release_receipt_json(*saga->release_receipt).dump()) {
          return fail("host release receipt scalar projection diverges");
        }
        require_event(
            "host-resource-release-receipt:" + request_id,
            "host.resource_release_receipt_recorded", saga->request.run_id,
            {{"request_id", request_id},
             {"receipt",
              resource_release_receipt_json(*saga->release_receipt)}});
      }
    }
    if (status != SQLITE_DONE) {
      return fail("could not scan host saga projections");
    }
    const auto require_projection_count = [&](std::string_view table,
                                              std::uint64_t expected) {
      Statement count(database,
                      "SELECT COUNT(*) FROM " + std::string(table));
      if (sqlite3_step(count.get()) != SQLITE_ROW ||
          static_cast<std::uint64_t>(sqlite3_column_int64(count.get(), 0)) !=
              expected ||
          sqlite3_step(count.get()) != SQLITE_DONE) {
        throw std::runtime_error(
            "host saga projection contains unreachable child rows");
      }
    };
    require_projection_count("host_resource_requests", requests);
    require_projection_count("host_resource_grants", grants);
    require_projection_count("host_resource_release_intents", release_intents);
    require_projection_count("host_resource_release_receipts",
                             release_receipts);
    const auto require_count = [&](std::string_view type,
                                   std::uint64_t expected) {
      Statement count(database, "SELECT COUNT(*) FROM events WHERE event_type=?");
      bind_text(count.get(), 1, std::string(type));
      if (sqlite3_step(count.get()) != SQLITE_ROW ||
          static_cast<std::uint64_t>(sqlite3_column_int64(count.get(), 0)) !=
              expected) {
        throw std::runtime_error("host saga chained event count diverges");
      }
    };
    require_count("host.resource_request_recorded", requests);
    require_count("host.resource_busy_recorded", busy_outcomes);
    require_count("host.resource_grant_recorded", grants);
    require_count("host.resource_release_intent_recorded", release_intents);
    require_count("host.resource_release_receipt_recorded", release_receipts);
    return true;
  } catch (const std::exception& error) {
    return fail(error.what());
  }
}

void replay_host_saga_projection(sqlite3* database, const Event& event) {
  if (event.event_type == "host.resource_request_recorded") {
    if (!event.payload.is_object() || event.payload.size() != 2U ||
        !event.payload.contains("concurrency_key") ||
        !event.payload.at("concurrency_key").is_string() ||
        !event.payload.contains("request")) {
      throw std::runtime_error("chained host request event is malformed");
    }
    const auto request =
        resource_request_from_json(event.payload.at("request"));
    const std::string concurrency_key =
        event.payload.at("concurrency_key").get<std::string>();
    if (request.run_id != event.run_id || concurrency_key.empty()) {
      throw std::runtime_error("chained host request identity diverges");
    }
    Statement insert(database, R"sql(
      INSERT INTO host_resource_requests(
        request_id, journal_id, run_id, concurrency_key, logical_lease_id,
        logical_fencing_token, request_digest, canonical_request_json
      ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
    )sql");
    bind_text(insert.get(), 1, request.request_id);
    bind_text(insert.get(), 2, request.journal_id);
    bind_text(insert.get(), 3, request.run_id);
    bind_text(insert.get(), 4, concurrency_key);
    bind_text(insert.get(), 5, request.logical_lease_id);
    bind_integer(insert.get(), 6,
                 checked_integer(request.logical_fencing_token,
                                 "logical_fencing_token"));
    bind_text(insert.get(), 7, request.canonical_request_digest);
    bind_text(insert.get(), 8, resource_request_json(request).dump());
    require_done(database, insert.get(), "replay host resource request");
    return;
  }
  const auto required_request_id = [&]() {
    if (!event.payload.is_object() || !event.payload.contains("request_id") ||
        !event.payload.at("request_id").is_string()) {
      throw std::runtime_error("chained host saga event has no request identity");
    }
    return event.payload.at("request_id").get<std::string>();
  };
  if (event.event_type == "host.resource_grant_recorded") {
    if (event.payload.size() != 2U || !event.payload.contains("grant")) {
      throw std::runtime_error("chained host grant event is malformed");
    }
    const std::string request_id = required_request_id();
    const auto grant =
        resource_bundle_grant_from_json(event.payload.at("grant"));
    if (grant.request_id != request_id || grant.run_id != event.run_id) {
      throw std::runtime_error("chained host grant identity diverges");
    }
    Statement insert(database, R"sql(
      INSERT INTO host_resource_grants(
        request_id, allocation_id, request_digest, grant_digest,
        canonical_grant_json
      ) VALUES(?, ?, ?, ?, ?)
    )sql");
    bind_text(insert.get(), 1, request_id);
    bind_text(insert.get(), 2, grant.allocation_id);
    bind_text(insert.get(), 3, grant.request_digest);
    bind_text(insert.get(), 4, grant.receipt_digest);
    bind_text(insert.get(), 5, resource_bundle_grant_json(grant).dump());
    require_done(database, insert.get(), "replay host grant receipt");
    return;
  }
  if (event.event_type == "host.resource_busy_recorded") {
    if (!event.payload.is_object() || event.payload.size() != 3U ||
        !event.payload.contains("request_digest") ||
        !event.payload.at("request_digest").is_string() ||
        !event.payload.contains("outcome_digest") ||
        !event.payload.at("outcome_digest").is_string()) {
      throw std::runtime_error("chained host busy outcome is malformed");
    }
    const std::string request_id = required_request_id();
    Statement request(database, R"sql(
      SELECT run_id, request_digest FROM host_resource_requests
      WHERE request_id=?
    )sql");
    bind_text(request.get(), 1, request_id);
    if (sqlite3_step(request.get()) != SQLITE_ROW ||
        column_text(request.get(), 0) != event.run_id ||
        column_text(request.get(), 1) !=
            event.payload.at("request_digest").get<std::string>() ||
        sqlite3_step(request.get()) != SQLITE_DONE) {
      throw std::runtime_error("chained host busy outcome identity diverges");
    }
    return;
  }
  if (event.event_type == "host.resource_release_intent_recorded") {
    if (event.payload.size() != 2U || !event.payload.contains("release")) {
      throw std::runtime_error("chained host release intent is malformed");
    }
    const std::string request_id = required_request_id();
    const auto release =
        resource_release_request_from_json(event.payload.at("release"));
    if (release.run_id != event.run_id) {
      throw std::runtime_error("chained host release intent identity diverges");
    }
    Statement insert(database, R"sql(
      INSERT INTO host_resource_release_intents(
        release_request_id, request_id, allocation_id, grant_digest,
        release_request_digest, canonical_release_request_json
      ) VALUES(?, ?, ?, ?, ?, ?)
    )sql");
    bind_text(insert.get(), 1, release.release_request_id);
    bind_text(insert.get(), 2, request_id);
    bind_text(insert.get(), 3, release.allocation_id);
    bind_text(insert.get(), 4, release.grant_digest);
    bind_text(insert.get(), 5, release.canonical_request_digest);
    bind_text(insert.get(), 6, resource_release_request_json(release).dump());
    require_done(database, insert.get(), "replay host release intent");
    return;
  }
  if (event.event_type == "host.resource_release_receipt_recorded") {
    if (event.payload.size() != 2U || !event.payload.contains("receipt")) {
      throw std::runtime_error("chained host release receipt is malformed");
    }
    const std::string request_id = required_request_id();
    const auto receipt =
        resource_release_receipt_from_json(event.payload.at("receipt"));
    Statement insert(database, R"sql(
      INSERT INTO host_resource_release_receipts(
        release_request_id, request_id, release_receipt_digest,
        canonical_release_receipt_json
      ) VALUES(?, ?, ?, ?)
    )sql");
    bind_text(insert.get(), 1, receipt.release_request_id);
    bind_text(insert.get(), 2, request_id);
    bind_text(insert.get(), 3, receipt.receipt_digest);
    bind_text(insert.get(), 4,
              resource_release_receipt_json(receipt).dump());
    require_done(database, insert.get(), "replay host release receipt");
  }
}

LeaseRenewalReceipt renewal_receipt_from_row(sqlite3_stmt* statement) {
  return {
      .concurrency_key = column_text(statement, 0),
      .owner_run_id = column_text(statement, 1),
      .lease_id = column_text(statement, 2),
      .fencing_token =
          static_cast<std::uint64_t>(sqlite3_column_int64(statement, 3)),
      .clock_domain = column_text(statement, 4),
      .boot_id = column_text(statement, 5),
      .acquired_boottime_ns = sqlite3_column_int64(statement, 6),
      .acquired_wall_time_ns = sqlite3_column_int64(statement, 7),
      .prior_expires_boottime_ns = sqlite3_column_int64(statement, 8),
      .new_expires_boottime_ns = sqlite3_column_int64(statement, 9),
      .prior_expires_wall_time_ns = sqlite3_column_int64(statement, 10),
      .new_expires_wall_time_ns = sqlite3_column_int64(statement, 11),
      .renewed_boottime_ns = sqlite3_column_int64(statement, 12),
      .renewed_wall_time_ns = sqlite3_column_int64(statement, 13),
  };
}

Dispatch dispatch_from_row(sqlite3_stmt* statement) {
  Dispatch dispatch{
      .dispatch_id = column_text(statement, 0),
      .run_id = column_text(statement, 1),
      .run_revision = static_cast<std::uint64_t>(sqlite3_column_int64(statement, 2)),
      .plan_revision = static_cast<std::uint64_t>(sqlite3_column_int64(statement, 3)),
      .node_id = column_text(statement, 4),
      .attempt_id = column_text(statement, 5),
      .component = column_text(statement, 6),
      .operation = column_text(statement, 7),
      .status = column_text(statement, 8) == "completed" ? DispatchStatus::completed
                                                        : DispatchStatus::prepared,
      .result_event_id = std::nullopt,
  };
  if (sqlite3_column_type(statement, 9) != SQLITE_NULL) {
    dispatch.result_event_id = column_text(statement, 9);
  }
  return dispatch;
}

bool same_dispatch_attempt(const Dispatch& left, const Dispatch& right) {
  return left.dispatch_id == right.dispatch_id && left.run_id == right.run_id &&
         left.plan_revision == right.plan_revision && left.node_id == right.node_id &&
         left.attempt_id == right.attempt_id && left.component == right.component &&
         left.operation == right.operation;
}

std::string command_status_name(ControlCommandStatus status) {
  switch (status) {
    case ControlCommandStatus::requested:
      return "requested";
    case ControlCommandStatus::applied:
      return "applied";
    case ControlCommandStatus::rejected:
      return "rejected";
    case ControlCommandStatus::restart_required:
      return "restart_required";
  }
  throw std::invalid_argument("invalid control command status");
}

ControlCommandStatus command_status(std::string_view value) {
  if (value == "requested") {
    return ControlCommandStatus::requested;
  }
  if (value == "applied") {
    return ControlCommandStatus::applied;
  }
  if (value == "rejected") {
    return ControlCommandStatus::rejected;
  }
  if (value == "restart_required") {
    return ControlCommandStatus::restart_required;
  }
  throw std::runtime_error("stored control command has an invalid status");
}

ControlCommand command_from_row(sqlite3_stmt* statement) {
  const auto apply = enum_from_string<ApplyPoint>(column_text(statement, 7));
  if (!apply) {
    throw std::runtime_error("stored control command has an invalid application point");
  }
  ControlCommand command{
      .command_id = column_text(statement, 0),
      .run_id = column_text(statement, 1),
      .idempotency_key = column_text(statement, 2),
      .expected_run_revision = static_cast<std::uint64_t>(sqlite3_column_int64(statement, 3)),
      .expected_control_revision = static_cast<std::uint64_t>(sqlite3_column_int64(statement, 4)),
      .control_revision = static_cast<std::uint64_t>(sqlite3_column_int64(statement, 5)),
      .plan_revision = static_cast<std::uint64_t>(sqlite3_column_int64(statement, 6)),
      .apply_point = *apply,
      .requires_pause = sqlite3_column_int(statement, 8) != 0,
      .assignments = nlohmann::json::parse(column_text(statement, 9)),
      .author = column_text(statement, 10),
      .reason = column_text(statement, 11),
      .status = command_status(column_text(statement, 12)),
      .effective_step = std::nullopt,
      .effective_values = nlohmann::json::parse(column_text(statement, 14)),
      .diagnostics = nlohmann::json::parse(column_text(statement, 15)),
      .acknowledgement = std::nullopt,
      .acknowledged_at_ns = std::nullopt,
  };
  if (sqlite3_column_type(statement, 13) != SQLITE_NULL) {
    command.effective_step = static_cast<std::uint64_t>(sqlite3_column_int64(statement, 13));
  }
  if (sqlite3_column_type(statement, 16) != SQLITE_NULL) {
    command.acknowledgement = ControlAcknowledgementIdentity{
        .concurrency_key = column_text(statement, 16),
        .lease_id = column_text(statement, 17),
        .fencing_token = static_cast<std::uint64_t>(sqlite3_column_int64(statement, 18)),
        .node_id = column_text(statement, 19),
        .attempt_id = column_text(statement, 20),
        .worker_sequence = static_cast<std::uint64_t>(sqlite3_column_int64(statement, 21)),
    };
    command.acknowledged_at_ns = sqlite3_column_int64(statement, 22);
  }
  return command;
}

bool same_command_request(const ControlCommand& left, const ControlCommand& right) {
  return left.run_id == right.run_id && left.idempotency_key == right.idempotency_key &&
         left.expected_run_revision == right.expected_run_revision &&
         left.expected_control_revision == right.expected_control_revision &&
         left.plan_revision == right.plan_revision && left.apply_point == right.apply_point &&
         left.requires_pause == right.requires_pause && left.assignments == right.assignments &&
         left.author == right.author && left.reason == right.reason;
}

int open_existing_directory_by_components(
    const std::filesystem::path& absolute_path) {
  if (!absolute_path.is_absolute()) {
    throw std::runtime_error("authority namespace directory is not absolute");
  }
  int current = ::open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (current < 0) {
    throw std::runtime_error("could not open filesystem root for authority validation");
  }
  for (const auto& part : absolute_path.relative_path()) {
    const std::string component = part.string();
    if (component.empty() || component == "." || component == ".." ||
        component.find('/') != std::string::npos) {
      (void)::close(current);
      throw std::runtime_error("authority namespace has a noncanonical component");
    }
    const int next = ::openat(
        current, component.c_str(),
        O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (next < 0) {
      const std::string message = std::strerror(errno);
      (void)::close(current);
      throw std::runtime_error("could not securely re-resolve authority namespace: " +
                               message);
    }
    (void)::close(current);
    current = next;
  }
  return current;
}

bool safe_authority_file(const struct stat& status, std::uint64_t owner_uid) {
  return S_ISREG(status.st_mode) &&
         static_cast<std::uint64_t>(status.st_uid) == owner_uid &&
         status.st_nlink == 1 &&
         (status.st_mode & (S_IWGRP | S_IWOTH)) == 0;
}

}  // namespace

Journal::Journal(const std::filesystem::path& path,
                 std::optional<JournalFileIdentity> expected_file,
                 HostGrantEnforcement host_grant_enforcement,
                 std::optional<HostIdentity> expected_host_grant_authority)
    : expected_file_(std::move(expected_file)),
      host_grant_enforcement_(host_grant_enforcement),
      expected_host_grant_authority_(
          std::move(expected_host_grant_authority)) {
  if (expected_host_grant_authority_) {
    const auto valid_host_identifier = [](std::string_view value) {
      return !value.empty() &&
             value.size() <= HostResourceBounds::maximum_identifier_bytes &&
             std::ranges::all_of(value, [](char character) {
               return std::isalnum(static_cast<unsigned char>(character)) != 0 ||
                      character == '.' || character == '_' || character == ':' ||
                      character == '/' || character == '-';
             });
    };
    if (!valid_host_identifier(expected_host_grant_authority_->host_id) ||
        !valid_host_identifier(expected_host_grant_authority_->boot_id)) {
      throw std::invalid_argument(
          "trusted host grant authority has a malformed host or boot identity");
    }
  }
  const auto parent = path.parent_path();
  if (!expected_file_ && !parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  const int open_flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE |
                         (expected_file_ ? 0 : SQLITE_OPEN_NOFOLLOW);
  if (sqlite3_open_v2(path.c_str(), &database_, open_flags,
                      nullptr) !=
      SQLITE_OK) {
    const std::string message = database_ ? sqlite3_errmsg(database_) : "unknown error";
    if (database_) {
      sqlite3_close(database_);
      database_ = nullptr;
    }
    throw std::runtime_error("sqlite open failed: " + message);
  }
  try {
    if (expected_file_) {
      require_file_identity(*expected_file_);
      if (sqlite3_set_authorizer(database_, &Journal::authorize_database_operation,
                                 this) != SQLITE_OK) {
        throw std::runtime_error("could not install journal authority boundary");
      }
      (void)sqlite3_commit_hook(database_, &Journal::authorize_commit, this);
    }
    initialize();
  } catch (...) {
    sqlite3_close(database_);
    database_ = nullptr;
    throw;
  }
}

void Journal::require_file_identity(const JournalFileIdentity& expected) const {
  require_namespace_identity(expected);
  int moved = 0;
  const int control = sqlite3_file_control(
      database_, "main", SQLITE_FCNTL_HAS_MOVED, &moved);
  const char* filename = sqlite3_db_filename(database_, "main");
  struct stat status {};
  if (control != SQLITE_OK || moved != 0 || filename == nullptr ||
      ::stat(filename, &status) != 0 || !S_ISREG(status.st_mode) ||
      static_cast<std::uint64_t>(status.st_dev) != expected.device ||
      static_cast<std::uint64_t>(status.st_ino) != expected.inode) {
    throw std::runtime_error(
        "SQLite journal file does not match the authority-locked inode");
  }
}

void Journal::require_namespace_identity(
    const JournalFileIdentity& expected) const {
  if (expected.directory_path.empty() || expected.journal_name.empty() ||
      expected.authority_name.empty()) {
    throw std::runtime_error("journal authority namespace identity is incomplete");
  }
  const int directory = open_existing_directory_by_components(
      std::filesystem::path(expected.directory_path));
  struct CloseDirectory final {
    int descriptor;
    ~CloseDirectory() { (void)::close(descriptor); }
  } close{directory};

  struct stat directory_status {};
  if (::fstat(directory, &directory_status) != 0 ||
      !S_ISDIR(directory_status.st_mode) ||
      static_cast<std::uint64_t>(directory_status.st_dev) !=
          expected.directory_device ||
      static_cast<std::uint64_t>(directory_status.st_ino) !=
          expected.directory_inode ||
      static_cast<std::uint64_t>(directory_status.st_uid) != expected.owner_uid ||
      (directory_status.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
    throw std::runtime_error("journal authority directory identity has moved");
  }

  const auto require_entry = [&](const std::string& name, std::uint64_t device,
                                 std::uint64_t inode,
                                 std::string_view description) {
    struct stat status {};
    if (::fstatat(directory, name.c_str(), &status, AT_SYMLINK_NOFOLLOW) != 0 ||
        !safe_authority_file(status, expected.owner_uid) ||
        static_cast<std::uint64_t>(status.st_dev) != device ||
        static_cast<std::uint64_t>(status.st_ino) != inode) {
      if (description == "database") {
        throw std::runtime_error(
            "SQLite journal file does not match the authority-locked inode");
      }
      throw std::runtime_error("journal " + std::string(description) +
                               " identity has moved");
    }
  };
  require_entry(expected.journal_name, expected.device, expected.inode,
                "database");
  require_entry(expected.authority_name, expected.authority_device,
                expected.authority_inode, "authority sidecar");

  // WAL and rollback auxiliaries are legitimately deleted and recreated by
  // SQLite, so their inode cannot be latched across an observed absence. Every
  // authoritative SQL boundary instead requires any present generation to be
  // an unaliased, non-symlink, owner-controlled regular file.
  for (const std::string_view suffix :
       {std::string_view{"-journal"}, std::string_view{"-wal"},
        std::string_view{"-shm"}}) {
    const std::string name = expected.journal_name + std::string(suffix);
    struct stat status {};
    if (::fstatat(directory, name.c_str(), &status, AT_SYMLINK_NOFOLLOW) != 0) {
      if (errno == ENOENT) continue;
      throw std::runtime_error("could not inspect SQLite authority auxiliary");
    }
    if (!safe_authority_file(status, expected.owner_uid)) {
      throw std::runtime_error("SQLite authority auxiliary is unsafe: " + name);
    }
  }
}

bool Journal::validate_authority_boundary() const noexcept {
  if (!expected_file_) return true;
  if (authority_poisoned_.load(std::memory_order_acquire)) return false;
  try {
    require_namespace_identity(*expected_file_);
    return true;
  } catch (...) {
    authority_poisoned_.store(true, std::memory_order_release);
    return false;
  }
}

void Journal::require_attested_authority() const {
  if (!expected_file_ || !expected_host_grant_authority_) {
    throw OperationPreconditionError(
        "journal fence inspection requires retained filesystem and host authority");
  }
  if (!validate_authority_boundary()) {
    throw OperationPreconditionError("journal filesystem authority is poisoned");
  }
  try {
    require_file_identity(*expected_file_);
  } catch (...) {
    authority_poisoned_.store(true, std::memory_order_release);
    throw OperationPreconditionError("journal filesystem authority moved");
  }
}

int Journal::authorize_database_operation(void* context, int action,
                                          const char*, const char*,
                                          const char*, const char*) noexcept {
  // SQLITE_READ and SQLITE_FUNCTION are subordinate callbacks of a statement
  // whose top-level SELECT/INSERT/UPDATE/etc. callback was already checked.
  if (action == SQLITE_READ || action == SQLITE_FUNCTION ||
      action == SQLITE_RECURSIVE) {
    return SQLITE_OK;
  }
  const auto* journal = static_cast<const Journal*>(context);
  return journal != nullptr && journal->validate_authority_boundary()
             ? SQLITE_OK
             : SQLITE_DENY;
}

int Journal::authorize_commit(void* context) noexcept {
  const auto* journal = static_cast<const Journal*>(context);
  return journal != nullptr && journal->validate_authority_boundary() ? 0 : 1;
}

Journal::~Journal() {
  if (database_) {
    sqlite3_close(database_);
  }
}

Journal::ReadSnapshot::ReadSnapshot(sqlite3* database) : database_(database) {
  if (database_ == nullptr || sqlite3_get_autocommit(database_) == 0) {
    throw std::logic_error("journal read snapshot requires an idle database connection");
  }
  char* error_message = nullptr;
  if (sqlite3_exec(database_, "BEGIN", nullptr, nullptr, &error_message) != SQLITE_OK) {
    const std::string message = error_message ? error_message : sqlite3_errmsg(database_);
    sqlite3_free(error_message);
    database_ = nullptr;
    throw std::runtime_error("sqlite read snapshot failed: " + message);
  }
}

Journal::ReadSnapshot::ReadSnapshot(ReadSnapshot&& other) noexcept
    : database_(std::exchange(other.database_, nullptr)) {}

Journal::ReadSnapshot::~ReadSnapshot() {
  if (database_ != nullptr) {
    sqlite3_exec(database_, "ROLLBACK", nullptr, nullptr, nullptr);
  }
}

Journal::ReadSnapshot Journal::read_snapshot() const {
  return ReadSnapshot(database_);
}

void Journal::initialize() {
  char* error_message = nullptr;
  const auto execute_sql = [&](std::string_view sql, std::string_view action) {
    const std::string owned(sql);
    char* message = nullptr;
    if (sqlite3_exec(database_, owned.c_str(), nullptr, nullptr, &message) !=
        SQLITE_OK) {
      const std::string detail = message ? message : sqlite3_errmsg(database_);
      sqlite3_free(message);
      throw std::runtime_error(std::string(action) + ": " + detail);
    }
  };
  execute_sql(kConnectionPragmas, "could not configure journal connection");
  bool has_metadata = false;
  {
    Statement metadata(database_, R"sql(
      SELECT 1 FROM sqlite_master WHERE type='table' AND name='journal_meta'
    )sql");
    has_metadata = sqlite3_step(metadata.get()) == SQLITE_ROW;
  }
  std::string stored_version;
  if (has_metadata) {
    Statement version(database_, "SELECT value FROM journal_meta WHERE key='schema_version'");
    if (sqlite3_step(version.get()) != SQLITE_ROW) {
      throw std::runtime_error("journal schema version is missing");
    }
    stored_version = column_text(version.get(), 0);
    if (stored_version != "1" && stored_version != "2" && stored_version != "3" &&
        stored_version != "4" && stored_version != "5" &&
        stored_version != "6" && stored_version != "7") {
      throw std::runtime_error("unsupported journal schema version");
    }
  }
  if (!has_metadata) {
    Transaction transaction(database_);
    if (!schema_snapshot(database_).empty()) {
      throw std::runtime_error(
          "refusing to initialize an unversioned nonempty journal database");
    }
    const auto pragma_integer = [&](std::string_view pragma) {
      Statement query(database_, std::string(pragma));
      if (sqlite3_step(query.get()) != SQLITE_ROW ||
          sqlite3_column_type(query.get(), 0) != SQLITE_INTEGER) {
        throw std::runtime_error("could not inspect unversioned SQLite database headers");
      }
      return sqlite3_column_int64(query.get(), 0);
    };
    if (pragma_integer("PRAGMA application_id") != 0 ||
        pragma_integer("PRAGMA user_version") != 0) {
      throw std::runtime_error(
          "refusing to initialize a SQLite database claimed by another application");
    }
    execute_sql(kSchema, "could not create journal schema v7");
    require_exact_schema(database_, canonical_schema_v7(), "v7");
    Statement insert(database_, R"sql(
      INSERT INTO journal_meta(key, value) VALUES('journal_id', ?)
    )sql");
    bind_text(insert.get(), 1, random_journal_id());
    require_done(database_, insert.get(), "initialize journal identity");
    transaction.commit();
    execute_sql(kWalPragma, "could not enable WAL for a new journal database");
    return;
  }
  const auto table_exists = [&](std::string_view table) {
    Statement exists(database_, R"sql(
      SELECT EXISTS(
        SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1
      )
    )sql");
    bind_text(exists.get(), 1, std::string(table));
    if (sqlite3_step(exists.get()) != SQLITE_ROW) {
      throw std::runtime_error("could not inspect journal tables");
    }
    return sqlite3_column_int(exists.get(), 0) != 0;
  };
  const auto table_columns = [&](std::string_view table) {
    std::vector<std::string> columns;
    Statement query(database_, "PRAGMA table_info(\"" + std::string(table) + "\")");
    while (sqlite3_step(query.get()) == SQLITE_ROW) {
      columns.push_back(column_text(query.get(), 1));
    }
    return columns;
  };
  const auto table_definition = [&](std::string_view table) {
    std::vector<std::string> definition;
    Statement query(database_, "PRAGMA table_info(\"" + std::string(table) + "\")");
    while (sqlite3_step(query.get()) == SQLITE_ROW) {
      definition.push_back(column_text(query.get(), 1) + "|" +
                           column_text(query.get(), 2) + "|" +
                           std::to_string(sqlite3_column_int(query.get(), 3)) + "|" +
                           std::to_string(sqlite3_column_int(query.get(), 5)));
    }
    return definition;
  };
  const auto require_columns = [&](std::string_view table,
                                   const std::vector<std::string>& expected,
                                   std::string_view version) {
    if (!table_exists(table) || table_columns(table) != expected) {
      throw std::runtime_error("journal schema " + std::string(version) +
                               " has a malformed " + std::string(table) +
                               " table");
    }
  };
  const auto require_definition = [&](std::string_view table,
                                      const std::vector<std::string>& expected,
                                      std::string_view version) {
    if (table_definition(table) != expected) {
      throw std::runtime_error("journal schema " + std::string(version) +
                               " has a malformed " + std::string(table) +
                               " definition");
    }
  };
  const auto require_schema_fragments = [&](
      std::string_view table, const std::vector<std::string_view>& fragments,
      std::string_view version) {
    Statement query(database_,
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name=?");
    bind_text(query.get(), 1, std::string(table));
    if (sqlite3_step(query.get()) != SQLITE_ROW) {
      throw std::runtime_error("journal schema " + std::string(version) +
                               " is missing " + std::string(table));
    }
    const std::string sql = column_text(query.get(), 0);
    if (std::ranges::any_of(fragments, [&](std::string_view fragment) {
          return sql.find(fragment) == std::string::npos;
        })) {
      throw std::runtime_error("journal schema " + std::string(version) +
                               " has malformed constraints for " +
                               std::string(table));
    }
  };
  const auto require_authority_metadata = [&](std::string_view version) {
    std::map<std::string, std::string> metadata;
    Statement query(database_, "SELECT key, value FROM journal_meta ORDER BY key");
    int status = SQLITE_ROW;
    while ((status = sqlite3_step(query.get())) == SQLITE_ROW) {
      metadata.emplace(column_text(query.get(), 0), column_text(query.get(), 1));
    }
    if (status != SQLITE_DONE) {
      throw std::runtime_error("journal authority metadata is unreadable");
    }
    const auto chain = metadata.find("chain_head");
    const auto identity = metadata.find("journal_id");
    const auto schema_version = metadata.find("schema_version");
    const auto valid_hash = [](std::string_view value) {
      return value.size() == 64U &&
             std::ranges::all_of(value, [](char character) {
               return (character >= '0' && character <= '9') ||
                      (character >= 'a' && character <= 'f');
             });
    };
    const bool extension_metadata_valid = std::ranges::all_of(
        metadata, [&](const auto& entry) {
          const auto& [key, value] = entry;
          if (key == "chain_head" || key == "journal_id" ||
              key == "schema_version")
            return true;
          const auto valid_hashed_extension = [&](std::string_view prefix) {
            return key.starts_with(prefix) &&
                   valid_hash_hex(std::string_view(key).substr(prefix.size())) &&
                   !value.empty() && value.size() <= 4096U;
          };
          return (key == kLegacyControllerAuthorityMetadataKey &&
                  !value.empty() && value.size() <= 4096U) ||
                 valid_hashed_extension(kLeaseAuthorityMetadataPrefix) ||
                 valid_hashed_extension(kControllerAuthorityMetadataPrefix) ||
                 valid_hashed_extension(kControllerIdentityMetadataPrefix);
        });
    std::string controller_scope_reason;
    if (chain == metadata.end() || identity == metadata.end() ||
        schema_version == metadata.end() || !extension_metadata_valid ||
        schema_version->second != version || !valid_hash(chain->second) ||
        !valid_journal_id(identity->second) ||
        !verify_controller_scope_metadata(database_,
                                          &controller_scope_reason)) {
      throw std::runtime_error(
          "established journal authority metadata is missing, malformed, or "
          "unexpected" +
          (controller_scope_reason.empty()
               ? std::string{}
               : ": " + controller_scope_reason));
    }
  };
  if (stored_version == "1" || stored_version == "2" || stored_version == "3") {
    // These schemas predate the complete authority and acknowledgement
    // invariants. Even an apparently empty journal may carry an old table
    // definition that CREATE TABLE IF NOT EXISTS cannot safely replace. Never
    // stamp it as a newer authority schema; preserve it read-only and create a
    // fresh journal or use a future exact-version export tool.
    throw std::runtime_error(
        "refusing to migrate a pre-v4 journal; preserve it read-only and create a new v5 authority journal");
  }
  std::unique_ptr<Transaction> migration_transaction;
  if (stored_version == "4") {
    migration_transaction = std::make_unique<Transaction>(database_);
    require_exact_schema(database_, canonical_schema_v4(), "v4");
    require_authority_metadata("4");
    std::string chain_reason;
    if (!verify_event_chain(&chain_reason)) {
      throw std::runtime_error(
          "refusing to migrate journal schema v4 with an invalid event chain: " +
          chain_reason);
    }
    for (const std::string_view table :
         std::array<std::string_view, 8>{
             "journal_meta", "events", "run_projection", "compiled_plans",
             "resource_leases", "resource_lease_releases", "node_dispatches",
             "control_commands"}) {
      if (!table_exists(table)) {
        throw std::runtime_error("journal schema v4 is partial: missing " +
                                 std::string(table));
      }
    }
    require_columns(
        "resource_leases",
        {"concurrency_key", "owner_run_id", "lease_id", "fencing_token",
         "acquired_at_ns", "expires_at_ns", "released_at_ns"},
        "v4");
    require_columns(
        "resource_lease_releases",
        {"concurrency_key", "owner_run_id", "lease_id", "fencing_token",
         "released_at_ns"},
        "v4");
    require_definition(
        "resource_leases",
        {"concurrency_key|TEXT|1|1", "owner_run_id|TEXT|1|0",
         "lease_id|TEXT|1|0", "fencing_token|INTEGER|1|0",
         "acquired_at_ns|INTEGER|1|0", "expires_at_ns|INTEGER|1|0",
         "released_at_ns|INTEGER|0|0"},
        "v4");
    require_definition(
        "resource_lease_releases",
        {"concurrency_key|TEXT|1|1", "owner_run_id|TEXT|1|0",
         "lease_id|TEXT|1|2", "fencing_token|INTEGER|1|3",
         "released_at_ns|INTEGER|1|0"},
        "v4");
    Statement legacy_leases(database_, R"sql(
      SELECT concurrency_key, owner_run_id, lease_id, fencing_token,
             acquired_at_ns, expires_at_ns, released_at_ns
      FROM resource_leases
    )sql");
    int legacy_lease_status = SQLITE_ROW;
    while ((legacy_lease_status = sqlite3_step(legacy_leases.get())) == SQLITE_ROW) {
      const bool valid_release =
          sqlite3_column_type(legacy_leases.get(), 6) == SQLITE_NULL ||
          (sqlite3_column_type(legacy_leases.get(), 6) == SQLITE_INTEGER &&
           sqlite3_column_int64(legacy_leases.get(), 6) >= 0);
      if (column_text(legacy_leases.get(), 0).empty() ||
          column_text(legacy_leases.get(), 1).empty() ||
          column_text(legacy_leases.get(), 2).empty() ||
          sqlite3_column_type(legacy_leases.get(), 3) != SQLITE_INTEGER ||
          sqlite3_column_int64(legacy_leases.get(), 3) <= 0 ||
          sqlite3_column_type(legacy_leases.get(), 4) != SQLITE_INTEGER ||
          sqlite3_column_int64(legacy_leases.get(), 4) < 0 ||
          sqlite3_column_type(legacy_leases.get(), 5) != SQLITE_INTEGER ||
          sqlite3_column_int64(legacy_leases.get(), 5) < 0 || !valid_release) {
        throw std::runtime_error("journal schema v4 has malformed lease history");
      }
    }
    if (legacy_lease_status != SQLITE_DONE) {
      throw std::runtime_error("journal schema v4 lease history is unreadable");
    }
    Statement legacy_releases(database_, R"sql(
      SELECT concurrency_key, owner_run_id, lease_id, fencing_token,
             released_at_ns
      FROM resource_lease_releases
    )sql");
    int legacy_release_status = SQLITE_ROW;
    while ((legacy_release_status = sqlite3_step(legacy_releases.get())) == SQLITE_ROW) {
      if (column_text(legacy_releases.get(), 0).empty() ||
          column_text(legacy_releases.get(), 1).empty() ||
          column_text(legacy_releases.get(), 2).empty() ||
          sqlite3_column_type(legacy_releases.get(), 3) != SQLITE_INTEGER ||
          sqlite3_column_int64(legacy_releases.get(), 3) <= 0 ||
          sqlite3_column_type(legacy_releases.get(), 4) != SQLITE_INTEGER ||
          sqlite3_column_int64(legacy_releases.get(), 4) < 0) {
        throw std::runtime_error("journal schema v4 has malformed release history");
      }
    }
    if (legacy_release_status != SQLITE_DONE) {
      throw std::runtime_error("journal schema v4 release history is unreadable");
    }
    constexpr std::string_view migration = R"sql(
      ALTER TABLE resource_leases RENAME TO resource_leases_v4;
      CREATE TABLE resource_leases (
        concurrency_key TEXT PRIMARY KEY,
        owner_run_id TEXT NOT NULL,
        lease_id TEXT NOT NULL,
        fencing_token INTEGER NOT NULL,
        clock_domain TEXT NOT NULL CHECK(clock_domain IN ('boottime/v1','legacy-wall/v1')),
        boot_id TEXT,
        acquired_boottime_ns INTEGER,
        expires_boottime_ns INTEGER,
        acquired_wall_time_ns INTEGER NOT NULL,
        expires_wall_time_ns INTEGER NOT NULL,
        released_wall_time_ns INTEGER,
        CHECK(
          (clock_domain='boottime/v1' AND boot_id IS NOT NULL AND
           acquired_boottime_ns IS NOT NULL AND expires_boottime_ns IS NOT NULL) OR
          (clock_domain='legacy-wall/v1' AND boot_id IS NULL AND
           acquired_boottime_ns IS NULL AND expires_boottime_ns IS NULL)
        )
      ) WITHOUT ROWID;
      INSERT INTO resource_leases(
        concurrency_key, owner_run_id, lease_id, fencing_token, clock_domain,
        boot_id, acquired_boottime_ns, expires_boottime_ns,
        acquired_wall_time_ns, expires_wall_time_ns, released_wall_time_ns
      )
      SELECT concurrency_key, owner_run_id, lease_id, fencing_token,
             'legacy-wall/v1', NULL, NULL, NULL,
             acquired_at_ns, expires_at_ns, released_at_ns
      FROM resource_leases_v4;
      DROP TABLE resource_leases_v4;

      ALTER TABLE resource_lease_releases RENAME TO resource_lease_releases_v4;
      CREATE TABLE resource_lease_releases (
        concurrency_key TEXT NOT NULL,
        owner_run_id TEXT NOT NULL,
        lease_id TEXT NOT NULL,
        fencing_token INTEGER NOT NULL,
        clock_domain TEXT NOT NULL CHECK(clock_domain IN ('boottime/v1','legacy-wall/v1')),
        boot_id TEXT,
        released_wall_time_ns INTEGER NOT NULL,
        CHECK(
          (clock_domain='boottime/v1' AND boot_id IS NOT NULL) OR
          (clock_domain='legacy-wall/v1' AND boot_id IS NULL)
        ),
        PRIMARY KEY(concurrency_key, lease_id, fencing_token)
      ) WITHOUT ROWID;
      INSERT INTO resource_lease_releases(
        concurrency_key, owner_run_id, lease_id, fencing_token, clock_domain,
        boot_id, released_wall_time_ns
      )
      SELECT concurrency_key, owner_run_id, lease_id, fencing_token,
             'legacy-wall/v1', NULL, released_at_ns
      FROM resource_lease_releases_v4;
      DROP TABLE resource_lease_releases_v4;
      UPDATE journal_meta SET value='5' WHERE key='schema_version';
    )sql";
    if (sqlite3_exec(database_, std::string(migration).c_str(), nullptr, nullptr,
                     &error_message) != SQLITE_OK) {
      const std::string message =
          error_message ? error_message : sqlite3_errmsg(database_);
      sqlite3_free(error_message);
      throw std::runtime_error("journal schema migration to version 5 failed: " +
                               message);
    }
    stored_version = "5";
  }
  if (stored_version == "5" && !migration_transaction) {
    migration_transaction = std::make_unique<Transaction>(database_);
  }
  std::map<std::string, ResourceLease> attested_leases;
  if (stored_version == "5" || stored_version == "6" ||
      stored_version == "7") {
    const std::string authority_version = stored_version;
    for (const std::string_view table :
         std::array<std::string_view, 8>{
             "journal_meta", "events", "run_projection", "compiled_plans",
             "resource_leases", "resource_lease_releases", "node_dispatches",
             "control_commands"}) {
      if (!table_exists(table)) {
        throw std::runtime_error("journal schema v" + authority_version +
                                 " is partial: missing " +
                                 std::string(table));
      }
    }
    if ((stored_version == "6" || stored_version == "7") &&
        !table_exists("resource_lease_renewals")) {
      throw std::runtime_error(
          "journal schema v6 is partial: missing resource_lease_renewals");
    }
    require_exact_schema(database_,
                         stored_version == "5" ? canonical_schema_v5()
                         : stored_version == "6" ? canonical_schema_v6()
                                                   : canonical_schema_v7(),
                         "v" + authority_version);
    require_authority_metadata(authority_version);
    require_columns(
        "resource_leases",
        {"concurrency_key", "owner_run_id", "lease_id", "fencing_token",
         "clock_domain", "boot_id", "acquired_boottime_ns",
         "expires_boottime_ns", "acquired_wall_time_ns",
         "expires_wall_time_ns", "released_wall_time_ns"},
        "v" + authority_version);
    require_columns(
        "resource_lease_releases",
        {"concurrency_key", "owner_run_id", "lease_id", "fencing_token",
         "clock_domain", "boot_id", "released_wall_time_ns"},
        "v" + authority_version);
    require_definition(
        "resource_leases",
        {"concurrency_key|TEXT|1|1", "owner_run_id|TEXT|1|0",
         "lease_id|TEXT|1|0", "fencing_token|INTEGER|1|0",
         "clock_domain|TEXT|1|0", "boot_id|TEXT|0|0",
         "acquired_boottime_ns|INTEGER|0|0",
         "expires_boottime_ns|INTEGER|0|0",
         "acquired_wall_time_ns|INTEGER|1|0",
         "expires_wall_time_ns|INTEGER|1|0",
         "released_wall_time_ns|INTEGER|0|0"},
        "v" + authority_version);
    require_definition(
        "resource_lease_releases",
        {"concurrency_key|TEXT|1|1", "owner_run_id|TEXT|1|0",
         "lease_id|TEXT|1|2", "fencing_token|INTEGER|1|3",
         "clock_domain|TEXT|1|0", "boot_id|TEXT|0|0",
         "released_wall_time_ns|INTEGER|1|0"},
        "v" + authority_version);
    require_schema_fragments(
        "resource_leases",
        {"CHECK(clock_domain IN ('boottime/v1','legacy-wall/v1'))",
         "(clock_domain='boottime/v1' AND boot_id IS NOT NULL AND",
         "(clock_domain='legacy-wall/v1' AND boot_id IS NULL AND",
         "WITHOUT ROWID"},
        "v" + authority_version);
    require_schema_fragments(
        "resource_lease_releases",
        {"CHECK(clock_domain IN ('boottime/v1','legacy-wall/v1'))",
         "(clock_domain='boottime/v1' AND boot_id IS NOT NULL)",
         "(clock_domain='legacy-wall/v1' AND boot_id IS NULL)",
         "WITHOUT ROWID"},
        "v" + authority_version);

    Statement leases(database_, R"sql(
      SELECT concurrency_key, owner_run_id, lease_id, fencing_token,
             clock_domain, boot_id, acquired_boottime_ns,
             expires_boottime_ns, acquired_wall_time_ns,
             expires_wall_time_ns, released_wall_time_ns
      FROM resource_leases
    )sql");
    int lease_status = SQLITE_ROW;
    while ((lease_status = sqlite3_step(leases.get())) == SQLITE_ROW) {
      const std::string domain = column_text(leases.get(), 4);
      const bool boot_scoped = domain == ResourceLease::kBootTimeDomain;
      const bool legacy = domain == ResourceLease::kLegacyWallDomain;
      const bool has_boot_id = sqlite3_column_type(leases.get(), 5) != SQLITE_NULL;
      const bool has_acquired_boot = sqlite3_column_type(leases.get(), 6) != SQLITE_NULL;
      const bool has_expires_boot = sqlite3_column_type(leases.get(), 7) != SQLITE_NULL;
      const bool valid_boot_scope =
          boot_scoped && has_boot_id && has_acquired_boot && has_expires_boot &&
          canonical_boot_id(column_text(leases.get(), 5)) &&
          sqlite3_column_int64(leases.get(), 6) >= 0 &&
          sqlite3_column_int64(leases.get(), 7) > sqlite3_column_int64(leases.get(), 6);
      const bool valid_legacy_scope =
          legacy && !has_boot_id && !has_acquired_boot && !has_expires_boot;
      const bool valid_release =
          sqlite3_column_type(leases.get(), 10) == SQLITE_NULL ||
          sqlite3_column_int64(leases.get(), 10) >= 0;
      if (column_text(leases.get(), 0).empty() ||
          column_text(leases.get(), 1).empty() ||
          column_text(leases.get(), 2).empty() ||
          sqlite3_column_int64(leases.get(), 3) <= 0 ||
          sqlite3_column_type(leases.get(), 8) != SQLITE_INTEGER ||
          sqlite3_column_type(leases.get(), 9) != SQLITE_INTEGER ||
          sqlite3_column_int64(leases.get(), 8) < 0 ||
          sqlite3_column_int64(leases.get(), 9) < 0 ||
          (!valid_boot_scope && !valid_legacy_scope) || !valid_release) {
        throw std::runtime_error("journal schema v5 has malformed lease authority data");
      }
      ResourceLease attested = lease_from_row(leases.get());
      if (!attested_leases.emplace(attested.concurrency_key,
                                   std::move(attested)).second) {
        throw std::runtime_error(
            "journal lease authority contains a duplicate concurrency key");
      }
    }
    if (lease_status != SQLITE_DONE) {
      throw std::runtime_error("journal schema v5 lease authority data is unreadable");
    }

    Statement releases(database_, R"sql(
      SELECT concurrency_key, owner_run_id, lease_id, fencing_token,
             clock_domain, boot_id, released_wall_time_ns
      FROM resource_lease_releases
    )sql");
    int release_status = SQLITE_ROW;
    while ((release_status = sqlite3_step(releases.get())) == SQLITE_ROW) {
      const std::string domain = column_text(releases.get(), 4);
      const bool has_boot_id = sqlite3_column_type(releases.get(), 5) != SQLITE_NULL;
      const bool valid_scope =
          (domain == ResourceLease::kBootTimeDomain && has_boot_id &&
           canonical_boot_id(column_text(releases.get(), 5))) ||
          (domain == ResourceLease::kLegacyWallDomain && !has_boot_id);
      if (column_text(releases.get(), 0).empty() ||
          column_text(releases.get(), 1).empty() ||
          column_text(releases.get(), 2).empty() ||
          sqlite3_column_int64(releases.get(), 3) <= 0 || !valid_scope ||
          sqlite3_column_type(releases.get(), 6) != SQLITE_INTEGER ||
          sqlite3_column_int64(releases.get(), 6) < 0) {
        throw std::runtime_error("journal schema v5 has malformed lease release data");
      }
    }
    if (release_status != SQLITE_DONE) {
      throw std::runtime_error("journal schema v5 lease release data is unreadable");
    }
    if (stored_version == "5") {
      std::string chain_reason;
      if (!verify_event_chain(&chain_reason)) {
        throw std::runtime_error(
            "refusing v5 journal migration with invalid event chain: " +
            chain_reason);
      }
      execute_sql(kSchemaV6Migration,
                  "journal schema migration to version 6 failed");
      stored_version = "6";
      require_exact_schema(database_, canonical_schema_v6(), "v6");
      require_authority_metadata("6");
    }
  }
  if (stored_version == "6" || stored_version == "7") {
    require_columns(
        "resource_lease_renewals",
        {"concurrency_key", "owner_run_id", "lease_id", "fencing_token",
         "clock_domain", "boot_id", "acquired_boottime_ns",
         "acquired_wall_time_ns", "prior_expires_boottime_ns",
         "new_expires_boottime_ns", "prior_expires_wall_time_ns",
         "new_expires_wall_time_ns", "renewed_boottime_ns",
         "renewed_wall_time_ns"},
        "v6");
    require_definition(
        "resource_lease_renewals",
        {"concurrency_key|TEXT|1|1", "owner_run_id|TEXT|1|0",
         "lease_id|TEXT|1|2", "fencing_token|INTEGER|1|3",
         "clock_domain|TEXT|1|0", "boot_id|TEXT|1|0",
         "acquired_boottime_ns|INTEGER|1|0",
         "acquired_wall_time_ns|INTEGER|1|0",
         "prior_expires_boottime_ns|INTEGER|1|4",
         "new_expires_boottime_ns|INTEGER|1|0",
         "prior_expires_wall_time_ns|INTEGER|1|0",
         "new_expires_wall_time_ns|INTEGER|1|0",
         "renewed_boottime_ns|INTEGER|1|0",
         "renewed_wall_time_ns|INTEGER|1|0"},
        "v6");

    struct RenewalChainState final {
      std::string owner_run_id;
      std::string clock_domain;
      std::string boot_id;
      std::int64_t acquired_boottime_ns{};
      std::int64_t acquired_wall_time_ns{};
      std::int64_t new_expires_boottime_ns{};
      std::int64_t new_expires_wall_time_ns{};
      std::int64_t renewed_boottime_ns{};
    };
    using RenewalChainKey =
        std::tuple<std::string, std::string, std::int64_t>;
    std::map<RenewalChainKey, RenewalChainState> chains;
    Statement renewals(database_, R"sql(
      SELECT concurrency_key, owner_run_id, lease_id, fencing_token,
             clock_domain, boot_id, acquired_boottime_ns,
             acquired_wall_time_ns, prior_expires_boottime_ns,
             new_expires_boottime_ns, prior_expires_wall_time_ns,
             new_expires_wall_time_ns, renewed_boottime_ns,
             renewed_wall_time_ns
      FROM resource_lease_renewals
      ORDER BY concurrency_key, lease_id, fencing_token,
               prior_expires_boottime_ns
    )sql");
    int renewal_status = SQLITE_ROW;
    while ((renewal_status = sqlite3_step(renewals.get())) == SQLITE_ROW) {
      const std::string concurrency_key = column_text(renewals.get(), 0);
      const std::string owner_run_id = column_text(renewals.get(), 1);
      const std::string lease_id = column_text(renewals.get(), 2);
      const std::string clock_domain = column_text(renewals.get(), 4);
      const std::string boot_id = column_text(renewals.get(), 5);
      const std::int64_t fencing_token = sqlite3_column_int64(renewals.get(), 3);
      const std::int64_t acquired_boot = sqlite3_column_int64(renewals.get(), 6);
      const std::int64_t acquired_wall = sqlite3_column_int64(renewals.get(), 7);
      const std::int64_t prior_boot = sqlite3_column_int64(renewals.get(), 8);
      const std::int64_t new_boot = sqlite3_column_int64(renewals.get(), 9);
      const std::int64_t prior_wall = sqlite3_column_int64(renewals.get(), 10);
      const std::int64_t new_wall = sqlite3_column_int64(renewals.get(), 11);
      const std::int64_t renewed_boot = sqlite3_column_int64(renewals.get(), 12);
      const std::int64_t renewed_wall = sqlite3_column_int64(renewals.get(), 13);
      const bool integer_fields =
          sqlite3_column_type(renewals.get(), 3) == SQLITE_INTEGER &&
          std::ranges::all_of(std::array{6, 7, 8, 9, 10, 11, 12, 13},
                              [&](int column) {
                                return sqlite3_column_type(renewals.get(), column) ==
                                       SQLITE_INTEGER;
                              });
      if (concurrency_key.empty() || owner_run_id.empty() || lease_id.empty() ||
          !integer_fields || fencing_token <= 0 ||
          clock_domain != ResourceLease::kBootTimeDomain ||
          !canonical_boot_id(boot_id) || acquired_boot < 0 ||
          acquired_wall < 0 || prior_boot <= acquired_boot ||
          new_boot <= prior_boot || renewed_boot < acquired_boot ||
          renewed_boot >= prior_boot || prior_wall < 0 || new_wall < 0 ||
          renewed_wall < 0 || new_wall < renewed_wall ||
          new_boot - renewed_boot != new_wall - renewed_wall) {
        throw std::runtime_error(
            "journal schema v6 has malformed lease renewal receipt data");
      }
      const RenewalChainKey chain_key{concurrency_key, lease_id,
                                      fencing_token};
      const auto found = chains.find(chain_key);
      if (found != chains.end() &&
          (found->second.owner_run_id != owner_run_id ||
           found->second.clock_domain != clock_domain ||
           found->second.boot_id != boot_id ||
           found->second.acquired_boottime_ns != acquired_boot ||
           found->second.acquired_wall_time_ns != acquired_wall ||
           found->second.new_expires_boottime_ns != prior_boot ||
           found->second.new_expires_wall_time_ns != prior_wall ||
           renewed_boot < found->second.renewed_boottime_ns)) {
        throw std::runtime_error(
            "journal schema v6 has a discontinuous lease renewal receipt chain");
      }
      chains[chain_key] = {.owner_run_id = owner_run_id,
                           .clock_domain = clock_domain,
                           .boot_id = boot_id,
                           .acquired_boottime_ns = acquired_boot,
                           .acquired_wall_time_ns = acquired_wall,
                           .new_expires_boottime_ns = new_boot,
                           .new_expires_wall_time_ns = new_wall,
                           .renewed_boottime_ns = renewed_boot};
    }
    if (renewal_status != SQLITE_DONE) {
      throw std::runtime_error(
          "journal schema v6 lease renewal receipt data is unreadable");
    }

    for (const auto& [key, chain] : chains) {
      const auto& [concurrency_key, lease_id, fencing_token] = key;
      const auto current = attested_leases.find(concurrency_key);
      if (current == attested_leases.end() ||
          current->second.fencing_token <
              static_cast<std::uint64_t>(fencing_token)) {
        throw std::runtime_error(
            "journal schema v6 renewal receipt has no possible current lease");
      }
      if (current->second.fencing_token ==
              static_cast<std::uint64_t>(fencing_token) &&
          (current->second.owner_run_id != chain.owner_run_id ||
           current->second.lease_id != lease_id ||
           current->second.clock_domain != chain.clock_domain ||
           current->second.boot_id != chain.boot_id ||
           current->second.acquired_boottime_ns !=
               chain.acquired_boottime_ns ||
           current->second.acquired_wall_time_ns !=
               chain.acquired_wall_time_ns ||
           current->second.expires_boottime_ns !=
               chain.new_expires_boottime_ns ||
           current->second.expires_wall_time_ns !=
               chain.new_expires_wall_time_ns)) {
        throw std::runtime_error(
            "journal schema v6 current lease disagrees with renewal receipts");
      }
    }
  }
  if (stored_version == "6") {
    if (!migration_transaction) {
      migration_transaction = std::make_unique<Transaction>(database_);
    }
    std::string chain_reason;
    if (!verify_event_chain(&chain_reason)) {
      throw std::runtime_error(
          "refusing v6 journal migration with invalid event chain: " +
          chain_reason);
    }
    execute_sql(kSchemaV7Migration,
                "journal schema migration to version 7 failed");
    stored_version = "7";
    require_exact_schema(database_, canonical_schema_v7(), "v7");
    require_authority_metadata("7");
  }
  if (stored_version == "7") {
    for (const std::string_view table :
         {"host_resource_requests", "host_resource_grants",
          "host_resource_release_intents", "host_resource_release_receipts"}) {
      if (!table_exists(table)) {
        throw std::runtime_error("journal schema v7 is partial: missing " +
                                 std::string(table));
      }
    }
  }
  if (migration_transaction) {
    migration_transaction->commit();
  }
  std::string saga_reason;
  if (!verify_event_chain(&saga_reason) ||
      !verify_host_saga_projection(database_, &saga_reason)) {
    throw std::runtime_error("journal host-grant saga is inconsistent: " +
                             saga_reason);
  }
  execute_sql(kWalPragma, "could not enable WAL for journal schema v7");
}

std::uint64_t Journal::append(const Event& event) {
  Transaction transaction(database_);
  const std::uint64_t sequence = append_uncommitted(event);
  transaction.commit();
  return sequence;
}

std::vector<std::uint64_t> Journal::append_batch(const std::vector<Event>& events) {
  if (events.empty()) {
    return {};
  }
  Transaction transaction(database_);
  std::vector<std::uint64_t> sequences;
  sequences.reserve(events.size());
  for (const auto& event : events) {
    sequences.push_back(append_uncommitted(event));
  }
  transaction.commit();
  return sequences;
}

RunCreationResult Journal::create_run(const CompiledPlan& plan,
                                      const std::vector<Event>& events) {
  const CompileResult verified = compile_document(plan.canonical_plan);
  if (!verified.valid() || !verified.plan || verified.plan->plan_hash != plan.plan_hash ||
      verified.plan->canonical_plan != plan.canonical_plan ||
      verified.plan->experiment.metadata.name != plan.experiment.metadata.name) {
    throw std::invalid_argument("run plan is not a valid canonical compiled plan");
  }
  if (events.empty() || events.front().event_type != "run.created" ||
      events.front().payload.value("plan_hash", std::string{}) != plan.plan_hash) {
    throw std::invalid_argument("run creation batch does not identify its compiled plan");
  }
  const Event& requested_creation = events.front();
  for (const Event& event : events) {
    if (event.run_id != requested_creation.run_id) {
      throw std::invalid_argument("run creation batch crosses run identities");
    }
  }

  Transaction transaction(database_);
  std::string chain_reason;
  if (!verify_chain(&chain_reason)) {
    throw std::runtime_error("refusing run creation: " + chain_reason);
  }

  std::optional<Event> stored_creation;
  {
    Statement existing_run(database_, R"sql(
      SELECT journal_sequence, event_id, run_id, run_revision, plan_revision, node_id,
             attempt_id, worker_sequence, event_type, event_version, wall_time_ns,
             monotonic_time_ns, optimizer_step, payload_json
      FROM events WHERE run_id=? AND event_type='run.created'
      ORDER BY journal_sequence
    )sql");
    bind_text(existing_run.get(), 1, requested_creation.run_id);
    const int status = sqlite3_step(existing_run.get());
    if (status == SQLITE_ROW) {
      stored_creation = event_from_row(existing_run.get());
      if (sqlite3_step(existing_run.get()) != SQLITE_DONE) {
        throw std::runtime_error("run has more than one durable creation event");
      }
    } else if (status != SQLITE_DONE) {
      throw std::runtime_error("could not read existing run creation: " +
                               std::string(sqlite3_errmsg(database_)));
    }
  }

  if (stored_creation && event_json(*stored_creation) != event_json(requested_creation)) {
    throw RunCreationConflict(
        "run already exists with a different run.created event");
  }

  Statement existing(database_, R"sql(
    SELECT experiment_name, canonical_plan_json FROM compiled_plans WHERE plan_hash=?
  )sql");
  bind_text(existing.get(), 1, plan.plan_hash);
  const int existing_status = sqlite3_step(existing.get());
  if (existing_status == SQLITE_ROW) {
    if (column_text(existing.get(), 0) != plan.experiment.metadata.name ||
        column_text(existing.get(), 1) != plan.canonical_plan.dump()) {
      throw std::invalid_argument("plan_hash already exists with different canonical content");
    }
  } else if (existing_status == SQLITE_DONE) {
    if (stored_creation) {
      throw std::runtime_error("durable run creation has no compiled plan");
    }
    Statement insert(database_, R"sql(
      INSERT INTO compiled_plans(plan_hash, experiment_name, canonical_plan_json)
      VALUES(?, ?, ?)
    )sql");
    bind_text(insert.get(), 1, plan.plan_hash);
    bind_text(insert.get(), 2, plan.experiment.metadata.name);
    bind_text(insert.get(), 3, plan.canonical_plan.dump());
    require_done(database_, insert.get(), "insert compiled plan");
  } else {
    throw std::runtime_error("could not read compiled plan: " +
                             std::string(sqlite3_errmsg(database_)));
  }

  if (stored_creation) {
    Statement projection(database_, "SELECT 1 FROM run_projection WHERE run_id=?");
    bind_text(projection.get(), 1, requested_creation.run_id);
    const int projection_status = sqlite3_step(projection.get());
    if (projection_status == SQLITE_DONE) {
      throw std::runtime_error("durable run creation has no projection");
    }
    if (projection_status != SQLITE_ROW) {
      throw std::runtime_error("could not read durable run projection: " +
                               std::string(sqlite3_errmsg(database_)));
    }
    transaction.commit();
    return {.disposition = RunCreationDisposition::replayed,
            .created_event = std::move(*stored_creation)};
  }

  {
    Statement orphaned_events(database_, "SELECT 1 FROM events WHERE run_id=? LIMIT 1");
    bind_text(orphaned_events.get(), 1, requested_creation.run_id);
    const int event_status = sqlite3_step(orphaned_events.get());
    if (event_status == SQLITE_ROW) {
      throw std::runtime_error("run has durable history without run.created");
    }
    if (event_status != SQLITE_DONE) {
      throw std::runtime_error("could not inspect existing run history: " +
                               std::string(sqlite3_errmsg(database_)));
    }
    Statement orphaned_projection(database_, "SELECT 1 FROM run_projection WHERE run_id=?");
    bind_text(orphaned_projection.get(), 1, requested_creation.run_id);
    const int projection_status = sqlite3_step(orphaned_projection.get());
    if (projection_status == SQLITE_ROW) {
      throw std::runtime_error("run projection exists without run.created");
    }
    if (projection_status != SQLITE_DONE) {
      throw std::runtime_error("could not inspect existing run projection: " +
                               std::string(sqlite3_errmsg(database_)));
    }
  }

  for (const auto& event : events) {
    append_uncommitted(event);
  }
  transaction.commit();
  return {.disposition = RunCreationDisposition::inserted,
          .created_event = requested_creation};
}

std::uint64_t Journal::append_uncommitted(const Event& event,
                                          bool allow_host_saga) {
  if (event.event_id.empty() || event.run_id.empty() || event.event_type.empty()) {
    throw std::invalid_argument("event_id, run_id, and event_type must not be empty");
  }
  if (!event.payload.is_object()) {
    throw std::invalid_argument("event payload must be an object");
  }
  if ((event.event_type.starts_with("host.resource_") ||
       event.event_type.starts_with("host.process_")) &&
      !allow_host_saga) {
    throw std::invalid_argument(
        "host resource/process events are reserved for typed saga authority");
  }
  if (event.worker_sequence > 0 && (event.node_id.empty() || event.attempt_id.empty())) {
    throw std::invalid_argument("sequenced worker events require node_id and attempt_id");
  }
  if (event.event_type == "node.entered" && (event.node_id.empty() || event.attempt_id.empty())) {
    throw std::invalid_argument("node.entered requires node_id and attempt_id");
  }
  const std::string event_content_hash = content_hash(event);
  {
    Statement duplicate(database_,
                        "SELECT journal_sequence, content_hash FROM events WHERE event_id=?");
    bind_text(duplicate.get(), 1, event.event_id);
    if (sqlite3_step(duplicate.get()) == SQLITE_ROW) {
      const auto sequence = static_cast<std::uint64_t>(sqlite3_column_int64(duplicate.get(), 0));
      if (column_text(duplicate.get(), 1) != event_content_hash) {
        throw std::invalid_argument("event_id already exists with different content");
      }
      return sequence;
    }
  }

  if (event.event_type != "run.created") {
    Statement revision(database_,
                       "SELECT run_revision FROM run_projection WHERE run_id=?");
    bind_text(revision.get(), 1, event.run_id);
    if (sqlite3_step(revision.get()) != SQLITE_ROW) {
      throw std::invalid_argument("run event precedes run.created for " + event.run_id);
    }
    const auto current_revision = static_cast<std::uint64_t>(sqlite3_column_int64(revision.get(), 0));
    if (event.run_revision < current_revision) {
      throw std::invalid_argument("run_revision must not move backward");
    }
    Statement plan_revision(database_,
                            "SELECT COALESCE(MAX(plan_revision), 0) FROM events WHERE run_id=?");
    bind_text(plan_revision.get(), 1, event.run_id);
    if (sqlite3_step(plan_revision.get()) != SQLITE_ROW) {
      throw std::runtime_error("could not read current plan revision");
    }
    const auto current_plan_revision =
        static_cast<std::uint64_t>(sqlite3_column_int64(plan_revision.get(), 0));
    if (event.plan_revision < current_plan_revision) {
      throw std::invalid_argument("plan_revision must not move backward");
    }
  }

  if (event.worker_sequence > 0) {
    Statement latest(database_, R"sql(
      SELECT COALESCE(MAX(worker_sequence), 0) FROM events
      WHERE run_id=? AND node_id=? AND attempt_id=?
    )sql");
    bind_text(latest.get(), 1, event.run_id);
    bind_text(latest.get(), 2, event.node_id);
    bind_text(latest.get(), 3, event.attempt_id);
    if (sqlite3_step(latest.get()) != SQLITE_ROW) {
      throw std::runtime_error("could not read latest worker sequence");
    }
    const auto previous_sequence = static_cast<std::uint64_t>(sqlite3_column_int64(latest.get(), 0));
    if (event.worker_sequence <= previous_sequence) {
      throw std::invalid_argument("worker_sequence must increase within an attempt");
    }
  }

  std::string previous_hash(64, '0');
  {
    Statement latest(database_, "SELECT chain_hash FROM events ORDER BY journal_sequence DESC LIMIT 1");
    if (sqlite3_step(latest.get()) == SQLITE_ROW) {
      previous_hash = column_text(latest.get(), 0);
    }
  }
  const std::string chain_hash = sha256_hex(previous_hash + ":" + event_content_hash);
  const std::string payload = event.payload.dump();
  Statement insert(database_, R"sql(
    INSERT INTO events(
      event_id, run_id, run_revision, plan_revision, node_id, attempt_id,
      worker_sequence, event_type, event_version, wall_time_ns, monotonic_time_ns,
      optimizer_step, payload_json, previous_hash, content_hash, chain_hash
    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  )sql");
  bind_text(insert.get(), 1, event.event_id);
  bind_text(insert.get(), 2, event.run_id);
  bind_integer(insert.get(), 3, checked_integer(event.run_revision, "run_revision"));
  bind_integer(insert.get(), 4, checked_integer(event.plan_revision, "plan_revision"));
  bind_text(insert.get(), 5, event.node_id);
  bind_text(insert.get(), 6, event.attempt_id);
  bind_integer(insert.get(), 7, checked_integer(event.worker_sequence, "worker_sequence"));
  bind_text(insert.get(), 8, event.event_type);
  bind_integer(insert.get(), 9, static_cast<std::int64_t>(event.event_version));
  bind_integer(insert.get(), 10, event.wall_time_ns);
  bind_integer(insert.get(), 11, checked_integer(event.monotonic_time_ns, "monotonic_time_ns"));
  if (event.optimizer_step) {
    bind_integer(insert.get(), 12, checked_integer(*event.optimizer_step, "optimizer_step"));
  } else if (sqlite3_bind_null(insert.get(), 12) != SQLITE_OK) {
    throw std::runtime_error("sqlite null bind failed");
  }
  bind_text(insert.get(), 13, payload);
  bind_text(insert.get(), 14, previous_hash);
  bind_text(insert.get(), 15, event_content_hash);
  bind_text(insert.get(), 16, chain_hash);
  require_done(database_, insert.get(), "append event");
  const auto journal_sequence = static_cast<std::uint64_t>(sqlite3_last_insert_rowid(database_));
  update_projection(database_, event, journal_sequence);
  {
    Statement update_head(database_, "UPDATE journal_meta SET value=? WHERE key='chain_head'");
    bind_text(update_head.get(), 1, chain_hash);
    require_done(database_, update_head.get(), "update journal chain head");
    if (sqlite3_changes(database_) != 1) {
      throw std::runtime_error("journal chain head is missing");
    }
  }
  return journal_sequence;
}

std::pair<std::string, std::uint64_t>
Journal::append_authority_event_uncommitted(const Event& event) {
  if (event.event_id.empty() || event.run_id.empty() ||
      !event.event_type.starts_with("authority.") ||
      event.run_revision != 0U || event.plan_revision != 0U ||
      !event.node_id.empty() || !event.attempt_id.empty() ||
      event.worker_sequence != 0U || event.event_version != 1U ||
      event.wall_time_ns < 0 || !event.payload.is_object()) {
    throw std::invalid_argument("journal authority event is noncanonical");
  }
  const std::string event_content_hash = content_hash(event);
  {
    Statement duplicate(
        database_,
        "SELECT content_hash, journal_sequence FROM events WHERE event_id=?");
    bind_text(duplicate.get(), 1, event.event_id);
    if (sqlite3_step(duplicate.get()) == SQLITE_ROW) {
      if (column_text(duplicate.get(), 0) != event_content_hash)
        throw std::runtime_error(
            "journal authority event identity has conflicting content");
      return {event_content_hash, static_cast<std::uint64_t>(
                                      sqlite3_column_int64(duplicate.get(), 1))};
    }
  }
  std::string previous_hash(64U, '0');
  {
    Statement latest(
        database_,
        "SELECT chain_hash FROM events ORDER BY journal_sequence DESC LIMIT 1");
    if (sqlite3_step(latest.get()) == SQLITE_ROW)
      previous_hash = column_text(latest.get(), 0);
  }
  const std::string chain_hash =
      sha256_hex(previous_hash + ":" + event_content_hash);
  Statement insert(database_, R"sql(
    INSERT INTO events(
      event_id, run_id, run_revision, plan_revision, node_id, attempt_id,
      worker_sequence, event_type, event_version, wall_time_ns,
      monotonic_time_ns, optimizer_step, payload_json, previous_hash,
      content_hash, chain_hash
    ) VALUES(?, ?, 0, 0, '', '', 0, ?, 1, ?, ?, NULL, ?, ?, ?, ?)
  )sql");
  bind_text(insert.get(), 1, event.event_id);
  bind_text(insert.get(), 2, event.run_id);
  bind_text(insert.get(), 3, event.event_type);
  bind_integer(insert.get(), 4, event.wall_time_ns);
  bind_integer(insert.get(), 5,
               checked_integer(event.monotonic_time_ns, "monotonic_time_ns"));
  bind_text(insert.get(), 6, event.payload.dump());
  bind_text(insert.get(), 7, previous_hash);
  bind_text(insert.get(), 8, event_content_hash);
  bind_text(insert.get(), 9, chain_hash);
  require_done(database_, insert.get(), "append journal authority event");
  const std::uint64_t journal_sequence = static_cast<std::uint64_t>(
      sqlite3_last_insert_rowid(database_));
  Statement update_head(database_,
                        "UPDATE journal_meta SET value=? WHERE key='chain_head'");
  bind_text(update_head.get(), 1, chain_hash);
  require_done(database_, update_head.get(), "update journal authority chain head");
  if (sqlite3_changes(database_) != 1)
    throw std::runtime_error("journal authority chain head is missing");
  return {event_content_hash, journal_sequence};
}

void Journal::record_lease_authority_acquisition_uncommitted(
    const ResourceLease& lease) {
  const LeaseAuthorityHead initial{
      .concurrency_key = lease.concurrency_key,
      .owner_run_id = lease.owner_run_id,
      .lease_id = lease.lease_id,
      .fencing_token = lease.fencing_token,
      .authority_revision = 0U,
      .head_event_sequence = 0U,
      .head_event_hash = {},
      .released = false};
  Event event{.event_id = lease_authority_event_id(
                  lease.concurrency_key, lease.fencing_token, 0U),
              .run_id = lease.owner_run_id,
              .run_revision = 0U,
              .plan_revision = 0U,
              .node_id = {},
              .attempt_id = {},
              .worker_sequence = 0U,
              .event_type = "authority.resource_lease_acquired",
              .event_version = 1U,
              .wall_time_ns = lease.acquired_wall_time_ns,
              .monotonic_time_ns = static_cast<std::uint64_t>(
                  lease.acquired_boottime_ns),
              .optimizer_step = std::nullopt,
              .payload = {{"acquired_boottime_ns",
                           lease.acquired_boottime_ns},
                          {"acquired_wall_time_ns", lease.acquired_wall_time_ns},
                          {"authority_revision", 0U},
                          {"boot_id", lease.boot_id},
                          {"clock_domain", lease.clock_domain},
                          {"concurrency_key", lease.concurrency_key},
                          {"expires_boottime_ns", lease.expires_boottime_ns},
                          {"expires_wall_time_ns", lease.expires_wall_time_ns},
                          {"fencing_token", lease.fencing_token},
                          {"lease_id", lease.lease_id},
                          {"operation", "acquired"},
                          {"owner_run_id", lease.owner_run_id},
                          {"previous_authority_hash", std::string(64U, '0')}}};
  LeaseAuthorityHead stored = initial;
  std::tie(stored.head_event_hash, stored.head_event_sequence) =
      append_authority_event_uncommitted(event);
  Statement insert(database_,
                   "INSERT INTO journal_meta(key, value) VALUES(?, ?)");
  bind_text(insert.get(), 1, lease_authority_metadata_key(
                                 lease.concurrency_key, lease.fencing_token));
  bind_text(insert.get(), 2, lease_authority_head_json(stored).dump());
  require_done(database_, insert.get(), "record lease acquisition authority head");
}

void Journal::record_lease_authority_renewal_uncommitted(
    const LeaseRenewalReceipt& renewal) {
  const std::string key = lease_authority_metadata_key(
      renewal.concurrency_key, renewal.fencing_token);
  Statement query(database_, "SELECT value FROM journal_meta WHERE key=?");
  bind_text(query.get(), 1, key);
  if (sqlite3_step(query.get()) != SQLITE_ROW)
    throw std::runtime_error("lease renewal has no acquisition authority root");
  const std::string prior_encoded = column_text(query.get(), 0);
  LeaseAuthorityHead head = parse_lease_authority_head(prior_encoded);
  if (head.concurrency_key != renewal.concurrency_key ||
      head.owner_run_id != renewal.owner_run_id ||
      head.lease_id != renewal.lease_id ||
      head.fencing_token != renewal.fencing_token || head.released ||
      head.authority_revision ==
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::runtime_error("lease renewal authority head is inexact");
  }
  ++head.authority_revision;
  Event event{.event_id = lease_authority_event_id(
                  renewal.concurrency_key, renewal.fencing_token,
                  head.authority_revision),
              .run_id = renewal.owner_run_id,
              .run_revision = 0U,
              .plan_revision = 0U,
              .node_id = {},
              .attempt_id = {},
              .worker_sequence = 0U,
              .event_type = "authority.resource_lease_renewed",
              .event_version = 1U,
              .wall_time_ns = renewal.renewed_wall_time_ns,
              .monotonic_time_ns = static_cast<std::uint64_t>(
                  renewal.renewed_boottime_ns),
              .optimizer_step = std::nullopt,
              .payload = {{"acquired_boottime_ns",
                           renewal.acquired_boottime_ns},
                          {"acquired_wall_time_ns", renewal.acquired_wall_time_ns},
                          {"authority_revision", head.authority_revision},
                          {"boot_id", renewal.boot_id},
                          {"clock_domain", renewal.clock_domain},
                          {"concurrency_key", renewal.concurrency_key},
                          {"fencing_token", renewal.fencing_token},
                          {"lease_id", renewal.lease_id},
                          {"new_expires_boottime_ns",
                           renewal.new_expires_boottime_ns},
                          {"new_expires_wall_time_ns",
                           renewal.new_expires_wall_time_ns},
                          {"operation", "renewed"},
                          {"owner_run_id", renewal.owner_run_id},
                          {"previous_authority_hash", head.head_event_hash},
                          {"prior_expires_boottime_ns",
                           renewal.prior_expires_boottime_ns},
                          {"prior_expires_wall_time_ns",
                           renewal.prior_expires_wall_time_ns},
                          {"renewed_boottime_ns", renewal.renewed_boottime_ns},
                          {"renewed_wall_time_ns", renewal.renewed_wall_time_ns}}};
  std::tie(head.head_event_hash, head.head_event_sequence) =
      append_authority_event_uncommitted(event);
  const std::string next_encoded = lease_authority_head_json(head).dump();
  Statement update(database_,
                   "UPDATE journal_meta SET value=? WHERE key=? AND value=?");
  bind_text(update.get(), 1, next_encoded);
  bind_text(update.get(), 2, key);
  bind_text(update.get(), 3, prior_encoded);
  require_done(database_, update.get(), "advance lease renewal authority head");
  if (sqlite3_changes(database_) != 1)
    throw std::runtime_error("lease renewal authority head changed concurrently");
}

void Journal::record_lease_authority_release_uncommitted(
    const ResourceLease& lease, std::int64_t released_wall_time_ns) {
  const std::string key = lease_authority_metadata_key(
      lease.concurrency_key, lease.fencing_token);
  Statement query(database_, "SELECT value FROM journal_meta WHERE key=?");
  bind_text(query.get(), 1, key);
  if (sqlite3_step(query.get()) != SQLITE_ROW)
    throw std::runtime_error("lease release has no acquisition authority root");
  const std::string prior_encoded = column_text(query.get(), 0);
  LeaseAuthorityHead head = parse_lease_authority_head(prior_encoded);
  if (head.concurrency_key != lease.concurrency_key ||
      head.owner_run_id != lease.owner_run_id ||
      head.lease_id != lease.lease_id ||
      head.fencing_token != lease.fencing_token || head.released ||
      head.authority_revision ==
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::runtime_error("lease release authority head is inexact");
  }
  ++head.authority_revision;
  Event event{.event_id = lease_authority_event_id(
                  lease.concurrency_key, lease.fencing_token,
                  head.authority_revision),
              .run_id = lease.owner_run_id,
              .run_revision = 0U,
              .plan_revision = 0U,
              .node_id = {},
              .attempt_id = {},
              .worker_sequence = 0U,
              .event_type = "authority.resource_lease_released",
              .event_version = 1U,
              .wall_time_ns = released_wall_time_ns,
              .monotonic_time_ns = 0U,
              .optimizer_step = std::nullopt,
              .payload = {{"authority_revision", head.authority_revision},
                          {"boot_id", lease.boot_id},
                          {"clock_domain", lease.clock_domain},
                          {"concurrency_key", lease.concurrency_key},
                          {"fencing_token", lease.fencing_token},
                          {"lease_id", lease.lease_id},
                          {"operation", "released"},
                          {"owner_run_id", lease.owner_run_id},
                          {"previous_authority_hash", head.head_event_hash},
                          {"released_wall_time_ns", released_wall_time_ns}}};
  std::tie(head.head_event_hash, head.head_event_sequence) =
      append_authority_event_uncommitted(event);
  head.released = true;
  Statement update(database_,
                   "UPDATE journal_meta SET value=? WHERE key=? AND value=?");
  bind_text(update.get(), 1, lease_authority_head_json(head).dump());
  bind_text(update.get(), 2, key);
  bind_text(update.get(), 3, prior_encoded);
  require_done(database_, update.get(), "advance lease release authority head");
  if (sqlite3_changes(database_) != 1)
    throw std::runtime_error("lease release authority head changed concurrently");
}

std::optional<Event> Journal::event(const std::string& event_id) const {
  Statement query(database_, R"sql(
    SELECT journal_sequence, event_id, run_id, run_revision, plan_revision, node_id,
           attempt_id, worker_sequence, event_type, event_version, wall_time_ns,
           monotonic_time_ns, optimizer_step, payload_json
    FROM events WHERE event_id=?
  )sql");
  bind_text(query.get(), 1, event_id);
  const int status = sqlite3_step(query.get());
  if (status == SQLITE_DONE) {
    return std::nullopt;
  }
  if (status != SQLITE_ROW) {
    throw std::runtime_error("could not read journal event: " + std::string(sqlite3_errmsg(database_)));
  }
  return event_from_row(query.get());
}

std::vector<Event> Journal::events_for_run(const std::string& run_id) const {
  Statement query(database_, R"sql(
    SELECT journal_sequence, event_id, run_id, run_revision, plan_revision, node_id,
           attempt_id, worker_sequence, event_type, event_version, wall_time_ns,
           monotonic_time_ns, optimizer_step, payload_json
    FROM events
    WHERE run_id=? AND event_type NOT LIKE 'authority.%'
    ORDER BY journal_sequence
  )sql");
  bind_text(query.get(), 1, run_id);
  std::vector<Event> events;
  int status = SQLITE_OK;
  while ((status = sqlite3_step(query.get())) == SQLITE_ROW) {
    events.push_back(event_from_row(query.get()));
  }
  if (status != SQLITE_DONE) {
    throw std::runtime_error("could not scan run events: " + std::string(sqlite3_errmsg(database_)));
  }
  return events;
}

std::uint64_t Journal::latest_worker_sequence(
    const std::string& run_id, const std::string& node_id,
    const std::string& attempt_id) const {
  if (run_id.empty() || node_id.empty() || attempt_id.empty()) {
    throw std::invalid_argument(
        "worker sequence lookup requires run, node, and attempt identity");
  }
  Statement query(database_, R"sql(
    SELECT COALESCE(MAX(worker_sequence), 0) FROM events
    WHERE run_id=? AND node_id=? AND attempt_id=?
  )sql");
  bind_text(query.get(), 1, run_id);
  bind_text(query.get(), 2, node_id);
  bind_text(query.get(), 3, attempt_id);
  if (sqlite3_step(query.get()) != SQLITE_ROW) {
    throw std::runtime_error("could not read latest worker sequence");
  }
  return static_cast<std::uint64_t>(sqlite3_column_int64(query.get(), 0));
}

std::optional<RunProjection> Journal::projection(const std::string& run_id) const {
  Statement query(database_, R"sql(
    SELECT run_id, experiment_name, plan_hash, desired_state, observed_state,
           current_node_id, current_attempt_id, run_revision, optimizer_step,
           last_heartbeat_ns, last_event_sequence, failure_summary
    FROM run_projection WHERE run_id=?
  )sql");
  bind_text(query.get(), 1, run_id);
  if (sqlite3_step(query.get()) != SQLITE_ROW) {
    return std::nullopt;
  }
  return run_projection_from_row(query.get());
}

std::vector<RunProjection> Journal::reconcilable_projections(
    std::string_view after_run_id, std::size_t limit) const {
  constexpr std::size_t kMaximumPageSize = 1'024U;
  if (limit == 0U || limit > kMaximumPageSize ||
      after_run_id.size() > 256U) {
    throw std::invalid_argument(
        "reconcilable projection page is outside its bounds");
  }
  Statement query(database_, R"sql(
    SELECT run_id, experiment_name, plan_hash, desired_state, observed_state,
           current_node_id, current_attempt_id, run_revision, optimizer_step,
           last_heartbeat_ns, last_event_sequence, failure_summary
    FROM run_projection
    WHERE run_id > ?
      AND desired_state IN ('queued', 'running')
      AND observed_state IN ('queued', 'acquiring', 'running')
    ORDER BY run_id
    LIMIT ?
  )sql");
  bind_text(query.get(), 1, std::string(after_run_id));
  bind_integer(query.get(), 2, static_cast<std::int64_t>(limit));
  std::vector<RunProjection> result;
  result.reserve(limit);
  int status = SQLITE_ROW;
  while ((status = sqlite3_step(query.get())) == SQLITE_ROW) {
    result.push_back(run_projection_from_row(query.get()));
  }
  if (status != SQLITE_DONE) {
    throw std::runtime_error("could not scan reconcilable run projections");
  }
  return result;
}

std::vector<RunProjection> Journal::run_projections(
    const RunProjectionQuery& input) const {
  constexpr std::size_t kMaximumPageSize = 1'001U;
  constexpr std::size_t kMaximumFilters = 64U;
  static const std::set<std::string, std::less<>> kObservedStates{
      "draft",      "validated", "queued",     "acquiring",
      "running",    "pausing",   "paused",     "recovering",
      "completing", "completed", "cancelling", "cancelled",
      "failing",    "failed",    "blocked"};
  if (input.limit == 0U || input.limit > kMaximumPageSize ||
      input.observed_states.size() > kMaximumFilters ||
      input.labels.size() > kMaximumFilters ||
      (input.after &&
       (input.after->last_event_sequence == 0U ||
        input.after->run_id.empty() || input.after->run_id.size() > 256U))) {
    throw std::invalid_argument("run projection query exceeds its bounds");
  }
  for (const std::string& state : input.observed_states) {
    if (!kObservedStates.contains(state)) {
      throw std::invalid_argument(
          "run projection query has an unknown observed state");
    }
  }
  for (const auto& [key, value] : input.labels) {
    if (key.empty() || key.size() > 128U || value.size() > 512U) {
      throw std::invalid_argument(
          "run projection label filter exceeds its bounds");
    }
  }

  std::string sql = R"sql(
    SELECT projection.run_id, projection.experiment_name,
           projection.plan_hash, projection.desired_state,
           projection.observed_state, projection.current_node_id,
           projection.current_attempt_id, projection.run_revision,
           projection.optimizer_step, projection.last_heartbeat_ns,
           projection.last_event_sequence, projection.failure_summary
    FROM run_projection AS projection
    JOIN compiled_plans AS plan ON plan.plan_hash=projection.plan_hash
    WHERE 1=1
  )sql";
  if (input.after) {
    sql += R"sql(
      AND (projection.last_event_sequence < ? OR
           (projection.last_event_sequence = ? AND projection.run_id > ?))
    )sql";
  }
  if (!input.observed_states.empty()) {
    sql += " AND projection.observed_state IN (";
    for (std::size_t index = 0U; index < input.observed_states.size(); ++index) {
      if (index != 0U) sql += ',';
      sql += '?';
    }
    sql += ')';
  }
  std::size_t label_index = 0U;
  for (const auto& [key, value] : input.labels) {
    (void)key;
    (void)value;
    sql += " AND EXISTS (SELECT 1 FROM json_each(";
    sql += "plan.canonical_plan_json, '$.metadata.labels') AS label_";
    sql += std::to_string(label_index++);
    sql += " WHERE label_" + std::to_string(label_index - 1U) +
           ".key=? AND label_" + std::to_string(label_index - 1U) +
           ".value=? AND label_" + std::to_string(label_index - 1U) +
           ".type='text')";
  }
  sql += R"sql(
    ORDER BY projection.last_event_sequence DESC, projection.run_id
    LIMIT ?
  )sql";
  Statement query(database_, sql);
  int parameter = 1;
  if (input.after) {
    bind_integer(query.get(), parameter++,
                 checked_integer(input.after->last_event_sequence,
                                 "run page sequence"));
    bind_integer(query.get(), parameter++,
                 checked_integer(input.after->last_event_sequence,
                                 "run page sequence"));
    bind_text(query.get(), parameter++, input.after->run_id);
  }
  for (const std::string& state : input.observed_states) {
    bind_text(query.get(), parameter++, state);
  }
  for (const auto& [key, value] : input.labels) {
    bind_text(query.get(), parameter++, key);
    bind_text(query.get(), parameter++, value);
  }
  bind_integer(query.get(), parameter,
               static_cast<std::int64_t>(input.limit));
  std::vector<RunProjection> result;
  result.reserve(input.limit);
  int status = SQLITE_ROW;
  while ((status = sqlite3_step(query.get())) == SQLITE_ROW) {
    result.push_back(run_projection_from_row(query.get()));
  }
  if (status != SQLITE_DONE) {
    throw std::runtime_error("could not list run projections");
  }
  return result;
}

std::vector<SequencedEvent> Journal::sequenced_events(
    const EventScanQuery& input) const {
  constexpr std::size_t kMaximumPageSize = 1'024U;
  constexpr std::size_t kMaximumFilters = 64U;
  if (input.limit == 0U || input.limit > kMaximumPageSize ||
      input.run_ids.size() > kMaximumFilters ||
      input.event_types.size() > kMaximumFilters) {
    throw std::invalid_argument("event scan query exceeds its bounds");
  }
  const auto invalid = [](const std::string& value) {
    return value.empty() || value.size() > 256U;
  };
  if (std::ranges::any_of(input.run_ids, invalid) ||
      std::ranges::any_of(input.event_types, invalid)) {
    throw std::invalid_argument("event scan filter is malformed");
  }
  std::string sql = R"sql(
    SELECT journal_sequence, event_id, run_id, run_revision, plan_revision,
           node_id, attempt_id, worker_sequence, event_type, event_version,
           wall_time_ns, monotonic_time_ns, optimizer_step, payload_json
    FROM events WHERE journal_sequence>?
  )sql";
  const auto append_filter = [&sql](std::string_view column,
                                    std::size_t count) {
    if (count == 0U) return;
    sql += " AND ";
    sql += column;
    sql += " IN (";
    for (std::size_t index = 0U; index < count; ++index) {
      if (index != 0U) sql += ',';
      sql += '?';
    }
    sql += ')';
  };
  append_filter("run_id", input.run_ids.size());
  append_filter("event_type", input.event_types.size());
  sql += " ORDER BY journal_sequence LIMIT ?";
  Statement query(database_, sql);
  int parameter = 1;
  bind_integer(query.get(), parameter++,
               checked_integer(input.after_journal_sequence,
                               "event scan sequence"));
  for (const std::string& run_id : input.run_ids) {
    bind_text(query.get(), parameter++, run_id);
  }
  for (const std::string& event_type : input.event_types) {
    bind_text(query.get(), parameter++, event_type);
  }
  bind_integer(query.get(), parameter,
               static_cast<std::int64_t>(input.limit));
  std::vector<SequencedEvent> result;
  result.reserve(input.limit);
  int status = SQLITE_ROW;
  while ((status = sqlite3_step(query.get())) == SQLITE_ROW) {
    result.push_back({
        .journal_sequence = static_cast<std::uint64_t>(
            sqlite3_column_int64(query.get(), 0)),
        .event = event_from_row(query.get()),
    });
  }
  if (status != SQLITE_DONE) {
    throw std::runtime_error("could not scan sequenced journal events");
  }
  return result;
}

std::optional<RunWallTimeBounds> Journal::run_wall_time_bounds(
    const std::string& run_id) const {
  if (run_id.empty() || run_id.size() > 256U) {
    throw std::invalid_argument("run wall-time query requires a bounded run ID");
  }
  Statement query(database_, R"sql(
    SELECT MIN(wall_time_ns), MAX(wall_time_ns)
    FROM events
    WHERE run_id=? AND event_type NOT LIKE 'authority.%'
  )sql");
  bind_text(query.get(), 1, run_id);
  if (sqlite3_step(query.get()) != SQLITE_ROW) {
    throw std::runtime_error("could not read run wall-time bounds");
  }
  if (sqlite3_column_type(query.get(), 0) == SQLITE_NULL ||
      sqlite3_column_type(query.get(), 1) == SQLITE_NULL) {
    return std::nullopt;
  }
  const RunWallTimeBounds result{
      .created_wall_time_ns = sqlite3_column_int64(query.get(), 0),
      .updated_wall_time_ns = sqlite3_column_int64(query.get(), 1),
  };
  if (result.created_wall_time_ns < 0 ||
      result.updated_wall_time_ns < result.created_wall_time_ns ||
      sqlite3_step(query.get()) != SQLITE_DONE) {
    throw std::runtime_error("run wall-time bounds are malformed");
  }
  return result;
}

std::optional<CompiledPlan> Journal::compiled_plan(const std::string& plan_hash) const {
  if (plan_hash.empty()) {
    throw std::invalid_argument("compiled plan hash must not be empty");
  }
  Statement query(database_, R"sql(
    SELECT experiment_name, canonical_plan_json FROM compiled_plans WHERE plan_hash=?
  )sql");
  bind_text(query.get(), 1, plan_hash);
  const int status = sqlite3_step(query.get());
  if (status == SQLITE_DONE) {
    return std::nullopt;
  }
  if (status != SQLITE_ROW) {
    throw std::runtime_error("could not read compiled plan: " +
                             std::string(sqlite3_errmsg(database_)));
  }

  const std::string experiment_name = column_text(query.get(), 0);
  const std::string canonical_text = column_text(query.get(), 1);
  try {
    const nlohmann::json canonical = nlohmann::json::parse(canonical_text);
    CompileResult compiled = compile_document(canonical);
    if (!compiled.valid() || !compiled.plan || compiled.plan->plan_hash != plan_hash ||
        compiled.plan->canonical_plan != canonical ||
        compiled.plan->experiment.metadata.name != experiment_name) {
      throw std::runtime_error("persisted compiled plan failed content-address verification");
    }
    return std::move(compiled.plan);
  } catch (const nlohmann::json::exception& exception) {
    throw std::runtime_error("persisted compiled plan is invalid JSON: " +
                             std::string(exception.what()));
  }
}

Dispatch Journal::prepare_dispatch(const Dispatch& dispatch, const Event& prepared_event) {
  return prepare_dispatch_impl(dispatch, prepared_event, std::nullopt, std::nullopt);
}

Dispatch Journal::prepare_fenced_dispatch(const Dispatch& dispatch,
                                          const Event& prepared_event,
                                          const WorkerLaunchTicket& launch,
                                          const AuthorityTimeSample& now) {
  return prepare_dispatch_impl(dispatch, prepared_event, launch, now);
}

Dispatch Journal::prepare_dispatch_impl(
    const Dispatch& dispatch, const Event& prepared_event,
    const std::optional<WorkerLaunchTicket>& launch,
    std::optional<AuthorityTimeSample> now) {
  if (dispatch.dispatch_id.empty() || dispatch.run_id.empty() || dispatch.node_id.empty() ||
      dispatch.attempt_id.empty() || dispatch.component.empty() || dispatch.operation.empty()) {
    throw std::invalid_argument("dispatch identity and operation fields must not be empty");
  }
  if (dispatch.status != DispatchStatus::prepared || dispatch.result_event_id) {
    throw std::invalid_argument("a new dispatch must be prepared without a result event");
  }
  if (prepared_event.event_type != "node.dispatch_prepared" ||
      prepared_event.run_id != dispatch.run_id || prepared_event.node_id != dispatch.node_id ||
      prepared_event.attempt_id != dispatch.attempt_id ||
      prepared_event.run_revision != dispatch.run_revision ||
      prepared_event.plan_revision != dispatch.plan_revision) {
    throw std::invalid_argument("dispatch preparation event does not match its dispatch");
  }
  const auto prepared_dispatch_id = prepared_event.payload.find("dispatch_id");
  if (prepared_dispatch_id == prepared_event.payload.end() || !prepared_dispatch_id->is_string() ||
      prepared_dispatch_id->get<std::string>() != dispatch.dispatch_id) {
    throw std::invalid_argument("dispatch preparation event has the wrong dispatch_id payload");
  }
  if (now) {
    require_authority_time(*now);
  }

  Transaction transaction(database_);
  if (launch) {
    if (!now || launch->run_id != dispatch.run_id ||
        launch->node_id != dispatch.node_id ||
        launch->attempt_id != dispatch.attempt_id || launch->fencing_token == 0) {
      throw std::invalid_argument("fenced dispatch has an invalid launch identity");
    }
    Statement lease(database_, R"sql(
      SELECT owner_run_id, lease_id, fencing_token, clock_domain, boot_id,
             expires_boottime_ns, released_wall_time_ns
      FROM resource_leases WHERE concurrency_key=?
        AND NOT EXISTS(
          SELECT 1 FROM resource_lease_releases AS release
          WHERE release.concurrency_key=resource_leases.concurrency_key
            AND release.owner_run_id=resource_leases.owner_run_id
            AND release.lease_id=resource_leases.lease_id
            AND release.fencing_token=resource_leases.fencing_token
        )
    )sql");
    bind_text(lease.get(), 1, launch->concurrency_key);
    if (sqlite3_step(lease.get()) != SQLITE_ROW ||
        column_text(lease.get(), 0) != launch->run_id ||
        column_text(lease.get(), 1) != launch->lease_id ||
        static_cast<std::uint64_t>(sqlite3_column_int64(lease.get(), 2)) !=
            launch->fencing_token ||
        column_text(lease.get(), 3) != ResourceLease::kBootTimeDomain ||
        column_text(lease.get(), 4) != now->boot_id ||
        sqlite3_column_int64(lease.get(), 5) <= now->boot.nanoseconds ||
        sqlite3_column_type(lease.get(), 6) != SQLITE_NULL) {
      throw OperationPreconditionError(
          "dispatch no longer owns its active worker lease");
    }
    require_live_host_grant_claim(
        launch->host_grant, launch->run_id, launch->concurrency_key,
        launch->lease_id, launch->fencing_token);
    const auto ready = event(dispatch.run_id + ":worker-launch:" +
                             dispatch.node_id + ":" + dispatch.attempt_id +
                             ":ready");
    if (!ready || ready->event_type != "worker.ready" ||
        ready->payload.value("launch_nonce", std::string{}) !=
            launch->launch_nonce ||
        ready->payload.value("lease_id", std::string{}) != launch->lease_id ||
        ready->payload.value("fencing_token", std::uint64_t{}) !=
            launch->fencing_token) {
      throw std::runtime_error("dispatch has no matching durable worker readiness");
    }
  }
  Statement projection(database_, R"sql(
    SELECT observed_state, run_revision, current_node_id, current_attempt_id
    FROM run_projection WHERE run_id=?
  )sql");
  bind_text(projection.get(), 1, dispatch.run_id);
  if (sqlite3_step(projection.get()) != SQLITE_ROW ||
      column_text(projection.get(), 0) != "running" ||
      static_cast<std::uint64_t>(sqlite3_column_int64(projection.get(), 1)) !=
          dispatch.run_revision ||
      column_text(projection.get(), 2) != dispatch.node_id ||
      column_text(projection.get(), 3) != dispatch.attempt_id) {
    throw OperationPreconditionError(
        "dispatch preparation is stale for the active run projection");
  }
  Statement existing(database_, R"sql(
    SELECT dispatch_id, run_id, run_revision, plan_revision, node_id, attempt_id,
           component, operation, status, result_event_id
    FROM node_dispatches WHERE dispatch_id=?
  )sql");
  bind_text(existing.get(), 1, dispatch.dispatch_id);
  const int existing_status = sqlite3_step(existing.get());
  if (existing_status == SQLITE_ROW) {
    Dispatch stored = dispatch_from_row(existing.get());
    if (!same_dispatch_attempt(stored, dispatch)) {
      throw std::invalid_argument("dispatch_id already exists with different content");
    }
    // Pause/resume advances the run revision without creating a new node
    // attempt. Reissue the still-prepared dispatch at the current revision;
    // its durable attempt identity and original preparation receipt remain
    // unchanged.
    if (stored.status == DispatchStatus::prepared) {
      stored.run_revision = dispatch.run_revision;
      Statement refresh(database_, R"sql(
        UPDATE node_dispatches SET run_revision=? WHERE dispatch_id=?
      )sql");
      bind_integer(refresh.get(), 1,
                   checked_integer(dispatch.run_revision, "run_revision"));
      bind_text(refresh.get(), 2, dispatch.dispatch_id);
      require_done(database_, refresh.get(), "refresh prepared dispatch revision");
      if (sqlite3_changes(database_) != 1) {
        throw std::runtime_error("dispatch revision refresh affected an unexpected row count");
      }
    }
    transaction.commit();
    return stored;
  }
  if (existing_status != SQLITE_DONE) {
    throw std::runtime_error("could not read dispatch: " + std::string(sqlite3_errmsg(database_)));
  }
  {
    Statement attempt(database_, R"sql(
      SELECT dispatch_id FROM node_dispatches WHERE run_id=? AND node_id=? AND attempt_id=?
    )sql");
    bind_text(attempt.get(), 1, dispatch.run_id);
    bind_text(attempt.get(), 2, dispatch.node_id);
    bind_text(attempt.get(), 3, dispatch.attempt_id);
    if (sqlite3_step(attempt.get()) == SQLITE_ROW) {
      throw std::invalid_argument("node attempt already has a different dispatch");
    }
  }

  append_uncommitted(prepared_event);
  Statement insert(database_, R"sql(
    INSERT INTO node_dispatches(
      dispatch_id, run_id, run_revision, plan_revision, node_id, attempt_id,
      component, operation, status, result_event_id
    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, 'prepared', NULL)
  )sql");
  bind_text(insert.get(), 1, dispatch.dispatch_id);
  bind_text(insert.get(), 2, dispatch.run_id);
  bind_integer(insert.get(), 3, checked_integer(dispatch.run_revision, "run_revision"));
  bind_integer(insert.get(), 4, checked_integer(dispatch.plan_revision, "plan_revision"));
  bind_text(insert.get(), 5, dispatch.node_id);
  bind_text(insert.get(), 6, dispatch.attempt_id);
  bind_text(insert.get(), 7, dispatch.component);
  bind_text(insert.get(), 8, dispatch.operation);
  require_done(database_, insert.get(), "insert node dispatch");
  transaction.commit();
  return dispatch;
}

void Journal::complete_dispatch(const std::string& dispatch_id, const std::string& result_event_id,
                                const std::vector<Event>& events) {
  complete_dispatch_impl(dispatch_id, result_event_id, events, std::nullopt,
                         std::nullopt);
}

void Journal::complete_fenced_dispatch(
    const std::string& dispatch_id, const std::string& result_event_id,
    const std::vector<Event>& events, const WorkerSessionIdentity& identity,
    const AuthorityTimeSample& now) {
  complete_dispatch_impl(dispatch_id, result_event_id, events, identity, now);
}

void Journal::append_fenced_worker_observation(
    const Event& observation, const WorkerSessionIdentity& identity,
    const AuthorityTimeSample& now) {
  require_authority_time(now);
  if (observation.run_id != identity.run_id ||
      observation.node_id != identity.node_id ||
      observation.attempt_id != identity.attempt_id ||
      observation.worker_sequence == 0U || identity.launch_nonce.empty() ||
      identity.concurrency_key.empty() || identity.lease_id.empty() ||
      identity.fencing_token == 0U) {
    throw std::invalid_argument(
        "worker observation has an invalid fenced session identity");
  }

  Transaction transaction(database_);
  const std::string launch_id = identity.run_id + ":worker-launch:" +
                                identity.node_id + ":" + identity.attempt_id;
  const auto binding = launch_binding(launch_id);
  if (!binding || binding->identity.run_id != identity.run_id ||
      binding->identity.node_id != identity.node_id ||
      binding->identity.attempt_id != identity.attempt_id ||
      binding->identity.launch_nonce != identity.launch_nonce ||
      binding->identity.concurrency_key != identity.concurrency_key ||
      binding->identity.lease_id != identity.lease_id ||
      binding->identity.fencing_token != identity.fencing_token) {
    throw OperationPreconditionError(
        "worker observation has no exact durable launch binding");
  }
  require_live_host_grant_claim(
      binding->identity.host_grant, identity.run_id,
      identity.concurrency_key, identity.lease_id, identity.fencing_token);

  Statement projection(database_, R"sql(
    SELECT observed_state, run_revision, current_node_id, current_attempt_id
    FROM run_projection WHERE run_id=?
  )sql");
  bind_text(projection.get(), 1, identity.run_id);
  if (sqlite3_step(projection.get()) != SQLITE_ROW ||
      column_text(projection.get(), 0) != "running" ||
      static_cast<std::uint64_t>(sqlite3_column_int64(projection.get(), 1)) !=
          observation.run_revision ||
      column_text(projection.get(), 2) != identity.node_id ||
      column_text(projection.get(), 3) != identity.attempt_id) {
    throw OperationPreconditionError(
        "worker observation is stale for the active run projection");
  }

  Statement dispatch(database_, R"sql(
    SELECT status FROM node_dispatches
    WHERE run_id=? AND node_id=? AND attempt_id=?
  )sql");
  bind_text(dispatch.get(), 1, identity.run_id);
  bind_text(dispatch.get(), 2, identity.node_id);
  bind_text(dispatch.get(), 3, identity.attempt_id);
  if (sqlite3_step(dispatch.get()) != SQLITE_ROW ||
      column_text(dispatch.get(), 0) != "prepared") {
    throw OperationPreconditionError(
        "worker observation has no active prepared dispatch");
  }

  Statement lease(database_, R"sql(
    SELECT owner_run_id, lease_id, fencing_token, clock_domain, boot_id,
           expires_boottime_ns, released_wall_time_ns
    FROM resource_leases WHERE concurrency_key=?
      AND NOT EXISTS(
        SELECT 1 FROM resource_lease_releases AS release
        WHERE release.concurrency_key=resource_leases.concurrency_key
          AND release.owner_run_id=resource_leases.owner_run_id
          AND release.lease_id=resource_leases.lease_id
          AND release.fencing_token=resource_leases.fencing_token
      )
  )sql");
  bind_text(lease.get(), 1, identity.concurrency_key);
  if (sqlite3_step(lease.get()) != SQLITE_ROW ||
      column_text(lease.get(), 0) != identity.run_id ||
      column_text(lease.get(), 1) != identity.lease_id ||
      static_cast<std::uint64_t>(sqlite3_column_int64(lease.get(), 2)) !=
          identity.fencing_token ||
      column_text(lease.get(), 3) != ResourceLease::kBootTimeDomain ||
      column_text(lease.get(), 4) != now.boot_id ||
      sqlite3_column_int64(lease.get(), 5) <= now.boot.nanoseconds ||
      sqlite3_column_type(lease.get(), 6) != SQLITE_NULL) {
    throw OperationPreconditionError(
        "worker observation lost its active lease fence");
  }

  const auto ready = event(launch_id + ":ready");
  if (!ready || ready->payload.value("launch_nonce", std::string{}) !=
                    identity.launch_nonce ||
      ready->payload.value("concurrency_key", std::string{}) !=
          identity.concurrency_key ||
      ready->payload.value("lease_id", std::string{}) != identity.lease_id ||
      ready->payload.value("fencing_token", std::uint64_t{}) !=
          identity.fencing_token) {
    throw OperationPreconditionError(
        "worker observation has no matching readiness receipt");
  }
  (void)append_uncommitted(observation);
  transaction.commit();
}

void Journal::complete_dispatch_impl(
    const std::string& dispatch_id, const std::string& result_event_id,
    const std::vector<Event>& events,
    const std::optional<WorkerSessionIdentity>& identity,
    std::optional<AuthorityTimeSample> now) {
  if (dispatch_id.empty() || result_event_id.empty() || events.empty()) {
    throw std::invalid_argument("dispatch completion requires IDs and journal events");
  }
  if (std::none_of(events.begin(), events.end(), [&](const Event& event) {
        return event.event_id == result_event_id;
      })) {
    throw std::invalid_argument("dispatch completion batch does not contain its result event");
  }
  if (now) {
    require_authority_time(*now);
  }
  Transaction transaction(database_);
  Statement query(database_, R"sql(
    SELECT dispatch_id, run_id, run_revision, plan_revision, node_id, attempt_id,
           component, operation, status, result_event_id
    FROM node_dispatches WHERE dispatch_id=?
  )sql");
  bind_text(query.get(), 1, dispatch_id);
  if (sqlite3_step(query.get()) != SQLITE_ROW) {
    throw std::invalid_argument("cannot complete an unknown dispatch");
  }
  const Dispatch stored = dispatch_from_row(query.get());
  if (identity) {
    const std::string launch_id = identity->run_id + ":worker-launch:" +
                                  identity->node_id + ":" +
                                  identity->attempt_id;
    const auto binding = launch_binding(launch_id);
    if (!binding || binding->identity.run_id != identity->run_id ||
        binding->identity.node_id != identity->node_id ||
        binding->identity.attempt_id != identity->attempt_id ||
        binding->identity.launch_nonce != identity->launch_nonce ||
        binding->identity.concurrency_key != identity->concurrency_key ||
        binding->identity.lease_id != identity->lease_id ||
        binding->identity.fencing_token != identity->fencing_token) {
      throw OperationPreconditionError(
          "dispatch completion has no exact durable launch binding");
    }
    require_live_host_grant_claim(
        binding->identity.host_grant, identity->run_id,
        identity->concurrency_key, identity->lease_id,
        identity->fencing_token);
  }
  if (stored.status == DispatchStatus::completed) {
    if (stored.result_event_id != std::optional<std::string>{result_event_id}) {
      throw std::invalid_argument("dispatch already completed with a different result event");
    }
    for (const Event& requested : events) {
      const auto durable = event(requested.event_id);
      if (!durable || event_json(*durable) != event_json(requested)) {
        throw std::invalid_argument(
            "completed dispatch retry differs from its durable event batch");
      }
    }
    transaction.commit();
    return;
  }
  Statement projection(database_, R"sql(
    SELECT observed_state, run_revision, current_node_id, current_attempt_id
    FROM run_projection WHERE run_id=?
  )sql");
  bind_text(projection.get(), 1, stored.run_id);
  if (sqlite3_step(projection.get()) != SQLITE_ROW ||
      column_text(projection.get(), 0) != "running" ||
      static_cast<std::uint64_t>(sqlite3_column_int64(projection.get(), 1)) !=
          stored.run_revision ||
      column_text(projection.get(), 2) != stored.node_id ||
      column_text(projection.get(), 3) != stored.attempt_id) {
    throw OperationPreconditionError(
        "dispatch completion is stale for the active run projection");
  }
  if (identity) {
    if (!now || identity->run_id != stored.run_id ||
        identity->node_id != stored.node_id ||
        identity->attempt_id != stored.attempt_id || identity->fencing_token == 0) {
      throw std::invalid_argument("dispatch completion has an invalid worker session");
    }
    Statement lease(database_, R"sql(
      SELECT owner_run_id, lease_id, fencing_token, clock_domain, boot_id,
             expires_boottime_ns, released_wall_time_ns
      FROM resource_leases WHERE concurrency_key=?
        AND NOT EXISTS(
          SELECT 1 FROM resource_lease_releases AS release
          WHERE release.concurrency_key=resource_leases.concurrency_key
            AND release.owner_run_id=resource_leases.owner_run_id
            AND release.lease_id=resource_leases.lease_id
            AND release.fencing_token=resource_leases.fencing_token
        )
    )sql");
    bind_text(lease.get(), 1, identity->concurrency_key);
    if (sqlite3_step(lease.get()) != SQLITE_ROW ||
        column_text(lease.get(), 0) != identity->run_id ||
        column_text(lease.get(), 1) != identity->lease_id ||
        static_cast<std::uint64_t>(sqlite3_column_int64(lease.get(), 2)) !=
            identity->fencing_token ||
        column_text(lease.get(), 3) != ResourceLease::kBootTimeDomain ||
        column_text(lease.get(), 4) != now->boot_id ||
        sqlite3_column_int64(lease.get(), 5) <= now->boot.nanoseconds ||
        sqlite3_column_type(lease.get(), 6) != SQLITE_NULL) {
      throw OperationPreconditionError(
          "dispatch completion lost its active lease fence");
    }
    const auto ready = event(identity->run_id + ":worker-launch:" +
                             identity->node_id + ":" + identity->attempt_id +
                             ":ready");
    if (!ready || ready->payload.value("launch_nonce", std::string{}) !=
                      identity->launch_nonce ||
        ready->payload.value("concurrency_key", std::string{}) !=
            identity->concurrency_key ||
        ready->payload.value("lease_id", std::string{}) != identity->lease_id ||
        ready->payload.value("fencing_token", std::uint64_t{}) !=
            identity->fencing_token) {
      throw std::runtime_error("dispatch completion has no matching readiness receipt");
    }
  }
  for (const Event& event : events) {
    if (event.run_id != stored.run_id) {
      throw std::invalid_argument("dispatch completion batch crosses run identities");
    }
    if (event.event_id == result_event_id &&
        (event.node_id != stored.node_id || event.attempt_id != stored.attempt_id ||
         event.worker_sequence == 0)) {
      throw std::invalid_argument("dispatch result event does not match the dispatched attempt");
    }
  }
  for (const Event& event : events) {
    append_uncommitted(event);
  }
  Statement update(database_, R"sql(
    UPDATE node_dispatches SET status='completed', result_event_id=?
    WHERE dispatch_id=? AND status='prepared'
  )sql");
  bind_text(update.get(), 1, result_event_id);
  bind_text(update.get(), 2, dispatch_id);
  require_done(database_, update.get(), "complete node dispatch");
  if (sqlite3_changes(database_) != 1) {
    throw std::runtime_error("dispatch completion affected an unexpected number of rows");
  }
  transaction.commit();
}

void Journal::complete_managed_builtin_dispatch(
    const Dispatch& dispatch, const ResourceLease& lease,
    const AuthorityTimeSample& now, bool release_lease,
    const std::vector<Event>& events) {
  require_lease_identity(lease.concurrency_key, lease.owner_run_id,
                         lease.lease_id);
  require_authority_time(now);
  if (lease.fencing_token == 0 || events.size() != 4U ||
      dispatch.status != DispatchStatus::prepared || dispatch.result_event_id) {
    throw std::invalid_argument(
        "managed builtin completion requires a prepared dispatch, lease, and four events");
  }
  const Event& result = events[0];
  const Event& transition = events[1];
  const auto receipt = std::ranges::find_if(events, [](const Event& event) {
    return event.event_type == "node.dispatch_completed";
  });
  const bool validation = dispatch.operation == "validate_artifact";
  const bool releasing = dispatch.operation == "release_resources";
  const nlohmann::json expected_result_payload =
      releasing
          ? nlohmann::json{{"concurrency_key", lease.concurrency_key},
                           {"lease_id", lease.lease_id},
                           {"fencing_token", lease.fencing_token}}
          : nlohmann::json::object();
  if ((!validation && !releasing) || release_lease != releasing ||
      dispatch.run_id != lease.owner_run_id || result.run_id != dispatch.run_id ||
      result.node_id != dispatch.node_id ||
      result.attempt_id != dispatch.attempt_id ||
      result.run_revision != dispatch.run_revision ||
      result.plan_revision != dispatch.plan_revision || result.worker_sequence != 0 ||
      result.event_id != dispatch.dispatch_id + ":builtin-result" ||
      (validation && result.event_type != "artifact.validated" &&
       result.event_type != "artifact.invalid") ||
      (releasing && result.event_type != "resource.released") ||
      result.wall_time_ns != now.wall.nanoseconds || result.monotonic_time_ns != 0 ||
      result.payload != expected_result_payload ||
      transition.event_id != result.event_id + ":transition" ||
      transition.event_type != "fsm.transitioned" ||
      transition.run_id != dispatch.run_id ||
      transition.node_id != dispatch.node_id ||
      transition.attempt_id != dispatch.attempt_id ||
      transition.worker_sequence != 0 ||
      transition.payload.value("cause_event_id", std::string{}) != result.event_id ||
      receipt == events.end() ||
      receipt->event_id != dispatch.dispatch_id + ":completed" ||
      receipt->run_id != dispatch.run_id || receipt->node_id != dispatch.node_id ||
      receipt->attempt_id != dispatch.attempt_id || receipt->worker_sequence != 0 ||
      receipt->payload != nlohmann::json{{"dispatch_id", dispatch.dispatch_id},
                                        {"result_event_id", result.event_id}} ||
      std::ranges::count_if(events, [](const Event& event) {
        return event.event_type == "node.dispatch_completed";
      }) != 1) {
    throw std::invalid_argument(
        "managed builtin completion batch is not canonical for its operation");
  }
  if (std::ranges::any_of(events, [&](const Event& event) {
        return event.run_id != dispatch.run_id ||
               event.plan_revision != dispatch.plan_revision;
      })) {
    throw std::invalid_argument("managed builtin completion crosses run identity");
  }

  Transaction transaction(database_);
  Statement query(database_, R"sql(
    SELECT dispatch_id, run_id, run_revision, plan_revision, node_id, attempt_id,
           component, operation, status, result_event_id
    FROM node_dispatches WHERE dispatch_id=?
  )sql");
  bind_text(query.get(), 1, dispatch.dispatch_id);
  if (sqlite3_step(query.get()) != SQLITE_ROW) {
    throw std::invalid_argument("managed builtin completion has no prepared dispatch");
  }
  const Dispatch stored = dispatch_from_row(query.get());
  if (!same_dispatch_attempt(stored, dispatch)) {
    throw std::invalid_argument("managed builtin dispatch identity changed");
  }
  if (stored.status == DispatchStatus::completed) {
    if (stored.result_event_id != std::optional<std::string>{result.event_id}) {
      throw std::invalid_argument(
          "managed builtin dispatch completed with another result");
    }
    for (const Event& requested : events) {
      const auto durable = event(requested.event_id);
      if (!durable || event_json(*durable) != event_json(requested)) {
        throw std::invalid_argument(
            "managed builtin retry differs from its durable event batch");
      }
    }
    if (release_lease &&
        !has_lease_release_receipt(
            lease.concurrency_key, lease.owner_run_id, lease.lease_id,
            lease.fencing_token, lease.clock_domain, lease.boot_id,
            result.wall_time_ns)) {
      throw std::runtime_error(
          "managed builtin release retry has no durable lease receipt");
    }
    transaction.commit();
    return;
  }
  Statement projection(database_, R"sql(
    SELECT observed_state, run_revision, current_node_id, current_attempt_id
    FROM run_projection WHERE run_id=?
  )sql");
  bind_text(projection.get(), 1, dispatch.run_id);
  if (sqlite3_step(projection.get()) != SQLITE_ROW ||
      column_text(projection.get(), 0) != "running" ||
      static_cast<std::uint64_t>(sqlite3_column_int64(projection.get(), 1)) !=
          dispatch.run_revision ||
      column_text(projection.get(), 2) != dispatch.node_id ||
      column_text(projection.get(), 3) != dispatch.attempt_id) {
    throw std::runtime_error(
        "managed builtin completion is stale for the active projection");
  }
  Statement current_lease(database_, R"sql(
    SELECT owner_run_id, lease_id, fencing_token, clock_domain, boot_id,
           acquired_boottime_ns, expires_boottime_ns, released_wall_time_ns
    FROM resource_leases WHERE concurrency_key=?
      AND NOT EXISTS(
        SELECT 1 FROM resource_lease_releases AS release
        WHERE release.concurrency_key=resource_leases.concurrency_key
          AND release.owner_run_id=resource_leases.owner_run_id
          AND release.lease_id=resource_leases.lease_id
          AND release.fencing_token=resource_leases.fencing_token
      )
  )sql");
  bind_text(current_lease.get(), 1, lease.concurrency_key);
  if (sqlite3_step(current_lease.get()) != SQLITE_ROW ||
      column_text(current_lease.get(), 0) != lease.owner_run_id ||
      column_text(current_lease.get(), 1) != lease.lease_id ||
      static_cast<std::uint64_t>(sqlite3_column_int64(current_lease.get(), 2)) !=
          lease.fencing_token ||
      column_text(current_lease.get(), 3) != ResourceLease::kBootTimeDomain ||
      lease.clock_domain != ResourceLease::kBootTimeDomain ||
      column_text(current_lease.get(), 4) != now.boot_id ||
      lease.boot_id != now.boot_id ||
      sqlite3_column_int64(current_lease.get(), 5) != lease.acquired_boottime_ns ||
      sqlite3_column_int64(current_lease.get(), 6) <= now.boot.nanoseconds ||
      sqlite3_column_type(current_lease.get(), 7) != SQLITE_NULL) {
    throw OperationPreconditionError(
        "managed builtin completion lost its active lease fence");
  }
  if (release_lease) {
    Statement release(database_, R"sql(
      UPDATE resource_leases SET released_wall_time_ns=?
      WHERE concurrency_key=? AND owner_run_id=? AND lease_id=? AND fencing_token=?
        AND clock_domain='boottime/v1' AND boot_id=?
        AND released_wall_time_ns IS NULL AND expires_boottime_ns>?
        AND NOT EXISTS(
          SELECT 1 FROM resource_lease_releases AS release
          WHERE release.concurrency_key=resource_leases.concurrency_key
            AND release.owner_run_id=resource_leases.owner_run_id
            AND release.lease_id=resource_leases.lease_id
            AND release.fencing_token=resource_leases.fencing_token
        )
    )sql");
    bind_integer(release.get(), 1, now.wall.nanoseconds);
    bind_text(release.get(), 2, lease.concurrency_key);
    bind_text(release.get(), 3, lease.owner_run_id);
    bind_text(release.get(), 4, lease.lease_id);
    bind_integer(release.get(), 5,
                 checked_integer(lease.fencing_token, "fencing_token"));
    bind_text(release.get(), 6, now.boot_id);
    bind_integer(release.get(), 7, now.boot.nanoseconds);
    require_done(database_, release.get(), "release managed builtin lease");
    if (sqlite3_changes(database_) != 1) {
      throw std::runtime_error(
          "managed builtin lease release affected an unexpected row count");
    }
    Statement release_receipt(database_, R"sql(
      INSERT INTO resource_lease_releases(
        concurrency_key, owner_run_id, lease_id, fencing_token, clock_domain,
        boot_id, released_wall_time_ns
      ) VALUES(?, ?, ?, ?, ?, ?, ?)
    )sql");
    bind_text(release_receipt.get(), 1, lease.concurrency_key);
    bind_text(release_receipt.get(), 2, lease.owner_run_id);
    bind_text(release_receipt.get(), 3, lease.lease_id);
    bind_integer(release_receipt.get(), 4,
                 checked_integer(lease.fencing_token, "fencing_token"));
    bind_text(release_receipt.get(), 5, ResourceLease::kBootTimeDomain);
    bind_text(release_receipt.get(), 6, now.boot_id);
    bind_integer(release_receipt.get(), 7, now.wall.nanoseconds);
    require_done(database_, release_receipt.get(),
                 "record managed builtin lease release");
    record_lease_authority_release_uncommitted(lease,
                                                now.wall.nanoseconds);
  }
  for (const Event& event : events) {
    append_uncommitted(event);
  }
  Statement update(database_, R"sql(
    UPDATE node_dispatches SET status='completed', result_event_id=?
    WHERE dispatch_id=? AND status='prepared'
  )sql");
  bind_text(update.get(), 1, result.event_id);
  bind_text(update.get(), 2, dispatch.dispatch_id);
  require_done(database_, update.get(), "complete managed builtin dispatch");
  if (sqlite3_changes(database_) != 1) {
    throw std::runtime_error(
        "managed builtin completion affected an unexpected dispatch count");
  }
  transaction.commit();
}

std::optional<Dispatch> Journal::dispatch(const std::string& dispatch_id) const {
  Statement query(database_, R"sql(
    SELECT dispatch_id, run_id, run_revision, plan_revision, node_id, attempt_id,
           component, operation, status, result_event_id
    FROM node_dispatches WHERE dispatch_id=?
  )sql");
  bind_text(query.get(), 1, dispatch_id);
  const int status = sqlite3_step(query.get());
  if (status == SQLITE_DONE) {
    return std::nullopt;
  }
  if (status != SQLITE_ROW) {
    throw std::runtime_error("could not read dispatch: " + std::string(sqlite3_errmsg(database_)));
  }
  return dispatch_from_row(query.get());
}

ControlSubmission Journal::submit_control_command(ControlCommand command) {
  if (command.command_id.empty() || command.run_id.empty() || command.idempotency_key.empty() ||
      command.author.empty() || command.reason.empty()) {
    throw std::invalid_argument("control command identity, author, and reason must not be empty");
  }
  if (!command.assignments.is_object() || command.assignments.empty() ||
      command.status != ControlCommandStatus::requested || command.control_revision != 0 ||
      command.effective_step || !command.effective_values.empty() || !command.diagnostics.empty() ||
      command.acknowledgement || command.acknowledged_at_ns) {
    throw std::invalid_argument("new control command has invalid request state");
  }
  Transaction transaction(database_);
  Statement existing(database_, R"sql(
    SELECT command_id, run_id, idempotency_key, expected_run_revision,
           expected_control_revision, control_revision, plan_revision, apply_point,
           requires_pause, assignments_json, author, reason, status, effective_step,
           effective_values_json, diagnostics_json, ack_concurrency_key, ack_lease_id,
           ack_fencing_token, ack_node_id, ack_attempt_id, ack_worker_sequence,
           acknowledged_at_ns
    FROM control_commands
    WHERE command_id=? OR (run_id=? AND idempotency_key=?)
  )sql");
  bind_text(existing.get(), 1, command.command_id);
  bind_text(existing.get(), 2, command.run_id);
  bind_text(existing.get(), 3, command.idempotency_key);
  const int existing_status = sqlite3_step(existing.get());
  if (existing_status == SQLITE_ROW) {
    ControlCommand stored = command_from_row(existing.get());
    if (!same_command_request(stored, command)) {
      throw std::invalid_argument("control command idempotency identity has different content");
    }
    transaction.commit();
    return {.command = std::move(stored), .inserted = false};
  }
  if (existing_status != SQLITE_DONE) {
    throw std::runtime_error("could not read existing control command");
  }

  Statement run(database_,
                "SELECT run_revision, observed_state FROM run_projection WHERE run_id=?");
  bind_text(run.get(), 1, command.run_id);
  if (sqlite3_step(run.get()) != SQLITE_ROW) {
    throw std::invalid_argument("cannot submit controls for an unknown run");
  }
  const auto current_run_revision =
      static_cast<std::uint64_t>(sqlite3_column_int64(run.get(), 0));
  if (current_run_revision != command.expected_run_revision) {
    throw std::invalid_argument("control command expected_run_revision conflict");
  }
  const std::string observed_state = column_text(run.get(), 1);
  if (observed_state != "running" && observed_state != "paused") {
    throw std::invalid_argument("cannot submit controls for an inactive run");
  }
  if (command.requires_pause && observed_state != "paused") {
    throw std::invalid_argument("control command requires a durably paused run");
  }
  Statement latest(database_, R"sql(
    SELECT COALESCE(MAX(control_revision), 0) FROM control_commands WHERE run_id=?
  )sql");
  bind_text(latest.get(), 1, command.run_id);
  if (sqlite3_step(latest.get()) != SQLITE_ROW) {
    throw std::runtime_error("could not read latest control revision");
  }
  const auto current_control_revision =
      static_cast<std::uint64_t>(sqlite3_column_int64(latest.get(), 0));
  if (current_control_revision != command.expected_control_revision) {
    throw std::invalid_argument("control command expected_control_revision conflict");
  }
  if (current_control_revision ==
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error("control revision is exhausted");
  }
  command.control_revision = current_control_revision + 1;
  Event requested{
      .event_id = command.command_id + ":requested",
      .run_id = command.run_id,
      .run_revision = current_run_revision,
      .plan_revision = command.plan_revision,
      .node_id = "",
      .attempt_id = "",
      .worker_sequence = 0,
      .event_type = "control.requested",
      .event_version = 1,
      .wall_time_ns = 0,
      .monotonic_time_ns = 0,
      .optimizer_step = std::nullopt,
      .payload = {{"command_id", command.command_id},
                  {"idempotency_key", command.idempotency_key},
                  {"expected_run_revision", command.expected_run_revision},
                  {"expected_control_revision", command.expected_control_revision},
                  {"control_revision", command.control_revision},
                  {"plan_revision", command.plan_revision},
                  {"apply_point", enum_to_string(command.apply_point)},
                  {"requires_pause", command.requires_pause},
                  {"assignments", command.assignments},
                  {"author", command.author},
                  {"reason", command.reason}},
  };
  append_uncommitted(requested);

  Statement insert(database_, R"sql(
    INSERT INTO control_commands(
      command_id, run_id, idempotency_key, expected_run_revision,
      expected_control_revision, control_revision, plan_revision, apply_point,
      requires_pause, assignments_json, author, reason, status, effective_step,
      effective_values_json, diagnostics_json, ack_concurrency_key, ack_lease_id,
      ack_fencing_token, ack_node_id, ack_attempt_id, ack_worker_sequence,
      acknowledged_at_ns
    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'requested', NULL, '{}', '[]',
             NULL, NULL, NULL, NULL, NULL, NULL, NULL)
  )sql");
  bind_text(insert.get(), 1, command.command_id);
  bind_text(insert.get(), 2, command.run_id);
  bind_text(insert.get(), 3, command.idempotency_key);
  bind_integer(insert.get(), 4,
               checked_integer(command.expected_run_revision, "expected_run_revision"));
  bind_integer(insert.get(), 5,
               checked_integer(command.expected_control_revision, "expected_control_revision"));
  bind_integer(insert.get(), 6, checked_integer(command.control_revision, "control_revision"));
  bind_integer(insert.get(), 7, checked_integer(command.plan_revision, "plan_revision"));
  bind_text(insert.get(), 8, enum_to_string(command.apply_point));
  bind_integer(insert.get(), 9, command.requires_pause ? 1 : 0);
  bind_text(insert.get(), 10, command.assignments.dump());
  bind_text(insert.get(), 11, command.author);
  bind_text(insert.get(), 12, command.reason);
  require_done(database_, insert.get(), "insert control command");
  transaction.commit();
  return {.command = std::move(command), .inserted = true};
}

ControlCommand Journal::acknowledge_control_command(
    const std::string& run_id, const std::string& command_id,
    const ControlAcknowledgementIdentity& identity, ControlCommandStatus status,
    std::optional<std::uint64_t> effective_step, nlohmann::json effective_values,
    nlohmann::json diagnostics, const AuthorityTimeSample& now) {
  if (run_id.empty() || command_id.empty() || identity.concurrency_key.empty() ||
      identity.lease_id.empty() || identity.fencing_token == 0 || identity.node_id.empty() ||
      identity.attempt_id.empty() || identity.worker_sequence == 0 ||
      status == ControlCommandStatus::requested ||
      !effective_values.is_object() || !diagnostics.is_array()) {
    throw std::invalid_argument("invalid control command acknowledgement");
  }
  require_authority_time(now);
  Transaction transaction(database_);
  const std::int64_t received_at_ns = now.wall.nanoseconds;
  Statement query(database_, R"sql(
    SELECT command_id, run_id, idempotency_key, expected_run_revision,
           expected_control_revision, control_revision, plan_revision, apply_point,
           requires_pause, assignments_json, author, reason, status, effective_step,
           effective_values_json, diagnostics_json, ack_concurrency_key, ack_lease_id,
           ack_fencing_token, ack_node_id, ack_attempt_id, ack_worker_sequence,
           acknowledged_at_ns
    FROM control_commands WHERE command_id=?
  )sql");
  bind_text(query.get(), 1, command_id);
  if (sqlite3_step(query.get()) != SQLITE_ROW) {
    throw std::invalid_argument("cannot acknowledge an unknown control command");
  }
  ControlCommand command = command_from_row(query.get());
  if (command.run_id != run_id) {
    throw std::invalid_argument("control command belongs to a different run");
  }
  if (host_grant_enforcement_ == HostGrantEnforcement::required) {
    const std::string launch_id = command.run_id + ":worker-launch:" +
                                  identity.node_id + ":" +
                                  identity.attempt_id;
    const auto binding = launch_binding(launch_id);
    if (!binding || binding->identity.run_id != command.run_id ||
        binding->identity.node_id != identity.node_id ||
        binding->identity.attempt_id != identity.attempt_id ||
        binding->identity.concurrency_key != identity.concurrency_key ||
        binding->identity.lease_id != identity.lease_id ||
        binding->identity.fencing_token != identity.fencing_token) {
      throw OperationPreconditionError(
          "control acknowledgement has no exact durable worker launch binding");
    }
    require_live_host_grant_claim(
        binding->identity.host_grant, command.run_id,
        identity.concurrency_key, identity.lease_id,
        identity.fencing_token);
  }
  if (command.status != ControlCommandStatus::requested) {
    if (command.status == status && command.effective_step == effective_step &&
        command.effective_values == effective_values && command.diagnostics == diagnostics &&
        command.acknowledgement == identity) {
      transaction.commit();
      return command;
    }
    throw std::invalid_argument("control command already has a different acknowledgement");
  }
  Statement next_pending(database_, R"sql(
    SELECT MIN(control_revision) FROM control_commands
    WHERE run_id=? AND status='requested'
  )sql");
  bind_text(next_pending.get(), 1, command.run_id);
  if (sqlite3_step(next_pending.get()) != SQLITE_ROW ||
      sqlite3_column_type(next_pending.get(), 0) == SQLITE_NULL) {
    throw std::runtime_error("control command pending order disappeared");
  }
  const auto next_revision =
      static_cast<std::uint64_t>(sqlite3_column_int64(next_pending.get(), 0));
  if (next_revision != command.control_revision) {
    throw std::invalid_argument("control commands must be acknowledged in revision order");
  }
  if (status == ControlCommandStatus::applied) {
    if (effective_values != command.assignments) {
      throw std::invalid_argument("applied control values differ from the atomic request");
    }
    if (command.apply_point == ApplyPoint::restart) {
      throw std::invalid_argument("restart controls cannot become effective in the current attempt");
    }
    if (command.apply_point != ApplyPoint::immediate && !effective_step) {
      throw std::invalid_argument("safe-point control acknowledgement requires an effective step");
    }
  } else if (!effective_values.empty() || effective_step) {
    throw std::invalid_argument(
        "non-applied control acknowledgement has effective values or step");
  }
  Statement run(database_, R"sql(
    SELECT run_revision, observed_state, current_node_id, current_attempt_id
    FROM run_projection WHERE run_id=?
  )sql");
  bind_text(run.get(), 1, command.run_id);
  if (sqlite3_step(run.get()) != SQLITE_ROW) {
    throw std::runtime_error("control command run projection disappeared");
  }
  const auto run_revision = static_cast<std::uint64_t>(sqlite3_column_int64(run.get(), 0));
  const std::string observed_state = column_text(run.get(), 1);
  if (identity.node_id != column_text(run.get(), 2) ||
      identity.attempt_id != column_text(run.get(), 3)) {
    throw std::invalid_argument("control acknowledgement is from a stale worker attempt");
  }
  Statement lease(database_, R"sql(
    SELECT 1 FROM resource_leases
    WHERE concurrency_key=? AND owner_run_id=? AND lease_id=? AND fencing_token=?
      AND clock_domain='boottime/v1' AND boot_id=?
      AND released_wall_time_ns IS NULL
      AND acquired_boottime_ns<=? AND expires_boottime_ns>?
      AND NOT EXISTS(
        SELECT 1 FROM resource_lease_releases AS release
        WHERE release.concurrency_key=resource_leases.concurrency_key
          AND release.owner_run_id=resource_leases.owner_run_id
          AND release.lease_id=resource_leases.lease_id
          AND release.fencing_token=resource_leases.fencing_token
      )
  )sql");
  bind_text(lease.get(), 1, identity.concurrency_key);
  bind_text(lease.get(), 2, command.run_id);
  bind_text(lease.get(), 3, identity.lease_id);
  bind_integer(lease.get(), 4, checked_integer(identity.fencing_token, "fencing_token"));
  bind_text(lease.get(), 5, now.boot_id);
  bind_integer(lease.get(), 6, now.boot.nanoseconds);
  bind_integer(lease.get(), 7, now.boot.nanoseconds);
  if (sqlite3_step(lease.get()) != SQLITE_ROW) {
    throw std::invalid_argument("control acknowledgement has no matching active fenced lease");
  }
  if (status == ControlCommandStatus::applied && command.requires_pause &&
      observed_state != "paused") {
    throw std::invalid_argument("pause-required control cannot become effective while running");
  }
  command.status = status;
  command.effective_step = effective_step;
  command.effective_values = std::move(effective_values);
  command.diagnostics = std::move(diagnostics);
  command.acknowledgement = identity;
  command.acknowledged_at_ns = received_at_ns;
  const std::string status_name = command_status_name(status);
  Event acknowledged{
      .event_id = command.command_id + ":ack",
      .run_id = command.run_id,
      .run_revision = run_revision,
      .plan_revision = command.plan_revision,
      .node_id = identity.node_id,
      .attempt_id = identity.attempt_id,
      .worker_sequence = identity.worker_sequence,
      .event_type = "control." + status_name,
      .event_version = 1,
      .wall_time_ns = received_at_ns,
      .monotonic_time_ns = 0,
      .optimizer_step = command.effective_step,
      .payload = {{"command_id", command.command_id},
                  {"control_revision", command.control_revision},
                  {"apply_point", enum_to_string(command.apply_point)},
                  {"status", status_name},
                  {"concurrency_key", identity.concurrency_key},
                  {"lease_id", identity.lease_id},
                  {"fencing_token", identity.fencing_token},
                  {"effective_values", command.effective_values},
                  {"diagnostics", command.diagnostics}},
  };
  append_uncommitted(acknowledged);
  Statement update(database_, R"sql(
    UPDATE control_commands
    SET status=?, effective_step=?, effective_values_json=?, diagnostics_json=?,
        ack_concurrency_key=?, ack_lease_id=?, ack_fencing_token=?, ack_node_id=?,
        ack_attempt_id=?, ack_worker_sequence=?, acknowledged_at_ns=?
    WHERE command_id=? AND status='requested'
  )sql");
  bind_text(update.get(), 1, status_name);
  if (command.effective_step) {
    bind_integer(update.get(), 2, checked_integer(*command.effective_step, "effective_step"));
  } else if (sqlite3_bind_null(update.get(), 2) != SQLITE_OK) {
    throw std::runtime_error("sqlite null bind failed");
  }
  bind_text(update.get(), 3, command.effective_values.dump());
  bind_text(update.get(), 4, command.diagnostics.dump());
  bind_text(update.get(), 5, identity.concurrency_key);
  bind_text(update.get(), 6, identity.lease_id);
  bind_integer(update.get(), 7, checked_integer(identity.fencing_token, "fencing_token"));
  bind_text(update.get(), 8, identity.node_id);
  bind_text(update.get(), 9, identity.attempt_id);
  bind_integer(update.get(), 10, checked_integer(identity.worker_sequence, "worker_sequence"));
  bind_integer(update.get(), 11, received_at_ns);
  bind_text(update.get(), 12, command.command_id);
  require_done(database_, update.get(), "acknowledge control command");
  if (sqlite3_changes(database_) != 1) {
    throw std::runtime_error("control acknowledgement affected an unexpected number of rows");
  }
  transaction.commit();
  return command;
}

std::optional<ControlCommand> Journal::control_command(const std::string& command_id) const {
  Statement query(database_, R"sql(
    SELECT command_id, run_id, idempotency_key, expected_run_revision,
           expected_control_revision, control_revision, plan_revision, apply_point,
           requires_pause, assignments_json, author, reason, status, effective_step,
           effective_values_json, diagnostics_json, ack_concurrency_key, ack_lease_id,
           ack_fencing_token, ack_node_id, ack_attempt_id, ack_worker_sequence,
           acknowledged_at_ns
    FROM control_commands WHERE command_id=?
  )sql");
  bind_text(query.get(), 1, command_id);
  const int status = sqlite3_step(query.get());
  if (status == SQLITE_DONE) {
    return std::nullopt;
  }
  if (status != SQLITE_ROW) {
    throw std::runtime_error("could not read control command");
  }
  return command_from_row(query.get());
}

std::vector<ControlCommand> Journal::pending_control_commands(
    const std::string& run_id,
    std::uint64_t after_control_revision) const {
  if (run_id.empty()) {
    throw std::invalid_argument(
        "pending control lookup requires a run identity");
  }
  Statement query(database_, R"sql(
    SELECT command_id, run_id, idempotency_key, expected_run_revision,
           expected_control_revision, control_revision, plan_revision,
           apply_point, requires_pause, assignments_json, author, reason,
           status, effective_step, effective_values_json, diagnostics_json,
           ack_concurrency_key, ack_lease_id, ack_fencing_token, ack_node_id,
           ack_attempt_id, ack_worker_sequence, acknowledged_at_ns
    FROM control_commands
    WHERE run_id=? AND status='requested' AND control_revision>?
    ORDER BY control_revision
  )sql");
  bind_text(query.get(), 1, run_id);
  bind_integer(query.get(), 2,
               checked_integer(after_control_revision,
                               "after_control_revision"));
  std::vector<ControlCommand> commands;
  int status = SQLITE_OK;
  while ((status = sqlite3_step(query.get())) == SQLITE_ROW) {
    commands.push_back(command_from_row(query.get()));
  }
  if (status != SQLITE_DONE) {
    throw std::runtime_error("could not scan pending control commands");
  }
  return commands;
}

std::vector<ControlCommand> Journal::control_commands(
    const std::string& run_id, std::size_t limit) const {
  constexpr std::size_t kMaximumControlHistory = 50U;
  if (run_id.empty() || run_id.size() > 256U || limit == 0U ||
      limit > kMaximumControlHistory) {
    throw std::invalid_argument(
        "control history requires a bounded run ID and limit");
  }
  Statement query(database_, R"sql(
    SELECT command_id, run_id, idempotency_key, expected_run_revision,
           expected_control_revision, control_revision, plan_revision,
           apply_point, requires_pause, assignments_json, author, reason,
           status, effective_step, effective_values_json, diagnostics_json,
           ack_concurrency_key, ack_lease_id, ack_fencing_token, ack_node_id,
           ack_attempt_id, ack_worker_sequence, acknowledged_at_ns
    FROM control_commands
    WHERE run_id=?
    ORDER BY control_revision DESC
    LIMIT ?
  )sql");
  bind_text(query.get(), 1, run_id);
  bind_integer(query.get(), 2, static_cast<std::int64_t>(limit));
  std::vector<ControlCommand> commands;
  commands.reserve(limit);
  int status = SQLITE_ROW;
  while ((status = sqlite3_step(query.get())) == SQLITE_ROW) {
    commands.push_back(command_from_row(query.get()));
  }
  if (status != SQLITE_DONE) {
    throw std::runtime_error("could not scan control command history");
  }
  return commands;
}

std::uint64_t Journal::latest_control_revision(const std::string& run_id) const {
  Statement query(database_, R"sql(
    SELECT COALESCE(MAX(control_revision), 0) FROM control_commands WHERE run_id=?
  )sql");
  bind_text(query.get(), 1, run_id);
  if (sqlite3_step(query.get()) != SQLITE_ROW) {
    throw std::runtime_error("could not read latest control revision");
  }
  return static_cast<std::uint64_t>(sqlite3_column_int64(query.get(), 0));
}

std::uint64_t Journal::latest_effective_control_revision(const std::string& run_id) const {
  Statement query(database_, R"sql(
    SELECT COALESCE(MAX(control_revision), 0) FROM control_commands
    WHERE run_id=? AND status='applied'
  )sql");
  bind_text(query.get(), 1, run_id);
  if (sqlite3_step(query.get()) != SQLITE_ROW) {
    throw std::runtime_error("could not read latest effective control revision");
  }
  return static_cast<std::uint64_t>(sqlite3_column_int64(query.get(), 0));
}

EffectiveControlSnapshot Journal::effective_controls(
    const std::string& run_id) const {
  if (run_id.empty())
    throw std::invalid_argument(
        "effective control snapshot requires a run identity");
  EffectiveControlSnapshot result;
  Statement query(database_, R"sql(
    SELECT control_revision, effective_values_json
    FROM control_commands
    WHERE run_id=? AND status='applied'
    ORDER BY control_revision
  )sql");
  bind_text(query.get(), 1, run_id);
  int status = SQLITE_ROW;
  while ((status = sqlite3_step(query.get())) == SQLITE_ROW) {
    const auto revision =
        static_cast<std::uint64_t>(sqlite3_column_int64(query.get(), 0));
    nlohmann::json values;
    try {
      values = nlohmann::json::parse(column_text(query.get(), 1));
    } catch (const nlohmann::json::exception&) {
      throw std::runtime_error(
          "effective control projection contains malformed JSON");
    }
    if (revision == 0U || revision <= result.revision ||
        !values.is_object() || values.empty()) {
      throw std::runtime_error(
          "effective control projection is not canonical");
    }
    for (const auto& [name, value] : values.items())
      result.values[name] = value;
    result.revision = revision;
  }
  if (status != SQLITE_DONE)
    throw std::runtime_error("could not read effective control snapshot");
  return result;
}

LeaseAcquireResult Journal::acquire_lease(const std::string& concurrency_key,
                                          const std::string& owner_run_id,
                                          const std::string& lease_id,
                                          const AuthorityTimeSample& now,
                                          std::int64_t timeout_ns) {
  return acquire_lease_with_events(concurrency_key, owner_run_id, lease_id, now,
                                   timeout_ns, {});
}

LeaseAcquireResult Journal::acquire_lease_with_events(
    const std::string& concurrency_key, const std::string& owner_run_id,
    const std::string& lease_id, const AuthorityTimeSample& now,
    std::int64_t timeout_ns,
    const std::vector<Event>& events) {
  require_lease_identity(concurrency_key, owner_run_id, lease_id);
  require_authority_time(now);
  const std::int64_t expires_boottime_ns =
      lease_expiration(now.boot.nanoseconds, timeout_ns);
  const std::int64_t expires_wall_time_ns =
      lease_expiration(now.wall.nanoseconds, timeout_ns);
  Transaction transaction(database_);
  bool replaying_acquisition = false;
  std::optional<Event> replayed_resource_event;
  if (!events.empty()) {
    std::string chain_reason;
    if (!verify_chain(&chain_reason)) {
      throw std::runtime_error("refusing run lease acquisition: " + chain_reason);
    }
    const auto requested_plan = events[0].payload.find("plan_hash");
    const std::string requested_plan_hash =
        requested_plan != events[0].payload.end() && requested_plan->is_string()
            ? requested_plan->get<std::string>()
            : std::string{};
    if (events.size() != 3U || events[0].run_id != owner_run_id ||
        events[1].run_id != owner_run_id || events[2].run_id != owner_run_id ||
        events[0].event_type != "run.desired_state_changed" ||
        events[0].payload != nlohmann::json{{"state", "running"},
                                           {"cause", "scheduler.lease_acquisition"},
                                           {"lease_id", lease_id},
                                           {"plan_hash", requested_plan_hash}} ||
        events[1].event_type != "resource.lease_acquired" ||
        events[1].payload != nlohmann::json{{"concurrency_key", concurrency_key},
                                           {"owner_run_id", owner_run_id},
                                           {"lease_id", lease_id}} ||
        events[2].event_type != "run.observed_state_changed" ||
        events[2].payload != nlohmann::json{{"state", "acquiring"},
                                           {"cause_event_id", events[1].event_id},
                                           {"concurrency_key", concurrency_key},
                                           {"lease_id", lease_id}} ||
        events[0].worker_sequence != 0 || events[1].worker_sequence != 0 ||
        events[2].worker_sequence != 0 || !events[0].node_id.empty() ||
        !events[1].node_id.empty() || !events[2].node_id.empty() ||
        !events[0].attempt_id.empty() || !events[1].attempt_id.empty() ||
        !events[2].attempt_id.empty()) {
      throw std::invalid_argument("lease acquisition requires its exact lifecycle event pair");
    }
    Statement projection(database_, R"sql(
      SELECT desired_state, observed_state, run_revision,
             current_node_id, current_attempt_id, plan_hash
      FROM run_projection WHERE run_id=?
    )sql");
    bind_text(projection.get(), 1, owner_run_id);
    if (sqlite3_step(projection.get()) != SQLITE_ROW) {
      throw std::invalid_argument("lease acquisition run does not exist");
    }
    const auto run_revision =
        static_cast<std::uint64_t>(sqlite3_column_int64(projection.get(), 2));
    const std::string desired_state = column_text(projection.get(), 0);
    const std::string observed_state = column_text(projection.get(), 1);
    const bool unassigned = column_text(projection.get(), 3).empty() &&
                            column_text(projection.get(), 4).empty();
    const bool plan_matches = !requested_plan_hash.empty() &&
                              requested_plan_hash == column_text(projection.get(), 5) &&
                              events[0].plan_revision == 1U &&
                              events[1].plan_revision == 1U &&
                              events[2].plan_revision == 1U;
    if (desired_state == "running" && observed_state == "acquiring") {
      if (!unassigned || !plan_matches || run_revision < events[2].run_revision ||
          events[0].run_revision + 1U != events[2].run_revision ||
          events[1].run_revision + 1U != events[2].run_revision) {
        throw std::invalid_argument("acquisition replay disagrees with the acquiring run");
      }
      const auto stored_desired = event(events[0].event_id);
      const auto stored_resource = event(events[1].event_id);
      const auto stored_observed = event(events[2].event_id);
      const auto same_envelope = [](const Event& stored, const Event& requested) {
        return stored.event_id == requested.event_id && stored.run_id == requested.run_id &&
               stored.run_revision == requested.run_revision &&
               stored.plan_revision == requested.plan_revision &&
               stored.node_id == requested.node_id && stored.attempt_id == requested.attempt_id &&
               stored.worker_sequence == requested.worker_sequence &&
               stored.event_type == requested.event_type &&
               stored.event_version == requested.event_version;
      };
      if (!stored_desired || !stored_resource || !stored_observed ||
          !same_envelope(*stored_desired, events[0]) ||
          stored_desired->payload != events[0].payload ||
          !same_envelope(*stored_resource, events[1]) ||
          !same_envelope(*stored_observed, events[2])) {
        throw std::invalid_argument("acquisition retry differs from durable lifecycle events");
      }
      nlohmann::json expected_resource = events[1].payload;
      expected_resource["fencing_token"] =
          stored_resource->payload.value("fencing_token", std::uint64_t{});
      expected_resource["clock_domain"] = ResourceLease::kBootTimeDomain;
      expected_resource["boot_id"] = now.boot_id;
      expected_resource["acquired_boottime_ns"] =
          stored_resource->payload.value("acquired_boottime_ns", std::int64_t{});
      expected_resource["expires_boottime_ns"] =
          stored_resource->payload.value("expires_boottime_ns", std::int64_t{});
      nlohmann::json expected_observed = events[2].payload;
      expected_observed["fencing_token"] =
          stored_resource->payload.value("fencing_token", std::uint64_t{});
      if (stored_resource->payload != expected_resource ||
          stored_observed->payload != expected_observed) {
        throw std::invalid_argument("acquisition retry differs from durable lease evidence");
      }
      replaying_acquisition = true;
      replayed_resource_event = *stored_resource;
    } else if (desired_state != "queued" || observed_state != "queued" || !unassigned ||
               !plan_matches || events[0].run_revision != run_revision + 1U ||
               events[1].run_revision != run_revision + 1U ||
               events[2].run_revision != run_revision + 2U) {
      throw std::invalid_argument("lease acquisition events disagree with the queued run");
    }
  }
  const auto append_events = [&](const ResourceLease& acquired_lease) {
    for (std::size_t index = 0; index < events.size(); ++index) {
      Event event = events[index];
      if (index == 1U) {
        event.wall_time_ns = acquired_lease.acquired_wall_time_ns;
        event.payload["clock_domain"] = acquired_lease.clock_domain;
        event.payload["boot_id"] = acquired_lease.boot_id;
        event.payload["acquired_boottime_ns"] = acquired_lease.acquired_boottime_ns;
        event.payload["expires_boottime_ns"] = acquired_lease.expires_boottime_ns;
        event.payload["fencing_token"] = acquired_lease.fencing_token;
      } else if (index == 2U) {
        event.payload["fencing_token"] = acquired_lease.fencing_token;
      }
      append_uncommitted(event);
    }
  };
  Statement query(database_, R"sql(
    SELECT concurrency_key, owner_run_id, lease_id, fencing_token,
           clock_domain, boot_id, acquired_boottime_ns, expires_boottime_ns,
           acquired_wall_time_ns, expires_wall_time_ns, released_wall_time_ns,
           EXISTS(
             SELECT 1 FROM resource_lease_releases AS release
             WHERE release.concurrency_key=resource_leases.concurrency_key
               AND release.owner_run_id=resource_leases.owner_run_id
               AND release.lease_id=resource_leases.lease_id
               AND release.fencing_token=resource_leases.fencing_token
           )
    FROM resource_leases WHERE concurrency_key=?
  )sql");
  bind_text(query.get(), 1, concurrency_key);
  const int status = sqlite3_step(query.get());
  if (status != SQLITE_ROW && status != SQLITE_DONE) {
    throw std::runtime_error("could not read resource lease: " + std::string(sqlite3_errmsg(database_)));
  }

  ResourceLease lease;
  if (replaying_acquisition) {
    if (status != SQLITE_ROW || !replayed_resource_event) {
      throw std::runtime_error("acquiring run has no durable lease row");
    }
    lease = lease_from_row(query.get());
    const bool released = sqlite3_column_type(query.get(), 10) != SQLITE_NULL ||
                          sqlite3_column_int(query.get(), 11) != 0;
    if (released || lease.owner_run_id != owner_run_id || lease.lease_id != lease_id ||
        lease.clock_domain != ResourceLease::kBootTimeDomain ||
        lease.boot_id != now.boot_id ||
        lease.fencing_token !=
            replayed_resource_event->payload.value("fencing_token", std::uint64_t{}) ||
        lease.acquired_boottime_ns !=
            replayed_resource_event->payload.value("acquired_boottime_ns", std::int64_t{})) {
      throw std::runtime_error("acquiring run no longer has its durable fenced lease");
    }
    transaction.commit();
    return {.status = LeaseAcquireStatus::already_owned, .lease = std::move(lease)};
  }
  if (status == SQLITE_DONE) {
    lease = ResourceLease{.concurrency_key = concurrency_key,
                          .owner_run_id = owner_run_id,
                          .lease_id = lease_id,
                          .fencing_token = 1,
                          .clock_domain = ResourceLease::kBootTimeDomain,
                          .boot_id = now.boot_id,
                          .acquired_boottime_ns = now.boot.nanoseconds,
                          .expires_boottime_ns = expires_boottime_ns,
                          .acquired_wall_time_ns = now.wall.nanoseconds,
                          .expires_wall_time_ns = expires_wall_time_ns};
    Statement insert(database_, R"sql(
      INSERT INTO resource_leases(
        concurrency_key, owner_run_id, lease_id, fencing_token,
        clock_domain, boot_id, acquired_boottime_ns, expires_boottime_ns,
        acquired_wall_time_ns, expires_wall_time_ns, released_wall_time_ns
      ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
    )sql");
    bind_text(insert.get(), 1, lease.concurrency_key);
    bind_text(insert.get(), 2, lease.owner_run_id);
    bind_text(insert.get(), 3, lease.lease_id);
    bind_integer(insert.get(), 4, checked_integer(lease.fencing_token, "fencing_token"));
    bind_text(insert.get(), 5, lease.clock_domain);
    bind_text(insert.get(), 6, lease.boot_id);
    bind_integer(insert.get(), 7, lease.acquired_boottime_ns);
    bind_integer(insert.get(), 8, lease.expires_boottime_ns);
    bind_integer(insert.get(), 9, lease.acquired_wall_time_ns);
    bind_integer(insert.get(), 10, lease.expires_wall_time_ns);
    require_done(database_, insert.get(), "insert resource lease");
    record_lease_authority_acquisition_uncommitted(lease);
    append_events(lease);
    transaction.commit();
    return {.status = LeaseAcquireStatus::acquired, .lease = std::move(lease)};
  }

  lease = lease_from_row(query.get());
  const bool released = sqlite3_column_type(query.get(), 10) != SQLITE_NULL ||
                        sqlite3_column_int(query.get(), 11) != 0;
  const bool active = !released &&
                      lease.clock_domain == ResourceLease::kBootTimeDomain &&
                      lease.boot_id == now.boot_id &&
                      lease.expires_boottime_ns > now.boot.nanoseconds;
  if (active) {
    const LeaseAcquireStatus disposition =
        lease.owner_run_id == owner_run_id && lease.lease_id == lease_id
            ? LeaseAcquireStatus::already_owned
            : LeaseAcquireStatus::busy;
    if (disposition == LeaseAcquireStatus::already_owned) {
      append_events(lease);
    }
    transaction.commit();
    return {.status = disposition, .lease = std::move(lease)};
  }
  if (lease.fencing_token == static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error("resource lease fencing token is exhausted");
  }
  ++lease.fencing_token;
  lease.owner_run_id = owner_run_id;
  lease.lease_id = lease_id;
  lease.clock_domain = ResourceLease::kBootTimeDomain;
  lease.boot_id = now.boot_id;
  lease.acquired_boottime_ns = now.boot.nanoseconds;
  lease.expires_boottime_ns = expires_boottime_ns;
  lease.acquired_wall_time_ns = now.wall.nanoseconds;
  lease.expires_wall_time_ns = expires_wall_time_ns;
  Statement replace(database_, R"sql(
    UPDATE resource_leases
    SET owner_run_id=?, lease_id=?, fencing_token=?, clock_domain=?, boot_id=?,
        acquired_boottime_ns=?, expires_boottime_ns=?, acquired_wall_time_ns=?,
        expires_wall_time_ns=?, released_wall_time_ns=NULL
    WHERE concurrency_key=?
  )sql");
  bind_text(replace.get(), 1, lease.owner_run_id);
  bind_text(replace.get(), 2, lease.lease_id);
  bind_integer(replace.get(), 3, checked_integer(lease.fencing_token, "fencing_token"));
  bind_text(replace.get(), 4, lease.clock_domain);
  bind_text(replace.get(), 5, lease.boot_id);
  bind_integer(replace.get(), 6, lease.acquired_boottime_ns);
  bind_integer(replace.get(), 7, lease.expires_boottime_ns);
  bind_integer(replace.get(), 8, lease.acquired_wall_time_ns);
  bind_integer(replace.get(), 9, lease.expires_wall_time_ns);
  bind_text(replace.get(), 10, concurrency_key);
  require_done(database_, replace.get(), "replace resource lease");
  if (sqlite3_changes(database_) != 1) {
    throw std::runtime_error("resource lease replacement affected an unexpected number of rows");
  }
  record_lease_authority_acquisition_uncommitted(lease);
  append_events(lease);
  transaction.commit();
  return {.status = LeaseAcquireStatus::acquired, .lease = std::move(lease)};
}

bool Journal::complete_builtin_admission(const ResourceLease& lease,
                                         const AuthorityTimeSample& now,
                                         const std::vector<Event>& events) {
  require_lease_identity(lease.concurrency_key, lease.owner_run_id,
                         lease.lease_id);
  require_authority_time(now);
  if (lease.fencing_token == 0 || events.size() != 2U) {
    throw std::invalid_argument("builtin admission requires a valid lease and event pair");
  }
  const Event& result = events[0];
  const Event& transition = events[1];
  if (result.run_id != lease.owner_run_id ||
      result.event_type != "resource.acquired" ||
      result.worker_sequence != 0 || result.node_id.empty() ||
      result.attempt_id.empty() ||
      result.payload != nlohmann::json{{"concurrency_key", lease.concurrency_key},
                                      {"lease_id", lease.lease_id},
                                      {"fencing_token", lease.fencing_token}} ||
      transition.run_id != lease.owner_run_id ||
      transition.event_type != "fsm.transitioned" ||
      transition.event_id != result.event_id + ":transition" ||
      transition.run_revision != result.run_revision + 1U ||
      transition.plan_revision != result.plan_revision ||
      transition.node_id != result.node_id ||
      transition.attempt_id != result.attempt_id ||
      transition.worker_sequence != 0 ||
      transition.payload.value("cause_event_id", std::string{}) !=
          result.event_id ||
      transition.payload.value("source", std::string{}) != result.node_id) {
    throw std::invalid_argument("builtin admission event pair is malformed");
  }

  Transaction transaction(database_);
  std::string chain_reason;
  if (!verify_chain(&chain_reason)) {
    throw std::runtime_error("refusing builtin admission: " + chain_reason);
  }
  Statement projection(database_, R"sql(
    SELECT desired_state, observed_state, run_revision,
           current_node_id, current_attempt_id
    FROM run_projection WHERE run_id=?
  )sql");
  bind_text(projection.get(), 1, lease.owner_run_id);
  if (sqlite3_step(projection.get()) != SQLITE_ROW ||
      column_text(projection.get(), 0) != "running" ||
      column_text(projection.get(), 1) != "acquiring" ||
      !column_text(projection.get(), 3).empty() ||
      !column_text(projection.get(), 4).empty()) {
    throw std::invalid_argument("builtin admission requires an unassigned acquiring run");
  }
  const auto run_revision =
      static_cast<std::uint64_t>(sqlite3_column_int64(projection.get(), 2));

  Statement current_lease(database_, R"sql(
    SELECT owner_run_id, lease_id, fencing_token, clock_domain, boot_id,
           acquired_boottime_ns, expires_boottime_ns, released_wall_time_ns
    FROM resource_leases WHERE concurrency_key=?
      AND NOT EXISTS(
        SELECT 1 FROM resource_lease_releases AS release
        WHERE release.concurrency_key=resource_leases.concurrency_key
          AND release.owner_run_id=resource_leases.owner_run_id
          AND release.lease_id=resource_leases.lease_id
          AND release.fencing_token=resource_leases.fencing_token
      )
  )sql");
  bind_text(current_lease.get(), 1, lease.concurrency_key);
  if (sqlite3_step(current_lease.get()) != SQLITE_ROW ||
      column_text(current_lease.get(), 0) != lease.owner_run_id ||
      column_text(current_lease.get(), 1) != lease.lease_id ||
      static_cast<std::uint64_t>(sqlite3_column_int64(current_lease.get(), 2)) !=
          lease.fencing_token ||
      column_text(current_lease.get(), 3) != ResourceLease::kBootTimeDomain ||
      lease.clock_domain != ResourceLease::kBootTimeDomain ||
      column_text(current_lease.get(), 4) != now.boot_id ||
      lease.boot_id != now.boot_id ||
      sqlite3_column_int64(current_lease.get(), 5) != lease.acquired_boottime_ns ||
      sqlite3_column_int64(current_lease.get(), 6) <= now.boot.nanoseconds ||
      sqlite3_column_type(current_lease.get(), 7) != SQLITE_NULL) {
    throw OperationPreconditionError(
        "builtin admission no longer owns its active fenced lease");
  }

  const auto stored_result = event(result.event_id);
  const auto stored_transition = event(transition.event_id);
  if (run_revision == transition.run_revision) {
    Event replay_result = result;
    Event replay_transition = transition;
    if (stored_result && stored_transition) {
      replay_result.wall_time_ns = stored_result->wall_time_ns;
      replay_result.monotonic_time_ns = stored_result->monotonic_time_ns;
      replay_transition.wall_time_ns = stored_transition->wall_time_ns;
      replay_transition.monotonic_time_ns = stored_transition->monotonic_time_ns;
    }
    if (!stored_result || !stored_transition ||
        event_json(*stored_result) != event_json(replay_result) ||
        event_json(*stored_transition) != event_json(replay_transition)) {
      throw std::invalid_argument("builtin admission replay differs from durable events");
    }
    transaction.commit();
    return false;
  }
  if (run_revision != result.run_revision || stored_result || stored_transition) {
    throw std::invalid_argument(
        "builtin admission disagrees with the acquiring revision: projection=" +
        std::to_string(run_revision) + ", result=" +
        std::to_string(result.run_revision) + ", transition=" +
        std::to_string(transition.run_revision) + ", stored_result=" +
        (stored_result ? "yes" : "no") + ", stored_transition=" +
        (stored_transition ? "yes" : "no"));
  }
  append_uncommitted(result);
  append_uncommitted(transition);
  transaction.commit();
  return true;
}

bool Journal::prepare_worker_launch(const WorkerLaunchTicket& launch,
                                    const AuthorityTimeSample& now,
                                    const Event& event) {
  require_lease_identity(launch.concurrency_key, launch.run_id,
                         launch.lease_id);
  nlohmann::json expected_payload{
      {"launch_nonce", launch.launch_nonce},
      {"adapter", launch.adapter},
      {"adapter_version", launch.adapter_version},
      {"code_fingerprint", launch.code_fingerprint},
      {"required_capabilities", launch.required_capabilities},
      {"concurrency_key", launch.concurrency_key},
      {"lease_id", launch.lease_id},
      {"fencing_token", launch.fencing_token},
  };
  if (launch.host_grant) {
    expected_payload["host_grant"] = encode_json(*launch.host_grant);
  }
  require_authority_time(now);
  if (launch.fencing_token == 0 || launch.node_id.empty() ||
      launch.attempt_id.empty() || launch.launch_nonce.empty() ||
      launch.adapter.empty() || launch.adapter_version.empty() ||
      launch.code_fingerprint.empty() || event.run_id != launch.run_id ||
      event.node_id != launch.node_id || event.attempt_id != launch.attempt_id ||
      event.worker_sequence != 0 ||
      event.event_type != "worker.launch_requested" ||
      event.payload != expected_payload) {
    throw std::invalid_argument("worker launch ticket or event is malformed");
  }
  Transaction transaction(database_);
  std::string chain_reason;
  if (!verify_chain(&chain_reason)) {
    throw std::runtime_error("refusing worker launch: " + chain_reason);
  }
  Statement projection(database_, R"sql(
    SELECT desired_state, observed_state, run_revision,
           current_node_id, current_attempt_id
    FROM run_projection WHERE run_id=?
  )sql");
  bind_text(projection.get(), 1, launch.run_id);
  if (sqlite3_step(projection.get()) != SQLITE_ROW ||
      column_text(projection.get(), 0) != "running" ||
      column_text(projection.get(), 1) != "acquiring" ||
      static_cast<std::uint64_t>(sqlite3_column_int64(projection.get(), 2)) !=
          event.run_revision ||
      !column_text(projection.get(), 3).empty() ||
      !column_text(projection.get(), 4).empty()) {
    throw std::invalid_argument("worker launch requires an unassigned acquiring run");
  }
  Statement lease(database_, R"sql(
    SELECT owner_run_id, lease_id, fencing_token, clock_domain, boot_id,
           expires_boottime_ns, released_wall_time_ns
    FROM resource_leases WHERE concurrency_key=?
      AND NOT EXISTS(
        SELECT 1 FROM resource_lease_releases AS release
        WHERE release.concurrency_key=resource_leases.concurrency_key
          AND release.owner_run_id=resource_leases.owner_run_id
          AND release.lease_id=resource_leases.lease_id
          AND release.fencing_token=resource_leases.fencing_token
      )
  )sql");
  bind_text(lease.get(), 1, launch.concurrency_key);
  if (sqlite3_step(lease.get()) != SQLITE_ROW ||
      column_text(lease.get(), 0) != launch.run_id ||
      column_text(lease.get(), 1) != launch.lease_id ||
      static_cast<std::uint64_t>(sqlite3_column_int64(lease.get(), 2)) !=
          launch.fencing_token ||
      column_text(lease.get(), 3) != ResourceLease::kBootTimeDomain ||
      column_text(lease.get(), 4) != now.boot_id ||
      sqlite3_column_int64(lease.get(), 5) <= now.boot.nanoseconds ||
      sqlite3_column_type(lease.get(), 6) != SQLITE_NULL) {
    throw OperationPreconditionError(
        "worker launch no longer owns its active lease");
  }
  require_live_host_grant_claim(
      launch.host_grant, launch.run_id, launch.concurrency_key,
      launch.lease_id, launch.fencing_token);
  const auto stored = this->event(event.event_id);
  if (stored) {
    Event replay = event;
    replay.wall_time_ns = stored->wall_time_ns;
    replay.monotonic_time_ns = stored->monotonic_time_ns;
    if (event_json(*stored) != event_json(replay)) {
      throw std::invalid_argument("worker launch retry differs from durable request");
    }
    transaction.commit();
    return false;
  }
  Statement conflicting(database_, R"sql(
    SELECT 1 FROM events
    WHERE run_id=? AND event_type='worker.launch_requested'
      AND node_id=? AND attempt_id=? LIMIT 1
  )sql");
  bind_text(conflicting.get(), 1, launch.run_id);
  bind_text(conflicting.get(), 2, launch.node_id);
  bind_text(conflicting.get(), 3, launch.attempt_id);
  if (sqlite3_step(conflicting.get()) == SQLITE_ROW) {
    throw std::invalid_argument("worker attempt already has another launch request");
  }
  append_uncommitted(event);
  transaction.commit();
  return true;
}

bool Journal::bind_worker_launch(const ResolvedLaunchSpec& binding,
                                 const AuthorityTimeSample& now,
                                 const Event& event) {
  const ResolvedLaunchSpec canonical = resolved_launch_spec_from_json(
      resolved_launch_spec_json(binding));
  const ResolvedLaunchIdentity& identity = canonical.identity;
  require_lease_identity(identity.concurrency_key, identity.run_id,
                         identity.lease_id);
  const nlohmann::json expected_payload{
      {"cause_event_id", identity.launch_event_id},
      {"spec", resolved_launch_spec_json(canonical)},
  };
  require_authority_time(now);
  if (identity.fencing_token == 0U ||
      identity.launch_event_id.empty() || identity.run_id.empty() ||
      identity.node_id.empty() || identity.attempt_id.empty() ||
      identity.launch_nonce.empty() || identity.adapter_key.adapter.empty() ||
      identity.adapter_key.version.empty() ||
      identity.code_fingerprint.empty() ||
      event.event_id != identity.launch_event_id + ":bound" ||
      event.run_id != identity.run_id || event.node_id != identity.node_id ||
      event.attempt_id != identity.attempt_id || event.worker_sequence != 0U ||
      event.event_type != "worker.launch_bound" ||
      event.payload != expected_payload) {
    throw std::invalid_argument(
        "worker host launch binding or event is malformed");
  }
  Transaction transaction(database_);
  std::string chain_reason;
  if (!verify_chain(&chain_reason)) {
    throw std::runtime_error("refusing host launch binding: " +
                             chain_reason);
  }
  const auto stored = this->event(event.event_id);
  if (stored) {
    Event replay = event;
    replay.wall_time_ns = stored->wall_time_ns;
    replay.monotonic_time_ns = stored->monotonic_time_ns;
    if (event_json(*stored) != event_json(replay)) {
      throw std::invalid_argument(
          "host launch binding retry differs from durable evidence");
    }
    transaction.commit();
    return false;
  }
  Statement projection(database_, R"sql(
    SELECT desired_state, observed_state, run_revision,
           current_node_id, current_attempt_id
    FROM run_projection WHERE run_id=?
  )sql");
  bind_text(projection.get(), 1, identity.run_id);
  if (sqlite3_step(projection.get()) != SQLITE_ROW ||
      column_text(projection.get(), 0) != "running" ||
      column_text(projection.get(), 1) != "acquiring" ||
      static_cast<std::uint64_t>(sqlite3_column_int64(projection.get(), 2)) !=
          event.run_revision ||
      !column_text(projection.get(), 3).empty() ||
      !column_text(projection.get(), 4).empty()) {
    throw std::invalid_argument(
        "host launch binding requires an unassigned acquiring run");
  }
  const auto launch = this->event(identity.launch_event_id);
  nlohmann::json expected_launch_payload{
      {"launch_nonce", identity.launch_nonce},
      {"adapter", identity.adapter_key.adapter},
      {"adapter_version", identity.adapter_key.version},
      {"code_fingerprint", identity.code_fingerprint},
      {"required_capabilities", identity.required_capabilities},
      {"concurrency_key", identity.concurrency_key},
      {"lease_id", identity.lease_id},
      {"fencing_token", identity.fencing_token},
  };
  if (identity.host_grant) {
    expected_launch_payload["host_grant"] = encode_json(*identity.host_grant);
  }
  if (!launch || launch->event_type != "worker.launch_requested" ||
      launch->run_id != identity.run_id ||
      launch->node_id != identity.node_id ||
      launch->attempt_id != identity.attempt_id ||
      launch->run_revision != event.run_revision ||
      launch->payload != expected_launch_payload) {
    throw std::invalid_argument(
        "host launch binding has no matching durable launch request");
  }
  Statement lease(database_, R"sql(
    SELECT 1 FROM resource_leases
    WHERE concurrency_key=? AND owner_run_id=? AND lease_id=?
      AND fencing_token=? AND clock_domain='boottime/v1' AND boot_id=?
      AND expires_boottime_ns>? AND released_wall_time_ns IS NULL
      AND NOT EXISTS(
        SELECT 1 FROM resource_lease_releases AS release
        WHERE release.concurrency_key=resource_leases.concurrency_key
          AND release.owner_run_id=resource_leases.owner_run_id
          AND release.lease_id=resource_leases.lease_id
          AND release.fencing_token=resource_leases.fencing_token
      )
  )sql");
  bind_text(lease.get(), 1, identity.concurrency_key);
  bind_text(lease.get(), 2, identity.run_id);
  bind_text(lease.get(), 3, identity.lease_id);
  bind_integer(lease.get(), 4,
               checked_integer(identity.fencing_token, "fencing_token"));
  bind_text(lease.get(), 5, now.boot_id);
  bind_integer(lease.get(), 6, now.boot.nanoseconds);
  if (sqlite3_step(lease.get()) != SQLITE_ROW) {
    throw OperationPreconditionError(
        "host launch binding no longer owns its active lease");
  }
  require_live_host_grant_claim(
      identity.host_grant, identity.run_id, identity.concurrency_key,
      identity.lease_id, identity.fencing_token);
  Statement conflicting(database_, R"sql(
    SELECT 1 FROM events
    WHERE run_id=? AND event_type='worker.launch_bound'
      AND node_id=? AND attempt_id=? LIMIT 1
  )sql");
  bind_text(conflicting.get(), 1, identity.run_id);
  bind_text(conflicting.get(), 2, identity.node_id);
  bind_text(conflicting.get(), 3, identity.attempt_id);
  if (sqlite3_step(conflicting.get()) == SQLITE_ROW) {
    throw std::invalid_argument(
        "worker attempt already has another host launch binding");
  }
  append_uncommitted(event);
  transaction.commit();
  return true;
}

std::optional<ResolvedLaunchSpec> Journal::launch_binding(
    const std::string& launch_event_id) const {
  if (launch_event_id.empty()) {
    throw std::invalid_argument("launch event identity must not be empty");
  }
  const auto bound = event(launch_event_id + ":bound");
  if (!bound) return std::nullopt;
  if (bound->event_type != "worker.launch_bound" ||
      bound->event_id != launch_event_id + ":bound" ||
      bound->event_version != 1U || bound->worker_sequence != 0U ||
      bound->payload.value("cause_event_id", std::string{}) !=
          launch_event_id ||
      !bound->payload.contains("spec")) {
    throw std::runtime_error("durable host launch binding is malformed");
  }
  ResolvedLaunchSpec spec =
      resolved_launch_spec_from_json(bound->payload.at("spec"));
  if (spec.identity.launch_event_id != launch_event_id ||
      spec.identity.run_id != bound->run_id ||
      spec.identity.node_id != bound->node_id ||
      spec.identity.attempt_id != bound->attempt_id) {
    throw std::runtime_error(
        "durable host launch binding disagrees with its event envelope");
  }
  return spec;
}

bool Journal::bind_worker_invocation(
    const WorkerInvocationSpec& invocation,
    const WorkerSessionIdentity& identity, const AuthorityTimeSample& now,
    const Event& event) {
  const std::string canonical_json =
      worker_invocation_canonical_json(invocation);
  const nlohmann::json expected_payload{
      {"canonical_invocation_json", canonical_json},
      {"dispatch_id", invocation.dispatch_id},
      {"invocation_digest", invocation.invocation_digest},
  };
  require_authority_time(now);
  if (invocation.run_id != identity.run_id ||
      invocation.node_id != identity.node_id ||
      invocation.attempt_id != identity.attempt_id ||
      event.event_id != invocation.dispatch_id + ":invocation" ||
      event.run_id != invocation.run_id ||
      event.node_id != invocation.node_id ||
      event.attempt_id != invocation.attempt_id ||
      event.run_revision == 0U ||
      event.plan_revision != invocation.plan_revision ||
      event.worker_sequence != 0U ||
      event.event_type != "worker.invocation_bound" ||
      event.event_version != 1U || event.monotonic_time_ns != 0U ||
      event.optimizer_step || event.payload != expected_payload) {
    throw std::invalid_argument(
        "worker invocation binding or event is malformed");
  }
  Transaction transaction(database_);
  std::string chain_reason;
  if (!verify_chain(&chain_reason))
    throw std::runtime_error("refusing worker invocation binding: " +
                             chain_reason);
  const auto stored = this->event(event.event_id);
  if (stored) {
    Event replay = event;
    replay.wall_time_ns = stored->wall_time_ns;
    replay.monotonic_time_ns = stored->monotonic_time_ns;
    if (event_json(*stored) != event_json(replay))
      throw std::invalid_argument(
          "worker invocation retry differs from durable evidence");
    transaction.commit();
    return false;
  }

  Statement projection(database_, R"sql(
    SELECT observed_state, run_revision, current_node_id, current_attempt_id
    FROM run_projection WHERE run_id=?
  )sql");
  bind_text(projection.get(), 1, identity.run_id);
  if (sqlite3_step(projection.get()) != SQLITE_ROW ||
      column_text(projection.get(), 0) != "running" ||
      static_cast<std::uint64_t>(sqlite3_column_int64(projection.get(), 1)) !=
          event.run_revision ||
      column_text(projection.get(), 2) != identity.node_id ||
      column_text(projection.get(), 3) != identity.attempt_id) {
    throw OperationPreconditionError(
        "worker invocation is stale for the active run projection");
  }
  Statement dispatch(database_, R"sql(
    SELECT run_revision, plan_revision, component, operation, status
    FROM node_dispatches WHERE dispatch_id=?
  )sql");
  bind_text(dispatch.get(), 1, invocation.dispatch_id);
  if (sqlite3_step(dispatch.get()) != SQLITE_ROW ||
      static_cast<std::uint64_t>(sqlite3_column_int64(dispatch.get(), 0)) !=
          event.run_revision ||
      static_cast<std::uint64_t>(sqlite3_column_int64(dispatch.get(), 1)) !=
          invocation.plan_revision ||
      column_text(dispatch.get(), 2).empty() ||
      column_text(dispatch.get(), 3) != invocation.adapter.operation ||
      column_text(dispatch.get(), 4) != "prepared") {
    throw OperationPreconditionError(
        "worker invocation has no exact prepared dispatch");
  }
  const std::string launch_id = identity.run_id + ":worker-launch:" +
                                identity.node_id + ":" + identity.attempt_id;
  const auto binding = launch_binding(launch_id);
  if (!binding || binding->identity.launch_nonce != identity.launch_nonce ||
      binding->identity.concurrency_key != identity.concurrency_key ||
      binding->identity.lease_id != identity.lease_id ||
      binding->identity.fencing_token != identity.fencing_token ||
      invocation.host_id != binding->identity.host.host_id ||
      binding->identity.adapter_key != invocation.adapter) {
    throw OperationPreconditionError(
        "worker invocation has no exact durable launch binding");
  }
  require_live_host_grant_claim(
      binding->identity.host_grant, identity.run_id,
      identity.concurrency_key, identity.lease_id, identity.fencing_token);
  Statement lease(database_, R"sql(
    SELECT boot_id, expires_boottime_ns, released_wall_time_ns
    FROM resource_leases WHERE concurrency_key=? AND owner_run_id=?
      AND lease_id=? AND fencing_token=? AND clock_domain='boottime/v1'
  )sql");
  bind_text(lease.get(), 1, identity.concurrency_key);
  bind_text(lease.get(), 2, identity.run_id);
  bind_text(lease.get(), 3, identity.lease_id);
  bind_integer(lease.get(), 4,
               checked_integer(identity.fencing_token, "fencing_token"));
  if (sqlite3_step(lease.get()) != SQLITE_ROW ||
      column_text(lease.get(), 0) != now.boot_id ||
      sqlite3_column_int64(lease.get(), 1) <= now.boot.nanoseconds ||
      sqlite3_column_type(lease.get(), 2) != SQLITE_NULL) {
    throw OperationPreconditionError(
        "worker invocation lost its active lease fence");
  }
  Statement sequenced(database_, R"sql(
    SELECT 1 FROM events WHERE run_id=? AND node_id=? AND attempt_id=?
      AND worker_sequence>0 LIMIT 1
  )sql");
  bind_text(sequenced.get(), 1, identity.run_id);
  bind_text(sequenced.get(), 2, identity.node_id);
  bind_text(sequenced.get(), 3, identity.attempt_id);
  if (sqlite3_step(sequenced.get()) == SQLITE_ROW)
    throw OperationPreconditionError(
        "worker invocation cannot be changed after worker observations");
  append_uncommitted(event);
  transaction.commit();
  return true;
}

std::optional<WorkerInvocationSpec> Journal::worker_invocation(
    const std::string& dispatch_id) const {
  if (dispatch_id.empty())
    throw std::invalid_argument(
        "worker invocation lookup requires a dispatch identity");
  const auto bound = event(dispatch_id + ":invocation");
  if (!bound) return std::nullopt;
  if (bound->event_type != "worker.invocation_bound" ||
      bound->event_version != 1U || bound->worker_sequence != 0U ||
      bound->event_id != dispatch_id + ":invocation" ||
      bound->payload.size() != 3U ||
      bound->payload.value("dispatch_id", std::string{}) != dispatch_id ||
      !bound->payload.contains("canonical_invocation_json") ||
      !bound->payload.at("canonical_invocation_json").is_string() ||
      !bound->payload.contains("invocation_digest") ||
      !bound->payload.at("invocation_digest").is_string()) {
    throw std::runtime_error("durable worker invocation is malformed");
  }
  const WorkerInvocationSpec invocation =
      worker_invocation_from_canonical_json(
          bound->payload.at("canonical_invocation_json").get_ref<
              const std::string&>());
  if (invocation.dispatch_id != dispatch_id ||
      invocation.run_id != bound->run_id ||
      invocation.node_id != bound->node_id ||
      invocation.attempt_id != bound->attempt_id ||
      invocation.plan_revision != bound->plan_revision ||
      invocation.invocation_digest !=
          bound->payload.at("invocation_digest").get<std::string>()) {
    throw std::runtime_error(
        "durable worker invocation disagrees with its event envelope");
  }
  return invocation;
}

WorkerReadinessDisposition Journal::accept_worker_ready(
    const WorkerLaunchTicket& launch, const WorkerHelloEvidence& hello,
    const AuthorityTimeSample& now, const std::vector<Event>& events) {
  require_lease_identity(launch.concurrency_key, launch.run_id,
                         launch.lease_id);
  require_authority_time(now);
  if (events.size() != 3U || hello.run_id != launch.run_id ||
      hello.node_id != launch.node_id || hello.attempt_id != launch.attempt_id ||
      hello.launch_nonce != launch.launch_nonce || hello.adapter != launch.adapter ||
      hello.adapter_version != launch.adapter_version ||
      hello.code_fingerprint != launch.code_fingerprint ||
      hello.concurrency_key != launch.concurrency_key ||
      hello.lease_id != launch.lease_id ||
      hello.fencing_token != launch.fencing_token ||
      hello.last_acked_controller_sequence != 0U ||
      !std::ranges::is_sorted(hello.capabilities) ||
      std::ranges::adjacent_find(hello.capabilities) != hello.capabilities.end() ||
      !std::ranges::includes(hello.capabilities,
                             launch.required_capabilities)) {
    throw std::invalid_argument("worker hello disagrees with its launch ticket");
  }
  const Event& ready = events[0];
  const Event& running = events[1];
  const Event& entered = events[2];
  if (ready.run_id != launch.run_id || ready.node_id != launch.node_id ||
      ready.attempt_id != launch.attempt_id || ready.worker_sequence != 0 ||
      ready.event_type != "worker.ready" ||
      running.run_id != launch.run_id || running.node_id != launch.node_id ||
      running.attempt_id != launch.attempt_id || running.worker_sequence != 0 ||
      running.event_type != "run.observed_state_changed" ||
      running.run_revision != ready.run_revision + 1U ||
      running.payload.value("state", std::string{}) != "running" ||
      running.payload.value("cause_event_id", std::string{}) != ready.event_id ||
      entered.run_id != launch.run_id || entered.node_id != launch.node_id ||
      entered.attempt_id != launch.attempt_id || entered.worker_sequence != 0 ||
      entered.event_type != "node.entered" ||
      entered.run_revision != running.run_revision) {
    throw std::invalid_argument("worker readiness event sequence is malformed");
  }

  Transaction transaction(database_);
  std::string chain_reason;
  if (!verify_chain(&chain_reason)) {
    throw std::runtime_error("refusing worker readiness: " + chain_reason);
  }
  require_live_host_grant_claim(
      launch.host_grant, launch.run_id, launch.concurrency_key,
      launch.lease_id, launch.fencing_token);
  Statement projection(database_, R"sql(
    SELECT desired_state, observed_state, run_revision,
           current_node_id, current_attempt_id
    FROM run_projection WHERE run_id=?
  )sql");
  bind_text(projection.get(), 1, launch.run_id);
  if (sqlite3_step(projection.get()) != SQLITE_ROW ||
      column_text(projection.get(), 0) != "running") {
    throw std::invalid_argument("worker readiness run projection is unavailable");
  }
  const std::string observed_state = column_text(projection.get(), 1);
  const auto run_revision =
      static_cast<std::uint64_t>(sqlite3_column_int64(projection.get(), 2));
  const std::string current_node = column_text(projection.get(), 3);
  const std::string current_attempt = column_text(projection.get(), 4);

  const auto launch_event = event(ready.payload.value("cause_event_id", std::string{}));
  nlohmann::json expected_launch_payload{
      {"launch_nonce", launch.launch_nonce},
      {"adapter", launch.adapter},
      {"adapter_version", launch.adapter_version},
      {"code_fingerprint", launch.code_fingerprint},
      {"required_capabilities", launch.required_capabilities},
      {"concurrency_key", launch.concurrency_key},
      {"lease_id", launch.lease_id},
      {"fencing_token", launch.fencing_token},
  };
  if (launch.host_grant) {
    expected_launch_payload["host_grant"] = encode_json(*launch.host_grant);
  }
  if (!launch_event || launch_event->event_type != "worker.launch_requested" ||
      launch_event->run_id != launch.run_id ||
      launch_event->node_id != launch.node_id ||
      launch_event->attempt_id != launch.attempt_id ||
      launch_event->payload != expected_launch_payload ||
      launch_event->run_revision != ready.run_revision) {
    throw std::invalid_argument("worker readiness has no matching durable launch");
  }
  const nlohmann::json expected_ready_payload{
      {"cause_event_id", launch_event->event_id},
      {"launch_nonce", hello.launch_nonce},
      {"adapter", hello.adapter},
      {"adapter_version", hello.adapter_version},
      {"code_fingerprint", hello.code_fingerprint},
      {"capabilities", hello.capabilities},
      {"last_acked_controller_sequence", hello.last_acked_controller_sequence},
      {"concurrency_key", hello.concurrency_key},
      {"lease_id", hello.lease_id},
      {"fencing_token", hello.fencing_token},
  };
  const nlohmann::json expected_running_payload{
      {"state", "running"},
      {"cause_event_id", ready.event_id},
      {"launch_nonce", launch.launch_nonce},
  };
  if (ready.payload != expected_ready_payload ||
      running.payload != expected_running_payload) {
    throw std::invalid_argument("worker readiness payload differs from verified hello");
  }
  Statement lease(database_, R"sql(
    SELECT owner_run_id, lease_id, fencing_token, clock_domain, boot_id,
           expires_boottime_ns, released_wall_time_ns
    FROM resource_leases WHERE concurrency_key=?
      AND NOT EXISTS(
        SELECT 1 FROM resource_lease_releases AS release
        WHERE release.concurrency_key=resource_leases.concurrency_key
          AND release.owner_run_id=resource_leases.owner_run_id
          AND release.lease_id=resource_leases.lease_id
          AND release.fencing_token=resource_leases.fencing_token
      )
  )sql");
  bind_text(lease.get(), 1, launch.concurrency_key);
  if (sqlite3_step(lease.get()) != SQLITE_ROW ||
      column_text(lease.get(), 0) != launch.run_id ||
      column_text(lease.get(), 1) != launch.lease_id ||
      static_cast<std::uint64_t>(sqlite3_column_int64(lease.get(), 2)) !=
          launch.fencing_token ||
      column_text(lease.get(), 3) != ResourceLease::kBootTimeDomain ||
      column_text(lease.get(), 4) != now.boot_id ||
      sqlite3_column_int64(lease.get(), 5) <= now.boot.nanoseconds ||
      sqlite3_column_type(lease.get(), 6) != SQLITE_NULL) {
    throw OperationPreconditionError(
        "worker readiness no longer owns its active lease");
  }

  if (observed_state == "running") {
    const auto stored_ready = event(ready.event_id);
    const auto stored_running = event(running.event_id);
    const auto stored_entered = event(entered.event_id);
    Event replay_ready = ready;
    Event replay_running = running;
    Event replay_entered = entered;
    if (stored_ready && stored_running && stored_entered) {
      replay_ready.wall_time_ns = stored_ready->wall_time_ns;
      replay_ready.monotonic_time_ns = stored_ready->monotonic_time_ns;
      replay_running.wall_time_ns = stored_running->wall_time_ns;
      replay_running.monotonic_time_ns = stored_running->monotonic_time_ns;
      replay_entered.wall_time_ns = stored_entered->wall_time_ns;
      replay_entered.monotonic_time_ns = stored_entered->monotonic_time_ns;
    }
    if (run_revision != entered.run_revision || current_node != launch.node_id ||
        current_attempt != launch.attempt_id || !stored_ready || !stored_running ||
        !stored_entered || event_json(*stored_ready) != event_json(replay_ready) ||
        event_json(*stored_running) != event_json(replay_running) ||
        event_json(*stored_entered) != event_json(replay_entered)) {
      throw std::invalid_argument("worker hello retry differs from durable readiness");
    }
    transaction.commit();
    return WorkerReadinessDisposition::replayed;
  }
  if (observed_state != "acquiring" || run_revision != ready.run_revision ||
      !current_node.empty() || !current_attempt.empty() || event(ready.event_id) ||
      event(running.event_id) || event(entered.event_id)) {
    throw std::invalid_argument("worker readiness disagrees with acquiring run");
  }
  for (const Event& readiness_event : events) {
    append_uncommitted(readiness_event);
  }
  transaction.commit();
  return WorkerReadinessDisposition::accepted;
}

bool Journal::renew_lease(const std::string& concurrency_key, const std::string& owner_run_id,
                          const std::string& lease_id, std::uint64_t fencing_token,
                          const AuthorityTimeSample& now,
                          std::int64_t timeout_ns) {
  require_lease_identity(concurrency_key, owner_run_id, lease_id);
  require_authority_time(now);
  const auto active = active_lease(concurrency_key, now);
  if (!active || active->owner_run_id != owner_run_id ||
      active->lease_id != lease_id || active->fencing_token != fencing_token) {
    return false;
  }
  const LeaseRenewalResult result = renew_lease_exact(*active, now, timeout_ns);
  return result.status != LeaseRenewalStatus::not_owned;
}

LeaseRenewalResult Journal::renew_lease_exact(
    const ResourceLease& expected, const AuthorityTimeSample& now,
    std::int64_t timeout_ns) {
  require_lease_identity(expected.concurrency_key, expected.owner_run_id,
                         expected.lease_id);
  require_authority_time(now);
  if (expected.fencing_token == 0U ||
      expected.clock_domain != ResourceLease::kBootTimeDomain ||
      !canonical_boot_id(expected.boot_id) ||
      expected.boot_id != now.boot_id || expected.acquired_boottime_ns < 0 ||
      expected.expires_boottime_ns <= expected.acquired_boottime_ns ||
      expected.acquired_wall_time_ns < 0 ||
      expected.expires_wall_time_ns < 0) {
    throw std::invalid_argument(
        "exact lease renewal requires a canonical boot-scoped lease");
  }
  const std::int64_t new_expires_boottime_ns =
      lease_expiration(now.boot.nanoseconds, timeout_ns);
  const std::int64_t new_expires_wall_time_ns =
      lease_expiration(now.wall.nanoseconds, timeout_ns);
  if (expected.expires_boottime_ns <= now.boot.nanoseconds ||
      new_expires_boottime_ns <= expected.expires_boottime_ns) {
    return {.status = LeaseRenewalStatus::not_owned, .receipt = std::nullopt};
  }
  const LeaseRenewalReceipt requested{
      .concurrency_key = expected.concurrency_key,
      .owner_run_id = expected.owner_run_id,
      .lease_id = expected.lease_id,
      .fencing_token = expected.fencing_token,
      .clock_domain = expected.clock_domain,
      .boot_id = expected.boot_id,
      .acquired_boottime_ns = expected.acquired_boottime_ns,
      .acquired_wall_time_ns = expected.acquired_wall_time_ns,
      .prior_expires_boottime_ns = expected.expires_boottime_ns,
      .new_expires_boottime_ns = new_expires_boottime_ns,
      .prior_expires_wall_time_ns = expected.expires_wall_time_ns,
      .new_expires_wall_time_ns = new_expires_wall_time_ns,
      .renewed_boottime_ns = now.boot.nanoseconds,
      .renewed_wall_time_ns = now.wall.nanoseconds,
  };

  Transaction transaction(database_);
  Statement replay(database_, R"sql(
    SELECT concurrency_key, owner_run_id, lease_id, fencing_token,
           clock_domain, boot_id, acquired_boottime_ns,
           acquired_wall_time_ns, prior_expires_boottime_ns,
           new_expires_boottime_ns, prior_expires_wall_time_ns,
           new_expires_wall_time_ns, renewed_boottime_ns,
           renewed_wall_time_ns
    FROM resource_lease_renewals
    WHERE concurrency_key=? AND lease_id=? AND fencing_token=?
      AND prior_expires_boottime_ns=?
  )sql");
  bind_text(replay.get(), 1, expected.concurrency_key);
  bind_text(replay.get(), 2, expected.lease_id);
  bind_integer(replay.get(), 3,
               checked_integer(expected.fencing_token, "fencing_token"));
  bind_integer(replay.get(), 4, expected.expires_boottime_ns);
  const int replay_status = sqlite3_step(replay.get());
  if (replay_status == SQLITE_ROW) {
    const LeaseRenewalReceipt stored = renewal_receipt_from_row(replay.get());
    if (stored != requested) {
      throw std::runtime_error(
          "lease renewal replay conflicts with its durable receipt");
    }
    transaction.commit();
    return {.status = LeaseRenewalStatus::replayed, .receipt = stored};
  }
  if (replay_status != SQLITE_DONE) {
    throw std::runtime_error("could not inspect lease renewal replay receipt: " +
                             std::string(sqlite3_errmsg(database_)));
  }

  Statement update(database_, R"sql(
    UPDATE resource_leases
    SET expires_boottime_ns=?, expires_wall_time_ns=?
    WHERE concurrency_key=? AND owner_run_id=? AND lease_id=? AND fencing_token=?
      AND clock_domain='boottime/v1' AND boot_id=?
      AND acquired_boottime_ns=? AND acquired_wall_time_ns=?
      AND expires_boottime_ns=? AND expires_wall_time_ns=?
      AND released_wall_time_ns IS NULL AND expires_boottime_ns>?
      AND NOT EXISTS(
        SELECT 1 FROM resource_lease_releases AS release
        WHERE release.concurrency_key=resource_leases.concurrency_key
          AND release.owner_run_id=resource_leases.owner_run_id
          AND release.lease_id=resource_leases.lease_id
          AND release.fencing_token=resource_leases.fencing_token
      )
  )sql");
  bind_integer(update.get(), 1, new_expires_boottime_ns);
  bind_integer(update.get(), 2, new_expires_wall_time_ns);
  bind_text(update.get(), 3, expected.concurrency_key);
  bind_text(update.get(), 4, expected.owner_run_id);
  bind_text(update.get(), 5, expected.lease_id);
  bind_integer(update.get(), 6,
               checked_integer(expected.fencing_token, "fencing_token"));
  bind_text(update.get(), 7, expected.boot_id);
  bind_integer(update.get(), 8, expected.acquired_boottime_ns);
  bind_integer(update.get(), 9, expected.acquired_wall_time_ns);
  bind_integer(update.get(), 10, expected.expires_boottime_ns);
  bind_integer(update.get(), 11, expected.expires_wall_time_ns);
  bind_integer(update.get(), 12, now.boot.nanoseconds);
  require_done(database_, update.get(), "renew resource lease");
  const bool renewed = sqlite3_changes(database_) == 1;
  if (!renewed) {
    transaction.commit();
    return {.status = LeaseRenewalStatus::not_owned, .receipt = std::nullopt};
  }

  Statement receipt(database_, R"sql(
    INSERT INTO resource_lease_renewals(
      concurrency_key, owner_run_id, lease_id, fencing_token, clock_domain,
      boot_id, acquired_boottime_ns, acquired_wall_time_ns,
      prior_expires_boottime_ns, new_expires_boottime_ns,
      prior_expires_wall_time_ns, new_expires_wall_time_ns,
      renewed_boottime_ns, renewed_wall_time_ns
    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  )sql");
  bind_text(receipt.get(), 1, requested.concurrency_key);
  bind_text(receipt.get(), 2, requested.owner_run_id);
  bind_text(receipt.get(), 3, requested.lease_id);
  bind_integer(receipt.get(), 4,
               checked_integer(requested.fencing_token, "fencing_token"));
  bind_text(receipt.get(), 5, requested.clock_domain);
  bind_text(receipt.get(), 6, requested.boot_id);
  bind_integer(receipt.get(), 7, requested.acquired_boottime_ns);
  bind_integer(receipt.get(), 8, requested.acquired_wall_time_ns);
  bind_integer(receipt.get(), 9, requested.prior_expires_boottime_ns);
  bind_integer(receipt.get(), 10, requested.new_expires_boottime_ns);
  bind_integer(receipt.get(), 11, requested.prior_expires_wall_time_ns);
  bind_integer(receipt.get(), 12, requested.new_expires_wall_time_ns);
  bind_integer(receipt.get(), 13, requested.renewed_boottime_ns);
  bind_integer(receipt.get(), 14, requested.renewed_wall_time_ns);
  require_done(database_, receipt.get(), "record resource lease renewal");
  record_lease_authority_renewal_uncommitted(requested);
  transaction.commit();
  return {.status = LeaseRenewalStatus::renewed, .receipt = requested};
}

bool Journal::release_lease(const std::string& concurrency_key, const std::string& owner_run_id,
                            const std::string& lease_id, std::uint64_t fencing_token,
                            const AuthorityTimeSample& now) {
  require_lease_identity(concurrency_key, owner_run_id, lease_id);
  require_authority_time(now);
  Transaction transaction(database_);
  Statement update(database_, R"sql(
    UPDATE resource_leases SET released_wall_time_ns=?
    WHERE concurrency_key=? AND owner_run_id=? AND lease_id=? AND fencing_token=?
      AND clock_domain='boottime/v1' AND boot_id=?
      AND released_wall_time_ns IS NULL
      AND NOT EXISTS(
        SELECT 1 FROM resource_lease_releases AS release
        WHERE release.concurrency_key=resource_leases.concurrency_key
          AND release.owner_run_id=resource_leases.owner_run_id
          AND release.lease_id=resource_leases.lease_id
          AND release.fencing_token=resource_leases.fencing_token
      )
  )sql");
  bind_integer(update.get(), 1, now.wall.nanoseconds);
  bind_text(update.get(), 2, concurrency_key);
  bind_text(update.get(), 3, owner_run_id);
  bind_text(update.get(), 4, lease_id);
  bind_integer(update.get(), 5, checked_integer(fencing_token, "fencing_token"));
  bind_text(update.get(), 6, now.boot_id);
  require_done(database_, update.get(), "release resource lease");
  const bool released = sqlite3_changes(database_) == 1;
  if (released) {
    Statement receipt(database_, R"sql(
      INSERT INTO resource_lease_releases(
        concurrency_key, owner_run_id, lease_id, fencing_token, clock_domain,
        boot_id, released_wall_time_ns
      ) VALUES(?, ?, ?, ?, 'boottime/v1', ?, ?)
    )sql");
    bind_text(receipt.get(), 1, concurrency_key);
    bind_text(receipt.get(), 2, owner_run_id);
    bind_text(receipt.get(), 3, lease_id);
    bind_integer(receipt.get(), 4,
                 checked_integer(fencing_token, "fencing_token"));
    bind_text(receipt.get(), 5, now.boot_id);
    bind_integer(receipt.get(), 6, now.wall.nanoseconds);
    require_done(database_, receipt.get(), "record resource lease release");
    Statement released_lease(database_, R"sql(
      SELECT concurrency_key, owner_run_id, lease_id, fencing_token,
             clock_domain, boot_id, acquired_boottime_ns,
             expires_boottime_ns, acquired_wall_time_ns,
             expires_wall_time_ns, released_wall_time_ns
      FROM resource_leases WHERE concurrency_key=?
    )sql");
    bind_text(released_lease.get(), 1, concurrency_key);
    if (sqlite3_step(released_lease.get()) != SQLITE_ROW)
      throw std::runtime_error("released lease projection disappeared");
    record_lease_authority_release_uncommitted(
        lease_from_row(released_lease.get()), now.wall.nanoseconds);
  }
  transaction.commit();
  return released;
}

std::optional<ResourceLease> Journal::active_lease(const std::string& concurrency_key,
                                                   const AuthorityTimeSample& now) const {
  if (concurrency_key.empty()) {
    throw std::invalid_argument("lease concurrency_key must not be empty");
  }
  require_authority_time(now);
  Statement query(database_, R"sql(
    SELECT concurrency_key, owner_run_id, lease_id, fencing_token,
           clock_domain, boot_id, acquired_boottime_ns, expires_boottime_ns,
           acquired_wall_time_ns, expires_wall_time_ns
    FROM resource_leases
    WHERE concurrency_key=? AND clock_domain='boottime/v1' AND boot_id=?
      AND released_wall_time_ns IS NULL AND expires_boottime_ns>?
      AND NOT EXISTS(
        SELECT 1 FROM resource_lease_releases AS release
        WHERE release.concurrency_key=resource_leases.concurrency_key
          AND release.owner_run_id=resource_leases.owner_run_id
          AND release.lease_id=resource_leases.lease_id
          AND release.fencing_token=resource_leases.fencing_token
      )
  )sql");
  bind_text(query.get(), 1, concurrency_key);
  bind_text(query.get(), 2, now.boot_id);
  bind_integer(query.get(), 3, now.boot.nanoseconds);
  const int status = sqlite3_step(query.get());
  if (status == SQLITE_DONE) {
    return std::nullopt;
  }
  if (status != SQLITE_ROW) {
    throw std::runtime_error("could not read active resource lease: " +
                             std::string(sqlite3_errmsg(database_)));
  }
  return lease_from_row(query.get());
}

HostGrantSagaSnapshot Journal::record_host_resource_request(
    const ResourceBundleRequest& request, const AuthorityTimeSample& now) {
  validate_resource_request(request);
  require_authority_time(now);
  if (!expected_host_grant_authority_) {
    throw OperationPreconditionError(
        "host resource request requires a configured trusted local host epoch");
  }
  if (request.journal_id != journal_id()) {
    throw OperationPreconditionError(
        "host resource request targets a different journal authority");
  }
  const std::string canonical = resource_request_json(request).dump();
  Transaction transaction(database_);
  std::string authority_reason;
  if (!verify_chain(&authority_reason)) {
    throw std::runtime_error("refusing host request saga: " + authority_reason);
  }
  if (const auto existing = load_host_grant_saga(database_, request.request_id)) {
    if (existing->request != request) {
      throw OperationPreconditionError(
          "host resource request_id already has different content");
    }
    transaction.commit();
    return *existing;
  }
  Statement lease(database_, R"sql(
    SELECT concurrency_key FROM resource_leases
    WHERE owner_run_id=? AND lease_id=? AND fencing_token=?
      AND clock_domain='boottime/v1' AND boot_id=?
      AND expires_boottime_ns>? AND released_wall_time_ns IS NULL
      AND NOT EXISTS(
        SELECT 1 FROM resource_lease_releases AS release
        WHERE release.concurrency_key=resource_leases.concurrency_key
          AND release.owner_run_id=resource_leases.owner_run_id
          AND release.lease_id=resource_leases.lease_id
          AND release.fencing_token=resource_leases.fencing_token
      )
    ORDER BY concurrency_key
  )sql");
  bind_text(lease.get(), 1, request.run_id);
  bind_text(lease.get(), 2, request.logical_lease_id);
  bind_integer(lease.get(), 3,
               checked_integer(request.logical_fencing_token,
                               "logical_fencing_token"));
  bind_text(lease.get(), 4, now.boot_id);
  bind_integer(lease.get(), 5, now.boot.nanoseconds);
  if (sqlite3_step(lease.get()) != SQLITE_ROW) {
    throw OperationPreconditionError(
        "host resource request has no matching live logical lease");
  }
  const std::string concurrency_key = column_text(lease.get(), 0);
  if (sqlite3_step(lease.get()) != SQLITE_DONE) {
    throw OperationPreconditionError(
        "host resource request lease identity is ambiguous");
  }
  Statement insert(database_, R"sql(
    INSERT INTO host_resource_requests(
      request_id, journal_id, run_id, concurrency_key, logical_lease_id,
      logical_fencing_token, request_digest, canonical_request_json
    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
  )sql");
  bind_text(insert.get(), 1, request.request_id);
  bind_text(insert.get(), 2, request.journal_id);
  bind_text(insert.get(), 3, request.run_id);
  bind_text(insert.get(), 4, concurrency_key);
  bind_text(insert.get(), 5, request.logical_lease_id);
  bind_integer(insert.get(), 6,
               checked_integer(request.logical_fencing_token,
                               "logical_fencing_token"));
  bind_text(insert.get(), 7, request.canonical_request_digest);
  bind_text(insert.get(), 8, canonical);
  require_done(database_, insert.get(), "record host resource request intent");
  (void)append_uncommitted(host_saga_event(
      database_, "host-resource-request:" + request.request_id,
      "host.resource_request_recorded", request.run_id, now.wall.nanoseconds,
      static_cast<std::uint64_t>(now.boot.nanoseconds),
      {{"concurrency_key", concurrency_key},
       {"request", resource_request_json(request)}}), true);
  auto result = load_host_grant_saga(database_, request.request_id);
  if (!result) throw std::runtime_error("host resource request insert vanished");
  transaction.commit();
  return *result;
}

HostGrantSagaSnapshot Journal::record_host_grant_receipt(
    const ResourceBundleGrant& grant) {
  const auto bounded = [](const std::string& value) {
    return !value.empty() &&
           value.size() <= HostResourceBounds::maximum_identifier_bytes;
  };
  if (grant.fences.empty() ||
      grant.fences.size() > HostResourceBounds::maximum_bundle_count) {
    throw OperationPreconditionError("host grant fence count exceeds its bound");
  }
  if (!bounded(grant.api_version) || !bounded(grant.allocation_id) ||
      !bounded(grant.request_id) || !bounded(grant.request_digest) ||
      !bounded(grant.journal_id) || !bounded(grant.run_id) ||
      !bounded(grant.logical_lease_id) || !bounded(grant.host_id) ||
      !bounded(grant.boot_id) || !bounded(grant.broker_epoch) ||
      !bounded(grant.previous_receipt_digest) ||
      !bounded(grant.receipt_digest)) {
    throw OperationPreconditionError("host grant identity exceeds its bound");
  }
  if (!expected_host_grant_authority_ ||
      grant.host_id != expected_host_grant_authority_->host_id ||
      grant.boot_id != expected_host_grant_authority_->boot_id) {
    throw OperationPreconditionError(
        "host grant receipt disagrees with the trusted local host epoch");
  }
  validate_resource_fence_shape(grant.fences,
                                HostResourceBounds::maximum_bundle_count);
  const std::string canonical = resource_bundle_grant_json(grant).dump();
  Transaction transaction(database_);
  std::string authority_reason;
  if (!verify_chain(&authority_reason)) {
    throw std::runtime_error("refusing host grant copy: " + authority_reason);
  }
  auto saga = load_host_grant_saga(database_, grant.request_id);
  if (!saga) {
    throw OperationPreconditionError(
        "host grant has no durable journal request intent");
  }
  if (saga->grant) {
    if (*saga->grant != grant) {
      throw OperationPreconditionError(
          "host grant receipt diverges from its durable copy");
    }
    transaction.commit();
    return *saga;
  }
  if (saga->busy_outcome_digest) {
    throw OperationPreconditionError(
        "host grant conflicts with the durable busy outcome");
  }
  const auto& request = saga->request;
  if (grant.fences.size() != request.count ||
      grant.fences.size() > HostResourceBounds::maximum_bundle_count ||
      grant.request_digest != request.canonical_request_digest ||
      grant.journal_id != request.journal_id || grant.run_id != request.run_id ||
      grant.logical_lease_id != request.logical_lease_id ||
      grant.logical_fencing_token != request.logical_fencing_token) {
    throw OperationPreconditionError(
        "host grant does not exactly match its journal request intent");
  }
  Statement insert(database_, R"sql(
    INSERT INTO host_resource_grants(
      request_id, allocation_id, request_digest, grant_digest,
      canonical_grant_json
    ) VALUES(?, ?, ?, ?, ?)
  )sql");
  bind_text(insert.get(), 1, grant.request_id);
  bind_text(insert.get(), 2, grant.allocation_id);
  bind_text(insert.get(), 3, grant.request_digest);
  bind_text(insert.get(), 4, grant.receipt_digest);
  bind_text(insert.get(), 5, canonical);
  require_done(database_, insert.get(), "copy exact host grant receipt");
  (void)append_uncommitted(host_saga_event(
      database_, "host-resource-grant:" + grant.request_id,
      "host.resource_grant_recorded", grant.run_id, grant.granted_wall_time_ns,
      static_cast<std::uint64_t>(grant.granted_boottime_ns),
      {{"request_id", grant.request_id},
       {"grant", resource_bundle_grant_json(grant)}}), true);
  saga = load_host_grant_saga(database_, grant.request_id);
  transaction.commit();
  return *saga;
}

HostGrantSagaSnapshot Journal::record_host_busy_outcome(
    const std::string& request_id, const std::string& outcome_digest,
    const AuthorityTimeSample& now) {
  require_authority_time(now);
  const auto valid_digest = [](std::string_view value) {
    return value.size() == 71U && value.starts_with("sha256:") &&
           std::ranges::all_of(value.substr(7U), [](char character) {
             return (character >= '0' && character <= '9') ||
                    (character >= 'a' && character <= 'f');
           });
  };
  if (!valid_digest(outcome_digest)) {
    throw OperationPreconditionError("host busy outcome digest is invalid");
  }
  Transaction transaction(database_);
  std::string authority_reason;
  if (!verify_chain(&authority_reason)) {
    throw std::runtime_error("refusing host busy copy: " + authority_reason);
  }
  auto saga = load_host_grant_saga(database_, request_id);
  if (!saga) {
    throw OperationPreconditionError(
        "host busy outcome has no durable journal request intent");
  }
  if (saga->grant) {
    throw OperationPreconditionError(
        "host busy outcome conflicts with a durable grant");
  }
  if (saga->busy_outcome_digest) {
    if (*saga->busy_outcome_digest != outcome_digest) {
      throw OperationPreconditionError(
          "host busy outcome diverges from its durable copy");
    }
    transaction.commit();
    return *saga;
  }
  (void)append_uncommitted(host_saga_event(
      database_, "host-resource-busy:" + request_id,
      "host.resource_busy_recorded", saga->request.run_id,
      now.wall.nanoseconds, static_cast<std::uint64_t>(now.boot.nanoseconds),
      {{"request_id", request_id},
       {"request_digest", saga->request.canonical_request_digest},
       {"outcome_digest", outcome_digest}}), true);
  saga = load_host_grant_saga(database_, request_id);
  transaction.commit();
  return *saga;
}

HostGrantSagaSnapshot Journal::record_host_release_intent(
    const std::string& request_id, const ResourceReleaseRequest& release,
    const AuthorityTimeSample& now) {
  require_authority_time(now);
  const auto bounded = [](const std::string& value) {
    return !value.empty() &&
           value.size() <= HostResourceBounds::maximum_identifier_bytes;
  };
  if (!bounded(request_id) || !bounded(release.api_version) ||
      !bounded(release.release_request_id) ||
      !bounded(release.allocation_id) || !bounded(release.grant_digest) ||
      !bounded(release.journal_id) || !bounded(release.run_id) ||
      !bounded(release.logical_lease_id) ||
      !bounded(release.canonical_request_digest)) {
    throw OperationPreconditionError(
        "host release request identity exceeds its bound");
  }
  const std::string canonical = resource_release_request_json(release).dump();
  Transaction transaction(database_);
  std::string authority_reason;
  if (!verify_chain(&authority_reason)) {
    throw std::runtime_error("refusing host release intent: " + authority_reason);
  }
  auto saga = load_host_grant_saga(database_, request_id);
  if (!saga || !saga->grant) {
    throw OperationPreconditionError(
        "host release has no durable exact grant receipt");
  }
  if (saga->release_intent) {
    if (*saga->release_intent != release) {
      throw OperationPreconditionError(
          "host release intent diverges from its durable copy");
    }
    transaction.commit();
    return *saga;
  }
  const auto& request = saga->request;
  const auto& grant = *saga->grant;
  if (release.allocation_id != grant.allocation_id ||
      release.grant_digest != grant.receipt_digest ||
      release.journal_id != request.journal_id ||
      release.run_id != request.run_id ||
      release.logical_lease_id != request.logical_lease_id ||
      release.logical_fencing_token != request.logical_fencing_token) {
    throw OperationPreconditionError(
        "host release intent does not exactly match the durable grant");
  }
  Statement insert(database_, R"sql(
    INSERT INTO host_resource_release_intents(
      release_request_id, request_id, allocation_id, grant_digest,
      release_request_digest, canonical_release_request_json
    ) VALUES(?, ?, ?, ?, ?, ?)
  )sql");
  bind_text(insert.get(), 1, release.release_request_id);
  bind_text(insert.get(), 2, request_id);
  bind_text(insert.get(), 3, release.allocation_id);
  bind_text(insert.get(), 4, release.grant_digest);
  bind_text(insert.get(), 5, release.canonical_request_digest);
  bind_text(insert.get(), 6, canonical);
  require_done(database_, insert.get(), "record host release intent");
  (void)append_uncommitted(host_saga_event(
      database_, "host-resource-release-intent:" + request_id,
      "host.resource_release_intent_recorded", release.run_id,
      now.wall.nanoseconds, static_cast<std::uint64_t>(now.boot.nanoseconds),
      {{"request_id", request_id},
       {"release", resource_release_request_json(release)}}),
      true);
  saga = load_host_grant_saga(database_, request_id);
  transaction.commit();
  return *saga;
}

HostGrantSagaSnapshot Journal::record_host_release_receipt(
    const std::string& request_id, const ResourceReleaseReceipt& receipt) {
  const auto bounded = [](const std::string& value) {
    return !value.empty() &&
           value.size() <= HostResourceBounds::maximum_identifier_bytes;
  };
  if (!bounded(request_id) || !bounded(receipt.api_version) ||
      !bounded(receipt.release_request_id) ||
      !bounded(receipt.release_request_digest) ||
      !bounded(receipt.allocation_id) || !bounded(receipt.grant_digest) ||
      !bounded(receipt.host_id) || !bounded(receipt.boot_id) ||
      !bounded(receipt.broker_epoch) ||
      !bounded(receipt.previous_receipt_digest) ||
      !bounded(receipt.receipt_digest)) {
    throw OperationPreconditionError(
        "host release receipt identity exceeds its bound");
  }
  const std::string canonical = resource_release_receipt_json(receipt).dump();
  Transaction transaction(database_);
  std::string authority_reason;
  if (!verify_chain(&authority_reason)) {
    throw std::runtime_error("refusing host release copy: " + authority_reason);
  }
  auto saga = load_host_grant_saga(database_, request_id);
  if (!saga || !saga->grant || !saga->release_intent) {
    throw OperationPreconditionError(
        "host release receipt has no durable release intent");
  }
  if (saga->release_receipt) {
    if (*saga->release_receipt != receipt) {
      throw OperationPreconditionError(
          "host release receipt diverges from its durable copy");
    }
    transaction.commit();
    return *saga;
  }
  const auto& grant = *saga->grant;
  const auto& intent = *saga->release_intent;
  if (receipt.release_request_id != intent.release_request_id ||
      receipt.release_request_digest != intent.canonical_request_digest ||
      receipt.allocation_id != grant.allocation_id ||
      receipt.grant_digest != grant.receipt_digest ||
      receipt.host_id != grant.host_id || receipt.boot_id != grant.boot_id ||
      receipt.broker_epoch != grant.broker_epoch) {
    throw OperationPreconditionError(
        "host release receipt does not exactly match its durable intent");
  }
  Statement insert(database_, R"sql(
    INSERT INTO host_resource_release_receipts(
      release_request_id, request_id, release_receipt_digest,
      canonical_release_receipt_json
    ) VALUES(?, ?, ?, ?)
  )sql");
  bind_text(insert.get(), 1, receipt.release_request_id);
  bind_text(insert.get(), 2, request_id);
  bind_text(insert.get(), 3, receipt.receipt_digest);
  bind_text(insert.get(), 4, canonical);
  require_done(database_, insert.get(), "copy exact host release receipt");
  (void)append_uncommitted(host_saga_event(
      database_, "host-resource-release-receipt:" + request_id,
      "host.resource_release_receipt_recorded", saga->request.run_id,
      receipt.released_wall_time_ns,
      static_cast<std::uint64_t>(receipt.released_boottime_ns),
      {{"request_id", request_id},
       {"receipt", resource_release_receipt_json(receipt)}}), true);
  saga = load_host_grant_saga(database_, request_id);
  transaction.commit();
  return *saga;
}

std::optional<HostGrantSagaSnapshot> Journal::host_grant_saga(
    const std::string& request_id) const {
  auto snapshot = read_snapshot();
  (void)snapshot;
  return load_host_grant_saga(database_, request_id);
}

std::optional<JournalResourceMutationIdentity>
Journal::host_resource_mutation_identity(
    const std::string& request_or_release_id) const {
  if (request_or_release_id.empty()) {
    throw std::invalid_argument(
        "host resource mutation identity must not be empty");
  }
  auto snapshot = read_snapshot();
  (void)snapshot;
  Statement query(database_, R"sql(
    SELECT request.request_id, request.run_id, request.concurrency_key,
           request.logical_lease_id, request.logical_fencing_token
    FROM host_resource_requests AS request
    LEFT JOIN host_resource_release_intents AS release
      ON release.request_id=request.request_id
    WHERE request.request_id=? OR release.release_request_id=?
  )sql");
  bind_text(query.get(), 1, request_or_release_id);
  bind_text(query.get(), 2, request_or_release_id);
  const int status = sqlite3_step(query.get());
  if (status == SQLITE_DONE) return std::nullopt;
  if (status != SQLITE_ROW ||
      sqlite3_column_type(query.get(), 4) != SQLITE_INTEGER ||
      sqlite3_column_int64(query.get(), 4) <= 0) {
    throw std::runtime_error(
        "host resource mutation identity is malformed");
  }
  JournalResourceMutationIdentity result{
      .request_id = column_text(query.get(), 0),
      .run_id = column_text(query.get(), 1),
      .concurrency_key = column_text(query.get(), 2),
      .logical_lease_id = column_text(query.get(), 3),
      .logical_fencing_token = static_cast<std::uint64_t>(
          sqlite3_column_int64(query.get(), 4)),
  };
  if (sqlite3_step(query.get()) != SQLITE_DONE) {
    throw std::runtime_error(
        "host resource mutation identity is ambiguous");
  }
  const auto saga = load_host_grant_saga(database_, result.request_id);
  if (!saga || saga->request.run_id != result.run_id ||
      saga->request.logical_lease_id != result.logical_lease_id ||
      saga->request.logical_fencing_token != result.logical_fencing_token) {
    throw std::runtime_error(
        "host resource mutation identity disagrees with its durable saga");
  }
  if (request_or_release_id != result.request_id &&
      (!saga->release_intent ||
       saga->release_intent->release_request_id != request_or_release_id)) {
    throw std::runtime_error(
        "host resource release identity disagrees with its durable saga");
  }
  return result;
}

namespace {

HostdProcessPreparedResult durable_prepared_result(
    HostdProcessPreparedResult result) {
  result.replayed = false;
  (void)hostd_process_prepared_canonical_json(result);
  return result;
}

HostdProcessCommittedResult durable_committed_result(
    HostdProcessCommittedResult result) {
  result.replayed = false;
  (void)hostd_process_committed_canonical_json(result);
  return result;
}

HostProcessExitResult durable_exit_result(HostProcessExitResult result) {
  result.replayed = false;
  (void)host_process_exit_receipt_json(result.receipt);
  return result;
}

void require_prepared_process_binding(
    const HostdProcessPrepareRequest& prepare,
    const HostdProcessPreparedResult& prepared) {
  (void)hostd_process_prepare_canonical_json(prepare);
  (void)hostd_process_prepared_canonical_json(prepared);
  const auto& launch = prepare.launch.identity;
  const auto& grant = prepare.grant;
  const auto& intent = prepared.intent;
  const auto& request = intent.request;
  const auto& spawn = prepared.spawn;
  if (request.launch_id != launch.launch_event_id ||
      request.allocation_id != grant.allocation_id ||
      request.grant_digest != grant.receipt_digest ||
      request.journal_id != grant.journal_id ||
      request.run_id != grant.run_id ||
      request.logical_lease_id != grant.logical_lease_id ||
      request.logical_fencing_token != grant.logical_fencing_token ||
      request.resolved_launch_digest != hostd_bound_process_launch_digest(
          prepare.launch, prepare.worker_bootstrap_digest) ||
      request.executable_path != launch.executable.source_path ||
      request.executable_digest != launch.executable.sealed_sha256 ||
      intent.host_id != grant.host_id || intent.boot_id != grant.boot_id ||
      intent.broker_epoch != grant.broker_epoch ||
      spawn.request.launch_id != request.launch_id ||
      spawn.request.launch_intent_digest != intent.receipt_digest ||
      spawn.request.boot_id != grant.boot_id ||
      spawn.request.cgroup_path != request.cgroup_path ||
      spawn.request.cgroup_device != request.cgroup_device ||
      spawn.request.cgroup_inode != request.cgroup_inode ||
      spawn.request.executable_digest != request.executable_digest ||
      spawn.host_id != grant.host_id ||
      spawn.broker_epoch != grant.broker_epoch) {
    throw OperationPreconditionError(
        "host process prepare receipt disagrees with its exact launch grant");
  }
}

HostdProcessCommitRequest expected_process_commit(
    const HostProcessSagaSnapshot& saga) {
  const auto& grant = saga.prepare.grant;
  return {.api_version = std::string(kHostdProcessCommitApiVersion),
          .launch_id = saga.prepare.launch.identity.launch_event_id,
          .allocation_id = grant.allocation_id,
          .grant_digest = grant.receipt_digest,
          .journal_id = grant.journal_id,
          .run_id = grant.run_id,
          .logical_lease_id = grant.logical_lease_id,
          .logical_fencing_token = grant.logical_fencing_token,
          .spawn_receipt_digest = saga.prepared.spawn.receipt_digest};
}

}  // namespace

std::optional<HostProcessSagaSnapshot> Journal::host_process_saga(
    const std::string& launch_id) const {
  if (launch_id.empty()) {
    throw std::invalid_argument("host process launch_id must not be empty");
  }
  const auto prepared_event = event(launch_id + ":host-process-prepared");
  const auto committed_event = event(launch_id + ":host-process-committed");
  const auto exited_event = event(launch_id + ":host-process-exited");
  if (!prepared_event) {
    if (committed_event || exited_event) {
      throw std::runtime_error(
          "host process terminal evidence exists without a durable prepare receipt");
    }
    return std::nullopt;
  }
  if (prepared_event->event_type != "host.process_prepared" ||
      prepared_event->payload.size() != 2U ||
      !prepared_event->payload.contains("prepare") ||
      !prepared_event->payload.contains("prepared")) {
    throw std::runtime_error("durable host process prepare event is malformed");
  }
  HostProcessSagaSnapshot result{
      .prepare = hostd_process_prepare_from_canonical_json(
          prepared_event->payload.at("prepare").dump()),
      .prepared = durable_prepared_result(
          hostd_process_prepared_from_canonical_json(
              prepared_event->payload.at("prepared").dump())),
      .commit = std::nullopt,
      .committed = std::nullopt,
      .exit_command = std::nullopt,
      .exited = std::nullopt,
  };
  require_prepared_process_binding(result.prepare, result.prepared);
  const auto& identity = result.prepare.launch.identity;
  if (identity.launch_event_id != launch_id ||
      prepared_event->run_id != identity.run_id ||
      prepared_event->node_id != identity.node_id ||
      prepared_event->attempt_id != identity.attempt_id) {
    throw std::runtime_error(
        "durable host process prepare event identity diverges");
  }
  if (!committed_event) {
    if (exited_event) {
      throw std::runtime_error(
          "host process exit exists without a durable exec commit");
    }
    return result;
  }
  if (committed_event->event_type != "host.process_committed" ||
      committed_event->payload.size() != 2U ||
      !committed_event->payload.contains("commit") ||
      !committed_event->payload.contains("committed") ||
      committed_event->run_id != identity.run_id ||
      committed_event->node_id != identity.node_id ||
      committed_event->attempt_id != identity.attempt_id) {
    throw std::runtime_error("durable host process commit event is malformed");
  }
  result.commit = hostd_process_commit_from_canonical_json(
      committed_event->payload.at("commit").dump());
  result.committed = durable_committed_result(
      hostd_process_committed_from_canonical_json(
          committed_event->payload.at("committed").dump()));
  const HostdProcessCommitRequest expected = expected_process_commit(result);
  if (*result.commit != expected ||
      result.committed->launch_id != expected.launch_id ||
      result.committed->spawn_receipt_digest !=
          expected.spawn_receipt_digest ||
      !result.committed->released_to_exec) {
    throw std::runtime_error(
        "durable host process commit disagrees with its prepare receipt");
  }
  if (!exited_event) return result;
  if (exited_event->event_type != "host.process_exited" ||
      exited_event->payload.size() != 2U ||
      !exited_event->payload.contains("exit_command") ||
      !exited_event->payload.contains("exited") ||
      exited_event->run_id != identity.run_id ||
      exited_event->node_id != identity.node_id ||
      exited_event->attempt_id != identity.attempt_id) {
    throw std::runtime_error("durable host process exit event is malformed");
  }
  result.exit_command = hostd_process_exit_from_canonical_json(
      exited_event->payload.at("exit_command").dump());
  result.exited = durable_exit_result(HostProcessExitResult{
      .receipt = host_process_exit_receipt_from_json(
          exited_event->payload.at("exited")),
      .replayed = false,
  });
  const auto& command = *result.exit_command;
  const auto& receipt = result.exited->receipt;
  const auto& spawn = result.prepared.spawn;
  const auto& grant = result.prepare.grant;
  if (command.launch_id != launch_id ||
      command.allocation_id != grant.allocation_id ||
      command.grant_digest != grant.receipt_digest ||
      command.journal_id != grant.journal_id ||
      command.run_id != grant.run_id ||
      command.logical_lease_id != grant.logical_lease_id ||
      command.logical_fencing_token != grant.logical_fencing_token ||
      command.spawn_receipt_digest != spawn.receipt_digest ||
      receipt.request.exit_request_id != command.exit_request_id ||
      receipt.request.launch_id != launch_id ||
      receipt.request.spawn_receipt_digest != spawn.receipt_digest ||
      receipt.request.host_pid != spawn.request.host_pid ||
      receipt.request.process_starttime_ticks !=
          spawn.request.process_starttime_ticks ||
      receipt.request.cgroup_path != spawn.request.cgroup_path ||
      receipt.request.cgroup_device != spawn.request.cgroup_device ||
      receipt.request.cgroup_inode != spawn.request.cgroup_inode ||
      receipt.host_id != grant.host_id || receipt.boot_id != grant.boot_id ||
      receipt.broker_epoch != grant.broker_epoch) {
    throw std::runtime_error(
        "durable host process exit disagrees with its launch receipt");
  }
  return result;
}

HostProcessSagaSnapshot Journal::record_host_process_prepared(
    const HostdProcessPrepareRequest& request,
    const HostdProcessPreparedResult& supplied_result,
    const AuthorityTimeSample& now) {
  require_authority_time(now);
  const HostdProcessPreparedResult result =
      durable_prepared_result(supplied_result);
  require_prepared_process_binding(request, result);
  const auto& identity = request.launch.identity;
  if (!identity.host_grant ||
      request.grant.journal_id != journal_id() ||
      request.grant.run_id != identity.run_id ||
      request.grant.logical_lease_id != identity.lease_id ||
      request.grant.logical_fencing_token != identity.fencing_token) {
    throw OperationPreconditionError(
        "host process prepare does not target this journal launch authority");
  }
  require_host_launch_eligible(*identity.host_grant, now);
  const auto binding = launch_binding(identity.launch_event_id);
  const auto grant_saga = host_grant_saga(identity.host_grant->request_id);
  if (!binding || *binding != request.launch || !grant_saga ||
      !grant_saga->grant || *grant_saga->grant != request.grant) {
    throw OperationPreconditionError(
        "host process prepare has no exact durable launch and grant binding");
  }
  if (const auto existing = host_process_saga(identity.launch_event_id)) {
    if (existing->prepare != request || existing->prepared != result) {
      throw OperationPreconditionError(
          "host process prepare replay diverges from its durable receipt");
    }
    return *existing;
  }
  Transaction transaction(database_);
  std::string reason;
  if (!verify_chain(&reason)) {
    throw std::runtime_error("refusing host process prepare copy: " + reason);
  }
  Event durable = host_saga_event(
      database_, identity.launch_event_id + ":host-process-prepared",
      "host.process_prepared", identity.run_id, now.wall.nanoseconds,
      static_cast<std::uint64_t>(now.boot.nanoseconds),
      {{"prepare", nlohmann::json::parse(
                       hostd_process_prepare_canonical_json(request))},
       {"prepared", nlohmann::json::parse(
                        hostd_process_prepared_canonical_json(result))}});
  durable.node_id = identity.node_id;
  durable.attempt_id = identity.attempt_id;
  (void)append_uncommitted(durable, true);
  transaction.commit();
  auto stored = host_process_saga(identity.launch_event_id);
  if (!stored) throw std::runtime_error("host process prepare copy vanished");
  return *stored;
}

HostProcessSagaSnapshot Journal::record_host_process_committed(
    const HostdProcessCommitRequest& request,
    const HostdProcessCommittedResult& supplied_result,
    const AuthorityTimeSample& now) {
  require_authority_time(now);
  (void)hostd_process_commit_canonical_json(request);
  const HostdProcessCommittedResult result =
      durable_committed_result(supplied_result);
  auto saga = host_process_saga(request.launch_id);
  if (!saga) {
    throw OperationPreconditionError(
        "host process commit requires a durable prepare receipt");
  }
  const HostdProcessCommitRequest expected = expected_process_commit(*saga);
  if (request != expected || result.launch_id != request.launch_id ||
      result.spawn_receipt_digest != request.spawn_receipt_digest ||
      !result.released_to_exec) {
    throw OperationPreconditionError(
        "host process commit disagrees with its durable prepare receipt");
  }
  const auto& identity = saga->prepare.launch.identity;
  if (!identity.host_grant) {
    throw OperationPreconditionError(
        "host process commit has no exact physical grant claim");
  }
  require_host_launch_eligible(*identity.host_grant, now);
  if (saga->committed) {
    if (*saga->commit != request || *saga->committed != result) {
      throw OperationPreconditionError(
          "host process commit replay diverges from its durable receipt");
    }
    return *saga;
  }
  Transaction transaction(database_);
  std::string reason;
  if (!verify_chain(&reason)) {
    throw std::runtime_error("refusing host process commit copy: " + reason);
  }
  Event durable = host_saga_event(
      database_, request.launch_id + ":host-process-committed",
      "host.process_committed", request.run_id, now.wall.nanoseconds,
      static_cast<std::uint64_t>(now.boot.nanoseconds),
      {{"commit", nlohmann::json::parse(
                      hostd_process_commit_canonical_json(request))},
       {"committed", nlohmann::json::parse(
                         hostd_process_committed_canonical_json(result))}});
  durable.node_id = identity.node_id;
  durable.attempt_id = identity.attempt_id;
  (void)append_uncommitted(durable, true);
  transaction.commit();
  saga = host_process_saga(request.launch_id);
  if (!saga || !saga->committed)
    throw std::runtime_error("host process commit copy vanished");
  return *saga;
}

HostProcessSagaSnapshot Journal::record_host_process_exited(
    const HostdProcessExitCommand& request,
    const HostProcessExitResult& supplied_result,
    const AuthorityTimeSample& now) {
  require_authority_time(now);
  (void)hostd_process_exit_canonical_json(request);
  const HostProcessExitResult result = durable_exit_result(supplied_result);
  auto saga = host_process_saga(request.launch_id);
  if (!saga || !saga->committed) {
    throw OperationPreconditionError(
        "host process exit requires a durable exec commit");
  }
  const auto& grant = saga->prepare.grant;
  const auto& spawn = saga->prepared.spawn;
  const auto& receipt = result.receipt;
  if (request.allocation_id != grant.allocation_id ||
      request.grant_digest != grant.receipt_digest ||
      request.journal_id != grant.journal_id ||
      request.run_id != grant.run_id ||
      request.logical_lease_id != grant.logical_lease_id ||
      request.logical_fencing_token != grant.logical_fencing_token ||
      request.spawn_receipt_digest != spawn.receipt_digest ||
      receipt.request.exit_request_id != request.exit_request_id ||
      receipt.request.launch_id != request.launch_id ||
      receipt.request.spawn_receipt_digest != spawn.receipt_digest ||
      receipt.request.host_pid != spawn.request.host_pid ||
      receipt.request.process_starttime_ticks !=
          spawn.request.process_starttime_ticks ||
      receipt.request.cgroup_path != spawn.request.cgroup_path ||
      receipt.request.cgroup_device != spawn.request.cgroup_device ||
      receipt.request.cgroup_inode != spawn.request.cgroup_inode ||
      receipt.host_id != grant.host_id || receipt.boot_id != grant.boot_id ||
      receipt.broker_epoch != grant.broker_epoch) {
    throw OperationPreconditionError(
        "host process exit result disagrees with its durable process saga");
  }
  if (const auto& identity = saga->prepare.launch.identity;
      !identity.host_grant) {
    throw OperationPreconditionError(
        "host process exit has no exact physical grant claim");
  } else {
    require_host_launch_eligible(*identity.host_grant, now);
  }
  if (saga->exited) {
    if (*saga->exit_command != request || *saga->exited != result) {
      throw OperationPreconditionError(
          "host process exit replay diverges from its durable receipt");
    }
    return *saga;
  }
  Transaction transaction(database_);
  std::string reason;
  if (!verify_chain(&reason)) {
    throw std::runtime_error("refusing host process exit copy: " + reason);
  }
  const auto& identity = saga->prepare.launch.identity;
  Event durable = host_saga_event(
      database_, request.launch_id + ":host-process-exited",
      "host.process_exited", request.run_id, now.wall.nanoseconds,
      static_cast<std::uint64_t>(now.boot.nanoseconds),
      {{"exit_command", nlohmann::json::parse(
                            hostd_process_exit_canonical_json(request))},
       {"exited", host_process_exit_receipt_json(result.receipt)}});
  durable.node_id = identity.node_id;
  durable.attempt_id = identity.attempt_id;
  (void)append_uncommitted(durable, true);
  transaction.commit();
  saga = host_process_saga(request.launch_id);
  if (!saga || !saga->exited)
    throw std::runtime_error("host process exit copy vanished");
  return *saga;
}

std::optional<HostLaunchGrantClaim> Journal::host_launch_grant_claim(
    const std::string& run_id, const std::string& concurrency_key,
    const std::string& lease_id, std::uint64_t fencing_token,
    const AuthorityTimeSample& now) const {
  require_authority_time(now);
  if (run_id.empty() || concurrency_key.empty() || lease_id.empty() ||
      fencing_token == 0U) {
    throw std::invalid_argument("host launch grant lookup is malformed");
  }
  std::optional<HostLaunchGrantClaim> claim;
  {
    auto snapshot = read_snapshot();
    (void)snapshot;
    Statement query(database_, R"sql(
      SELECT request.request_id, grant.canonical_grant_json
      FROM host_resource_requests AS request
      JOIN host_resource_grants AS grant ON grant.request_id=request.request_id
      LEFT JOIN host_resource_release_intents AS release
        ON release.request_id=request.request_id
      WHERE request.run_id=? AND request.concurrency_key=?
        AND request.logical_lease_id=? AND request.logical_fencing_token=?
        AND release.request_id IS NULL
      ORDER BY request.request_id
    )sql");
    bind_text(query.get(), 1, run_id);
    bind_text(query.get(), 2, concurrency_key);
    bind_text(query.get(), 3, lease_id);
    bind_integer(query.get(), 4,
                 checked_integer(fencing_token, "fencing_token"));
    const int status = sqlite3_step(query.get());
    if (status == SQLITE_ROW) {
      const auto grant = resource_bundle_grant_from_json(
          nlohmann::json::parse(column_text(query.get(), 1)));
      claim = HostLaunchGrantClaim{
          .request_id = column_text(query.get(), 0),
          .grant_digest = grant.receipt_digest,
          .fences = grant.fences,
      };
      if (grant.request_id != claim->request_id || grant.run_id != run_id ||
          grant.logical_lease_id != lease_id ||
          grant.logical_fencing_token != fencing_token ||
          sqlite3_step(query.get()) != SQLITE_DONE) {
        throw OperationPreconditionError(
            "host launch grant authority is ambiguous or mismatched");
      }
    } else if (status != SQLITE_DONE) {
      throw std::runtime_error("could not inspect host launch grant authority");
    }
  }
  if (!claim) {
    if (host_grant_enforcement_ == HostGrantEnforcement::required) {
      throw OperationPreconditionError(
          "external worker launch requires an exact durable host grant");
    }
    return std::nullopt;
  }
  require_live_host_grant_claim(claim, run_id, concurrency_key, lease_id,
                                fencing_token);
  require_host_launch_eligible(*claim, now);
  return claim;
}

void Journal::require_host_launch_eligible(
    const HostLaunchGrantClaim& claim, const AuthorityTimeSample& now) const {
  require_authority_time(now);
  auto snapshot = read_snapshot();
  (void)snapshot;
  const auto saga = load_host_grant_saga(database_, claim.request_id);
  if (!saga || !saga->grant) {
    throw OperationPreconditionError(
        "launch requires both durable request and exact host grant receipt");
  }
  if (saga->release_intent || saga->release_receipt) {
    throw OperationPreconditionError(
        "launch is blocked after durable host release intent");
  }
  if (!expected_host_grant_authority_ ||
      saga->grant->host_id != expected_host_grant_authority_->host_id ||
      saga->grant->boot_id != expected_host_grant_authority_->boot_id) {
    throw OperationPreconditionError(
        "launch grant disagrees with the trusted local host epoch");
  }
  if (claim.grant_digest != saga->grant->receipt_digest ||
      claim.fences != saga->grant->fences) {
    throw OperationPreconditionError(
        "launch host grant digest or physical fences do not exactly match");
  }
  Statement lease(database_, R"sql(
    SELECT 1
    FROM host_resource_requests AS request
    JOIN resource_leases AS lease
      ON lease.concurrency_key=request.concurrency_key
     AND lease.owner_run_id=request.run_id
     AND lease.lease_id=request.logical_lease_id
     AND lease.fencing_token=request.logical_fencing_token
    WHERE request.request_id=? AND lease.clock_domain='boottime/v1'
      AND lease.boot_id=? AND lease.expires_boottime_ns>?
      AND lease.released_wall_time_ns IS NULL
      AND NOT EXISTS(
        SELECT 1 FROM resource_lease_releases AS release
        WHERE release.concurrency_key=lease.concurrency_key
          AND release.owner_run_id=lease.owner_run_id
          AND release.lease_id=lease.lease_id
          AND release.fencing_token=lease.fencing_token
      )
  )sql");
  bind_text(lease.get(), 1, claim.request_id);
  bind_text(lease.get(), 2, now.boot_id);
  bind_integer(lease.get(), 3, now.boot.nanoseconds);
  if (sqlite3_step(lease.get()) != SQLITE_ROW) {
    throw OperationPreconditionError(
        "launch logical lease is stale, released, or fenced");
  }
}

void Journal::require_live_host_grant_claim(
    const std::optional<HostLaunchGrantClaim>& claim,
    const std::string& run_id, const std::string& concurrency_key,
    const std::string& lease_id, std::uint64_t fencing_token) const {
  if (!claim) {
    if (host_grant_enforcement_ == HostGrantEnforcement::required) {
      throw OperationPreconditionError(
          "worker authority has no exact physical host grant claim");
    }
    return;
  }
  if (!expected_host_grant_authority_) {
    throw OperationPreconditionError(
        "host grant claim has no configured trusted host epoch");
  }
  const auto saga = load_host_grant_saga(database_, claim->request_id);
  Statement request(database_, R"sql(
    SELECT 1 FROM host_resource_requests
    WHERE request_id=? AND run_id=? AND concurrency_key=?
      AND logical_lease_id=? AND logical_fencing_token=?
  )sql");
  bind_text(request.get(), 1, claim->request_id);
  bind_text(request.get(), 2, run_id);
  bind_text(request.get(), 3, concurrency_key);
  bind_text(request.get(), 4, lease_id);
  bind_integer(request.get(), 5,
               checked_integer(fencing_token, "fencing_token"));
  if (!saga || !saga->grant || saga->release_intent ||
      saga->busy_outcome_digest ||
      saga->grant->receipt_digest != claim->grant_digest ||
      saga->grant->fences != claim->fences ||
      saga->grant->host_id != expected_host_grant_authority_->host_id ||
      saga->grant->boot_id != expected_host_grant_authority_->boot_id ||
      sqlite3_step(request.get()) != SQLITE_ROW ||
      sqlite3_step(request.get()) != SQLITE_DONE) {
    throw OperationPreconditionError(
        "worker physical host grant is released, stale, or mismatched");
  }
}

bool Journal::has_lease_release_receipt(
    const std::string& concurrency_key, const std::string& owner_run_id,
    const std::string& lease_id, std::uint64_t fencing_token,
    std::string_view clock_domain, std::string_view boot_id,
    std::int64_t released_wall_time_ns) const {
  require_lease_identity(concurrency_key, owner_run_id, lease_id);
  if (fencing_token == 0 ||
      clock_domain != ResourceLease::kBootTimeDomain ||
      !canonical_boot_id(boot_id) || released_wall_time_ns < 0) {
    throw std::invalid_argument("lease release receipt identity is invalid");
  }
  Statement query(database_, R"sql(
    SELECT 1 FROM resource_lease_releases
    WHERE concurrency_key=? AND owner_run_id=? AND lease_id=?
      AND fencing_token=? AND clock_domain=? AND boot_id=?
      AND released_wall_time_ns=?
  )sql");
  bind_text(query.get(), 1, concurrency_key);
  bind_text(query.get(), 2, owner_run_id);
  bind_text(query.get(), 3, lease_id);
  bind_integer(query.get(), 4,
               checked_integer(fencing_token, "fencing_token"));
  bind_text(query.get(), 5, std::string(clock_domain));
  bind_text(query.get(), 6, std::string(boot_id));
  bind_integer(query.get(), 7, released_wall_time_ns);
  const int status = sqlite3_step(query.get());
  if (status == SQLITE_ROW) return true;
  if (status != SQLITE_DONE) {
    throw std::runtime_error("could not read lease release receipt");
  }
  return false;
}

std::uint64_t Journal::event_count() const {
  Statement query(
      database_,
      "SELECT COUNT(*) FROM events WHERE event_type NOT LIKE 'authority.%'");
  if (sqlite3_step(query.get()) != SQLITE_ROW) {
    throw std::runtime_error("could not count journal events");
  }
  return static_cast<std::uint64_t>(sqlite3_column_int64(query.get(), 0));
}

std::string Journal::journal_id() const {
  Statement query(database_, "SELECT value FROM journal_meta WHERE key='journal_id'");
  if (sqlite3_step(query.get()) != SQLITE_ROW) {
    throw std::runtime_error("journal identity is missing");
  }
  const std::string identity = column_text(query.get(), 0);
  if (!valid_journal_id(identity)) {
    throw std::runtime_error("journal identity is malformed");
  }
  return identity;
}

JournalAuthoritySnapshot Journal::journal_authority_snapshot() const {
  require_attested_authority();
  auto read = read_snapshot();
  (void)read;
  std::string controller_scope_reason;
  if (!verify_controller_scope_metadata(database_, &controller_scope_reason)) {
    authority_poisoned_.store(true, std::memory_order_release);
    throw OperationPreconditionError(
        "journal controller scope authority is corrupt: " +
        controller_scope_reason);
  }
  Statement query(database_,
                  "SELECT value FROM journal_meta WHERE key='journal_id'");
  if (sqlite3_step(query.get()) != SQLITE_ROW) {
    throw OperationPreconditionError("journal identity is missing");
  }
  const std::string identity = column_text(query.get(), 0);
  if (!valid_journal_id(identity) || sqlite3_step(query.get()) != SQLITE_DONE) {
    throw OperationPreconditionError("journal identity is malformed or ambiguous");
  }
  require_attested_authority();
  return {.file = *expected_file_,
          .host = *expected_host_grant_authority_,
          .journal_id = identity};
}

JournalLogicalFenceSnapshot Journal::journal_logical_fence_snapshot(
    const std::string& concurrency_key, const std::string& owner_run_id,
    const std::string& lease_id, std::uint64_t fencing_token,
    const AuthorityTimeSample& now) const {
  require_lease_identity(concurrency_key, owner_run_id, lease_id);
  require_authority_time(now);
  if (fencing_token == 0U ||
      fencing_token >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument("journal fence token is outside its bound");
  }
  require_attested_authority();
  if (now.boot_id != expected_host_grant_authority_->boot_id) {
    throw OperationPreconditionError(
        "journal fence time disagrees with retained host boot authority");
  }

  auto read = read_snapshot();
  (void)read;
  const auto corrupt = [&](std::string_view message) -> void {
    authority_poisoned_.store(true, std::memory_order_release);
    throw OperationPreconditionError(std::string(message));
  };
  std::string chain_reason;
  if (!verify_event_chain(&chain_reason))
    corrupt("journal logical fence event authority is corrupt");
  Statement metadata(database_,
                     "SELECT value FROM journal_meta WHERE key='journal_id'");
  if (sqlite3_step(metadata.get()) != SQLITE_ROW) {
    throw OperationPreconditionError("journal fence identity is missing");
  }
  const std::string identity = column_text(metadata.get(), 0);
  if (!valid_journal_id(identity) ||
      sqlite3_step(metadata.get()) != SQLITE_DONE) {
    throw OperationPreconditionError(
        "journal fence identity is malformed or ambiguous");
  }

  Statement lease(database_, R"sql(
    SELECT concurrency_key, owner_run_id, lease_id, fencing_token,
           clock_domain, boot_id, acquired_boottime_ns, expires_boottime_ns,
           acquired_wall_time_ns, expires_wall_time_ns, released_wall_time_ns
    FROM resource_leases
    WHERE concurrency_key=?
  )sql");
  bind_text(lease.get(), 1, concurrency_key);
  if (sqlite3_step(lease.get()) != SQLITE_ROW) {
    throw OperationPreconditionError(
        "journal logical fence is missing or superseded");
  }
  const bool scalar_types_exact =
      sqlite3_column_type(lease.get(), 0) == SQLITE_TEXT &&
      sqlite3_column_type(lease.get(), 1) == SQLITE_TEXT &&
      sqlite3_column_type(lease.get(), 2) == SQLITE_TEXT &&
      sqlite3_column_type(lease.get(), 3) == SQLITE_INTEGER &&
      sqlite3_column_type(lease.get(), 4) == SQLITE_TEXT &&
      sqlite3_column_type(lease.get(), 5) == SQLITE_TEXT &&
      sqlite3_column_type(lease.get(), 6) == SQLITE_INTEGER &&
      sqlite3_column_type(lease.get(), 7) == SQLITE_INTEGER &&
      sqlite3_column_type(lease.get(), 8) == SQLITE_INTEGER &&
      sqlite3_column_type(lease.get(), 9) == SQLITE_INTEGER &&
      (sqlite3_column_type(lease.get(), 10) == SQLITE_NULL ||
       sqlite3_column_type(lease.get(), 10) == SQLITE_INTEGER);
  const ResourceLease retained = lease_from_row(lease.get());
  const bool row_released = sqlite3_column_type(lease.get(), 10) != SQLITE_NULL;
  const std::int64_t row_released_wall_time_ns =
      row_released ? sqlite3_column_int64(lease.get(), 10) : 0;
  if (retained.owner_run_id != owner_run_id || retained.lease_id != lease_id ||
      retained.fencing_token != fencing_token)
    throw OperationPreconditionError(
        "journal logical fence is missing or superseded");
  if (!scalar_types_exact || retained.concurrency_key != concurrency_key ||
      retained.clock_domain != ResourceLease::kBootTimeDomain ||
      retained.boot_id != now.boot_id || retained.acquired_boottime_ns < 0 ||
      retained.expires_boottime_ns <= retained.acquired_boottime_ns ||
      retained.acquired_wall_time_ns < 0 ||
      retained.expires_wall_time_ns < 0 ||
      (row_released && row_released_wall_time_ns < 0) ||
      sqlite3_step(lease.get()) != SQLITE_DONE)
    corrupt("journal logical fence projection is malformed");

  const std::string authority_key =
      lease_authority_metadata_key(concurrency_key, fencing_token);
  Statement authority_head(database_,
                           "SELECT value FROM journal_meta WHERE key=?");
  bind_text(authority_head.get(), 1, authority_key);
  if (sqlite3_step(authority_head.get()) != SQLITE_ROW)
    corrupt("journal logical fence has no acquisition authority root");
  LeaseAuthorityHead head;
  try {
    head = parse_lease_authority_head(column_text(authority_head.get(), 0));
  } catch (...) {
    corrupt("journal logical fence authority head is corrupt");
  }
  if (head.concurrency_key != concurrency_key ||
      head.owner_run_id != owner_run_id || head.lease_id != lease_id ||
      head.fencing_token != fencing_token ||
      sqlite3_step(authority_head.get()) != SQLITE_DONE)
    corrupt("journal logical fence authority head is inexact");

  const auto exact_authority_event = [&](std::uint64_t revision)
      -> std::optional<Event> {
    return event(lease_authority_event_id(concurrency_key, fencing_token,
                                          revision));
  };
  const auto root = exact_authority_event(0U);
  if (!root || root->run_id != owner_run_id || root->run_revision != 0U ||
      root->plan_revision != 0U || !root->node_id.empty() ||
      !root->attempt_id.empty() || root->worker_sequence != 0U ||
      root->event_type != "authority.resource_lease_acquired" ||
      root->event_version != 1U ||
      root->wall_time_ns != retained.acquired_wall_time_ns ||
      root->monotonic_time_ns !=
          static_cast<std::uint64_t>(retained.acquired_boottime_ns) ||
      root->optimizer_step || !root->payload.is_object())
    corrupt("journal logical fence acquisition authority is missing or torn");
  std::int64_t initial_expires_boot = 0;
  std::int64_t initial_expires_wall = 0;
  try {
    if (!root->payload.at("expires_boottime_ns").is_number_integer() ||
        !root->payload.at("expires_wall_time_ns").is_number_integer())
      corrupt("journal logical fence acquisition expiry is malformed");
    initial_expires_boot =
        root->payload.at("expires_boottime_ns").get<std::int64_t>();
    initial_expires_wall =
        root->payload.at("expires_wall_time_ns").get<std::int64_t>();
  } catch (const nlohmann::json::exception&) {
    corrupt("journal logical fence acquisition expiry is missing");
  }
  const nlohmann::json expected_root{
      {"acquired_boottime_ns", retained.acquired_boottime_ns},
      {"acquired_wall_time_ns", retained.acquired_wall_time_ns},
      {"authority_revision", 0U},
      {"boot_id", retained.boot_id},
      {"clock_domain", retained.clock_domain},
      {"concurrency_key", retained.concurrency_key},
      {"expires_boottime_ns", initial_expires_boot},
      {"expires_wall_time_ns", initial_expires_wall},
      {"fencing_token", retained.fencing_token},
      {"lease_id", retained.lease_id},
      {"operation", "acquired"},
      {"owner_run_id", retained.owner_run_id},
      {"previous_authority_hash", std::string(64U, '0')}};
  if (root->payload != expected_root ||
      initial_expires_boot <= retained.acquired_boottime_ns ||
      initial_expires_wall < 0)
    corrupt("journal logical fence acquisition root disagrees with projection");

  const auto latest = exact_authority_event(head.authority_revision);
  if (!latest || latest->run_id != owner_run_id ||
      latest->run_revision != 0U || latest->plan_revision != 0U ||
      !latest->node_id.empty() || !latest->attempt_id.empty() ||
      latest->worker_sequence != 0U || latest->event_version != 1U ||
      latest->optimizer_step || content_hash(*latest) != head.head_event_hash)
    corrupt("journal logical fence authority head event is missing or torn");
  Statement latest_sequence(
      database_, "SELECT journal_sequence FROM events WHERE event_id=?");
  bind_text(latest_sequence.get(), 1, latest->event_id);
  if (sqlite3_step(latest_sequence.get()) != SQLITE_ROW ||
      sqlite3_column_type(latest_sequence.get(), 0) != SQLITE_INTEGER ||
      sqlite3_column_int64(latest_sequence.get(), 0) <= 0 ||
      static_cast<std::uint64_t>(
          sqlite3_column_int64(latest_sequence.get(), 0)) !=
          head.head_event_sequence ||
      sqlite3_step(latest_sequence.get()) != SQLITE_DONE)
    corrupt("journal logical fence authority sequence is torn");
  if (head.authority_revision ==
      std::numeric_limits<std::uint64_t>::max())
    corrupt("journal logical fence authority revision is exhausted");
  Statement later_authority(database_,
                            "SELECT 1 FROM events WHERE event_id=?");
  bind_text(later_authority.get(), 1,
            lease_authority_event_id(concurrency_key, fencing_token,
                                     head.authority_revision + 1U));
  const int later_status = sqlite3_step(later_authority.get());
  if (later_status == SQLITE_ROW)
    corrupt("journal logical fence authority head was rolled back");
  if (later_status != SQLITE_DONE)
    corrupt("journal logical fence later authority is unreadable");
  if (head.authority_revision > 0U) {
    const auto previous = exact_authority_event(head.authority_revision - 1U);
    if (!previous || !latest->payload.is_object() ||
        latest->payload.value("previous_authority_hash", std::string{}) !=
            content_hash(*previous))
      corrupt("journal logical fence authority revision chain is torn");
  } else if (content_hash(*root) != head.head_event_hash) {
    corrupt("journal logical fence acquisition head is inexact");
  }

  Statement release(database_, R"sql(
    SELECT clock_domain, boot_id, released_wall_time_ns
    FROM resource_lease_releases
    WHERE concurrency_key=? AND owner_run_id=? AND lease_id=?
      AND fencing_token=?
  )sql");
  bind_text(release.get(), 1, retained.concurrency_key);
  bind_text(release.get(), 2, retained.owner_run_id);
  bind_text(release.get(), 3, retained.lease_id);
  bind_integer(release.get(), 4,
               static_cast<std::int64_t>(retained.fencing_token));
  const int release_status = sqlite3_step(release.get());
  const bool release_receipt = release_status == SQLITE_ROW;
  if (release_status != SQLITE_ROW && release_status != SQLITE_DONE)
    corrupt("journal logical fence release authority is unreadable");
  if (release_receipt) {
    if (sqlite3_column_type(release.get(), 0) != SQLITE_TEXT ||
        sqlite3_column_type(release.get(), 1) != SQLITE_TEXT ||
        sqlite3_column_type(release.get(), 2) != SQLITE_INTEGER ||
        column_text(release.get(), 0) != retained.clock_domain ||
        column_text(release.get(), 1) != retained.boot_id ||
        sqlite3_column_int64(release.get(), 2) !=
            row_released_wall_time_ns ||
        sqlite3_step(release.get()) != SQLITE_DONE)
      corrupt("journal logical fence release projection is torn");
  }
  if (head.released) {
    const nlohmann::json expected_release{
        {"authority_revision", head.authority_revision},
        {"boot_id", retained.boot_id},
        {"clock_domain", retained.clock_domain},
        {"concurrency_key", retained.concurrency_key},
        {"fencing_token", retained.fencing_token},
        {"lease_id", retained.lease_id},
        {"operation", "released"},
        {"owner_run_id", retained.owner_run_id},
        {"previous_authority_hash",
         latest->payload.value("previous_authority_hash", std::string{})},
        {"released_wall_time_ns", row_released_wall_time_ns}};
    if (!row_released || !release_receipt ||
        latest->event_type != "authority.resource_lease_released" ||
        latest->wall_time_ns != row_released_wall_time_ns ||
        latest->monotonic_time_ns != 0U ||
        latest->payload != expected_release)
      corrupt("journal logical fence release closure is torn");
    throw OperationPreconditionError(
        "journal logical fence is durably released");
  }
  if (row_released || release_receipt)
    corrupt("journal logical fence release evidence lacks authority closure");

  if (head.authority_revision == 0U) {
    if (latest->event_type != "authority.resource_lease_acquired" ||
        retained.expires_boottime_ns != initial_expires_boot ||
        retained.expires_wall_time_ns != initial_expires_wall)
      corrupt("journal logical fence acquisition projection was extended");
  } else {
    Statement renewal(database_, R"sql(
      SELECT concurrency_key, owner_run_id, lease_id, fencing_token,
             clock_domain, boot_id, acquired_boottime_ns,
             acquired_wall_time_ns, prior_expires_boottime_ns,
             new_expires_boottime_ns, prior_expires_wall_time_ns,
             new_expires_wall_time_ns, renewed_boottime_ns,
             renewed_wall_time_ns
      FROM resource_lease_renewals
      WHERE concurrency_key=? AND owner_run_id=? AND lease_id=?
        AND fencing_token=? AND prior_expires_boottime_ns=?
    )sql");
    bind_text(renewal.get(), 1, retained.concurrency_key);
    bind_text(renewal.get(), 2, retained.owner_run_id);
    bind_text(renewal.get(), 3, retained.lease_id);
    bind_integer(renewal.get(), 4,
                 static_cast<std::int64_t>(retained.fencing_token));
    const std::int64_t prior_expiry = latest->payload.value(
        "prior_expires_boottime_ns", std::int64_t{-1});
    bind_integer(renewal.get(), 5, prior_expiry);
    if (sqlite3_step(renewal.get()) != SQLITE_ROW)
      corrupt("journal logical fence renewal receipt is missing");
    const LeaseRenewalReceipt receipt = renewal_receipt_from_row(renewal.get());
    if (sqlite3_step(renewal.get()) != SQLITE_DONE)
      corrupt("journal logical fence renewal receipt is ambiguous");
    const nlohmann::json expected_renewal{
        {"acquired_boottime_ns", receipt.acquired_boottime_ns},
        {"acquired_wall_time_ns", receipt.acquired_wall_time_ns},
        {"authority_revision", head.authority_revision},
        {"boot_id", receipt.boot_id},
        {"clock_domain", receipt.clock_domain},
        {"concurrency_key", receipt.concurrency_key},
        {"fencing_token", receipt.fencing_token},
        {"lease_id", receipt.lease_id},
        {"new_expires_boottime_ns", receipt.new_expires_boottime_ns},
        {"new_expires_wall_time_ns", receipt.new_expires_wall_time_ns},
        {"operation", "renewed"},
        {"owner_run_id", receipt.owner_run_id},
        {"previous_authority_hash",
         latest->payload.value("previous_authority_hash", std::string{})},
        {"prior_expires_boottime_ns", receipt.prior_expires_boottime_ns},
        {"prior_expires_wall_time_ns", receipt.prior_expires_wall_time_ns},
        {"renewed_boottime_ns", receipt.renewed_boottime_ns},
        {"renewed_wall_time_ns", receipt.renewed_wall_time_ns}};
    if (latest->event_type != "authority.resource_lease_renewed" ||
        latest->payload != expected_renewal ||
        receipt.concurrency_key != retained.concurrency_key ||
        receipt.owner_run_id != retained.owner_run_id ||
        receipt.lease_id != retained.lease_id ||
        receipt.fencing_token != retained.fencing_token ||
        receipt.clock_domain != retained.clock_domain ||
        receipt.boot_id != retained.boot_id ||
        receipt.acquired_boottime_ns != retained.acquired_boottime_ns ||
        receipt.acquired_wall_time_ns != retained.acquired_wall_time_ns ||
        receipt.new_expires_boottime_ns != retained.expires_boottime_ns ||
        receipt.new_expires_wall_time_ns != retained.expires_wall_time_ns)
      corrupt("journal logical fence renewal authority is torn");
  }
  if (retained.expires_boottime_ns <= now.boot.nanoseconds)
    throw OperationPreconditionError("journal logical fence is expired");

  require_attested_authority();
  return {.authority = {.file = *expected_file_,
                        .host = *expected_host_grant_authority_,
                        .journal_id = identity},
          .lease = retained,
          .authority_revision = head.authority_revision,
          .authority_event_sequence = head.head_event_sequence,
          .authority_event_hash = head.head_event_hash};
}

JournalControllerFence Journal::register_hostd_controller_fence(
    const JournalControllerFence& requested,
    const AuthorityTimeSample& now) {
  require_lease_identity(requested.concurrency_key, requested.run_id,
                         requested.logical_lease_id);
  require_authority_time(now);
  const auto bounded_identifier = [](std::string_view value) {
    return !value.empty() && value.size() <= 192U &&
           std::ranges::all_of(value, [](unsigned char character) {
             return (character >= 'a' && character <= 'z') ||
                    (character >= 'A' && character <= 'Z') ||
                    (character >= '0' && character <= '9') ||
                    character == '.' || character == '_' || character == '-' ||
                    character == ':' || character == '/' || character == '@';
           });
  };
  if (!bounded_identifier(requested.broker_epoch) ||
      !bounded_identifier(requested.controller_id) ||
      requested.controller_generation == 0U ||
      requested.controller_generation >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
      requested.logical_fencing_token == 0U ||
      requested.logical_fencing_token >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument("hostd controller fence is noncanonical");
  }
  const JournalLogicalFenceSnapshot live = journal_logical_fence_snapshot(
      requested.concurrency_key, requested.run_id,
      requested.logical_lease_id, requested.logical_fencing_token, now);

  require_attested_authority();
  Transaction transaction(database_);
  std::string chain_reason;
  if (!verify_event_chain(&chain_reason)) {
    authority_poisoned_.store(true, std::memory_order_release);
    throw OperationPreconditionError(
        "hostd controller registration found corrupt journal authority");
  }

  {
    Statement legacy(database_, "SELECT 1 FROM journal_meta WHERE key=?");
    bind_text(legacy.get(), 1,
              std::string(kLegacyControllerAuthorityMetadataKey));
    const int legacy_status = sqlite3_step(legacy.get());
    if (legacy_status == SQLITE_ROW) {
      authority_poisoned_.store(true, std::memory_order_release);
      throw OperationPreconditionError(
          "legacy global hostd controller authority cannot be safely scoped");
    }
    if (legacy_status != SQLITE_DONE)
      throw std::runtime_error(
          "could not inspect legacy hostd controller authority");
  }

  const std::string authority_metadata_key =
      controller_authority_metadata_key(requested.concurrency_key);

  std::optional<nlohmann::json> prior;
  std::string prior_encoded;
  {
    Statement query(database_, "SELECT value FROM journal_meta WHERE key=?");
    bind_text(query.get(), 1, authority_metadata_key);
    const int status = sqlite3_step(query.get());
    if (status == SQLITE_ROW) {
      prior_encoded = column_text(query.get(), 0);
      try {
        prior = nlohmann::json::parse(prior_encoded);
      } catch (...) {
        authority_poisoned_.store(true, std::memory_order_release);
        throw OperationPreconditionError(
            "hostd controller authority head is malformed");
      }
      const bool exact_shape =
          prior->is_object() && prior->size() == 9U &&
          prior->contains("broker_epoch") &&
          prior->contains("concurrency_key") &&
          prior->contains("controller_generation") &&
          prior->contains("controller_id") &&
          prior->contains("event_sequence") &&
          prior->contains("event_hash") &&
          prior->contains("logical_fencing_token") &&
          prior->contains("logical_lease_id") && prior->contains("run_id");
      if (!exact_shape || prior->dump() != prior_encoded ||
          !prior->at("broker_epoch").is_string() ||
          !prior->at("concurrency_key").is_string() ||
          !prior->at("controller_generation").is_number_unsigned() ||
          !prior->at("controller_id").is_string() ||
          !prior->at("event_sequence").is_number_unsigned() ||
          !prior->at("event_hash").is_string() ||
          !prior->at("logical_fencing_token").is_number_unsigned() ||
          !prior->at("logical_lease_id").is_string() ||
          !prior->at("run_id").is_string() ||
          !valid_hash_hex(prior->at("event_hash").get<std::string>())) {
        authority_poisoned_.store(true, std::memory_order_release);
        throw OperationPreconditionError(
            "hostd controller authority head is noncanonical");
      }
    } else if (status != SQLITE_DONE) {
      throw std::runtime_error("could not read hostd controller authority head");
    }
  }
  if (!prior) {
    Statement history(database_, R"sql(
      SELECT event_id FROM events
      WHERE event_type='authority.hostd_controller_registered'
      ORDER BY journal_sequence
    )sql");
    const std::string scope_event_prefix =
        "journal-hostd-controller-" +
        controller_scope_identity(requested.concurrency_key) + "-";
    bool scope_history_exists = false;
    int history_status = SQLITE_ROW;
    while ((history_status = sqlite3_step(history.get())) == SQLITE_ROW) {
      if (column_text(history.get(), 0).starts_with(scope_event_prefix)) {
        scope_history_exists = true;
        break;
      }
    }
    if (history_status != SQLITE_ROW && history_status != SQLITE_DONE)
      throw std::runtime_error(
          "could not inspect hostd controller authority history");
    if (scope_history_exists) {
      authority_poisoned_.store(true, std::memory_order_release);
      throw OperationPreconditionError(
          "hostd controller authority head was deleted");
    }
  } else {
    const std::uint64_t prior_generation =
        prior->at("controller_generation").get<std::uint64_t>();
    const std::string prior_event_id =
        controller_event_id(requested.concurrency_key, prior_generation);
    const auto prior_event = event(prior_event_id);
    Statement prior_sequence(
        database_, "SELECT journal_sequence FROM events WHERE event_id=?");
    bind_text(prior_sequence.get(), 1, prior_event_id);
    const nlohmann::json expected_payload{
        {"broker_epoch", prior->at("broker_epoch")},
        {"concurrency_key", prior->at("concurrency_key")},
        {"controller_generation", prior->at("controller_generation")},
        {"controller_id", prior->at("controller_id")},
        {"logical_fencing_token", prior->at("logical_fencing_token")},
        {"logical_lease_id", prior->at("logical_lease_id")},
        {"operation", "hostd_controller_registered"},
        {"previous_controller_hash",
         prior_event && prior_event->payload.is_object()
             ? prior_event->payload.value("previous_controller_hash",
                                          std::string{})
             : std::string{}},
        {"run_id", prior->at("run_id")}};
    if (!prior_event ||
        prior_event->run_id != prior->at("run_id").get<std::string>() ||
        prior_event->run_revision != 0U || prior_event->plan_revision != 0U ||
        !prior_event->node_id.empty() || !prior_event->attempt_id.empty() ||
        prior_event->worker_sequence != 0U ||
        prior_event->event_type != "authority.hostd_controller_registered" ||
        prior_event->event_version != 1U || prior_event->optimizer_step ||
        !prior_event->payload.is_object() ||
        !valid_hash_hex(expected_payload.at("previous_controller_hash")
                            .get<std::string>()) ||
        prior_event->payload != expected_payload ||
        content_hash(*prior_event) !=
            prior->at("event_hash").get<std::string>() ||
        sqlite3_step(prior_sequence.get()) != SQLITE_ROW ||
        sqlite3_column_type(prior_sequence.get(), 0) != SQLITE_INTEGER ||
        sqlite3_column_int64(prior_sequence.get(), 0) <= 0 ||
        static_cast<std::uint64_t>(
            sqlite3_column_int64(prior_sequence.get(), 0)) !=
            prior->at("event_sequence").get<std::uint64_t>() ||
        sqlite3_step(prior_sequence.get()) != SQLITE_DONE) {
      authority_poisoned_.store(true, std::memory_order_release);
      throw OperationPreconditionError(
          "hostd controller authority event is missing or torn");
    }
    if (prior_generation ==
        std::numeric_limits<std::uint64_t>::max()) {
      authority_poisoned_.store(true, std::memory_order_release);
      throw OperationPreconditionError(
          "hostd controller generation authority is exhausted");
    }
    Statement later_controller(database_,
                               "SELECT 1 FROM events WHERE event_id=?");
    bind_text(later_controller.get(), 1,
              controller_event_id(requested.concurrency_key,
                                  prior_generation + 1U));
    const int later_status = sqlite3_step(later_controller.get());
    if (later_status == SQLITE_ROW) {
      authority_poisoned_.store(true, std::memory_order_release);
      throw OperationPreconditionError(
          "hostd controller authority head was rolled back");
    }
    if (later_status != SQLITE_DONE)
      throw std::runtime_error("could not read later controller authority");
    if (prior_generation ==
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
        requested.controller_generation != prior_generation + 1U ||
        requested.controller_id ==
            prior->at("controller_id").get<std::string>()) {
      throw OperationPreconditionError(
          "hostd controller generation is reused, stale, or noncontiguous");
    }
  }
  const std::string controller_identity_key =
      controller_identity_metadata_key(requested.concurrency_key,
                                       requested.controller_id);
  {
    Statement reused(database_, "SELECT 1 FROM journal_meta WHERE key=?");
    bind_text(reused.get(), 1, controller_identity_key);
    if (sqlite3_step(reused.get()) == SQLITE_ROW)
      throw OperationPreconditionError("hostd controller identity was reused");
  }

  Statement lease(database_, R"sql(
    SELECT expires_boottime_ns, released_wall_time_ns
    FROM resource_leases
    WHERE concurrency_key=? AND owner_run_id=? AND lease_id=?
      AND fencing_token=? AND clock_domain='boottime/v1' AND boot_id=?
      AND NOT EXISTS(
        SELECT 1 FROM resource_lease_releases AS release
        WHERE release.concurrency_key=resource_leases.concurrency_key
          AND release.owner_run_id=resource_leases.owner_run_id
          AND release.lease_id=resource_leases.lease_id
          AND release.fencing_token=resource_leases.fencing_token)
  )sql");
  bind_text(lease.get(), 1, requested.concurrency_key);
  bind_text(lease.get(), 2, requested.run_id);
  bind_text(lease.get(), 3, requested.logical_lease_id);
  bind_integer(lease.get(), 4,
               static_cast<std::int64_t>(requested.logical_fencing_token));
  bind_text(lease.get(), 5, now.boot_id);
  if (sqlite3_step(lease.get()) != SQLITE_ROW ||
      sqlite3_column_type(lease.get(), 0) != SQLITE_INTEGER ||
      sqlite3_column_int64(lease.get(), 0) <= now.boot.nanoseconds ||
      sqlite3_column_type(lease.get(), 1) != SQLITE_NULL ||
      sqlite3_step(lease.get()) != SQLITE_DONE)
    throw OperationPreconditionError(
        "hostd controller registration lost its live logical fence");
  Statement lease_head(database_, "SELECT value FROM journal_meta WHERE key=?");
  bind_text(lease_head.get(), 1,
            lease_authority_metadata_key(requested.concurrency_key,
                                         requested.logical_fencing_token));
  if (sqlite3_step(lease_head.get()) != SQLITE_ROW) {
    authority_poisoned_.store(true, std::memory_order_release);
    throw OperationPreconditionError(
        "hostd controller lease authority head disappeared");
  }
  LeaseAuthorityHead current_head;
  try {
    current_head = parse_lease_authority_head(column_text(lease_head.get(), 0));
  } catch (...) {
    authority_poisoned_.store(true, std::memory_order_release);
    throw OperationPreconditionError(
        "hostd controller lease authority head is corrupt");
  }
  if (current_head.authority_revision != live.authority_revision ||
      current_head.head_event_sequence != live.authority_event_sequence ||
      current_head.head_event_hash != live.authority_event_hash ||
      current_head.released) {
    throw OperationPreconditionError(
        "hostd controller lease authority advanced during registration");
  }

  const std::string previous_controller_hash =
      prior ? prior->at("event_hash").get<std::string>()
            : std::string(64U, '0');
  Event event{
      .event_id = controller_event_id(requested.concurrency_key,
                                      requested.controller_generation),
      .run_id = requested.run_id,
      .run_revision = 0U,
      .plan_revision = 0U,
      .node_id = {},
      .attempt_id = {},
      .worker_sequence = 0U,
      .event_type = "authority.hostd_controller_registered",
      .event_version = 1U,
      .wall_time_ns = now.wall.nanoseconds,
      .monotonic_time_ns =
          static_cast<std::uint64_t>(now.boot.nanoseconds),
      .optimizer_step = std::nullopt,
      .payload = {{"broker_epoch", requested.broker_epoch},
                  {"concurrency_key", requested.concurrency_key},
                  {"controller_generation", requested.controller_generation},
                  {"controller_id", requested.controller_id},
                  {"logical_fencing_token",
                   requested.logical_fencing_token},
                  {"logical_lease_id", requested.logical_lease_id},
                  {"operation", "hostd_controller_registered"},
                  {"previous_controller_hash", previous_controller_hash},
                  {"run_id", requested.run_id}}};
  const auto [event_hash, event_sequence] =
      append_authority_event_uncommitted(event);
  const nlohmann::json next{{"broker_epoch", requested.broker_epoch},
                            {"concurrency_key", requested.concurrency_key},
                            {"controller_generation",
                             requested.controller_generation},
                            {"controller_id", requested.controller_id},
                            {"event_sequence", event_sequence},
                            {"event_hash", event_hash},
                            {"logical_fencing_token",
                             requested.logical_fencing_token},
                            {"logical_lease_id", requested.logical_lease_id},
                            {"run_id", requested.run_id}};
  if (prior) {
    Statement update(database_,
                     "UPDATE journal_meta SET value=? WHERE key=? AND value=?");
    bind_text(update.get(), 1, next.dump());
    bind_text(update.get(), 2, authority_metadata_key);
    bind_text(update.get(), 3, prior_encoded);
    require_done(database_, update.get(), "advance hostd controller authority");
    if (sqlite3_changes(database_) != 1)
      throw std::runtime_error("hostd controller authority changed concurrently");
  } else {
    Statement insert(database_,
                     "INSERT INTO journal_meta(key, value) VALUES(?, ?)");
    bind_text(insert.get(), 1, authority_metadata_key);
    bind_text(insert.get(), 2, next.dump());
    require_done(database_, insert.get(), "create hostd controller authority");
  }
  Statement identity(database_,
                     "INSERT INTO journal_meta(key, value) VALUES(?, ?)");
  bind_text(identity.get(), 1, controller_identity_key);
  bind_text(identity.get(), 2, requested.controller_id);
  require_done(database_, identity.get(), "retain hostd controller identity");
  transaction.commit();
  require_attested_authority();
  return requested;
}

std::optional<JournalControllerFence>
Journal::current_hostd_controller_fence(
    const std::string& concurrency_key) const {
  if (concurrency_key.empty()) {
    throw std::invalid_argument(
        "hostd controller concurrency_key must not be empty");
  }
  require_attested_authority();
  std::optional<JournalControllerFence> result;
  {
    auto read = read_snapshot();
    (void)read;
    Statement legacy(database_, "SELECT 1 FROM journal_meta WHERE key=?");
    bind_text(legacy.get(), 1,
              std::string(kLegacyControllerAuthorityMetadataKey));
    const int legacy_status = sqlite3_step(legacy.get());
    if (legacy_status == SQLITE_ROW) {
      authority_poisoned_.store(true, std::memory_order_release);
      throw OperationPreconditionError(
          "legacy global hostd controller authority cannot be safely scoped");
    }
    if (legacy_status != SQLITE_DONE) {
      throw std::runtime_error(
          "legacy hostd controller authority is unreadable");
    }
    Statement query(database_, "SELECT value FROM journal_meta WHERE key=?");
    bind_text(query.get(), 1,
              controller_authority_metadata_key(concurrency_key));
    const int status = sqlite3_step(query.get());
    if (status == SQLITE_DONE) return std::nullopt;
    if (status != SQLITE_ROW) {
      throw std::runtime_error("could not read hostd controller authority");
    }
    const std::string encoded = column_text(query.get(), 0);
    if (sqlite3_step(query.get()) != SQLITE_DONE) {
      throw std::runtime_error("hostd controller authority is ambiguous");
    }
    try {
      const nlohmann::json head = nlohmann::json::parse(encoded);
      if (!head.is_object() || head.size() != 9U || head.dump() != encoded ||
          !head.at("broker_epoch").is_string() ||
          !head.at("run_id").is_string() ||
          !head.at("concurrency_key").is_string() ||
          !head.at("controller_id").is_string() ||
          !head.at("controller_generation").is_number_unsigned() ||
          !head.at("logical_lease_id").is_string() ||
          !head.at("logical_fencing_token").is_number_unsigned() ||
          !head.at("event_sequence").is_number_unsigned() ||
          !head.at("event_hash").is_string() ||
          head.at("concurrency_key").get<std::string>() != concurrency_key) {
        throw std::runtime_error("noncanonical controller head");
      }
      result = JournalControllerFence{
          .broker_epoch = head.at("broker_epoch").get<std::string>(),
          .run_id = head.at("run_id").get<std::string>(),
          .concurrency_key = concurrency_key,
          .controller_id = head.at("controller_id").get<std::string>(),
          .controller_generation =
              head.at("controller_generation").get<std::uint64_t>(),
          .logical_lease_id =
              head.at("logical_lease_id").get<std::string>(),
          .logical_fencing_token =
              head.at("logical_fencing_token").get<std::uint64_t>(),
      };
    } catch (...) {
      authority_poisoned_.store(true, std::memory_order_release);
      throw OperationPreconditionError(
          "hostd controller authority head is malformed");
    }
  }
  require_current_hostd_controller_fence(*result);
  return result;
}

void Journal::require_current_hostd_controller_fence(
    const JournalControllerFence& requested) const {
  require_attested_authority();
  auto read = read_snapshot();
  (void)read;
  const auto corrupt = [&](std::string_view message) -> void {
    authority_poisoned_.store(true, std::memory_order_release);
    throw OperationPreconditionError(std::string(message));
  };
  std::string chain_reason;
  if (!verify_event_chain(&chain_reason))
    corrupt("hostd controller event authority is corrupt");

  {
    Statement legacy(database_, "SELECT 1 FROM journal_meta WHERE key=?");
    bind_text(legacy.get(), 1,
              std::string(kLegacyControllerAuthorityMetadataKey));
    const int legacy_status = sqlite3_step(legacy.get());
    if (legacy_status == SQLITE_ROW)
      corrupt("legacy global hostd controller authority cannot be safely scoped");
    if (legacy_status != SQLITE_DONE)
      corrupt("legacy hostd controller authority is unreadable");
  }

  Statement query(database_, "SELECT value FROM journal_meta WHERE key=?");
  bind_text(query.get(), 1,
            controller_authority_metadata_key(requested.concurrency_key));
  if (sqlite3_step(query.get()) != SQLITE_ROW)
    corrupt("hostd controller authority head is missing");
  const std::string encoded = column_text(query.get(), 0);
  if (sqlite3_step(query.get()) != SQLITE_DONE)
    corrupt("hostd controller authority head is ambiguous");

  nlohmann::json head;
  try {
    head = nlohmann::json::parse(encoded);
    const bool exact_shape =
        head.is_object() && head.size() == 9U &&
        head.contains("broker_epoch") && head.contains("concurrency_key") &&
        head.contains("controller_generation") &&
        head.contains("controller_id") && head.contains("event_sequence") &&
        head.contains("event_hash") &&
        head.contains("logical_fencing_token") &&
        head.contains("logical_lease_id") && head.contains("run_id");
    if (!exact_shape || head.dump() != encoded ||
        !head.at("broker_epoch").is_string() ||
        !head.at("concurrency_key").is_string() ||
        !head.at("controller_generation").is_number_unsigned() ||
        !head.at("controller_id").is_string() ||
        !head.at("event_sequence").is_number_unsigned() ||
        !head.at("event_hash").is_string() ||
        !head.at("logical_fencing_token").is_number_unsigned() ||
        !head.at("logical_lease_id").is_string() ||
        !head.at("run_id").is_string() ||
        !valid_hash_hex(head.at("event_hash").get<std::string>()))
      corrupt("hostd controller authority head is noncanonical");
  } catch (const OperationPreconditionError&) {
    throw;
  } catch (...) {
    corrupt("hostd controller authority head is malformed");
  }

  const std::uint64_t generation =
      head.at("controller_generation").get<std::uint64_t>();
  const std::string authority_event_id =
      controller_event_id(requested.concurrency_key, generation);
  const auto authority_event = event(authority_event_id);
  Statement authority_sequence(
      database_, "SELECT journal_sequence FROM events WHERE event_id=?");
  bind_text(authority_sequence.get(), 1, authority_event_id);
  const std::string previous_hash =
      authority_event && authority_event->payload.is_object()
          ? authority_event->payload.value("previous_controller_hash",
                                           std::string{})
          : std::string{};
  const nlohmann::json expected_payload{
      {"broker_epoch", head.at("broker_epoch")},
      {"concurrency_key", head.at("concurrency_key")},
      {"controller_generation", head.at("controller_generation")},
      {"controller_id", head.at("controller_id")},
      {"logical_fencing_token", head.at("logical_fencing_token")},
      {"logical_lease_id", head.at("logical_lease_id")},
      {"operation", "hostd_controller_registered"},
      {"previous_controller_hash", previous_hash},
      {"run_id", head.at("run_id")}};
  if (!authority_event ||
      authority_event->run_id != head.at("run_id").get<std::string>() ||
      authority_event->run_revision != 0U ||
      authority_event->plan_revision != 0U ||
      !authority_event->node_id.empty() ||
      !authority_event->attempt_id.empty() ||
      authority_event->worker_sequence != 0U ||
      authority_event->event_type !=
          "authority.hostd_controller_registered" ||
      authority_event->event_version != 1U || authority_event->optimizer_step ||
      !valid_hash_hex(previous_hash) ||
      authority_event->payload != expected_payload ||
      content_hash(*authority_event) !=
          head.at("event_hash").get<std::string>() ||
      sqlite3_step(authority_sequence.get()) != SQLITE_ROW ||
      sqlite3_column_type(authority_sequence.get(), 0) != SQLITE_INTEGER ||
      sqlite3_column_int64(authority_sequence.get(), 0) <= 0 ||
      static_cast<std::uint64_t>(
          sqlite3_column_int64(authority_sequence.get(), 0)) !=
          head.at("event_sequence").get<std::uint64_t>() ||
      sqlite3_step(authority_sequence.get()) != SQLITE_DONE)
    corrupt("hostd controller authority event is missing or torn");
  if (generation == std::numeric_limits<std::uint64_t>::max())
    corrupt("hostd controller generation authority is exhausted");
  Statement later_controller(database_,
                             "SELECT 1 FROM events WHERE event_id=?");
  bind_text(later_controller.get(), 1,
            controller_event_id(requested.concurrency_key, generation + 1U));
  const int later_status = sqlite3_step(later_controller.get());
  if (later_status == SQLITE_ROW)
    corrupt("hostd controller authority head was rolled back");
  if (later_status != SQLITE_DONE)
    corrupt("hostd controller later authority is unreadable");

  const std::string identity_key = controller_identity_metadata_key(
      requested.concurrency_key,
      head.at("controller_id").get<std::string>());
  Statement identity(database_, "SELECT value FROM journal_meta WHERE key=?");
  bind_text(identity.get(), 1, identity_key);
  if (sqlite3_step(identity.get()) != SQLITE_ROW ||
      sqlite3_column_type(identity.get(), 0) != SQLITE_TEXT ||
      column_text(identity.get(), 0) !=
          head.at("controller_id").get<std::string>() ||
      sqlite3_step(identity.get()) != SQLITE_DONE)
    corrupt("hostd controller identity retention is torn");

  if (head.at("broker_epoch").get<std::string>() != requested.broker_epoch ||
      head.at("run_id").get<std::string>() != requested.run_id ||
      head.at("concurrency_key").get<std::string>() !=
          requested.concurrency_key ||
      head.at("controller_id").get<std::string>() != requested.controller_id ||
      generation != requested.controller_generation ||
      head.at("logical_lease_id").get<std::string>() !=
          requested.logical_lease_id ||
      head.at("logical_fencing_token").get<std::uint64_t>() !=
          requested.logical_fencing_token)
    throw OperationPreconditionError(
        "hostd controller fence is no longer current");
  require_attested_authority();
}

bool Journal::verify_event_chain(std::string* reason) const {
  Statement query(database_, R"sql(
    WITH head(value) AS (
      SELECT value FROM journal_meta WHERE key='chain_head'
    )
    SELECT events.journal_sequence, events.event_id, events.run_id,
           events.run_revision, events.plan_revision, events.node_id,
           events.attempt_id, events.worker_sequence, events.event_type,
           events.event_version, events.wall_time_ns, events.monotonic_time_ns,
           events.optimizer_step, events.payload_json, events.previous_hash,
           events.content_hash, events.chain_hash, head.value
    FROM head LEFT JOIN events ON TRUE
    ORDER BY events.journal_sequence
  )sql");
  std::string expected_previous(64, '0');
  std::optional<std::string> stored_head;
  int status = SQLITE_OK;
  while ((status = sqlite3_step(query.get())) == SQLITE_ROW) {
    if (!stored_head) {
      stored_head = column_text(query.get(), 17);
    }
    if (sqlite3_column_type(query.get(), 0) == SQLITE_NULL) {
      continue;
    }
    const auto sequence = static_cast<std::uint64_t>(sqlite3_column_int64(query.get(), 0));
    Event event;
    try {
      event = event_from_row(query.get());
    } catch (const std::exception& exception) {
      if (reason) {
        *reason = "invalid event at journal sequence " + std::to_string(sequence) + ": " + exception.what();
      }
      return false;
    }
    const std::string stored_previous = column_text(query.get(), 14);
    const std::string stored_content = column_text(query.get(), 15);
    const std::string stored_chain = column_text(query.get(), 16);
    const std::string expected_content = content_hash(event);
    const std::string expected_chain = sha256_hex(expected_previous + ":" + expected_content);
    if (stored_previous != expected_previous || stored_content != expected_content ||
        stored_chain != expected_chain) {
      if (reason) {
        *reason = "hash-chain mismatch at journal sequence " + std::to_string(sequence);
      }
      return false;
    }
    expected_previous = stored_chain;
  }
  if (status != SQLITE_DONE) {
    if (reason) {
      *reason = "could not scan the journal: " + std::string(sqlite3_errmsg(database_));
    }
    return false;
  }
  if (!stored_head || *stored_head != expected_previous) {
    if (reason) {
      *reason = "journal head does not match the final event (possible tail truncation)";
    }
    return false;
  }
  if (reason) {
    reason->clear();
  }
  return true;
}

bool Journal::verify_chain(std::string* reason) const {
  if (!verify_event_chain(reason)) return false;
  return verify_host_saga_projection(database_, reason);
}

std::uint64_t Journal::rebuild_projections() {
  std::string reason;
  Transaction transaction(database_);
  if (!verify_event_chain(&reason)) {
    throw std::runtime_error("refusing replay: " + reason);
  }
  constexpr std::string_view reset_host_saga = R"sql(
    DROP TRIGGER IF EXISTS host_resource_requests_no_update;
    DROP TRIGGER IF EXISTS host_resource_requests_no_delete;
    DROP TRIGGER IF EXISTS host_resource_grants_no_update;
    DROP TRIGGER IF EXISTS host_resource_grants_no_delete;
    DROP TRIGGER IF EXISTS host_resource_release_intents_no_update;
    DROP TRIGGER IF EXISTS host_resource_release_intents_no_delete;
    DROP TRIGGER IF EXISTS host_resource_release_receipts_no_update;
    DROP TRIGGER IF EXISTS host_resource_release_receipts_no_delete;
    DELETE FROM host_resource_release_receipts;
    DELETE FROM host_resource_release_intents;
    DELETE FROM host_resource_grants;
    DELETE FROM host_resource_requests;
  )sql";
  if (sqlite3_exec(database_, std::string(reset_host_saga).c_str(), nullptr,
                   nullptr, nullptr) != SQLITE_OK) {
    throw std::runtime_error("could not clear host saga projections: " +
                             std::string(sqlite3_errmsg(database_)));
  }
  if (sqlite3_exec(database_, "DELETE FROM run_projection", nullptr, nullptr, nullptr) != SQLITE_OK) {
    throw std::runtime_error("could not clear run projections");
  }
  if (sqlite3_exec(database_, "DELETE FROM control_commands", nullptr, nullptr, nullptr) !=
      SQLITE_OK) {
    throw std::runtime_error("could not clear control command projections");
  }
  Statement query(database_, R"sql(
    SELECT journal_sequence, event_id, run_id, run_revision, plan_revision, node_id,
           attempt_id, worker_sequence, event_type, event_version, wall_time_ns,
           monotonic_time_ns, optimizer_step, payload_json
    FROM events ORDER BY journal_sequence
  )sql");
  std::uint64_t replayed = 0;
  int status = SQLITE_OK;
  while ((status = sqlite3_step(query.get())) == SQLITE_ROW) {
    const auto sequence = static_cast<std::uint64_t>(sqlite3_column_int64(query.get(), 0));
    const Event event = event_from_row(query.get());
    if (!event.event_type.starts_with("authority.")) {
      update_projection(database_, event, sequence);
      update_control_projection(database_, event);
      replay_host_saga_projection(database_, event);
      ++replayed;
    }
  }
  if (status != SQLITE_DONE) {
    throw std::runtime_error("could not scan events while rebuilding projections: " +
                             std::string(sqlite3_errmsg(database_)));
  }
  constexpr std::string_view restore_host_saga_triggers = R"sql(
    CREATE TRIGGER host_resource_requests_no_update BEFORE UPDATE ON host_resource_requests
    BEGIN SELECT RAISE(ABORT, 'host resource requests are immutable'); END;
    CREATE TRIGGER host_resource_requests_no_delete BEFORE DELETE ON host_resource_requests
    BEGIN SELECT RAISE(ABORT, 'host resource requests are immutable'); END;
    CREATE TRIGGER host_resource_grants_no_update BEFORE UPDATE ON host_resource_grants
    BEGIN SELECT RAISE(ABORT, 'host resource grants are immutable'); END;
    CREATE TRIGGER host_resource_grants_no_delete BEFORE DELETE ON host_resource_grants
    BEGIN SELECT RAISE(ABORT, 'host resource grants are immutable'); END;
    CREATE TRIGGER host_resource_release_intents_no_update
    BEFORE UPDATE ON host_resource_release_intents
    BEGIN SELECT RAISE(ABORT, 'host resource release intents are immutable'); END;
    CREATE TRIGGER host_resource_release_intents_no_delete
    BEFORE DELETE ON host_resource_release_intents
    BEGIN SELECT RAISE(ABORT, 'host resource release intents are immutable'); END;
    CREATE TRIGGER host_resource_release_receipts_no_update
    BEFORE UPDATE ON host_resource_release_receipts
    BEGIN SELECT RAISE(ABORT, 'host resource release receipts are immutable'); END;
    CREATE TRIGGER host_resource_release_receipts_no_delete
    BEFORE DELETE ON host_resource_release_receipts
    BEGIN SELECT RAISE(ABORT, 'host resource release receipts are immutable'); END;
  )sql";
  if (sqlite3_exec(database_, std::string(restore_host_saga_triggers).c_str(),
                   nullptr, nullptr, nullptr) != SQLITE_OK) {
    throw std::runtime_error("could not restore host saga immutability: " +
                             std::string(sqlite3_errmsg(database_)));
  }
  if (!verify_host_saga_projection(database_, &reason)) {
    throw std::runtime_error("rebuilt host saga projection is invalid: " + reason);
  }
  if (!verify_event_chain(&reason)) {
    throw std::runtime_error("rebuilt journal chain changed: " + reason);
  }
  transaction.commit();
  return replayed;
}

nlohmann::json event_json(const Event& event) {
  nlohmann::json output{{"event_id", event.event_id},
                        {"run_id", event.run_id},
                        {"run_revision", event.run_revision},
                        {"plan_revision", event.plan_revision},
                        {"node_id", event.node_id},
                        {"attempt_id", event.attempt_id},
                        {"worker_sequence", event.worker_sequence},
                        {"event_type", event.event_type},
                        {"event_version", event.event_version},
                        {"wall_time_ns", event.wall_time_ns},
                        {"monotonic_time_ns", event.monotonic_time_ns},
                        {"payload", event.payload}};
  if (event.optimizer_step) {
    output["optimizer_step"] = *event.optimizer_step;
  }
  return output;
}

Event event_from_json(const nlohmann::json& input) {
  static const std::set<std::string> allowed{
      "event_id",          "run_id",       "run_revision",    "plan_revision",
      "node_id",           "attempt_id",   "worker_sequence", "event_type",
      "event_version",     "wall_time_ns", "monotonic_time_ns", "optimizer_step",
      "payload"};
  if (!input.is_object()) {
    throw std::invalid_argument("event document must be an object");
  }
  for (auto iterator = input.begin(); iterator != input.end(); ++iterator) {
    if (!allowed.contains(iterator.key())) {
      throw std::invalid_argument("unknown event field: " + iterator.key());
    }
  }
  Event event;
  try {
    event.event_id = input.at("event_id").get<std::string>();
    event.run_id = input.at("run_id").get<std::string>();
    event.run_revision = input.value("run_revision", 0ULL);
    event.plan_revision = input.value("plan_revision", 0ULL);
    event.node_id = input.value("node_id", "");
    event.attempt_id = input.value("attempt_id", "");
    event.worker_sequence = input.value("worker_sequence", 0ULL);
    event.event_type = input.at("event_type").get<std::string>();
    event.event_version = input.value("event_version", 1U);
    event.wall_time_ns = input.value("wall_time_ns", std::int64_t{0});
    event.monotonic_time_ns = input.value("monotonic_time_ns", 0ULL);
    if (input.contains("optimizer_step")) {
      event.optimizer_step = input.at("optimizer_step").get<std::uint64_t>();
    }
    event.payload = input.value("payload", nlohmann::json::object());
  } catch (const nlohmann::json::exception& exception) {
    throw std::invalid_argument(std::string("invalid event document: ") + exception.what());
  }
  return event;
}

nlohmann::json projection_json(const RunProjection& projection) {
  return {{"run_id", projection.run_id},
          {"experiment_name", projection.experiment_name},
          {"plan_hash", projection.plan_hash},
          {"desired_state", projection.desired_state},
          {"observed_state", projection.observed_state},
          {"current_node_id", projection.current_node_id},
          {"current_attempt_id", projection.current_attempt_id},
          {"run_revision", projection.run_revision},
          {"optimizer_step", projection.optimizer_step},
          {"last_heartbeat_ns", projection.last_heartbeat_ns},
          {"last_event_sequence", projection.last_event_sequence},
          {"failure_summary", projection.failure_summary}};
}

}  // namespace trainvm
