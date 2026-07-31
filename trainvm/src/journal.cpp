#include "trainvm/journal.hpp"

#include "trainvm/document.hpp"

#include <sqlite3.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace trainvm {
namespace {

constexpr std::string_view kSchema = R"sql(
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;
PRAGMA foreign_keys=ON;
PRAGMA busy_timeout=5000;
PRAGMA trusted_schema=OFF;

CREATE TABLE IF NOT EXISTS journal_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
) WITHOUT ROWID;

INSERT INTO journal_meta(key, value) VALUES('schema_version', '1')
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
      current_node_id=CASE WHEN ? IN ('completed','failed','cancelled') THEN '' ELSE current_node_id END,
      current_attempt_id=CASE WHEN ? IN ('completed','failed','cancelled') THEN '' ELSE current_attempt_id END
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

}  // namespace

Journal::Journal(const std::filesystem::path& path) {
  const auto parent = path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  if (sqlite3_open_v2(path.c_str(), &database_, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr) !=
      SQLITE_OK) {
    const std::string message = database_ ? sqlite3_errmsg(database_) : "unknown error";
    if (database_) {
      sqlite3_close(database_);
      database_ = nullptr;
    }
    throw std::runtime_error("sqlite open failed: " + message);
  }
  initialize();
}

Journal::~Journal() {
  if (database_) {
    sqlite3_close(database_);
  }
}

void Journal::initialize() {
  char* error_message = nullptr;
  const std::string schema(kSchema);
  if (sqlite3_exec(database_, schema.c_str(), nullptr, nullptr, &error_message) != SQLITE_OK) {
    const std::string message = error_message ? error_message : sqlite3_errmsg(database_);
    sqlite3_free(error_message);
    throw std::runtime_error("journal initialization failed: " + message);
  }
  Statement version(database_, "SELECT value FROM journal_meta WHERE key='schema_version'");
  if (sqlite3_step(version.get()) != SQLITE_ROW || column_text(version.get(), 0) != "1") {
    throw std::runtime_error("unsupported journal schema version");
  }
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

std::uint64_t Journal::append_uncommitted(const Event& event) {
  if (event.event_id.empty() || event.run_id.empty() || event.event_type.empty()) {
    throw std::invalid_argument("event_id, run_id, and event_type must not be empty");
  }
  if (!event.payload.is_object()) {
    throw std::invalid_argument("event payload must be an object");
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
    FROM events WHERE run_id=? ORDER BY journal_sequence
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
  return RunProjection{
      .run_id = column_text(query.get(), 0),
      .experiment_name = column_text(query.get(), 1),
      .plan_hash = column_text(query.get(), 2),
      .desired_state = column_text(query.get(), 3),
      .observed_state = column_text(query.get(), 4),
      .current_node_id = column_text(query.get(), 5),
      .current_attempt_id = column_text(query.get(), 6),
      .run_revision = static_cast<std::uint64_t>(sqlite3_column_int64(query.get(), 7)),
      .optimizer_step = static_cast<std::uint64_t>(sqlite3_column_int64(query.get(), 8)),
      .last_heartbeat_ns = sqlite3_column_int64(query.get(), 9),
      .last_event_sequence = static_cast<std::uint64_t>(sqlite3_column_int64(query.get(), 10)),
      .failure_summary = column_text(query.get(), 11),
  };
}

std::uint64_t Journal::event_count() const {
  Statement query(database_, "SELECT COUNT(*) FROM events");
  if (sqlite3_step(query.get()) != SQLITE_ROW) {
    throw std::runtime_error("could not count journal events");
  }
  return static_cast<std::uint64_t>(sqlite3_column_int64(query.get(), 0));
}

bool Journal::verify_chain(std::string* reason) const {
  Statement query(database_, R"sql(
    SELECT journal_sequence, event_id, run_id, run_revision, plan_revision, node_id,
           attempt_id, worker_sequence, event_type, event_version, wall_time_ns,
           monotonic_time_ns, optimizer_step, payload_json, previous_hash,
           content_hash, chain_hash
    FROM events ORDER BY journal_sequence
  )sql");
  std::string expected_previous(64, '0');
  int status = SQLITE_OK;
  while ((status = sqlite3_step(query.get())) == SQLITE_ROW) {
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
  Statement head(database_, "SELECT value FROM journal_meta WHERE key='chain_head'");
  if (sqlite3_step(head.get()) != SQLITE_ROW || column_text(head.get(), 0) != expected_previous) {
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

std::uint64_t Journal::rebuild_projections() {
  std::string reason;
  if (!verify_chain(&reason)) {
    throw std::runtime_error("refusing replay: " + reason);
  }
  Transaction transaction(database_);
  if (sqlite3_exec(database_, "DELETE FROM run_projection", nullptr, nullptr, nullptr) != SQLITE_OK) {
    throw std::runtime_error("could not clear run projections");
  }
  Statement query(database_, R"sql(
    SELECT journal_sequence, event_id, run_id, run_revision, plan_revision, node_id,
           attempt_id, worker_sequence, event_type, event_version, wall_time_ns,
           monotonic_time_ns, optimizer_step, payload_json
    FROM events ORDER BY journal_sequence
  )sql");
  std::uint64_t replayed = 0;
  while (sqlite3_step(query.get()) == SQLITE_ROW) {
    const auto sequence = static_cast<std::uint64_t>(sqlite3_column_int64(query.get(), 0));
    update_projection(database_, event_from_row(query.get()), sequence);
    ++replayed;
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
