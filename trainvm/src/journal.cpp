#include "trainvm/journal.hpp"

#include "trainvm/document.hpp"

#include <sqlite3.h>

#include <openssl/rand.h>

#include <algorithm>
#include <array>
#include <chrono>
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

INSERT INTO journal_meta(key, value) VALUES('schema_version', '4')
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
  acquired_at_ns INTEGER NOT NULL,
  expires_at_ns INTEGER NOT NULL,
  released_at_ns INTEGER
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
      .acquired_at_ns = sqlite3_column_int64(statement, 4),
      .expires_at_ns = sqlite3_column_int64(statement, 5),
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
  try {
    initialize();
  } catch (...) {
    sqlite3_close(database_);
    database_ = nullptr;
    throw;
  }
}

Journal::~Journal() {
  if (database_) {
    sqlite3_close(database_);
  }
}

void Journal::initialize() {
  char* error_message = nullptr;
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
        stored_version != "4") {
      throw std::runtime_error("unsupported journal schema version");
    }
    if (stored_version == "4") {
      Statement identity(database_, "SELECT value FROM journal_meta WHERE key='journal_id'");
      if (sqlite3_step(identity.get()) != SQLITE_ROW ||
          !valid_journal_id(column_text(identity.get(), 0))) {
        throw std::runtime_error("established v4 journal identity is missing or malformed");
      }
    }
  }
  if (stored_version == "1" || stored_version == "2" || stored_version == "3") {
    // These versions predate the complete authority and acknowledgement invariants.
    // Rewriting a journal that already contains history would silently bless data
    // that cannot satisfy the v4 trust model. Empty legacy databases may be upgraded,
    // but nonempty ones must be preserved and explicitly exported by a future,
    // version-aware migration tool.
    const auto table_has_rows = [&](std::string_view table) {
      Statement exists(database_, R"sql(
        SELECT EXISTS(
          SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1
        )
      )sql");
      bind_text(exists.get(), 1, std::string(table));
      if (sqlite3_step(exists.get()) != SQLITE_ROW) {
        throw std::runtime_error("could not inspect legacy journal tables");
      }
      if (sqlite3_column_int(exists.get(), 0) == 0) {
        return false;
      }
      Statement rows(database_, "SELECT EXISTS(SELECT 1 FROM \"" + std::string(table) +
                                    "\" LIMIT 1)");
      if (sqlite3_step(rows.get()) != SQLITE_ROW) {
        throw std::runtime_error("could not inspect legacy journal contents");
      }
      return sqlite3_column_int(rows.get(), 0) != 0;
    };
    bool has_durable_state = false;
    for (const std::string_view table :
         std::array<std::string_view, 6>{"events", "run_projection", "compiled_plans",
                                         "resource_leases", "node_dispatches",
                                         "control_commands"}) {
      has_durable_state = has_durable_state || table_has_rows(table);
    }
    if (has_durable_state) {
      throw std::runtime_error(
          "refusing to migrate a nonempty pre-v4 journal; preserve it read-only and create a "
          "new v4 authority journal");
    }
  }
  const std::string schema(kSchema);
  error_message = nullptr;
  if (sqlite3_exec(database_, schema.c_str(), nullptr, nullptr, &error_message) != SQLITE_OK) {
    const std::string message = error_message ? error_message : sqlite3_errmsg(database_);
    sqlite3_free(error_message);
    throw std::runtime_error("journal initialization failed: " + message);
  }
  if (!has_metadata) {
    stored_version = "4";
  }
  if (stored_version == "1" || stored_version == "2" || stored_version == "3") {
    Transaction transaction(database_);
    const auto column_exists = [&](std::string_view wanted) {
      Statement columns(database_, "PRAGMA table_info(control_commands)");
      while (sqlite3_step(columns.get()) == SQLITE_ROW) {
        if (column_text(columns.get(), 1) == wanted) {
          return true;
        }
      }
      return false;
    };
    for (const auto& [name, declaration] :
         std::vector<std::pair<std::string_view, std::string_view>>{
             {"ack_concurrency_key", "TEXT"}, {"ack_lease_id", "TEXT"},
             {"ack_fencing_token", "INTEGER"}, {"ack_node_id", "TEXT"},
             {"ack_attempt_id", "TEXT"}, {"ack_worker_sequence", "INTEGER"},
             {"acknowledged_at_ns", "INTEGER"}}) {
      if (!column_exists(name)) {
        const std::string sql = "ALTER TABLE control_commands ADD COLUMN " +
                                std::string(name) + " " + std::string(declaration);
        if (sqlite3_exec(database_, sql.c_str(), nullptr, nullptr, nullptr) != SQLITE_OK) {
          throw std::runtime_error("journal schema migration to version 4 failed: " +
                                   std::string(sqlite3_errmsg(database_)));
        }
      }
    }
    Statement migrate(database_, "UPDATE journal_meta SET value='4' WHERE key='schema_version'");
    require_done(database_, migrate.get(), "migrate journal schema to version 4");
    transaction.commit();
  }
  if (!has_metadata || stored_version != "4") {
    Transaction transaction(database_);
    Statement insert(database_, R"sql(
      INSERT INTO journal_meta(key, value) VALUES('journal_id', ?)
      ON CONFLICT(key) DO UPDATE SET value=excluded.value
    )sql");
    bind_text(insert.get(), 1, random_journal_id());
    require_done(database_, insert.get(), "initialize journal identity");
    transaction.commit();
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

  Transaction transaction(database_);
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
  if (dispatch_id.empty() || result_event_id.empty() || events.empty()) {
    throw std::invalid_argument("dispatch completion requires IDs and journal events");
  }
  if (std::none_of(events.begin(), events.end(), [&](const Event& event) {
        return event.event_id == result_event_id;
      })) {
    throw std::invalid_argument("dispatch completion batch does not contain its result event");
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
  if (stored.status == DispatchStatus::completed) {
    if (stored.result_event_id != std::optional<std::string>{result_event_id}) {
      throw std::invalid_argument("dispatch already completed with a different result event");
    }
    transaction.commit();
    return;
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
    nlohmann::json diagnostics) {
  if (run_id.empty() || command_id.empty() || identity.concurrency_key.empty() ||
      identity.lease_id.empty() || identity.fencing_token == 0 || identity.node_id.empty() ||
      identity.attempt_id.empty() || identity.worker_sequence == 0 ||
      status == ControlCommandStatus::requested ||
      !effective_values.is_object() || !diagnostics.is_array()) {
    throw std::invalid_argument("invalid control command acknowledgement");
  }
  Transaction transaction(database_);
  const auto received_at_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::system_clock::now().time_since_epoch())
                                  .count();
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
      AND released_at_ns IS NULL AND acquired_at_ns<=? AND expires_at_ns>?
  )sql");
  bind_text(lease.get(), 1, identity.concurrency_key);
  bind_text(lease.get(), 2, command.run_id);
  bind_text(lease.get(), 3, identity.lease_id);
  bind_integer(lease.get(), 4, checked_integer(identity.fencing_token, "fencing_token"));
  bind_integer(lease.get(), 5, received_at_ns);
  bind_integer(lease.get(), 6, received_at_ns);
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

LeaseAcquireResult Journal::acquire_lease(const std::string& concurrency_key,
                                          const std::string& owner_run_id,
                                          const std::string& lease_id, std::int64_t now_ns,
                                          std::int64_t timeout_ns) {
  require_lease_identity(concurrency_key, owner_run_id, lease_id);
  const std::int64_t expires_at_ns = lease_expiration(now_ns, timeout_ns);
  Transaction transaction(database_);
  Statement query(database_, R"sql(
    SELECT concurrency_key, owner_run_id, lease_id, fencing_token,
           acquired_at_ns, expires_at_ns, released_at_ns
    FROM resource_leases WHERE concurrency_key=?
  )sql");
  bind_text(query.get(), 1, concurrency_key);
  const int status = sqlite3_step(query.get());
  if (status != SQLITE_ROW && status != SQLITE_DONE) {
    throw std::runtime_error("could not read resource lease: " + std::string(sqlite3_errmsg(database_)));
  }

  ResourceLease lease;
  if (status == SQLITE_DONE) {
    lease = ResourceLease{.concurrency_key = concurrency_key,
                          .owner_run_id = owner_run_id,
                          .lease_id = lease_id,
                          .fencing_token = 1,
                          .acquired_at_ns = now_ns,
                          .expires_at_ns = expires_at_ns};
    Statement insert(database_, R"sql(
      INSERT INTO resource_leases(
        concurrency_key, owner_run_id, lease_id, fencing_token,
        acquired_at_ns, expires_at_ns, released_at_ns
      ) VALUES(?, ?, ?, ?, ?, ?, NULL)
    )sql");
    bind_text(insert.get(), 1, lease.concurrency_key);
    bind_text(insert.get(), 2, lease.owner_run_id);
    bind_text(insert.get(), 3, lease.lease_id);
    bind_integer(insert.get(), 4, checked_integer(lease.fencing_token, "fencing_token"));
    bind_integer(insert.get(), 5, lease.acquired_at_ns);
    bind_integer(insert.get(), 6, lease.expires_at_ns);
    require_done(database_, insert.get(), "insert resource lease");
    transaction.commit();
    return {.status = LeaseAcquireStatus::acquired, .lease = std::move(lease)};
  }

  lease = lease_from_row(query.get());
  const bool released = sqlite3_column_type(query.get(), 6) != SQLITE_NULL;
  const bool active = !released && lease.expires_at_ns > now_ns;
  if (active) {
    const LeaseAcquireStatus disposition =
        lease.owner_run_id == owner_run_id && lease.lease_id == lease_id
            ? LeaseAcquireStatus::already_owned
            : LeaseAcquireStatus::busy;
    transaction.commit();
    return {.status = disposition, .lease = std::move(lease)};
  }
  if (lease.fencing_token == static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error("resource lease fencing token is exhausted");
  }
  ++lease.fencing_token;
  lease.owner_run_id = owner_run_id;
  lease.lease_id = lease_id;
  lease.acquired_at_ns = now_ns;
  lease.expires_at_ns = expires_at_ns;
  Statement replace(database_, R"sql(
    UPDATE resource_leases
    SET owner_run_id=?, lease_id=?, fencing_token=?, acquired_at_ns=?,
        expires_at_ns=?, released_at_ns=NULL
    WHERE concurrency_key=?
  )sql");
  bind_text(replace.get(), 1, lease.owner_run_id);
  bind_text(replace.get(), 2, lease.lease_id);
  bind_integer(replace.get(), 3, checked_integer(lease.fencing_token, "fencing_token"));
  bind_integer(replace.get(), 4, lease.acquired_at_ns);
  bind_integer(replace.get(), 5, lease.expires_at_ns);
  bind_text(replace.get(), 6, concurrency_key);
  require_done(database_, replace.get(), "replace resource lease");
  if (sqlite3_changes(database_) != 1) {
    throw std::runtime_error("resource lease replacement affected an unexpected number of rows");
  }
  transaction.commit();
  return {.status = LeaseAcquireStatus::acquired, .lease = std::move(lease)};
}

bool Journal::renew_lease(const std::string& concurrency_key, const std::string& owner_run_id,
                          const std::string& lease_id, std::uint64_t fencing_token,
                          std::int64_t now_ns, std::int64_t timeout_ns) {
  require_lease_identity(concurrency_key, owner_run_id, lease_id);
  const std::int64_t expires_at_ns = lease_expiration(now_ns, timeout_ns);
  Transaction transaction(database_);
  Statement update(database_, R"sql(
    UPDATE resource_leases SET expires_at_ns=?
    WHERE concurrency_key=? AND owner_run_id=? AND lease_id=? AND fencing_token=?
      AND released_at_ns IS NULL AND expires_at_ns>?
  )sql");
  bind_integer(update.get(), 1, expires_at_ns);
  bind_text(update.get(), 2, concurrency_key);
  bind_text(update.get(), 3, owner_run_id);
  bind_text(update.get(), 4, lease_id);
  bind_integer(update.get(), 5, checked_integer(fencing_token, "fencing_token"));
  bind_integer(update.get(), 6, now_ns);
  require_done(database_, update.get(), "renew resource lease");
  const bool renewed = sqlite3_changes(database_) == 1;
  transaction.commit();
  return renewed;
}

bool Journal::release_lease(const std::string& concurrency_key, const std::string& owner_run_id,
                            const std::string& lease_id, std::uint64_t fencing_token,
                            std::int64_t now_ns) {
  require_lease_identity(concurrency_key, owner_run_id, lease_id);
  Transaction transaction(database_);
  Statement update(database_, R"sql(
    UPDATE resource_leases SET released_at_ns=?
    WHERE concurrency_key=? AND owner_run_id=? AND lease_id=? AND fencing_token=?
      AND released_at_ns IS NULL
  )sql");
  bind_integer(update.get(), 1, now_ns);
  bind_text(update.get(), 2, concurrency_key);
  bind_text(update.get(), 3, owner_run_id);
  bind_text(update.get(), 4, lease_id);
  bind_integer(update.get(), 5, checked_integer(fencing_token, "fencing_token"));
  require_done(database_, update.get(), "release resource lease");
  const bool released = sqlite3_changes(database_) == 1;
  transaction.commit();
  return released;
}

std::optional<ResourceLease> Journal::active_lease(const std::string& concurrency_key,
                                                   std::int64_t now_ns) const {
  if (concurrency_key.empty()) {
    throw std::invalid_argument("lease concurrency_key must not be empty");
  }
  Statement query(database_, R"sql(
    SELECT concurrency_key, owner_run_id, lease_id, fencing_token,
           acquired_at_ns, expires_at_ns
    FROM resource_leases
    WHERE concurrency_key=? AND released_at_ns IS NULL AND expires_at_ns>?
  )sql");
  bind_text(query.get(), 1, concurrency_key);
  bind_integer(query.get(), 2, now_ns);
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

std::uint64_t Journal::event_count() const {
  Statement query(database_, "SELECT COUNT(*) FROM events");
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
    update_projection(database_, event, sequence);
    update_control_projection(database_, event);
    ++replayed;
  }
  if (status != SQLITE_DONE) {
    throw std::runtime_error("could not scan events while rebuilding projections: " +
                             std::string(sqlite3_errmsg(database_)));
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
