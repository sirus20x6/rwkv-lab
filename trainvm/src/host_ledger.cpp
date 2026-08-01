#include "trainvm/host_ledger.hpp"

#include <openssl/evp.h>
#include <openssl/rand.h>
#include <sqlite3.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <dirent.h>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <set>
#include <sstream>
#include <string_view>
#include <sys/stat.h>
#include <tuple>
#include <unistd.h>
#include <utility>

#include "trainvm/reflection_json.hpp"

namespace trainvm {
namespace {

constexpr std::string_view kGenesisDigest =
    "sha256:0000000000000000000000000000000000000000000000000000000000000000";

constexpr std::string_view kSchemaV1 = R"sql(
CREATE TABLE ledger_identity (
  singleton INTEGER PRIMARY KEY CHECK(singleton=1),
  schema_version INTEGER NOT NULL CHECK(schema_version=1),
  ledger_id TEXT NOT NULL UNIQUE,
  host_id TEXT NOT NULL,
  created_boot_id TEXT NOT NULL,
  created_wall_time_ns INTEGER NOT NULL CHECK(created_wall_time_ns >= 0)
) WITHOUT ROWID;
CREATE TABLE ledger_chain_head (
  singleton INTEGER PRIMARY KEY CHECK(singleton=1),
  last_sequence INTEGER NOT NULL CHECK(last_sequence >= 0),
  chain_hash TEXT NOT NULL,
  FOREIGN KEY(singleton) REFERENCES ledger_identity(singleton)
) WITHOUT ROWID;
CREATE TABLE ledger_records (
  ledger_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
  record_id TEXT NOT NULL UNIQUE,
  record_type TEXT NOT NULL CHECK(record_type IN (
    'broker.started','client.registered','inventory.observed',
    'startup.audit_committed','bundle.requested','bundle.granted',
    'bundle.denied','bundle.release_requested','bundle.released'
  )),
  subject_id TEXT NOT NULL,
  canonical_json TEXT NOT NULL,
  receipt_digest TEXT NOT NULL UNIQUE,
  content_hash TEXT NOT NULL,
  previous_hash TEXT NOT NULL,
  chain_hash TEXT NOT NULL UNIQUE
);
CREATE TABLE projection_head (
  singleton INTEGER PRIMARY KEY CHECK(singleton=1),
  applied_sequence INTEGER NOT NULL CHECK(applied_sequence >= 0),
  applied_chain_hash TEXT NOT NULL
) WITHOUT ROWID;
CREATE TABLE current_inventory (
  singleton INTEGER PRIMARY KEY CHECK(singleton=1),
  inventory_digest TEXT NOT NULL,
  canonical_json TEXT NOT NULL
) WITHOUT ROWID;
CREATE TABLE resource_generations (
  resource_key TEXT PRIMARY KEY,
  generation INTEGER NOT NULL CHECK(generation >= 0),
  last_allocation_id TEXT,
  last_grant_digest TEXT
) WITHOUT ROWID;
CREATE TABLE request_outcomes (
  request_id TEXT PRIMARY KEY,
  request_digest TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('granted','busy')),
  allocation_id TEXT,
  outcome_digest TEXT NOT NULL,
  canonical_outcome_json TEXT NOT NULL,
  CHECK((status='granted') = (allocation_id IS NOT NULL))
) WITHOUT ROWID;
CREATE TABLE allocations (
  allocation_id TEXT PRIMARY KEY,
  request_id TEXT NOT NULL UNIQUE,
  request_digest TEXT NOT NULL,
  grant_digest TEXT NOT NULL UNIQUE,
  journal_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  logical_lease_id TEXT NOT NULL,
  logical_fencing_token INTEGER NOT NULL CHECK(logical_fencing_token > 0),
  host_id TEXT NOT NULL,
  grant_boot_id TEXT NOT NULL,
  broker_epoch TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('active','released')),
  release_digest TEXT
) WITHOUT ROWID;
CREATE TABLE allocation_resources (
  allocation_id TEXT NOT NULL,
  resource_key TEXT NOT NULL,
  resource_json TEXT NOT NULL,
  generation INTEGER NOT NULL CHECK(generation > 0),
  inventory_digest TEXT NOT NULL,
  topology_digest TEXT NOT NULL,
  PRIMARY KEY(allocation_id, resource_key),
  UNIQUE(resource_key, generation),
  FOREIGN KEY(allocation_id) REFERENCES allocations(allocation_id)
) WITHOUT ROWID;
CREATE TABLE active_resource_grants (
  resource_key TEXT PRIMARY KEY,
  allocation_id TEXT NOT NULL,
  generation INTEGER NOT NULL CHECK(generation > 0),
  grant_digest TEXT NOT NULL,
  FOREIGN KEY(allocation_id, resource_key)
    REFERENCES allocation_resources(allocation_id, resource_key)
) WITHOUT ROWID;
CREATE TABLE release_outcomes (
  release_request_id TEXT PRIMARY KEY,
  release_request_digest TEXT NOT NULL,
  allocation_id TEXT NOT NULL UNIQUE,
  grant_digest TEXT NOT NULL,
  release_receipt_digest TEXT NOT NULL UNIQUE,
  canonical_release_json TEXT NOT NULL
) WITHOUT ROWID;
CREATE TRIGGER ledger_records_no_update BEFORE UPDATE ON ledger_records
BEGIN SELECT RAISE(ABORT, 'ledger records are immutable'); END;
CREATE TRIGGER ledger_records_no_delete BEFORE DELETE ON ledger_records
BEGIN SELECT RAISE(ABORT, 'ledger records are immutable'); END;
)sql";

// Version 2 is an additive migration. In particular, ledger_records and every
// v1 canonical/hash-chain byte remain untouched. ledger_identity.schema_version
// intentionally continues to describe that immutable core v1 schema; the
// extension row and PRAGMA user_version identify the additive audit schema.
constexpr std::string_view kStartupAuditSchemaV2 = R"sql(
CREATE TABLE ledger_schema_extensions (
  feature TEXT PRIMARY KEY,
  schema_version INTEGER NOT NULL CHECK(schema_version > 1)
) WITHOUT ROWID;
CREATE TABLE startup_audit_outcomes (
  audit_id TEXT PRIMARY KEY,
  report_digest TEXT NOT NULL UNIQUE,
  canonical_report_json TEXT NOT NULL,
  record_receipt_digest TEXT NOT NULL UNIQUE,
  record_sequence INTEGER NOT NULL UNIQUE CHECK(record_sequence > 0),
  record_chain_hash TEXT NOT NULL UNIQUE,
  record_previous_sequence INTEGER NOT NULL
    CHECK(record_previous_sequence >= 0 AND
          record_previous_sequence + 1 = record_sequence),
  record_previous_hash TEXT NOT NULL,
  receipt_digest TEXT NOT NULL UNIQUE,
  canonical_receipt_json TEXT NOT NULL
) WITHOUT ROWID;
CREATE TRIGGER startup_audit_outcomes_no_update
BEFORE UPDATE ON startup_audit_outcomes
BEGIN SELECT RAISE(ABORT, 'startup audit outcomes are immutable'); END;
CREATE TRIGGER startup_audit_outcomes_no_delete
BEFORE DELETE ON startup_audit_outcomes
BEGIN SELECT RAISE(ABORT, 'startup audit outcomes are immutable'); END;
)sql";

// Version 3 adds a ledger-owned grant-admission fence. Finalization itself does
// not append to the v1 hash chain: it atomically binds the exact committed
// audit head and occupancy, while every subsequent request records which
// active epoch authorized it. Outcomes already present at migration are frozen
// into an immutable legacy closure; policy-unconfigured ledgers explicitly
// mark their direct-request compatibility mode. Every other request outcome
// must have an epoch authorization, so deleting an authorization cannot turn
// it into apparently valid legacy history. Releases remain exact grant-bound
// cleanup and do not require an epoch, so recovery can always reduce occupancy.
constexpr std::string_view kAdmissionEpochSchemaV3 = R"sql(
CREATE TABLE admission_epochs (
  epoch_digest TEXT PRIMARY KEY,
  api_version TEXT NOT NULL,
  audit_id TEXT NOT NULL UNIQUE
    REFERENCES startup_audit_outcomes(audit_id),
  report_digest TEXT NOT NULL UNIQUE,
  audit_receipt_digest TEXT NOT NULL UNIQUE,
  host_id TEXT NOT NULL,
  boot_id TEXT NOT NULL,
  broker_epoch TEXT NOT NULL,
  inventory_digest TEXT NOT NULL,
  audit_record_sequence INTEGER NOT NULL CHECK(audit_record_sequence > 0),
  audit_record_chain_hash TEXT NOT NULL,
  finalized_occupancy_digest TEXT NOT NULL,
  finalized_boottime_ns INTEGER NOT NULL CHECK(finalized_boottime_ns >= 0),
  finalized_wall_time_ns INTEGER NOT NULL CHECK(finalized_wall_time_ns >= 0),
  canonical_json TEXT NOT NULL
) WITHOUT ROWID;
CREATE TABLE active_admission_epoch (
  singleton INTEGER PRIMARY KEY CHECK(singleton=1),
  epoch_digest TEXT NOT NULL UNIQUE
    REFERENCES admission_epochs(epoch_digest)
) WITHOUT ROWID;
CREATE TABLE request_admission_epochs (
  request_id TEXT PRIMARY KEY,
  request_digest TEXT NOT NULL,
  epoch_digest TEXT NOT NULL REFERENCES admission_epochs(epoch_digest)
) WITHOUT ROWID;
CREATE TABLE request_admission_exemptions (
  request_id TEXT PRIMARY KEY,
  request_digest TEXT NOT NULL,
  reason TEXT NOT NULL CHECK(reason IN ('pre_v3', 'policy_unconfigured'))
) WITHOUT ROWID;
CREATE TRIGGER admission_epochs_no_update BEFORE UPDATE ON admission_epochs
BEGIN SELECT RAISE(ABORT, 'admission epochs are immutable'); END;
CREATE TRIGGER admission_epochs_no_delete BEFORE DELETE ON admission_epochs
BEGIN SELECT RAISE(ABORT, 'admission epochs are immutable'); END;
CREATE TRIGGER request_admission_epochs_no_update
BEFORE UPDATE ON request_admission_epochs
BEGIN SELECT RAISE(ABORT, 'request admission epochs are immutable'); END;
CREATE TRIGGER request_admission_epochs_no_delete
BEFORE DELETE ON request_admission_epochs
BEGIN SELECT RAISE(ABORT, 'request admission epochs are immutable'); END;
CREATE TRIGGER request_admission_exemptions_no_update
BEFORE UPDATE ON request_admission_exemptions
BEGIN SELECT RAISE(ABORT, 'request admission exemptions are immutable'); END;
CREATE TRIGGER request_admission_exemptions_no_delete
BEFORE DELETE ON request_admission_exemptions
BEGIN SELECT RAISE(ABORT, 'request admission exemptions are immutable'); END;
)sql";

class Statement final {
 public:
  Statement(sqlite3* database, std::string_view sql) : database_(database) {
    const std::string owned(sql);
    if (sqlite3_prepare_v2(database, owned.c_str(), -1, &statement_, nullptr) !=
        SQLITE_OK) {
      throw HostLedgerError("sqlite prepare failed: " +
                            std::string(sqlite3_errmsg(database)));
    }
  }
  ~Statement() { sqlite3_finalize(statement_); }
  Statement(const Statement&) = delete;
  Statement& operator=(const Statement&) = delete;
  [[nodiscard]] sqlite3_stmt* get() const { return statement_; }

 private:
  sqlite3* database_{};
  sqlite3_stmt* statement_{};
};

class Transaction final {
 public:
  explicit Transaction(sqlite3* database) : database_(database) {
    execute("BEGIN IMMEDIATE", "begin");
  }
  ~Transaction() {
    if (!committed_) sqlite3_exec(database_, "ROLLBACK", nullptr, nullptr, nullptr);
  }
  void commit() {
    execute("COMMIT", "commit");
    committed_ = true;
  }

 private:
  void execute(const char* sql, std::string_view action) {
    char* error = nullptr;
    if (sqlite3_exec(database_, sql, nullptr, nullptr, &error) != SQLITE_OK) {
      const std::string detail = error ? error : sqlite3_errmsg(database_);
      sqlite3_free(error);
      throw HostLedgerError("sqlite " + std::string(action) + " failed: " +
                            detail);
    }
  }
  sqlite3* database_{};
  bool committed_{};
};

class ReadTransaction final {
 public:
  explicit ReadTransaction(sqlite3* database) : database_(database) {
    execute("BEGIN", "begin read transaction");
  }
  ~ReadTransaction() {
    if (!committed_)
      sqlite3_exec(database_, "ROLLBACK", nullptr, nullptr, nullptr);
  }
  void commit() {
    execute("COMMIT", "commit read transaction");
    committed_ = true;
  }

 private:
  void execute(const char* sql, std::string_view action) {
    char* error = nullptr;
    if (sqlite3_exec(database_, sql, nullptr, nullptr, &error) != SQLITE_OK) {
      const std::string detail = error ? error : sqlite3_errmsg(database_);
      sqlite3_free(error);
      throw HostLedgerError("sqlite " + std::string(action) + " failed: " +
                            detail);
    }
  }
  sqlite3* database_{};
  bool committed_{};
};

void execute(sqlite3* database, std::string_view sql,
             std::string_view description) {
  const std::string owned(sql);
  char* error = nullptr;
  if (sqlite3_exec(database, owned.c_str(), nullptr, nullptr, &error) !=
      SQLITE_OK) {
    const std::string detail = error ? error : sqlite3_errmsg(database);
    sqlite3_free(error);
    throw HostLedgerError(std::string(description) + ": " + detail);
  }
}

void bind_text(sqlite3_stmt* statement, int index, const std::string& value) {
  if (sqlite3_bind_text(statement, index, value.data(),
                        static_cast<int>(value.size()), SQLITE_TRANSIENT) !=
      SQLITE_OK) {
    throw HostLedgerError("sqlite text bind failed");
  }
}

void bind_integer(sqlite3_stmt* statement, int index, std::int64_t value) {
  if (sqlite3_bind_int64(statement, index, value) != SQLITE_OK) {
    throw HostLedgerError("sqlite integer bind failed");
  }
}

std::string column_text(sqlite3_stmt* statement, int index) {
  const auto* value = sqlite3_column_text(statement, index);
  return value ? reinterpret_cast<const char*>(value) : std::string{};
}

void require_done(sqlite3* database, sqlite3_stmt* statement,
                  std::string_view action) {
  if (sqlite3_step(statement) != SQLITE_DONE) {
    throw HostLedgerError(std::string(action) + ": " +
                          sqlite3_errmsg(database));
  }
}

std::int64_t checked_integer(std::uint64_t value, std::string_view field) {
  if (value >
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    throw HostLedgerError(std::string(field) + " exceeds SQLite range");
  }
  return static_cast<std::int64_t>(value);
}

struct EvpDeleter {
  void operator()(EVP_MD_CTX* value) const { EVP_MD_CTX_free(value); }
};

std::string sha256(std::string_view domain, std::string_view value) {
  std::unique_ptr<EVP_MD_CTX, EvpDeleter> context(EVP_MD_CTX_new());
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1 ||
      EVP_DigestUpdate(context.get(), domain.data(), domain.size()) != 1) {
    throw HostLedgerError("could not initialize ledger digest");
  }
  const char separator = '\0';
  if (EVP_DigestUpdate(context.get(), &separator, 1U) != 1 ||
      EVP_DigestUpdate(context.get(), value.data(), value.size()) != 1) {
    throw HostLedgerError("could not update ledger digest");
  }
  std::array<unsigned char, EVP_MAX_MD_SIZE> bytes{};
  unsigned int length = 0;
  if (EVP_DigestFinal_ex(context.get(), bytes.data(), &length) != 1) {
    throw HostLedgerError("could not finalize ledger digest");
  }
  static constexpr char digits[] = "0123456789abcdef";
  std::string result = "sha256:";
  result.reserve(7U + static_cast<std::size_t>(length) * 2U);
  for (unsigned int index = 0; index < length; ++index) {
    result.push_back(digits[bytes[index] >> 4U]);
    result.push_back(digits[bytes[index] & 0x0fU]);
  }
  return result;
}

bool valid_digest(std::string_view value) {
  return value.size() == 71U && value.starts_with("sha256:") &&
         std::ranges::all_of(value.substr(7U), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

struct DirectoryDeleter final {
  void operator()(DIR* value) const {
    if (value != nullptr) (void)::closedir(value);
  }
};

std::set<int> open_file_descriptors() {
  std::unique_ptr<DIR, DirectoryDeleter> directory(::opendir("/proc/self/fd"));
  if (!directory) {
    throw HostLedgerError("could not inspect process descriptors");
  }
  const int directory_fd = ::dirfd(directory.get());
  std::set<int> result;
  errno = 0;
  while (const auto* entry = ::readdir(directory.get())) {
    const std::string_view name(entry->d_name);
    int descriptor = -1;
    const auto parsed =
        std::from_chars(name.data(), name.data() + name.size(), descriptor);
    if (parsed.ec == std::errc{} && parsed.ptr == name.data() + name.size() &&
        descriptor >= 0 && descriptor != directory_fd) {
      result.insert(descriptor);
    }
  }
  if (errno != 0) {
    throw HostLedgerError("could not enumerate process descriptors");
  }
  return result;
}

int find_new_database_descriptor(
    const std::set<int>& before,
    const HostLedgerFileIdentity& expected_identity) {
  for (const int descriptor : open_file_descriptors()) {
    if (before.contains(descriptor)) continue;
    struct stat status {};
    if (::fstat(descriptor, &status) == 0 &&
        static_cast<std::uint64_t>(status.st_dev) == expected_identity.device &&
        static_cast<std::uint64_t>(status.st_ino) == expected_identity.inode) {
      return descriptor;
    }
  }
  throw HostLedgerError(
      "SQLite did not retain the pinned host-ledger database inode");
}

std::mutex& sqlite_open_mutex() {
  static std::mutex value;
  return value;
}

std::string random_id(std::string_view prefix) {
  std::array<unsigned char, 16> bytes{};
  if (RAND_bytes(bytes.data(), static_cast<int>(bytes.size())) != 1) {
    throw HostLedgerError("could not generate ledger identity");
  }
  static constexpr char digits[] = "0123456789abcdef";
  std::string result(prefix);
  for (const unsigned char byte : bytes) {
    result.push_back(digits[byte >> 4U]);
    result.push_back(digits[byte & 0x0fU]);
  }
  return result;
}

nlohmann::json sealed_json(std::string_view domain, nlohmann::json value) {
  value.erase("receipt_digest");
  value["receipt_digest"] = sha256(domain, value.dump());
  return value;
}

bool valid_time(const HostLedgerTime& now) {
  return now.boottime_ns >= 0 && now.wall_time_ns >= 0;
}

bool same_occupied_resources(const ResourceOccupancySnapshot& left,
                             const ResourceOccupancySnapshot& right) {
  return left.api_version == right.api_version &&
         left.host_id == right.host_id && left.boot_id == right.boot_id &&
         left.inventory_digest == right.inventory_digest &&
         left.active_fences == right.active_fences;
}

nlohmann::json admission_epoch_json(
    const HostStartupAuditReport& report,
    const HostStartupAuditReceipt& receipt,
    const ResourceOccupancySnapshot& finalized_occupancy,
    const HostLedgerTime& now) {
  nlohmann::json value{
      {"api_version", kHostLedgerAdmissionEpochApiVersion},
      {"audit_id", report.audit_id},
      {"report_digest", report.report_digest},
      {"audit_receipt_digest", receipt.receipt_digest},
      {"host_id", report.host_id},
      {"boot_id", report.boot_id},
      {"broker_epoch", report.broker_epoch},
      {"inventory_digest", report.inventory.inventory_digest},
      {"audit_record_sequence", receipt.committed_ledger_head.ledger_sequence},
      {"audit_record_chain_hash", receipt.committed_ledger_head.chain_hash},
      {"finalized_occupancy_digest", finalized_occupancy.occupancy_digest},
      {"finalized_boottime_ns", now.boottime_ns},
      {"finalized_wall_time_ns", now.wall_time_ns},
  };
  value["epoch_digest"] = sha256("trainvm.host-ledger-admission-epoch/v1",
                                  value.dump());
  return value;
}

nlohmann::json grant_digest_json(const ResourceBundleGrant& grant) {
  return {{"api_version", grant.api_version},
          {"allocation_id", grant.allocation_id},
          {"request_id", grant.request_id},
          {"request_digest", grant.request_digest},
          {"journal_id", grant.journal_id},
          {"run_id", grant.run_id},
          {"logical_lease_id", grant.logical_lease_id},
          {"logical_fencing_token", grant.logical_fencing_token},
          {"host_id", grant.host_id},
          {"boot_id", grant.boot_id},
          {"broker_epoch", grant.broker_epoch},
          {"fences", encode_json(grant.fences)},
          {"granted_boottime_ns", grant.granted_boottime_ns},
          {"granted_wall_time_ns", grant.granted_wall_time_ns},
          {"previous_receipt_digest", grant.previous_receipt_digest}};
}

nlohmann::json release_digest_json(const ResourceReleaseReceipt& receipt) {
  return {{"api_version", receipt.api_version},
          {"release_request_id", receipt.release_request_id},
          {"release_request_digest", receipt.release_request_digest},
          {"allocation_id", receipt.allocation_id},
          {"grant_digest", receipt.grant_digest},
          {"host_id", receipt.host_id},
          {"boot_id", receipt.boot_id},
          {"broker_epoch", receipt.broker_epoch},
          {"released_boottime_ns", receipt.released_boottime_ns},
          {"released_wall_time_ns", receipt.released_wall_time_ns},
          {"previous_receipt_digest", receipt.previous_receipt_digest}};
}

nlohmann::json release_request_digest_json(
    const ResourceReleaseRequest& request) {
  return {{"api_version", request.api_version},
          {"release_request_id", request.release_request_id},
          {"allocation_id", request.allocation_id},
          {"grant_digest", request.grant_digest},
          {"journal_id", request.journal_id},
          {"run_id", request.run_id},
          {"logical_lease_id", request.logical_lease_id},
          {"logical_fencing_token", request.logical_fencing_token}};
}

template <typename T>
T strict_decode(const nlohmann::json& source, std::string_view description) {
  T result;
  std::vector<Diagnostic> diagnostics;
  if (!decode_json(source, result, "", diagnostics)) {
    throw HostLedgerError(std::string(description) + " decoding failed");
  }
  return result;
}

using SchemaSnapshot = std::map<std::string, std::string>;

SchemaSnapshot schema_snapshot(sqlite3* database) {
  SchemaSnapshot result;
  Statement query(database, R"sql(
    SELECT type, name, sql FROM sqlite_master
    WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name
  )sql");
  int status = SQLITE_OK;
  while ((status = sqlite3_step(query.get())) == SQLITE_ROW) {
    result.emplace(column_text(query.get(), 0) + "\n" +
                       column_text(query.get(), 1),
                   column_text(query.get(), 2));
  }
  if (status != SQLITE_DONE) throw HostLedgerError("schema scan failed");
  return result;
}

const SchemaSnapshot& canonical_schema_v1() {
  static const SchemaSnapshot snapshot = [] {
    sqlite3* database = nullptr;
    if (sqlite3_open(":memory:", &database) != SQLITE_OK) {
      throw HostLedgerError("could not construct canonical ledger schema");
    }
    try {
      execute(database, kSchemaV1, "canonical v1 schema creation failed");
      auto value = schema_snapshot(database);
      sqlite3_close(database);
      return value;
    } catch (...) {
      sqlite3_close(database);
      throw;
    }
  }();
  return snapshot;
}

const SchemaSnapshot& canonical_schema_v2() {
  static const SchemaSnapshot snapshot = [] {
    sqlite3* database = nullptr;
    if (sqlite3_open(":memory:", &database) != SQLITE_OK) {
      throw HostLedgerError("could not construct canonical ledger schema");
    }
    try {
      execute(database, kSchemaV1, "canonical v1 schema creation failed");
      execute(database, kStartupAuditSchemaV2,
              "canonical startup-audit schema creation failed");
      auto value = schema_snapshot(database);
      sqlite3_close(database);
      return value;
    } catch (...) {
      sqlite3_close(database);
      throw;
    }
  }();
  return snapshot;
}

const SchemaSnapshot& canonical_schema_v3() {
  static const SchemaSnapshot snapshot = [] {
    sqlite3* database = nullptr;
    if (sqlite3_open(":memory:", &database) != SQLITE_OK) {
      throw HostLedgerError("could not construct canonical ledger schema");
    }
    try {
      execute(database, kSchemaV1, "canonical v1 schema creation failed");
      execute(database, kStartupAuditSchemaV2,
              "canonical startup-audit schema creation failed");
      execute(database, kAdmissionEpochSchemaV3,
              "canonical admission-epoch schema creation failed");
      auto value = schema_snapshot(database);
      sqlite3_close(database);
      return value;
    } catch (...) {
      sqlite3_close(database);
      throw;
    }
  }();
  return snapshot;
}

}  // namespace

struct SQLiteHostLedger::Implementation final {
  explicit Implementation(
      std::optional<HostStartupAuditPolicy> trusted_policy)
      : trusted_startup_audit_policy(std::move(trusted_policy)) {}

  sqlite3* database{};
  std::shared_ptr<HostLedgerFilesystemAuthority> authority;
  int sqlite_database_fd{-1};
  HostInventoryReceipt inventory;
  IHostLedgerFaultInjector* faults{};
  const std::optional<HostStartupAuditPolicy> trusted_startup_audit_policy;
  mutable std::mutex mutex;

  ~Implementation() {
    if (database != nullptr) sqlite3_close(database);
  }

  void fault(HostLedgerFaultPoint point) const {
    if (faults != nullptr) faults->hit(point);
  }

  void attest_authority() const {
    if (!authority || sqlite_database_fd < 0) {
      throw HostLedgerError("host-ledger filesystem authority is unavailable");
    }
    const auto attestation = authority->attest_after_open();
    struct stat status {};
    if (::fstat(sqlite_database_fd, &status) != 0 ||
        static_cast<std::uint64_t>(status.st_dev) !=
            attestation.database_file.device ||
        static_cast<std::uint64_t>(status.st_ino) !=
            attestation.database_file.inode) {
      throw HostLedgerError(
          "SQLite host-ledger descriptor no longer matches authority");
    }
    (void)authority->validate_auxiliary_files();
  }

  nlohmann::json make_record(std::string type, std::string id,
                             std::string subject,
                             nlohmann::json payload) const {
    return sealed_json(
        "trainvm.host-ledger-record/v1",
        {{"api_version", "trainvm.host-ledger-record/v1"},
         {"record_id", std::move(id)},
         {"record_type", std::move(type)},
         {"subject_id", std::move(subject)},
         {"host_id", inventory.host_id},
         {"boot_id", inventory.boot_id},
         {"broker_epoch", inventory.broker_epoch},
         {"payload", std::move(payload)}});
  }

  std::string append_record(const nlohmann::json& record) {
    const std::string canonical = record.dump();
    const std::string content =
        sha256("trainvm.host-ledger-content/v1", canonical);
    Statement head(database,
                   "SELECT last_sequence, chain_hash FROM ledger_chain_head "
                   "WHERE singleton=1");
    if (sqlite3_step(head.get()) != SQLITE_ROW) {
      throw HostLedgerError("ledger chain head is missing");
    }
    const std::int64_t previous_sequence = sqlite3_column_int64(head.get(), 0);
    const std::string previous = column_text(head.get(), 1);
    const std::string chain = sha256(
        "trainvm.host-ledger-chain/v1", previous + "\n" + content);
    Statement insert(database, R"sql(
      INSERT INTO ledger_records(
        record_id, record_type, subject_id, canonical_json, receipt_digest,
        content_hash, previous_hash, chain_hash
      ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
    )sql");
    bind_text(insert.get(), 1, record.at("record_id").get<std::string>());
    bind_text(insert.get(), 2, record.at("record_type").get<std::string>());
    bind_text(insert.get(), 3, record.at("subject_id").get<std::string>());
    bind_text(insert.get(), 4, canonical);
    bind_text(insert.get(), 5, record.at("receipt_digest").get<std::string>());
    bind_text(insert.get(), 6, content);
    bind_text(insert.get(), 7, previous);
    bind_text(insert.get(), 8, chain);
    require_done(database, insert.get(), "ledger record append failed");
    const std::int64_t sequence = sqlite3_last_insert_rowid(database);
    Statement update(database, R"sql(
      UPDATE ledger_chain_head SET last_sequence=?, chain_hash=?
      WHERE singleton=1 AND last_sequence=? AND chain_hash=?
    )sql");
    bind_integer(update.get(), 1, sequence);
    bind_text(update.get(), 2, chain);
    bind_integer(update.get(), 3, previous_sequence);
    bind_text(update.get(), 4, previous);
    require_done(database, update.get(), "ledger chain update failed");
    if (sqlite3_changes(database) != 1) {
      throw HostLedgerError("ledger chain update lost its CAS row");
    }
    return record.at("receipt_digest").get<std::string>();
  }

  HostLedgerChainHead chain_head_unlocked() const {
    Statement query(database,
                    "SELECT last_sequence, chain_hash FROM ledger_chain_head "
                    "WHERE singleton=1");
    if (sqlite3_step(query.get()) != SQLITE_ROW) {
      throw HostLedgerError("ledger chain head is missing");
    }
    const auto sequence = sqlite3_column_int64(query.get(), 0);
    if (sequence < 0) throw HostLedgerError("ledger sequence is negative");
    return {.ledger_sequence = static_cast<std::uint64_t>(sequence),
            .chain_hash = column_text(query.get(), 1)};
  }

  void sync_projection_head() {
    Statement update(database, R"sql(
      UPDATE projection_head
      SET applied_sequence=(SELECT last_sequence FROM ledger_chain_head),
          applied_chain_hash=(SELECT chain_hash FROM ledger_chain_head)
      WHERE singleton=1
    )sql");
    require_done(database, update.get(), "projection head update failed");
    if (sqlite3_changes(database) != 1) {
      throw HostLedgerError("projection head update lost its CAS row");
    }
  }

  ResourceOccupancySnapshot occupancy_unlocked() const {
    ResourceOccupancySnapshot value{
        .api_version = std::string(kHostResourceOccupancyApiVersion),
        .host_id = inventory.host_id,
        .boot_id = inventory.boot_id,
        .inventory_digest = inventory.inventory_digest,
        .ledger_sequence = 0,
        .active_fences = {},
        .occupancy_digest = {},
    };
    Statement head(database,
                   "SELECT last_sequence FROM ledger_chain_head WHERE singleton=1");
    if (sqlite3_step(head.get()) != SQLITE_ROW) {
      throw HostLedgerError("ledger head is missing");
    }
    value.ledger_sequence =
        static_cast<std::uint64_t>(sqlite3_column_int64(head.get(), 0));
    Statement query(database, R"sql(
      SELECT resources.resource_json, resources.generation,
             resources.inventory_digest, resources.topology_digest
      FROM active_resource_grants AS active
      JOIN allocation_resources AS resources
        ON resources.allocation_id=active.allocation_id
       AND resources.resource_key=active.resource_key
      ORDER BY active.resource_key
    )sql");
    int status = SQLITE_OK;
    while ((status = sqlite3_step(query.get())) == SQLITE_ROW) {
      HostResourceId id = strict_decode<HostResourceId>(
          nlohmann::json::parse(column_text(query.get(), 0)), "resource id");
      value.active_fences.push_back(
          {.resource = std::move(id),
           .generation = static_cast<std::uint64_t>(
               sqlite3_column_int64(query.get(), 1)),
           .inventory_digest = column_text(query.get(), 2),
           .topology_digest = column_text(query.get(), 3)});
    }
    if (status != SQLITE_DONE) throw HostLedgerError("occupancy query failed");
    return seal_resource_occupancy(inventory, std::move(value));
  }

  bool verify_unlocked(std::string* reason,
                       bool bind_instance_inventory = true,
                       unsigned int schema_version = 3U) const {
    const auto fail = [&](std::string message) {
      if (reason != nullptr) *reason = std::move(message);
      return false;
    };
    try {
      attest_authority();
      int defensive = 0;
      if (sqlite3_db_config(database, SQLITE_DBCONFIG_DEFENSIVE, -1,
                            &defensive) != SQLITE_OK ||
          defensive != 1) {
        return fail("SQLite defensive mode is not active");
      }
      if ((schema_version == 1U &&
           schema_snapshot(database) != canonical_schema_v1()) ||
          (schema_version == 2U &&
           schema_snapshot(database) != canonical_schema_v2()) ||
          (schema_version == 3U &&
           schema_snapshot(database) != canonical_schema_v3()) ||
          (schema_version != 1U && schema_version != 2U &&
           schema_version != 3U)) {
        return fail("ledger schema does not exactly match its declared version");
      }
      Statement application(database, "PRAGMA application_id");
      Statement version(database, "PRAGMA user_version");
      Statement journal_mode(database, "PRAGMA journal_mode");
      Statement synchronous(database, "PRAGMA synchronous");
      Statement foreign_keys(database, "PRAGMA foreign_keys");
      Statement trusted_schema(database, "PRAGMA trusted_schema");
      if (sqlite3_step(application.get()) != SQLITE_ROW ||
          sqlite3_column_int64(application.get(), 0) != 0x5456484cLL ||
          sqlite3_step(version.get()) != SQLITE_ROW ||
          sqlite3_column_int64(version.get(), 0) !=
              static_cast<std::int64_t>(schema_version) ||
          sqlite3_step(journal_mode.get()) != SQLITE_ROW ||
          column_text(journal_mode.get(), 0) != "wal" ||
          sqlite3_step(synchronous.get()) != SQLITE_ROW ||
          sqlite3_column_int64(synchronous.get(), 0) != 2 ||
          sqlite3_step(foreign_keys.get()) != SQLITE_ROW ||
          sqlite3_column_int64(foreign_keys.get(), 0) != 1 ||
          sqlite3_step(trusted_schema.get()) != SQLITE_ROW ||
          sqlite3_column_int64(trusted_schema.get(), 0) != 0) {
        return fail("ledger application, schema, or durability mode is invalid");
      }
      Statement identity(database,
                         "SELECT schema_version, host_id FROM ledger_identity "
                         "WHERE singleton=1");
      if (sqlite3_step(identity.get()) != SQLITE_ROW ||
          sqlite3_column_int64(identity.get(), 0) != 1 ||
          column_text(identity.get(), 1) != inventory.host_id) {
        return fail("ledger identity is missing or belongs to another host");
      }
      std::string previous(kGenesisDigest);
      std::int64_t last_sequence = 0;
      std::int64_t expected_sequence = 0;
      std::map<std::string, ResourceBundleRequest> request_evidence;
      std::map<std::string, ResourceBundleGrant> grant_evidence;
      std::map<std::string, nlohmann::json> busy_evidence;
      std::map<std::string, nlohmann::json> release_request_evidence;
      std::map<std::string, ResourceReleaseReceipt> release_evidence;
      std::optional<HostInventoryReceipt> replay_inventory;
      std::map<std::string, std::vector<ResourceFence>> replay_active_grants;
      std::map<std::string, HostStartupAuditReport> startup_audit_evidence;
      std::map<std::string,
               std::tuple<std::uint64_t, std::string, std::uint64_t,
                          std::string, std::string>>
          startup_audit_record_evidence;
      Statement records(database, R"sql(
        SELECT ledger_sequence, record_id, record_type, subject_id,
               canonical_json, receipt_digest, content_hash, previous_hash,
               chain_hash FROM ledger_records ORDER BY ledger_sequence
      )sql");
      int status = SQLITE_OK;
      while ((status = sqlite3_step(records.get())) == SQLITE_ROW) {
        last_sequence = sqlite3_column_int64(records.get(), 0);
        ++expected_sequence;
        const std::string canonical = column_text(records.get(), 4);
        const nlohmann::json record = nlohmann::json::parse(canonical);
        auto digest_value = record;
        digest_value.erase("receipt_digest");
        const std::string expected_receipt =
            sha256("trainvm.host-ledger-record/v1", digest_value.dump());
        const std::string expected_content =
            sha256("trainvm.host-ledger-content/v1", canonical);
        const std::string expected_chain = sha256(
            "trainvm.host-ledger-chain/v1", previous + "\n" + expected_content);
        if (last_sequence != expected_sequence || canonical != record.dump() ||
            record.value("record_id", std::string{}) !=
                column_text(records.get(), 1) ||
            record.value("record_type", std::string{}) !=
                column_text(records.get(), 2) ||
            record.value("subject_id", std::string{}) !=
                column_text(records.get(), 3) ||
            record.value("receipt_digest", std::string{}) != expected_receipt ||
            column_text(records.get(), 5) != expected_receipt ||
            column_text(records.get(), 6) != expected_content ||
            column_text(records.get(), 7) != previous ||
            column_text(records.get(), 8) != expected_chain) {
          return fail("ledger hash-chain mismatch at sequence " +
                      std::to_string(last_sequence));
        }
        const std::string record_type =
            record.at("record_type").get<std::string>();
        const nlohmann::json& payload = record.at("payload");
        if (record_type == "inventory.observed") {
          const HostInventoryReceipt observed = host_inventory_from_json(payload);
          if (record.value("record_id", std::string{}) !=
                  "inventory:" + observed.receipt_digest ||
              record.value("subject_id", std::string{}) !=
                  observed.inventory_digest ||
              record.value("host_id", std::string{}) != observed.host_id ||
              record.value("boot_id", std::string{}) != observed.boot_id ||
              record.value("broker_epoch", std::string{}) !=
                  observed.broker_epoch) {
            return fail("inventory record wrapper diverges from its evidence");
          }
          replay_inventory = observed;
        } else if (record_type == "startup.audit_committed") {
          if (schema_version < 2U) {
            return fail("v1 ledger contains v2 startup-audit evidence");
          }
          const HostStartupAuditReport report =
              decode_untrusted_host_startup_audit_report(payload);
          if (!replay_inventory.has_value()) {
            return fail("startup audit precedes inventory authority evidence");
          }
          std::vector<ResourceFence> active_fences;
          for (const auto& [allocation_id, fences] : replay_active_grants) {
            (void)allocation_id;
            active_fences.insert(active_fences.end(), fences.begin(),
                                 fences.end());
          }
          const auto replay_occupancy = seal_resource_occupancy(
              *replay_inventory,
              {.api_version = std::string(kHostResourceOccupancyApiVersion),
               .host_id = replay_inventory->host_id,
               .boot_id = replay_inventory->boot_id,
               .inventory_digest = replay_inventory->inventory_digest,
               .ledger_sequence =
                   static_cast<std::uint64_t>(last_sequence - 1),
               .active_fences = std::move(active_fences),
               .occupancy_digest = {}});
          const HostLedgerChainHead actual_predecessor{
              .ledger_sequence = static_cast<std::uint64_t>(last_sequence - 1),
              .chain_hash = previous};
          if (!record.is_object() || record.size() != 9U ||
              record.value("api_version", std::string{}) !=
                  "trainvm.host-ledger-record/v1" ||
              report.inventory != *replay_inventory ||
              report.pre_audit_occupancy != replay_occupancy ||
              report.post_audit_occupancy != replay_occupancy ||
              report.ledger_head_before != actual_predecessor ||
              report.ledger_head_after_observation != actual_predecessor ||
              !startup_audit_evidence.emplace(report.audit_id, report).second ||
              !startup_audit_record_evidence
                   .emplace(report.audit_id,
                            std::tuple{static_cast<std::uint64_t>(last_sequence),
                                       expected_chain,
                                       actual_predecessor.ledger_sequence,
                                       actual_predecessor.chain_hash,
                                       expected_receipt})
                   .second ||
              record.value("record_id", std::string{}) !=
                  "startup-audit:" + report.audit_id ||
              record.value("subject_id", std::string{}) !=
                  report.broker_instance_id ||
              record.value("host_id", std::string{}) != report.host_id ||
              record.value("boot_id", std::string{}) != report.boot_id ||
              record.value("broker_epoch", std::string{}) !=
                  report.broker_epoch) {
            return fail("duplicate or malformed startup-audit authority evidence");
          }
        } else if (record_type == "bundle.requested") {
          const ResourceBundleRequest request =
              resource_request_from_json(payload);
          if (!request_evidence.emplace(request.request_id, request).second) {
            return fail("duplicate bundle request authority evidence");
          }
        } else if (record_type == "bundle.granted") {
          const ResourceBundleGrant grant =
              resource_bundle_grant_from_json(payload);
          if (!grant_evidence.emplace(grant.allocation_id, grant).second ||
              !replay_active_grants.emplace(grant.allocation_id, grant.fences)
                   .second) {
            return fail("duplicate bundle grant authority evidence");
          }
        } else if (record_type == "bundle.denied") {
          const std::string request_id =
              payload.value("request_id", std::string{});
          if (request_id.empty() ||
              !busy_evidence.emplace(request_id, payload).second) {
            return fail("duplicate or malformed busy authority evidence");
          }
        } else if (record_type == "bundle.release_requested") {
          const std::string release_request_id =
              payload.value("release_request_id", std::string{});
          if (release_request_id.empty() ||
              !release_request_evidence
                   .emplace(release_request_id, payload)
                   .second) {
            return fail("duplicate or malformed release request evidence");
          }
        } else if (record_type == "bundle.released") {
          const ResourceReleaseReceipt receipt =
              resource_release_receipt_from_json(payload);
          if (!release_evidence.emplace(receipt.allocation_id, receipt).second ||
              replay_active_grants.erase(receipt.allocation_id) != 1U) {
            return fail("duplicate bundle release authority evidence");
          }
        }
        previous = expected_chain;
      }
      if (status != SQLITE_DONE) return fail("ledger record scan failed");
      Statement heads(database, R"sql(
        SELECT chain.last_sequence, chain.chain_hash,
               projection.applied_sequence, projection.applied_chain_hash
        FROM ledger_chain_head AS chain JOIN projection_head AS projection
          ON chain.singleton=projection.singleton WHERE chain.singleton=1
      )sql");
      if (sqlite3_step(heads.get()) != SQLITE_ROW ||
          sqlite3_column_int64(heads.get(), 0) != last_sequence ||
          column_text(heads.get(), 1) != previous ||
          sqlite3_column_int64(heads.get(), 2) != last_sequence ||
          column_text(heads.get(), 3) != previous) {
        return fail("ledger or projection head does not match the chain tail");
      }
      Statement current(database,
                        "SELECT inventory_digest, canonical_json "
                        "FROM current_inventory WHERE singleton=1");
      if (sqlite3_step(current.get()) != SQLITE_ROW) {
        return fail("current inventory projection is invalid");
      }
      const std::string persisted_inventory_digest = column_text(current.get(), 0);
      const std::string persisted_inventory_json = column_text(current.get(), 1);
      const HostInventoryReceipt persisted_inventory = host_inventory_from_json(
          nlohmann::json::parse(persisted_inventory_json));
      if (persisted_inventory_digest != persisted_inventory.inventory_digest ||
          persisted_inventory_json != host_inventory_json(persisted_inventory).dump()) {
        return fail("current inventory digest or canonical bytes are inconsistent");
      }
      if (persisted_inventory.host_id != inventory.host_id) {
        return fail("current inventory belongs to another host");
      }
      if (bind_instance_inventory && persisted_inventory != inventory) {
        return fail("ledger instance is stale relative to current inventory");
      }
      Statement latest_inventory(database, R"sql(
        SELECT canonical_json FROM ledger_records
        WHERE record_type='inventory.observed'
        ORDER BY ledger_sequence DESC LIMIT 1
      )sql");
      if (sqlite3_step(latest_inventory.get()) != SQLITE_ROW) {
        return fail("inventory projection has no authority record");
      }
      const nlohmann::json inventory_record = nlohmann::json::parse(
          column_text(latest_inventory.get(), 0));
      if (inventory_record.at("payload") !=
          host_inventory_json(persisted_inventory)) {
        return fail("inventory projection diverges from its authority record");
      }
      for (const auto& resource : persisted_inventory.resources) {
        Statement generation(database, R"sql(
          SELECT 1 FROM resource_generations WHERE resource_key=?
        )sql");
        bind_text(generation.get(), 1, canonical_resource_key(resource.id));
        if (sqlite3_step(generation.get()) != SQLITE_ROW) {
          return fail("inventory resource has no persistent generation");
        }
      }
      std::set<std::string> current_resource_keys;
      for (const auto& resource : persisted_inventory.resources) {
        current_resource_keys.insert(canonical_resource_key(resource.id));
      }
      Statement generations(database, R"sql(
        SELECT generation.resource_key, generation.generation,
               generation.last_allocation_id, generation.last_grant_digest,
               COALESCE((SELECT MAX(history.generation)
                         FROM allocation_resources AS history
                         WHERE history.resource_key=generation.resource_key), 0),
               resource.allocation_id, allocation.grant_digest,
               EXISTS(SELECT 1 FROM allocation_resources AS history
                      WHERE history.resource_key=generation.resource_key)
        FROM resource_generations AS generation
        LEFT JOIN allocation_resources AS resource
          ON resource.resource_key=generation.resource_key
         AND resource.generation=generation.generation
        LEFT JOIN allocations AS allocation
          ON allocation.allocation_id=resource.allocation_id
        ORDER BY generation.resource_key
      )sql");
      while ((status = sqlite3_step(generations.get())) == SQLITE_ROW) {
        const std::string resource_key = column_text(generations.get(), 0);
        const std::int64_t generation = sqlite3_column_int64(generations.get(), 1);
        const bool last_allocation_null =
            sqlite3_column_type(generations.get(), 2) == SQLITE_NULL;
        const bool last_grant_null =
            sqlite3_column_type(generations.get(), 3) == SQLITE_NULL;
        const std::int64_t maximum = sqlite3_column_int64(generations.get(), 4);
        const bool has_history = sqlite3_column_int(generations.get(), 7) != 0;
        if (generation != maximum || generation < 0 ||
            (generation == 0 &&
             (!last_allocation_null || !last_grant_null)) ||
            (generation > 0 &&
             (last_allocation_null || last_grant_null ||
              column_text(generations.get(), 2) !=
                  column_text(generations.get(), 5) ||
              column_text(generations.get(), 3) !=
                  column_text(generations.get(), 6))) ||
            (!current_resource_keys.contains(resource_key) && !has_history)) {
          return fail("resource generation projection diverges from grant evidence");
        }
      }
      if (status != SQLITE_DONE) return fail("resource generation scan failed");
      Statement active(database, R"sql(
        SELECT 1 FROM active_resource_grants AS active
        LEFT JOIN allocations AS allocation
          ON allocation.allocation_id=active.allocation_id
        LEFT JOIN allocation_resources AS resource
          ON resource.allocation_id=active.allocation_id
         AND resource.resource_key=active.resource_key
        WHERE allocation.status IS NULL OR allocation.status!='active'
           OR resource.generation IS NULL
           OR resource.generation!=active.generation
           OR allocation.grant_digest!=active.grant_digest LIMIT 1
      )sql");
      if (sqlite3_step(active.get()) == SQLITE_ROW) {
        return fail("active grant projection is inconsistent");
      }
      Statement active_resources(database, R"sql(
        SELECT resource.resource_json, resource.topology_digest
        FROM active_resource_grants AS active
        JOIN allocation_resources AS resource
          ON resource.allocation_id=active.allocation_id
         AND resource.resource_key=active.resource_key
        ORDER BY active.resource_key
      )sql");
      while ((status = sqlite3_step(active_resources.get())) == SQLITE_ROW) {
        const HostResourceId active_id = strict_decode<HostResourceId>(
            nlohmann::json::parse(column_text(active_resources.get(), 0)),
            "active resource");
        const auto observed = std::ranges::find_if(
            persisted_inventory.resources, [&](const auto& resource) {
              return resource.id.stable_id == active_id.stable_id;
            });
        if (observed == persisted_inventory.resources.end() ||
            observed->id != active_id ||
            column_text(active_resources.get(), 1) !=
                persisted_inventory.topology_digest) {
          return fail(
              "active resource vanished, changed identity, or degraded topology");
        }
      }
      if (status != SQLITE_DONE) return fail("active resource scan failed");
      Statement incomplete_active(database, R"sql(
        SELECT 1 FROM allocations AS allocation
        JOIN allocation_resources AS resource
          ON resource.allocation_id=allocation.allocation_id
        LEFT JOIN active_resource_grants AS active
          ON active.allocation_id=resource.allocation_id
         AND active.resource_key=resource.resource_key
        WHERE allocation.status='active' AND active.resource_key IS NULL
        LIMIT 1
      )sql");
      if (sqlite3_step(incomplete_active.get()) == SQLITE_ROW) {
        return fail("active allocation is missing part of its resource bundle");
      }
      Statement released(database, R"sql(
        SELECT 1 FROM allocations AS allocation
        WHERE allocation.status='released' AND EXISTS(
          SELECT 1 FROM active_resource_grants AS active
          WHERE active.allocation_id=allocation.allocation_id
        ) LIMIT 1
      )sql");
      if (sqlite3_step(released.get()) == SQLITE_ROW) {
        return fail("released allocation retains an active resource");
      }
      Statement outcomes(database, R"sql(
        SELECT request_id, request_digest, status, allocation_id,
               outcome_digest, canonical_outcome_json
        FROM request_outcomes ORDER BY request_id
      )sql");
      while ((status = sqlite3_step(outcomes.get())) == SQLITE_ROW) {
        const std::string outcome_status = column_text(outcomes.get(), 2);
        const nlohmann::json outcome =
            nlohmann::json::parse(column_text(outcomes.get(), 5));
        const std::string request_id = column_text(outcomes.get(), 0);
        const auto request_record = request_evidence.find(request_id);
        if (request_record == request_evidence.end() ||
            request_record->second.canonical_request_digest !=
                column_text(outcomes.get(), 1)) {
          return fail("request outcome has no exact request authority evidence");
        }
        if (outcome_status == "busy") {
          auto unsigned_outcome = outcome;
          unsigned_outcome.erase("receipt_digest");
          if (outcome.value("request_id", std::string{}) !=
                  column_text(outcomes.get(), 0) ||
              outcome.value("request_digest", std::string{}) !=
                  column_text(outcomes.get(), 1) ||
              outcome.value("status", std::string{}) != "busy" ||
              outcome.value("receipt_digest", std::string{}) !=
                  column_text(outcomes.get(), 4) ||
              column_text(outcomes.get(), 4) !=
                  sha256("trainvm.host-ledger-busy/v1",
                         unsigned_outcome.dump()) ||
              !busy_evidence.contains(request_id) ||
              busy_evidence.at(request_id) != outcome ||
              std::ranges::any_of(grant_evidence, [&](const auto& entry) {
                return entry.second.request_id == request_id;
              })) {
            return fail("busy request projection is not canonical");
          }
          continue;
        }
        const ResourceBundleGrant grant =
            resource_bundle_grant_from_json(outcome);
        if (grant.request_id != column_text(outcomes.get(), 0) ||
            grant.request_digest != column_text(outcomes.get(), 1) ||
            grant.allocation_id != column_text(outcomes.get(), 3) ||
            grant.receipt_digest != column_text(outcomes.get(), 4) ||
            !grant_evidence.contains(grant.allocation_id) ||
            grant_evidence.at(grant.allocation_id) != grant ||
            busy_evidence.contains(request_id)) {
          return fail("grant outcome projection is not canonical");
        }
        Statement allocation(database, R"sql(
          SELECT request_id, request_digest, grant_digest, journal_id, run_id,
                 logical_lease_id, logical_fencing_token, host_id,
                 grant_boot_id, broker_epoch
          FROM allocations
          WHERE allocation_id=?
        )sql");
        bind_text(allocation.get(), 1, grant.allocation_id);
        if (sqlite3_step(allocation.get()) != SQLITE_ROW ||
            column_text(allocation.get(), 0) != grant.request_id ||
            column_text(allocation.get(), 1) != grant.request_digest ||
            column_text(allocation.get(), 2) != grant.receipt_digest ||
            column_text(allocation.get(), 3) != grant.journal_id ||
            column_text(allocation.get(), 4) != grant.run_id ||
            column_text(allocation.get(), 5) != grant.logical_lease_id ||
            static_cast<std::uint64_t>(
                sqlite3_column_int64(allocation.get(), 6)) !=
                grant.logical_fencing_token ||
            column_text(allocation.get(), 7) != grant.host_id ||
            column_text(allocation.get(), 8) != grant.boot_id ||
            column_text(allocation.get(), 9) != grant.broker_epoch) {
          return fail("grant outcome has no exact allocation projection");
        }
        Statement resources(database, R"sql(
          SELECT resource_key, resource_json, generation, inventory_digest,
                 topology_digest FROM allocation_resources
          WHERE allocation_id=? ORDER BY resource_key
        )sql");
        bind_text(resources.get(), 1, grant.allocation_id);
        std::vector<ResourceFence> projected;
        int resource_status = SQLITE_OK;
        while ((resource_status = sqlite3_step(resources.get())) == SQLITE_ROW) {
          projected.push_back(
              {.resource = strict_decode<HostResourceId>(
                   nlohmann::json::parse(column_text(resources.get(), 1)),
                   "projected resource"),
               .generation = static_cast<std::uint64_t>(
                   sqlite3_column_int64(resources.get(), 2)),
               .inventory_digest = column_text(resources.get(), 3),
               .topology_digest = column_text(resources.get(), 4)});
          if (canonical_resource_key(projected.back().resource) !=
              column_text(resources.get(), 0)) {
            return fail("allocation resource key is not canonical");
          }
        }
        if (resource_status != SQLITE_DONE || projected != grant.fences) {
          return fail("allocation resources diverge from the grant receipt");
        }
      }
      if (status != SQLITE_DONE) return fail("request outcome scan failed");
      Statement outcome_count(database, "SELECT COUNT(*) FROM request_outcomes");
      if (sqlite3_step(outcome_count.get()) != SQLITE_ROW ||
          static_cast<std::size_t>(
              sqlite3_column_int64(outcome_count.get(), 0)) !=
              request_evidence.size() ||
          grant_evidence.size() + busy_evidence.size() !=
              request_evidence.size()) {
        return fail("request authority evidence and outcomes are not closed");
      }
      Statement releases(database, R"sql(
        SELECT release_request_id, release_request_digest, allocation_id, grant_digest,
               release_receipt_digest, canonical_release_json
        FROM release_outcomes ORDER BY release_request_id
      )sql");
      while ((status = sqlite3_step(releases.get())) == SQLITE_ROW) {
        const ResourceReleaseReceipt receipt =
            resource_release_receipt_from_json(
                nlohmann::json::parse(column_text(releases.get(), 5)));
        if (receipt.release_request_id != column_text(releases.get(), 0) ||
            receipt.release_request_digest != column_text(releases.get(), 1) ||
            receipt.allocation_id != column_text(releases.get(), 2) ||
            receipt.grant_digest != column_text(releases.get(), 3) ||
            receipt.receipt_digest != column_text(releases.get(), 4) ||
            !release_request_evidence.contains(receipt.release_request_id) ||
            sha256("trainvm.host-resource-release-request/v1",
                   release_request_evidence.at(receipt.release_request_id).dump()) !=
                receipt.release_request_digest ||
            !release_evidence.contains(receipt.allocation_id) ||
            release_evidence.at(receipt.allocation_id) != receipt) {
          return fail("release outcome projection is not canonical");
        }
        Statement allocation(database, R"sql(
          SELECT status, release_digest FROM allocations WHERE allocation_id=?
        )sql");
        bind_text(allocation.get(), 1, receipt.allocation_id);
        if (sqlite3_step(allocation.get()) != SQLITE_ROW ||
            column_text(allocation.get(), 0) != "released" ||
            column_text(allocation.get(), 1) != receipt.receipt_digest) {
          return fail("release receipt has no exact terminal allocation");
        }
      }
      if (status != SQLITE_DONE) return fail("release outcome scan failed");
      Statement allocation_closure(database, R"sql(
        SELECT 1 FROM allocations AS allocation
        LEFT JOIN request_outcomes AS outcome
          ON outcome.request_id=allocation.request_id
        WHERE outcome.status IS NULL OR outcome.status!='granted'
           OR outcome.allocation_id!=allocation.allocation_id
           OR outcome.request_digest!=allocation.request_digest
           OR outcome.outcome_digest!=allocation.grant_digest
           OR NOT EXISTS(
             SELECT 1 FROM allocation_resources AS resource
             WHERE resource.allocation_id=allocation.allocation_id
           )
           OR (allocation.status='active' AND
               (allocation.release_digest IS NOT NULL OR EXISTS(
                 SELECT 1 FROM release_outcomes AS release
                 WHERE release.allocation_id=allocation.allocation_id
               )))
           OR (allocation.status='released' AND
               (allocation.release_digest IS NULL OR
                (SELECT COUNT(*) FROM release_outcomes AS release
                 WHERE release.allocation_id=allocation.allocation_id)!=1 OR
                NOT EXISTS(
                  SELECT 1 FROM release_outcomes AS release
                  WHERE release.allocation_id=allocation.allocation_id
                    AND release.release_receipt_digest=allocation.release_digest
                )))
        LIMIT 1
      )sql");
      if (sqlite3_step(allocation_closure.get()) == SQLITE_ROW) {
        return fail("allocation projections are not bidirectionally closed");
      }
      Statement allocation_count(database, "SELECT COUNT(*) FROM allocations");
      Statement release_count(database, "SELECT COUNT(*) FROM release_outcomes");
      if (sqlite3_step(allocation_count.get()) != SQLITE_ROW ||
          static_cast<std::size_t>(
              sqlite3_column_int64(allocation_count.get(), 0)) !=
              grant_evidence.size() ||
          sqlite3_step(release_count.get()) != SQLITE_ROW ||
          static_cast<std::size_t>(
              sqlite3_column_int64(release_count.get(), 0)) !=
              release_evidence.size() ||
          release_request_evidence.size() != release_evidence.size()) {
        return fail("grant/release authority evidence has missing projections");
      }
      if (schema_version >= 2U) {
        Statement extension(database, R"sql(
          SELECT schema_version FROM ledger_schema_extensions
          WHERE feature='startup_audit'
        )sql");
        if (sqlite3_step(extension.get()) != SQLITE_ROW ||
            sqlite3_column_int64(extension.get(), 0) != 2) {
          return fail("startup-audit schema extension marker is invalid");
        }
        Statement extension_count(database,
                                  "SELECT COUNT(*) FROM ledger_schema_extensions");
        if (sqlite3_step(extension_count.get()) != SQLITE_ROW ||
            sqlite3_column_int64(extension_count.get(), 0) !=
                (schema_version == 2U ? 1 : 2)) {
          return fail("unknown ledger schema extension is present");
        }
        Statement audits(database, R"sql(
          SELECT audit_id, report_digest, canonical_report_json,
                 record_receipt_digest, record_sequence, record_chain_hash,
                 record_previous_sequence, record_previous_hash,
                 receipt_digest, canonical_receipt_json
          FROM startup_audit_outcomes ORDER BY audit_id
        )sql");
        std::size_t audit_projection_count = 0U;
        while ((status = sqlite3_step(audits.get())) == SQLITE_ROW) {
          ++audit_projection_count;
          const std::string audit_id = column_text(audits.get(), 0);
          const auto evidence = startup_audit_evidence.find(audit_id);
          const auto record_evidence =
              startup_audit_record_evidence.find(audit_id);
          if (evidence == startup_audit_evidence.end() ||
              record_evidence == startup_audit_record_evidence.end()) {
            return fail("startup-audit projection has no authority record");
          }
          const HostStartupAuditReport projected_report =
              decode_untrusted_host_startup_audit_report(
                  nlohmann::json::parse(column_text(audits.get(), 2)));
          const HostStartupAuditReceipt projected_receipt =
              decode_untrusted_host_startup_audit_receipt(
                  nlohmann::json::parse(column_text(audits.get(), 9)),
                  projected_report);
          const auto& [record_sequence, record_chain_hash,
                       record_previous_sequence, record_previous_hash,
                       record_receipt_digest] = record_evidence->second;
          if (projected_report != evidence->second ||
              column_text(audits.get(), 1) != projected_report.report_digest ||
              column_text(audits.get(), 2) !=
                  host_startup_audit_report_json(projected_report).dump() ||
              column_text(audits.get(), 3) != record_receipt_digest ||
              static_cast<std::uint64_t>(sqlite3_column_int64(audits.get(), 4)) !=
                  record_sequence ||
              column_text(audits.get(), 5) != record_chain_hash ||
              static_cast<std::uint64_t>(sqlite3_column_int64(audits.get(), 6)) !=
                  record_previous_sequence ||
              column_text(audits.get(), 7) != record_previous_hash ||
              projected_report.ledger_head_before !=
                  HostLedgerChainHead{.ledger_sequence = record_previous_sequence,
                                      .chain_hash = record_previous_hash} ||
              column_text(audits.get(), 8) != projected_receipt.receipt_digest ||
              column_text(audits.get(), 9) !=
                  host_startup_audit_receipt_json(projected_receipt,
                                                  projected_report)
                      .dump() ||
              projected_receipt.commit_record_digest !=
                  record_receipt_digest ||
              projected_receipt.committed_ledger_head !=
                  HostLedgerChainHead{.ledger_sequence = record_sequence,
                                      .chain_hash = record_chain_hash}) {
            return fail("startup-audit projection is not canonical");
          }
        }
        if (status != SQLITE_DONE ||
            audit_projection_count != startup_audit_evidence.size()) {
          return fail("startup-audit evidence and projections are not closed");
        }
        if (schema_version == 3U) {
          Statement admission_extension(database, R"sql(
            SELECT schema_version FROM ledger_schema_extensions
            WHERE feature='admission_epoch'
          )sql");
          if (sqlite3_step(admission_extension.get()) != SQLITE_ROW ||
              sqlite3_column_int64(admission_extension.get(), 0) != 3 ||
              sqlite3_step(admission_extension.get()) != SQLITE_DONE) {
            return fail("admission-epoch schema extension marker is invalid");
          }
          Statement epochs(database, R"sql(
            SELECT epoch.epoch_digest, epoch.api_version, epoch.audit_id,
                   epoch.report_digest, epoch.audit_receipt_digest,
                   epoch.host_id, epoch.boot_id, epoch.broker_epoch,
                   epoch.inventory_digest, epoch.audit_record_sequence,
                   epoch.audit_record_chain_hash,
                   epoch.finalized_occupancy_digest,
                   epoch.finalized_boottime_ns, epoch.finalized_wall_time_ns,
                   epoch.canonical_json, audit.canonical_report_json,
                   audit.canonical_receipt_json
            FROM admission_epochs AS epoch
            JOIN startup_audit_outcomes AS audit
              ON audit.audit_id=epoch.audit_id
            ORDER BY epoch.epoch_digest
          )sql");
          std::set<std::string> epoch_digests;
          while ((status = sqlite3_step(epochs.get())) == SQLITE_ROW) {
            const HostStartupAuditReport report =
                decode_untrusted_host_startup_audit_report(
                    nlohmann::json::parse(column_text(epochs.get(), 15)));
            const HostStartupAuditReceipt receipt =
                decode_untrusted_host_startup_audit_receipt(
                    nlohmann::json::parse(column_text(epochs.get(), 16)),
                    report);
            ResourceOccupancySnapshot finalized = report.post_audit_occupancy;
            finalized.ledger_sequence =
                receipt.committed_ledger_head.ledger_sequence;
            finalized.occupancy_digest.clear();
            finalized = seal_resource_occupancy(report.inventory,
                                                std::move(finalized));
            const HostLedgerTime finalized_at{
                .boottime_ns = sqlite3_column_int64(epochs.get(), 12),
                .wall_time_ns = sqlite3_column_int64(epochs.get(), 13)};
            const nlohmann::json canonical = admission_epoch_json(
                report, receipt, finalized, finalized_at);
            const std::string epoch_digest = column_text(epochs.get(), 0);
            const bool blocking = std::ranges::any_of(
                report.findings, [](const HostStartupAuditFinding& finding) {
                  return finding.severity ==
                         HostStartupAuditFindingSeverity::blocking;
                });
            if (!epoch_digests.insert(epoch_digest).second ||
                report.disposition != HostStartupAuditDisposition::passed ||
                blocking ||
                column_text(epochs.get(), 1) !=
                    kHostLedgerAdmissionEpochApiVersion ||
                column_text(epochs.get(), 2) != report.audit_id ||
                column_text(epochs.get(), 3) != report.report_digest ||
                column_text(epochs.get(), 4) != receipt.receipt_digest ||
                column_text(epochs.get(), 5) != report.host_id ||
                column_text(epochs.get(), 6) != report.boot_id ||
                column_text(epochs.get(), 7) != report.broker_epoch ||
                column_text(epochs.get(), 8) !=
                    report.inventory.inventory_digest ||
                static_cast<std::uint64_t>(
                    sqlite3_column_int64(epochs.get(), 9)) !=
                    receipt.committed_ledger_head.ledger_sequence ||
                column_text(epochs.get(), 10) !=
                    receipt.committed_ledger_head.chain_hash ||
                column_text(epochs.get(), 11) !=
                    finalized.occupancy_digest ||
                !valid_time(finalized_at) ||
                column_text(epochs.get(), 14) != canonical.dump() ||
                epoch_digest != canonical.at("epoch_digest").get<std::string>()) {
              return fail("admission epoch projection is not canonical");
            }
          }
          if (status != SQLITE_DONE) return fail("admission epoch scan failed");
          Statement active_epoch(database, R"sql(
            SELECT active.epoch_digest FROM active_admission_epoch AS active
            JOIN admission_epochs AS epoch
              ON epoch.epoch_digest=active.epoch_digest
            WHERE active.singleton=1
          )sql");
          const int active_status = sqlite3_step(active_epoch.get());
          if ((active_status != SQLITE_DONE && active_status != SQLITE_ROW) ||
              (active_status == SQLITE_ROW &&
               (!epoch_digests.contains(column_text(active_epoch.get(), 0)) ||
                sqlite3_step(active_epoch.get()) != SQLITE_DONE))) {
            return fail("active admission epoch projection is invalid");
          }
          Statement authorizations(database, R"sql(
            SELECT authorization.request_id, authorization.request_digest,
                   authorization.epoch_digest, outcome.request_digest
            FROM request_admission_epochs AS authorization
            LEFT JOIN request_outcomes AS outcome
              ON outcome.request_id=authorization.request_id
            ORDER BY authorization.request_id
          )sql");
          std::set<std::string> covered_requests;
          while ((status = sqlite3_step(authorizations.get())) == SQLITE_ROW) {
            const std::string request_id =
                column_text(authorizations.get(), 0);
            const auto evidence = request_evidence.find(request_id);
            if (!covered_requests.insert(request_id).second ||
                evidence == request_evidence.end() ||
                evidence->second.canonical_request_digest !=
                    column_text(authorizations.get(), 1) ||
                !epoch_digests.contains(column_text(authorizations.get(), 2)) ||
                sqlite3_column_type(authorizations.get(), 3) == SQLITE_NULL ||
                column_text(authorizations.get(), 3) !=
                    column_text(authorizations.get(), 1)) {
              return fail("request admission authorization is not closed");
            }
          }
          if (status != SQLITE_DONE)
            return fail("request admission authorization scan failed");
          Statement exemptions(database, R"sql(
            SELECT exemption.request_id, exemption.request_digest,
                   exemption.reason, outcome.request_digest,
                   authorization.request_id
            FROM request_admission_exemptions AS exemption
            LEFT JOIN request_outcomes AS outcome
              ON outcome.request_id=exemption.request_id
            LEFT JOIN request_admission_epochs AS authorization
              ON authorization.request_id=exemption.request_id
            ORDER BY exemption.request_id
          )sql");
          while ((status = sqlite3_step(exemptions.get())) == SQLITE_ROW) {
            const std::string request_id = column_text(exemptions.get(), 0);
            const auto evidence = request_evidence.find(request_id);
            const std::string exemption_reason = column_text(exemptions.get(), 2);
            if (!covered_requests.insert(request_id).second ||
                evidence == request_evidence.end() ||
                evidence->second.canonical_request_digest !=
                    column_text(exemptions.get(), 1) ||
                (exemption_reason != "pre_v3" &&
                 exemption_reason != "policy_unconfigured") ||
                sqlite3_column_type(exemptions.get(), 3) == SQLITE_NULL ||
                column_text(exemptions.get(), 3) !=
                    column_text(exemptions.get(), 1) ||
                sqlite3_column_type(exemptions.get(), 4) != SQLITE_NULL) {
              return fail("request admission exemption is not canonical");
            }
          }
          if (status != SQLITE_DONE)
            return fail("request admission exemption scan failed");
          if (covered_requests.size() != request_evidence.size()) {
            return fail(
                "request outcome is missing admission authorization or durable exemption");
          }
        }
      } else if (!startup_audit_evidence.empty()) {
        return fail("v1 ledger contains startup-audit evidence");
      }
      if (reason != nullptr) reason->clear();
      return true;
    } catch (const std::exception& error) {
      return fail(error.what());
    }
  }
};

SQLiteHostLedger::SQLiteHostLedger(
    std::shared_ptr<HostLedgerFilesystemAuthority> authority,
    HostInventoryReceipt inventory,
    IHostLedgerFaultInjector* fault_injector,
    std::optional<HostStartupAuditPolicy> trusted_startup_audit_policy)
    : implementation_(std::make_unique<Implementation>(
          std::move(trusted_startup_audit_policy))) {
  if (implementation_->trusted_startup_audit_policy) {
    try {
      if (canonicalize_host_startup_audit_policy(
              *implementation_->trusted_startup_audit_policy) !=
          *implementation_->trusted_startup_audit_policy) {
        throw HostLedgerError(
            "trusted startup-audit policy is not canonical");
      }
    } catch (const HostStartupAuditError&) {
      throw HostLedgerError("trusted startup-audit policy is invalid");
    }
  }
  if (!authority) {
    throw HostLedgerError("host ledger requires filesystem authority");
  }
  validate_host_inventory(inventory);
  implementation_->authority = std::move(authority);
  implementation_->inventory = std::move(inventory);
  implementation_->faults = fault_injector;
  {
    std::scoped_lock open_guard(sqlite_open_mutex());
    const auto before = open_file_descriptors();
    const auto before_attestation =
        implementation_->authority->attest_before_open();
    (void)implementation_->authority->validate_auxiliary_files();
    const auto& path = implementation_->authority->ledger_path();
    if (sqlite3_open_v2(path.c_str(), &implementation_->database,
                        SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE |
                            SQLITE_OPEN_FULLMUTEX | SQLITE_OPEN_NOFOLLOW,
                        nullptr) != SQLITE_OK) {
      throw HostLedgerError("could not open host ledger");
    }
    implementation_->sqlite_database_fd =
        find_new_database_descriptor(before,
                                     before_attestation.database_file);
    implementation_->attest_authority();
  }
  int defensive = 0;
  if (sqlite3_db_config(implementation_->database,
                        SQLITE_DBCONFIG_DEFENSIVE, 1, &defensive) != SQLITE_OK ||
      defensive != 1) {
    throw HostLedgerError("could not enable SQLite defensive mode");
  }
  (void)sqlite3_extended_result_codes(implementation_->database, 1);
  execute(implementation_->database, R"sql(
    PRAGMA synchronous=FULL;
    PRAGMA foreign_keys=ON;
    PRAGMA busy_timeout=5000;
    PRAGMA trusted_schema=OFF;
  )sql", "could not configure host ledger");
  const bool initialized = [&] {
    Statement exists(implementation_->database, R"sql(
      SELECT EXISTS(SELECT 1 FROM sqlite_master
                    WHERE type='table' AND name='ledger_identity')
    )sql");
    if (sqlite3_step(exists.get()) != SQLITE_ROW) {
      throw HostLedgerError("could not inspect host ledger");
    }
    return sqlite3_column_int(exists.get(), 0) != 0;
  }();
  if (!initialized) {
    if (!schema_snapshot(implementation_->database).empty()) {
      throw HostLedgerError("refusing unversioned nonempty host ledger");
    }
    execute(implementation_->database, "PRAGMA journal_mode=WAL;",
            "could not enable host ledger WAL");
    Transaction transaction(implementation_->database);
    execute(implementation_->database, kSchemaV1,
            "could not create host ledger core v1");
    execute(implementation_->database, kStartupAuditSchemaV2,
            "could not create startup-audit schema v2");
    execute(implementation_->database, kAdmissionEpochSchemaV3,
            "could not create admission-epoch schema v3");
    execute(implementation_->database,
            "PRAGMA application_id=0x5456484c; PRAGMA user_version=3;",
            "could not mark host ledger schema");
    Statement identity(implementation_->database, R"sql(
      INSERT INTO ledger_identity(singleton, schema_version, ledger_id, host_id,
                                  created_boot_id, created_wall_time_ns)
      VALUES(1, 1, ?, ?, ?, 0)
    )sql");
    bind_text(identity.get(), 1, random_id("ledger-"));
    bind_text(identity.get(), 2, implementation_->inventory.host_id);
    bind_text(identity.get(), 3, implementation_->inventory.boot_id);
    require_done(implementation_->database, identity.get(),
                 "host ledger identity insert failed");
    execute(implementation_->database,
            "INSERT INTO ledger_schema_extensions(feature, schema_version) "
            "VALUES('startup_audit', 2);"
            "INSERT INTO ledger_schema_extensions(feature, schema_version) "
            "VALUES('admission_epoch', 3);",
            "host ledger extension marker insert failed");
    Statement head(implementation_->database, R"sql(
      INSERT INTO ledger_chain_head VALUES(1, 0, ?)
    )sql");
    bind_text(head.get(), 1, std::string(kGenesisDigest));
    require_done(implementation_->database, head.get(),
                 "host ledger head insert failed");
    Statement projection(implementation_->database, R"sql(
      INSERT INTO projection_head VALUES(1, 0, ?)
    )sql");
    bind_text(projection.get(), 1, std::string(kGenesisDigest));
    require_done(implementation_->database, projection.get(),
                 "projection head insert failed");
    const auto record = implementation_->make_record(
        "inventory.observed",
        "inventory:" + implementation_->inventory.receipt_digest,
        implementation_->inventory.inventory_digest,
        host_inventory_json(implementation_->inventory));
    implementation_->append_record(record);
    Statement current(implementation_->database, R"sql(
      INSERT INTO current_inventory(singleton, inventory_digest, canonical_json)
      VALUES(1, ?, ?)
    )sql");
    bind_text(current.get(), 1, implementation_->inventory.inventory_digest);
    bind_text(current.get(), 2,
              host_inventory_json(implementation_->inventory).dump());
    require_done(implementation_->database, current.get(),
                 "inventory projection insert failed");
    for (const auto& resource : implementation_->inventory.resources) {
      Statement generation(implementation_->database, R"sql(
        INSERT INTO resource_generations(resource_key, generation)
        VALUES(?, 0)
      )sql");
      bind_text(generation.get(), 1, canonical_resource_key(resource.id));
      require_done(implementation_->database, generation.get(),
                   "resource generation insert failed");
    }
    implementation_->sync_projection_head();
    std::string reason;
    if (!implementation_->verify_unlocked(&reason)) {
      throw HostLedgerError("new host ledger failed verification: " + reason);
    }
    transaction.commit();
  } else {
    const SchemaSnapshot opened_schema =
        schema_snapshot(implementation_->database);
    Statement opened_version(implementation_->database, "PRAGMA user_version");
    if (sqlite3_step(opened_version.get()) != SQLITE_ROW) {
      throw HostLedgerError("could not read host ledger schema version");
    }
    const std::int64_t user_version =
        sqlite3_column_int64(opened_version.get(), 0);
    if (opened_schema == canonical_schema_v1() && user_version == 1) {
      Transaction migration(implementation_->database);
      std::string migration_reason;
      if (!implementation_->verify_unlocked(&migration_reason, false, 1U)) {
        throw HostLedgerError("refusing startup-audit migration: " +
                              migration_reason);
      }
      execute(implementation_->database, kStartupAuditSchemaV2,
              "could not create startup-audit schema v2");
      implementation_->fault(
          HostLedgerFaultPoint::after_startup_audit_migration_schema);
      execute(implementation_->database,
              "INSERT INTO ledger_schema_extensions(feature, schema_version) "
              "VALUES('startup_audit', 2); PRAGMA user_version=2;",
              "could not mark startup-audit schema v2");
      if (!implementation_->verify_unlocked(&migration_reason, false, 2U)) {
        throw HostLedgerError("startup-audit migration verification failed: " +
                              migration_reason);
      }
      migration.commit();
    } else if ((opened_schema != canonical_schema_v2() || user_version != 2) &&
               (opened_schema != canonical_schema_v3() || user_version != 3)) {
      throw HostLedgerError(
          "refusing unknown or partially migrated host ledger schema");
    }
    const std::int64_t current_schema_version = [&] {
      Statement current_version(implementation_->database,
                                "PRAGMA user_version");
      if (sqlite3_step(current_version.get()) != SQLITE_ROW)
        throw HostLedgerError("could not reread host ledger schema version");
      return sqlite3_column_int64(current_version.get(), 0);
    }();
    if (current_schema_version == 2) {
      Transaction migration(implementation_->database);
      std::string migration_reason;
      if (!implementation_->verify_unlocked(&migration_reason, false, 2U)) {
        throw HostLedgerError("refusing admission-epoch migration: " +
                              migration_reason);
      }
      execute(implementation_->database, kAdmissionEpochSchemaV3,
              "could not create admission-epoch schema v3");
      execute(implementation_->database,
              "INSERT INTO request_admission_exemptions("
              "request_id, request_digest, reason) "
              "SELECT request_id, request_digest, 'pre_v3' "
              "FROM request_outcomes;",
              "could not freeze pre-admission request outcome closure");
      execute(implementation_->database,
              "INSERT INTO ledger_schema_extensions(feature, schema_version) "
              "VALUES('admission_epoch', 3); PRAGMA user_version=3;",
              "could not mark admission-epoch schema v3");
      if (!implementation_->verify_unlocked(&migration_reason, false, 3U)) {
        throw HostLedgerError("admission-epoch migration verification failed: " +
                              migration_reason);
      }
      migration.commit();
    }
    Transaction transaction(implementation_->database);
    std::string reason;
    if (!implementation_->verify_unlocked(&reason, false)) {
      throw HostLedgerError("refusing host ledger open: " + reason);
    }
    const std::string persisted_inventory_json = [&] {
      Statement current(implementation_->database,
                        "SELECT canonical_json FROM current_inventory "
                        "WHERE singleton=1");
      if (sqlite3_step(current.get()) != SQLITE_ROW) {
        throw HostLedgerError("current inventory projection is missing");
      }
      return column_text(current.get(), 0);
    }();
    if (persisted_inventory_json !=
        host_inventory_json(implementation_->inventory).dump()) {
      const auto record = implementation_->make_record(
          "inventory.observed",
          "inventory:" + implementation_->inventory.receipt_digest,
          implementation_->inventory.inventory_digest,
          host_inventory_json(implementation_->inventory));
      implementation_->append_record(record);
      Statement update(implementation_->database, R"sql(
        UPDATE current_inventory SET inventory_digest=?, canonical_json=?
        WHERE singleton=1
      )sql");
      bind_text(update.get(), 1, implementation_->inventory.inventory_digest);
      bind_text(update.get(), 2,
                host_inventory_json(implementation_->inventory).dump());
      require_done(implementation_->database, update.get(),
                   "inventory projection update failed");
      for (const auto& resource : implementation_->inventory.resources) {
        Statement generation(implementation_->database, R"sql(
          INSERT INTO resource_generations(resource_key, generation)
          VALUES(?, 0) ON CONFLICT(resource_key) DO NOTHING
        )sql");
        bind_text(generation.get(), 1, canonical_resource_key(resource.id));
        require_done(implementation_->database, generation.get(),
                     "resource generation update failed");
      }
      implementation_->sync_projection_head();
    }
    if (!implementation_->verify_unlocked(&reason)) {
      throw HostLedgerError("published inventory failed verification: " + reason);
    }
    transaction.commit();
  }
}

SQLiteHostLedger::~SQLiteHostLedger() = default;

BundleRequestResult SQLiteHostLedger::request_bundle(
    const ResourceBundleRequest& request, const HostLedgerTime& now) {
  if (implementation_->trusted_startup_audit_policy) {
    throw HostLedgerConflict(
        "bundle admission is sealed until an exact startup audit is finalized");
  }
  return request_bundle_authorized(request, now, nullptr);
}

BundleRequestResult SQLiteHostLedger::request_bundle(
    const ResourceBundleRequest& request, const HostLedgerTime& now,
    const HostLedgerAdmissionEpoch& admission_epoch) {
  if (!implementation_->trusted_startup_audit_policy) {
    throw HostLedgerConflict(
        "an admission epoch is invalid for a ledger without startup policy");
  }
  return request_bundle_authorized(request, now, &admission_epoch);
}

std::optional<BundleRequestResult> SQLiteHostLedger::reconcile_bundle_outcome(
    const ResourceBundleRequest& request) const {
  validate_resource_request(request);
  std::scoped_lock lock(implementation_->mutex);
  ReadTransaction transaction(implementation_->database);
  std::string reason;
  if (!implementation_->verify_unlocked(&reason)) {
    throw HostLedgerError("refusing request outcome reconciliation: " + reason);
  }

  Statement outcome(implementation_->database, R"sql(
    SELECT request_digest, status, canonical_outcome_json, outcome_digest
    FROM request_outcomes WHERE request_id=?
  )sql");
  bind_text(outcome.get(), 1, request.request_id);
  const int status = sqlite3_step(outcome.get());
  if (status == SQLITE_DONE) {
    transaction.commit();
    return std::nullopt;
  }
  if (status != SQLITE_ROW)
    throw HostLedgerError("request outcome reconciliation query failed");
  if (column_text(outcome.get(), 0) != request.canonical_request_digest) {
    throw HostLedgerConflict("request_id already has different content");
  }

  const std::string outcome_status = column_text(outcome.get(), 1);
  if (outcome_status != "granted" && outcome_status != "busy")
    throw HostLedgerError("request outcome reconciliation status is invalid");
  BundleRequestResult result{
      .status = outcome_status == "granted" ? BundleRequestStatus::granted
                                             : BundleRequestStatus::busy,
      .grant = std::nullopt,
      .outcome_digest = column_text(outcome.get(), 3),
      .replayed = true};
  if (result.status == BundleRequestStatus::granted) {
    result.grant = resource_bundle_grant_from_json(
        nlohmann::json::parse(column_text(outcome.get(), 2)));
    if (result.grant->request_id != request.request_id ||
        result.grant->request_digest != request.canonical_request_digest ||
        result.grant->journal_id != request.journal_id ||
        result.grant->run_id != request.run_id ||
        result.grant->logical_lease_id != request.logical_lease_id ||
        result.grant->logical_fencing_token !=
            request.logical_fencing_token ||
        result.grant->receipt_digest != result.outcome_digest) {
      throw HostLedgerError(
          "reconciled grant does not exactly match request attribution");
    }
  } else if (nlohmann::json::parse(column_text(outcome.get(), 2))
                 .value("status", std::string{}) != "busy") {
    throw HostLedgerError("reconciled busy outcome is malformed");
  }
  if (sqlite3_step(outcome.get()) != SQLITE_DONE)
    throw HostLedgerError("request outcome reconciliation is not unique");
  transaction.commit();
  return result;
}

BundleRequestResult SQLiteHostLedger::request_bundle_authorized(
    const ResourceBundleRequest& request, const HostLedgerTime& now,
    const HostLedgerAdmissionEpoch* admission_epoch) {
  validate_resource_request(request);
  if (!valid_time(now)) throw HostLedgerError("ledger time is invalid");
  std::scoped_lock lock(implementation_->mutex);
  Transaction transaction(implementation_->database);
  std::string reason;
  if (!implementation_->verify_unlocked(&reason)) {
    throw HostLedgerError("refusing bundle request: " + reason);
  }
  if (admission_epoch != nullptr) {
    Statement active(implementation_->database, R"sql(
      SELECT epoch.host_id, epoch.boot_id, epoch.broker_epoch,
             epoch.inventory_digest
      FROM active_admission_epoch AS active
      JOIN admission_epochs AS epoch
        ON epoch.epoch_digest=active.epoch_digest
      WHERE active.singleton=1 AND active.epoch_digest=?
    )sql");
    bind_text(active.get(), 1, admission_epoch->epoch_digest_);
    if (sqlite3_step(active.get()) != SQLITE_ROW ||
        column_text(active.get(), 0) != implementation_->inventory.host_id ||
        column_text(active.get(), 1) != implementation_->inventory.boot_id ||
        column_text(active.get(), 2) != implementation_->inventory.broker_epoch ||
        column_text(active.get(), 3) !=
            implementation_->inventory.inventory_digest ||
        sqlite3_step(active.get()) != SQLITE_DONE) {
      throw HostLedgerConflict("admission epoch is absent, stale, or inexact");
    }
  }
  Statement replay(implementation_->database, R"sql(
    SELECT request_digest, status, canonical_outcome_json, outcome_digest
    FROM request_outcomes WHERE request_id=?
  )sql");
  bind_text(replay.get(), 1, request.request_id);
  const int replay_status = sqlite3_step(replay.get());
  if (replay_status == SQLITE_ROW) {
    if (column_text(replay.get(), 0) != request.canonical_request_digest) {
      throw HostLedgerConflict("request_id already has different content");
    }
    if (admission_epoch != nullptr) {
      Statement authorization(implementation_->database, R"sql(
        SELECT request_digest, epoch_digest
        FROM request_admission_epochs WHERE request_id=?
      )sql");
      bind_text(authorization.get(), 1, request.request_id);
      if (sqlite3_step(authorization.get()) != SQLITE_ROW ||
          column_text(authorization.get(), 0) !=
              request.canonical_request_digest ||
          column_text(authorization.get(), 1) !=
              admission_epoch->epoch_digest_ ||
          sqlite3_step(authorization.get()) != SQLITE_DONE) {
        throw HostLedgerConflict(
            "request replay is not bound to the active admission epoch");
      }
    }
    const std::string status = column_text(replay.get(), 1);
    BundleRequestResult result{.status = status == "granted"
                                            ? BundleRequestStatus::granted
                                            : BundleRequestStatus::busy,
                               .grant = std::nullopt,
                               .outcome_digest = column_text(replay.get(), 3),
                               .replayed = true};
    if (result.status == BundleRequestStatus::granted) {
      result.grant = resource_bundle_grant_from_json(
          nlohmann::json::parse(column_text(replay.get(), 2)));
    }
    transaction.commit();
    return result;
  }
  if (replay_status != SQLITE_DONE) {
    throw HostLedgerError("request replay query failed");
  }

  if (admission_epoch != nullptr) {
    Statement authorization(implementation_->database, R"sql(
      INSERT INTO request_admission_epochs(
        request_id, request_digest, epoch_digest
      ) VALUES(?, ?, ?)
    )sql");
    bind_text(authorization.get(), 1, request.request_id);
    bind_text(authorization.get(), 2, request.canonical_request_digest);
    bind_text(authorization.get(), 3, admission_epoch->epoch_digest_);
    require_done(implementation_->database, authorization.get(),
                 "request admission authorization insert failed");
  } else {
    Statement exemption(implementation_->database, R"sql(
      INSERT INTO request_admission_exemptions(
        request_id, request_digest, reason
      ) VALUES(?, ?, 'policy_unconfigured')
    )sql");
    bind_text(exemption.get(), 1, request.request_id);
    bind_text(exemption.get(), 2, request.canonical_request_digest);
    require_done(implementation_->database, exemption.get(),
                 "request admission exemption insert failed");
  }

  const auto request_record = implementation_->make_record(
      "bundle.requested", "request:" + request.request_id, request.request_id,
      resource_request_json(request));
  const std::string request_receipt =
      implementation_->append_record(request_record);
  implementation_->fault(HostLedgerFaultPoint::after_request_record);

  const ResourceOccupancySnapshot occupancy =
      implementation_->occupancy_unlocked();
  const auto selection = select_host_resources(implementation_->inventory,
                                               request, occupancy);
  if (!selection) {
    const nlohmann::json outcome = sealed_json(
        "trainvm.host-ledger-busy/v1",
        {{"api_version", "trainvm.host-resource-decision/v1"},
         {"request_id", request.request_id},
         {"request_digest", request.canonical_request_digest},
         {"status", "busy"},
         {"inventory_digest", implementation_->inventory.inventory_digest},
         {"previous_receipt_digest", request_receipt}});
    const auto busy_record = implementation_->make_record(
        "bundle.denied", "busy:" + request.request_id, request.request_id,
        outcome);
    implementation_->append_record(busy_record);
    Statement insert(implementation_->database, R"sql(
      INSERT INTO request_outcomes(
        request_id, request_digest, status, allocation_id, outcome_digest,
        canonical_outcome_json
      ) VALUES(?, ?, 'busy', NULL, ?, ?)
    )sql");
    bind_text(insert.get(), 1, request.request_id);
    bind_text(insert.get(), 2, request.canonical_request_digest);
    bind_text(insert.get(), 3, outcome.at("receipt_digest").get<std::string>());
    bind_text(insert.get(), 4, outcome.dump());
    require_done(implementation_->database, insert.get(),
                 "busy outcome insert failed");
    implementation_->sync_projection_head();
    implementation_->fault(HostLedgerFaultPoint::before_commit);
    transaction.commit();
    return {.status = BundleRequestStatus::busy,
            .grant = std::nullopt,
            .outcome_digest = outcome.at("receipt_digest").get<std::string>(),
            .replayed = false};
  }

  ResourceBundleGrant grant{
      .api_version = std::string(kHostLedgerGrantApiVersion),
      .allocation_id = random_id("allocation-"),
      .request_id = request.request_id,
      .request_digest = request.canonical_request_digest,
      .journal_id = request.journal_id,
      .run_id = request.run_id,
      .logical_lease_id = request.logical_lease_id,
      .logical_fencing_token = request.logical_fencing_token,
      .host_id = implementation_->inventory.host_id,
      .boot_id = implementation_->inventory.boot_id,
      .broker_epoch = implementation_->inventory.broker_epoch,
      .fences = {},
      .granted_boottime_ns = now.boottime_ns,
      .granted_wall_time_ns = now.wall_time_ns,
      .previous_receipt_digest = request_receipt,
      .receipt_digest = {},
  };
  for (const auto& resource : selection->resources) {
    const std::string key = canonical_resource_key(resource.id);
    Statement query(implementation_->database,
                    "SELECT generation FROM resource_generations WHERE resource_key=?");
    bind_text(query.get(), 1, key);
    if (sqlite3_step(query.get()) != SQLITE_ROW) {
      throw HostLedgerError("selected resource has no generation row");
    }
    const std::uint64_t prior =
        static_cast<std::uint64_t>(sqlite3_column_int64(query.get(), 0));
    if (prior ==
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      throw HostLedgerError("resource generation is exhausted");
    }
    grant.fences.push_back({.resource = resource.id,
                            .generation = prior + 1U,
                            .inventory_digest =
                                implementation_->inventory.inventory_digest,
                            .topology_digest =
                                implementation_->inventory.topology_digest});
  }
  grant.receipt_digest = sha256("trainvm.host-resource-grant/v1",
                                grant_digest_json(grant).dump());
  for (const ResourceFence& fence : grant.fences) {
    Statement update(implementation_->database, R"sql(
      UPDATE resource_generations
      SET generation=?, last_allocation_id=?, last_grant_digest=?
      WHERE resource_key=? AND generation=?
    )sql");
    bind_integer(update.get(), 1,
                 checked_integer(fence.generation, "generation"));
    bind_text(update.get(), 2, grant.allocation_id);
    bind_text(update.get(), 3, grant.receipt_digest);
    bind_text(update.get(), 4, canonical_resource_key(fence.resource));
    bind_integer(update.get(), 5,
                 checked_integer(fence.generation - 1U, "generation"));
    require_done(implementation_->database, update.get(),
                 "generation CAS failed");
    if (sqlite3_changes(implementation_->database) != 1) {
      throw HostLedgerConflict("resource generation CAS lost");
    }
    implementation_->fault(HostLedgerFaultPoint::after_generation_update);
  }
  Statement allocation(implementation_->database, R"sql(
    INSERT INTO allocations(
      allocation_id, request_id, request_digest, grant_digest, journal_id,
      run_id, logical_lease_id, logical_fencing_token, host_id, grant_boot_id,
      broker_epoch, status, release_digest
    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', NULL)
  )sql");
  bind_text(allocation.get(), 1, grant.allocation_id);
  bind_text(allocation.get(), 2, grant.request_id);
  bind_text(allocation.get(), 3, grant.request_digest);
  bind_text(allocation.get(), 4, grant.receipt_digest);
  bind_text(allocation.get(), 5, grant.journal_id);
  bind_text(allocation.get(), 6, grant.run_id);
  bind_text(allocation.get(), 7, grant.logical_lease_id);
  bind_integer(allocation.get(), 8,
               checked_integer(grant.logical_fencing_token,
                               "logical_fencing_token"));
  bind_text(allocation.get(), 9, grant.host_id);
  bind_text(allocation.get(), 10, grant.boot_id);
  bind_text(allocation.get(), 11, grant.broker_epoch);
  require_done(implementation_->database, allocation.get(),
               "allocation insert failed");
  for (std::size_t index = 0; index < grant.fences.size(); ++index) {
    const ResourceFence& fence = grant.fences[index];
    Statement resource(implementation_->database, R"sql(
      INSERT INTO allocation_resources(
        allocation_id, resource_key, resource_json, generation,
        inventory_digest, topology_digest
      ) VALUES(?, ?, ?, ?, ?, ?)
    )sql");
    bind_text(resource.get(), 1, grant.allocation_id);
    bind_text(resource.get(), 2, canonical_resource_key(fence.resource));
    bind_text(resource.get(), 3, encode_json(fence.resource).dump());
    bind_integer(resource.get(), 4,
                 checked_integer(fence.generation, "generation"));
    bind_text(resource.get(), 5, fence.inventory_digest);
    bind_text(resource.get(), 6, fence.topology_digest);
    require_done(implementation_->database, resource.get(),
                 "allocation resource insert failed");
    Statement active(implementation_->database, R"sql(
      INSERT INTO active_resource_grants(
        resource_key, allocation_id, generation, grant_digest
      ) VALUES(?, ?, ?, ?)
    )sql");
    bind_text(active.get(), 1, canonical_resource_key(fence.resource));
    bind_text(active.get(), 2, grant.allocation_id);
    bind_integer(active.get(), 3,
                 checked_integer(fence.generation, "generation"));
    bind_text(active.get(), 4, grant.receipt_digest);
    require_done(implementation_->database, active.get(),
                 "active resource insert failed");
    (void)index;
  }
  const nlohmann::json grant_json = resource_bundle_grant_json(grant);
  const auto grant_record = implementation_->make_record(
      "bundle.granted", "grant:" + grant.allocation_id, grant.allocation_id,
      grant_json);
  implementation_->append_record(grant_record);
  Statement outcome(implementation_->database, R"sql(
    INSERT INTO request_outcomes(
      request_id, request_digest, status, allocation_id, outcome_digest,
      canonical_outcome_json
    ) VALUES(?, ?, 'granted', ?, ?, ?)
  )sql");
  bind_text(outcome.get(), 1, request.request_id);
  bind_text(outcome.get(), 2, request.canonical_request_digest);
  bind_text(outcome.get(), 3, grant.allocation_id);
  bind_text(outcome.get(), 4, grant.receipt_digest);
  bind_text(outcome.get(), 5, grant_json.dump());
  require_done(implementation_->database, outcome.get(),
               "grant outcome insert failed");
  implementation_->fault(HostLedgerFaultPoint::after_grant_projection);
  implementation_->sync_projection_head();
  implementation_->fault(HostLedgerFaultPoint::before_commit);
  transaction.commit();
  return {.status = BundleRequestStatus::granted,
          .grant = std::move(grant),
          .outcome_digest = grant_json.at("receipt_digest").get<std::string>(),
          .replayed = false};
}

BundleReleaseResult SQLiteHostLedger::release_bundle(
    const ResourceReleaseRequest& request, const HostLedgerTime& now) {
  const ResourceReleaseRequest sealed = seal_resource_release_request(request);
  if (sealed != request) {
    throw HostLedgerConflict("release request digest is not canonical");
  }
  if (!valid_time(now)) throw HostLedgerError("ledger time is invalid");
  std::scoped_lock lock(implementation_->mutex);
  Transaction transaction(implementation_->database);
  std::string reason;
  if (!implementation_->verify_unlocked(&reason)) {
    throw HostLedgerError("refusing bundle release: " + reason);
  }
  Statement replay(implementation_->database, R"sql(
    SELECT release_request_digest, canonical_release_json
    FROM release_outcomes WHERE release_request_id=?
  )sql");
  bind_text(replay.get(), 1, request.release_request_id);
  const int replay_status = sqlite3_step(replay.get());
  if (replay_status == SQLITE_ROW) {
    if (column_text(replay.get(), 0) != request.canonical_request_digest) {
      throw HostLedgerConflict("release_request_id has different content");
    }
    auto receipt = resource_release_receipt_from_json(
        nlohmann::json::parse(column_text(replay.get(), 1)));
    transaction.commit();
    return {.receipt = std::move(receipt), .replayed = true};
  }
  if (replay_status != SQLITE_DONE) {
    throw HostLedgerError("release replay query failed");
  }
  Statement allocation(implementation_->database, R"sql(
    SELECT grant_digest, journal_id, run_id, logical_lease_id,
           logical_fencing_token, host_id, grant_boot_id, status
    FROM allocations WHERE allocation_id=?
  )sql");
  bind_text(allocation.get(), 1, request.allocation_id);
  if (sqlite3_step(allocation.get()) != SQLITE_ROW) {
    throw HostLedgerConflict("allocation does not exist");
  }
  if (column_text(allocation.get(), 0) != request.grant_digest ||
      column_text(allocation.get(), 1) != request.journal_id ||
      column_text(allocation.get(), 2) != request.run_id ||
      column_text(allocation.get(), 3) != request.logical_lease_id ||
      static_cast<std::uint64_t>(sqlite3_column_int64(allocation.get(), 4)) !=
          request.logical_fencing_token ||
      column_text(allocation.get(), 5) != implementation_->inventory.host_id ||
      column_text(allocation.get(), 6) != implementation_->inventory.boot_id ||
      column_text(allocation.get(), 7) != "active") {
    throw HostLedgerConflict("release CAS does not match the active grant");
  }
  const auto request_record = implementation_->make_record(
      "bundle.release_requested",
      "release-request:" + request.release_request_id,
      request.release_request_id, release_request_digest_json(request));
  const std::string previous = implementation_->append_record(request_record);
  ResourceReleaseReceipt receipt{
      .api_version = std::string(kHostLedgerReleaseApiVersion),
      .release_request_id = request.release_request_id,
      .release_request_digest = request.canonical_request_digest,
      .allocation_id = request.allocation_id,
      .grant_digest = request.grant_digest,
      .host_id = implementation_->inventory.host_id,
      .boot_id = implementation_->inventory.boot_id,
      .broker_epoch = implementation_->inventory.broker_epoch,
      .released_boottime_ns = now.boottime_ns,
      .released_wall_time_ns = now.wall_time_ns,
      .previous_receipt_digest = previous,
      .receipt_digest = {},
  };
  receipt.receipt_digest = sha256("trainvm.host-resource-release/v1",
                                  release_digest_json(receipt).dump());
  Statement expected(implementation_->database, R"sql(
    SELECT COUNT(*) FROM allocation_resources WHERE allocation_id=?
  )sql");
  bind_text(expected.get(), 1, request.allocation_id);
  if (sqlite3_step(expected.get()) != SQLITE_ROW) {
    throw HostLedgerError("could not count allocation resources");
  }
  const std::int64_t resource_count = sqlite3_column_int64(expected.get(), 0);
  Statement remove(implementation_->database,
                   "DELETE FROM active_resource_grants WHERE allocation_id=?");
  bind_text(remove.get(), 1, request.allocation_id);
  require_done(implementation_->database, remove.get(),
               "active resource release failed");
  if (sqlite3_changes(implementation_->database) != resource_count ||
      resource_count <= 0) {
    throw HostLedgerConflict("release did not own the exact active bundle");
  }
  Statement update(implementation_->database, R"sql(
    UPDATE allocations SET status='released', release_digest=?
    WHERE allocation_id=? AND status='active' AND grant_digest=?
  )sql");
  bind_text(update.get(), 1, receipt.receipt_digest);
  bind_text(update.get(), 2, request.allocation_id);
  bind_text(update.get(), 3, request.grant_digest);
  require_done(implementation_->database, update.get(),
               "allocation release CAS failed");
  if (sqlite3_changes(implementation_->database) != 1) {
    throw HostLedgerConflict("allocation release CAS lost");
  }
  const nlohmann::json receipt_json = resource_release_receipt_json(receipt);
  const auto release_record = implementation_->make_record(
      "bundle.released", "release:" + request.allocation_id,
      request.allocation_id, receipt_json);
  implementation_->append_record(release_record);
  Statement outcome(implementation_->database, R"sql(
    INSERT INTO release_outcomes(
      release_request_id, release_request_digest, allocation_id, grant_digest,
      release_receipt_digest, canonical_release_json
    ) VALUES(?, ?, ?, ?, ?, ?)
  )sql");
  bind_text(outcome.get(), 1, request.release_request_id);
  bind_text(outcome.get(), 2, request.canonical_request_digest);
  bind_text(outcome.get(), 3, request.allocation_id);
  bind_text(outcome.get(), 4, request.grant_digest);
  bind_text(outcome.get(), 5, receipt.receipt_digest);
  bind_text(outcome.get(), 6, receipt_json.dump());
  require_done(implementation_->database, outcome.get(),
               "release outcome insert failed");
  implementation_->fault(HostLedgerFaultPoint::after_release_record);
  implementation_->sync_projection_head();
  implementation_->fault(HostLedgerFaultPoint::before_commit);
  transaction.commit();
  return {.receipt = std::move(receipt), .replayed = false};
}

HostStartupAuditLedgerCommitResult SQLiteHostLedger::commit_startup_audit(
    const HostStartupAuditReport& report, const HostLedgerTime& now) {
  if (!implementation_->trusted_startup_audit_policy) {
    throw HostLedgerError(
        "startup-audit authority was not configured for this ledger");
  }
  validate_host_startup_audit_report(report);
  if (report.policy != *implementation_->trusted_startup_audit_policy) {
    throw HostLedgerConflict(
        "startup-audit report policy differs from configured trusted policy");
  }
  if (!valid_time(now) || now.boottime_ns < report.observed_end_boottime_ns) {
    throw HostLedgerError("startup-audit commit time is invalid");
  }
  std::scoped_lock lock(implementation_->mutex);

  const auto reread = [&](bool replayed) {
    Transaction reread_transaction(implementation_->database);
    std::string reread_reason;
    if (!implementation_->verify_unlocked(&reread_reason)) {
      throw HostLedgerError("committed startup audit failed re-read: " +
                            reread_reason);
    }
    Statement query(implementation_->database, R"sql(
      SELECT report_digest, canonical_report_json, canonical_receipt_json
      FROM startup_audit_outcomes WHERE audit_id=?
    )sql");
    bind_text(query.get(), 1, report.audit_id);
    if (sqlite3_step(query.get()) != SQLITE_ROW ||
        column_text(query.get(), 0) != report.report_digest) {
      throw HostLedgerError("committed startup audit cannot be re-read exactly");
    }
    const HostStartupAuditReport persisted_report =
        decode_untrusted_host_startup_audit_report(
            nlohmann::json::parse(column_text(query.get(), 1)));
    const HostStartupAuditReceipt persisted_receipt =
        decode_untrusted_host_startup_audit_receipt(
            nlohmann::json::parse(column_text(query.get(), 2)),
            persisted_report);
    if (persisted_report != report) {
      throw HostLedgerError("startup-audit re-read report changed content");
    }
    reread_transaction.commit();
    return HostStartupAuditLedgerCommitResult{.receipt = persisted_receipt,
                                              .replayed = replayed};
  };

  Transaction transaction(implementation_->database);
  std::string reason;
  if (!implementation_->verify_unlocked(&reason)) {
    throw HostLedgerError("refusing startup-audit commit: " + reason);
  }
  Statement replay(implementation_->database, R"sql(
    SELECT report_digest FROM startup_audit_outcomes WHERE audit_id=?
  )sql");
  bind_text(replay.get(), 1, report.audit_id);
  const int replay_status = sqlite3_step(replay.get());
  if (replay_status == SQLITE_ROW) {
    if (column_text(replay.get(), 0) != report.report_digest) {
      throw HostLedgerConflict("audit_id already has different content");
    }
    transaction.commit();
    return reread(true);
  }
  if (replay_status != SQLITE_DONE) {
    throw HostLedgerError("startup-audit replay query failed");
  }
  const HostLedgerChainHead before = implementation_->chain_head_unlocked();
  const ResourceOccupancySnapshot occupancy =
      implementation_->occupancy_unlocked();
  if (report.host_id != implementation_->inventory.host_id ||
      report.boot_id != implementation_->inventory.boot_id ||
      report.broker_epoch != implementation_->inventory.broker_epoch ||
      report.inventory != implementation_->inventory ||
      report.ledger_head_before != before ||
      report.ledger_head_after_observation != before ||
      report.pre_audit_occupancy != occupancy ||
      report.post_audit_occupancy != occupancy) {
    throw HostLedgerConflict(
        "startup-audit report lost its host-ledger evidence CAS");
  }

  const nlohmann::json canonical_report =
      host_startup_audit_report_json(report);
  if (decode_untrusted_host_startup_audit_report(canonical_report) != report) {
    throw HostLedgerError(
        "startup-audit canonical report is not precommit-readable");
  }
  const auto record = implementation_->make_record(
      "startup.audit_committed", "startup-audit:" + report.audit_id,
      report.broker_instance_id, canonical_report);
  const std::string record_receipt = implementation_->append_record(record);
  implementation_->fault(HostLedgerFaultPoint::after_startup_audit_record);
  const HostLedgerChainHead committed = implementation_->chain_head_unlocked();
  HostStartupAuditReceipt receipt = canonicalize_host_startup_audit_receipt(
      {.api_version = std::string(kHostStartupAuditReceiptApiVersion),
       .audit_id = report.audit_id,
       .report_digest = report.report_digest,
       .host_id = report.host_id,
       .boot_id = report.boot_id,
       .broker_epoch = report.broker_epoch,
       .broker_instance_id = report.broker_instance_id,
       .inventory_digest = report.inventory.inventory_digest,
       .topology_digest = report.inventory.topology_digest,
       .pre_occupancy_digest = report.pre_audit_occupancy.occupancy_digest,
       .post_occupancy_digest = report.post_audit_occupancy.occupancy_digest,
       .policy_digest = report.policy.policy_digest,
       .findings_digest = report.findings_digest,
       .disposition = report.disposition,
       .ledger_head_before = before,
       .committed_ledger_head = committed,
       .commit_record_digest = record_receipt,
       .committed_boottime_ns = now.boottime_ns,
       .committed_wall_time_ns = now.wall_time_ns,
       .receipt_digest = {}},
      report);
  const nlohmann::json canonical_receipt =
      host_startup_audit_receipt_json(receipt, report);
  if (decode_untrusted_host_startup_audit_receipt(canonical_receipt, report) !=
      receipt) {
    throw HostLedgerError(
        "startup-audit canonical receipt is not precommit-readable");
  }
  Statement projection(implementation_->database, R"sql(
    INSERT INTO startup_audit_outcomes(
      audit_id, report_digest, canonical_report_json, record_receipt_digest,
      record_sequence, record_chain_hash, record_previous_sequence,
      record_previous_hash, receipt_digest, canonical_receipt_json
    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  )sql");
  bind_text(projection.get(), 1, report.audit_id);
  bind_text(projection.get(), 2, report.report_digest);
  bind_text(projection.get(), 3, canonical_report.dump());
  bind_text(projection.get(), 4, record_receipt);
  bind_integer(projection.get(), 5,
               checked_integer(committed.ledger_sequence, "ledger sequence"));
  bind_text(projection.get(), 6, committed.chain_hash);
  bind_integer(projection.get(), 7,
               checked_integer(before.ledger_sequence, "ledger sequence"));
  bind_text(projection.get(), 8, before.chain_hash);
  bind_text(projection.get(), 9, receipt.receipt_digest);
  bind_text(projection.get(), 10, canonical_receipt.dump());
  require_done(implementation_->database, projection.get(),
               "startup-audit projection insert failed");
  implementation_->fault(HostLedgerFaultPoint::after_startup_audit_projection);
  implementation_->sync_projection_head();
  implementation_->fault(HostLedgerFaultPoint::before_commit);
  transaction.commit();
  implementation_->fault(HostLedgerFaultPoint::after_startup_audit_commit);
  return reread(false);
}

HostLedgerAdmissionFinalizeResult
SQLiteHostLedger::finalize_startup_admission(
    const HostStartupAuditReport& report,
    const HostStartupAuditReceipt& receipt, const HostLedgerTime& now) {
  if (!implementation_->trusted_startup_audit_policy) {
    throw HostLedgerError(
        "startup-admission authority was not configured for this ledger");
  }
  validate_host_startup_audit_report(report);
  validate_host_startup_audit_receipt(receipt, report);
  if (report.policy != *implementation_->trusted_startup_audit_policy) {
    throw HostLedgerConflict(
        "startup-admission report policy differs from trusted policy");
  }
  const bool blocking = std::ranges::any_of(
      report.findings, [](const HostStartupAuditFinding& finding) {
        return finding.severity == HostStartupAuditFindingSeverity::blocking;
      });
  if (report.disposition != HostStartupAuditDisposition::passed || blocking) {
    throw HostLedgerConflict(
        "failed or blocking startup audit cannot authorize admission");
  }
  if (!valid_time(now) || now.boottime_ns < receipt.committed_boottime_ns ||
      now.wall_time_ns < receipt.committed_wall_time_ns) {
    throw HostLedgerError("startup-admission finalize time is invalid");
  }

  std::scoped_lock lock(implementation_->mutex);
  const auto reread = [&](std::string_view epoch_digest, bool replayed) {
    Transaction reread_transaction(implementation_->database);
    std::string reread_reason;
    if (!implementation_->verify_unlocked(&reread_reason)) {
      throw HostLedgerError("finalized admission epoch failed re-read: " +
                            reread_reason);
    }
    Statement query(implementation_->database, R"sql(
      SELECT epoch.canonical_json, active.epoch_digest
      FROM admission_epochs AS epoch
      JOIN active_admission_epoch AS active
        ON active.epoch_digest=epoch.epoch_digest AND active.singleton=1
      WHERE epoch.epoch_digest=? AND epoch.audit_id=?
        AND epoch.report_digest=? AND epoch.audit_receipt_digest=?
    )sql");
    bind_text(query.get(), 1, std::string(epoch_digest));
    bind_text(query.get(), 2, report.audit_id);
    bind_text(query.get(), 3, report.report_digest);
    bind_text(query.get(), 4, receipt.receipt_digest);
    if (sqlite3_step(query.get()) != SQLITE_ROW ||
        column_text(query.get(), 1) != epoch_digest ||
        nlohmann::json::parse(column_text(query.get(), 0))
                .at("epoch_digest")
                .get<std::string>() != epoch_digest ||
        sqlite3_step(query.get()) != SQLITE_DONE) {
      throw HostLedgerError(
          "finalized admission epoch cannot be re-read exactly");
    }
    reread_transaction.commit();
    return HostLedgerAdmissionFinalizeResult{
        .epoch = HostLedgerAdmissionEpoch(std::string(epoch_digest)),
        .replayed = replayed};
  };

  Transaction transaction(implementation_->database);
  std::string reason;
  if (!implementation_->verify_unlocked(&reason)) {
    throw HostLedgerError("refusing startup-admission finalize: " + reason);
  }
  if (report.host_id != implementation_->inventory.host_id ||
      report.boot_id != implementation_->inventory.boot_id ||
      report.broker_epoch != implementation_->inventory.broker_epoch ||
      report.inventory != implementation_->inventory) {
    throw HostLedgerConflict(
        "startup-admission report does not name the current ledger identity");
  }

  Statement persisted(implementation_->database, R"sql(
    SELECT report_digest, receipt_digest, canonical_report_json,
           canonical_receipt_json
    FROM startup_audit_outcomes WHERE audit_id=?
  )sql");
  bind_text(persisted.get(), 1, report.audit_id);
  if (sqlite3_step(persisted.get()) != SQLITE_ROW ||
      column_text(persisted.get(), 0) != report.report_digest ||
      column_text(persisted.get(), 1) != receipt.receipt_digest ||
      column_text(persisted.get(), 2) !=
          host_startup_audit_report_json(report).dump() ||
      column_text(persisted.get(), 3) !=
          host_startup_audit_receipt_json(receipt, report).dump() ||
      sqlite3_step(persisted.get()) != SQLITE_DONE) {
    throw HostLedgerConflict(
        "startup-admission audit receipt is not exactly persisted");
  }

  std::optional<std::string> prior_active_digest;
  Statement active(implementation_->database, R"sql(
    SELECT epoch.epoch_digest, epoch.audit_id, epoch.report_digest,
           epoch.audit_receipt_digest, epoch.host_id, epoch.boot_id,
           epoch.broker_epoch
    FROM active_admission_epoch AS active
    JOIN admission_epochs AS epoch ON epoch.epoch_digest=active.epoch_digest
    WHERE active.singleton=1
  )sql");
  const int active_status = sqlite3_step(active.get());
  if (active_status == SQLITE_ROW) {
    prior_active_digest = column_text(active.get(), 0);
    const bool exact_replay =
        column_text(active.get(), 1) == report.audit_id &&
        column_text(active.get(), 2) == report.report_digest &&
        column_text(active.get(), 3) == receipt.receipt_digest;
    const bool same_runtime_epoch =
        column_text(active.get(), 4) == report.host_id &&
        column_text(active.get(), 5) == report.boot_id &&
        column_text(active.get(), 6) == report.broker_epoch;
    if (sqlite3_step(active.get()) != SQLITE_DONE) {
      throw HostLedgerError("active admission epoch is not singular");
    }
    if (exact_replay) {
      const std::string epoch_digest = *prior_active_digest;
      transaction.commit();
      return reread(epoch_digest, true);
    }
    if (same_runtime_epoch) {
      throw HostLedgerConflict(
          "this host boot and broker already have another admission epoch");
    }
  } else if (active_status != SQLITE_DONE) {
    throw HostLedgerError("active admission epoch query failed");
  }

  const HostLedgerChainHead current_head =
      implementation_->chain_head_unlocked();
  const ResourceOccupancySnapshot current_occupancy =
      implementation_->occupancy_unlocked();
  if (report.host_id != implementation_->inventory.host_id ||
      report.boot_id != implementation_->inventory.boot_id ||
      report.broker_epoch != implementation_->inventory.broker_epoch ||
      report.inventory != implementation_->inventory ||
      current_head != receipt.committed_ledger_head ||
      current_occupancy.ledger_sequence != current_head.ledger_sequence ||
      !same_occupied_resources(current_occupancy,
                               report.post_audit_occupancy)) {
    throw HostLedgerConflict(
        "startup-admission finalize lost its exact ledger/occupancy CAS");
  }
  ResourceOccupancySnapshot finalized_occupancy =
      report.post_audit_occupancy;
  finalized_occupancy.ledger_sequence = current_head.ledger_sequence;
  finalized_occupancy.occupancy_digest.clear();
  finalized_occupancy = seal_resource_occupancy(
      implementation_->inventory, std::move(finalized_occupancy));
  if (finalized_occupancy != current_occupancy) {
    throw HostLedgerConflict(
        "startup-admission occupancy cannot be sealed at committed head");
  }
  const nlohmann::json canonical =
      admission_epoch_json(report, receipt, finalized_occupancy, now);
  const std::string epoch_digest =
      canonical.at("epoch_digest").get<std::string>();
  Statement insert(implementation_->database, R"sql(
    INSERT INTO admission_epochs(
      epoch_digest, api_version, audit_id, report_digest,
      audit_receipt_digest, host_id, boot_id, broker_epoch, inventory_digest,
      audit_record_sequence, audit_record_chain_hash,
      finalized_occupancy_digest, finalized_boottime_ns,
      finalized_wall_time_ns, canonical_json
    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  )sql");
  bind_text(insert.get(), 1, epoch_digest);
  bind_text(insert.get(), 2,
            std::string(kHostLedgerAdmissionEpochApiVersion));
  bind_text(insert.get(), 3, report.audit_id);
  bind_text(insert.get(), 4, report.report_digest);
  bind_text(insert.get(), 5, receipt.receipt_digest);
  bind_text(insert.get(), 6, report.host_id);
  bind_text(insert.get(), 7, report.boot_id);
  bind_text(insert.get(), 8, report.broker_epoch);
  bind_text(insert.get(), 9, report.inventory.inventory_digest);
  bind_integer(insert.get(), 10,
               checked_integer(current_head.ledger_sequence,
                               "audit record sequence"));
  bind_text(insert.get(), 11, current_head.chain_hash);
  bind_text(insert.get(), 12, finalized_occupancy.occupancy_digest);
  bind_integer(insert.get(), 13, now.boottime_ns);
  bind_integer(insert.get(), 14, now.wall_time_ns);
  bind_text(insert.get(), 15, canonical.dump());
  require_done(implementation_->database, insert.get(),
               "admission epoch insert failed");
  if (prior_active_digest) {
    Statement activate(implementation_->database, R"sql(
      UPDATE active_admission_epoch SET epoch_digest=?
      WHERE singleton=1 AND epoch_digest=?
    )sql");
    bind_text(activate.get(), 1, epoch_digest);
    bind_text(activate.get(), 2, *prior_active_digest);
    require_done(implementation_->database, activate.get(),
                 "active admission epoch update failed");
    if (sqlite3_changes(implementation_->database) != 1) {
      throw HostLedgerConflict("active admission epoch CAS lost");
    }
  } else {
    Statement activate(implementation_->database,
                       "INSERT INTO active_admission_epoch VALUES(1, ?)");
    bind_text(activate.get(), 1, epoch_digest);
    require_done(implementation_->database, activate.get(),
                 "active admission epoch insert failed");
  }
  if (!implementation_->verify_unlocked(&reason)) {
    throw HostLedgerError("new admission epoch failed verification: " + reason);
  }
  transaction.commit();
  implementation_->fault(HostLedgerFaultPoint::after_admission_finalize_commit);
  return reread(epoch_digest, false);
}

HostLedgerChainHead SQLiteHostLedger::chain_head() const {
  std::scoped_lock lock(implementation_->mutex);
  Transaction transaction(implementation_->database);
  std::string reason;
  if (!implementation_->verify_unlocked(&reason)) {
    throw HostLedgerError("refusing chain-head read: " + reason);
  }
  const HostLedgerChainHead result = implementation_->chain_head_unlocked();
  transaction.commit();
  return result;
}

ResourceOccupancySnapshot SQLiteHostLedger::occupancy() const {
  std::scoped_lock lock(implementation_->mutex);
  Transaction transaction(implementation_->database);
  std::string reason;
  if (!implementation_->verify_unlocked(&reason)) {
    throw HostLedgerError("refusing occupancy read: " + reason);
  }
  auto result = implementation_->occupancy_unlocked();
  transaction.commit();
  return result;
}

std::uint64_t SQLiteHostLedger::generation(
    const HostResourceId& resource) const {
  std::scoped_lock lock(implementation_->mutex);
  Transaction transaction(implementation_->database);
  std::string reason;
  if (!implementation_->verify_unlocked(&reason)) {
    throw HostLedgerError("refusing generation read: " + reason);
  }
  Statement query(implementation_->database,
                  "SELECT generation FROM resource_generations WHERE resource_key=?");
  bind_text(query.get(), 1, canonical_resource_key(resource));
  if (sqlite3_step(query.get()) != SQLITE_ROW) {
    throw HostLedgerError("resource generation is unknown");
  }
  const auto result =
      static_cast<std::uint64_t>(sqlite3_column_int64(query.get(), 0));
  transaction.commit();
  return result;
}

std::uint64_t SQLiteHostLedger::record_count() const {
  std::scoped_lock lock(implementation_->mutex);
  Transaction transaction(implementation_->database);
  std::string reason;
  if (!implementation_->verify_unlocked(&reason)) {
    throw HostLedgerError("refusing record count read: " + reason);
  }
  Statement query(implementation_->database, "SELECT COUNT(*) FROM ledger_records");
  if (sqlite3_step(query.get()) != SQLITE_ROW) {
    throw HostLedgerError("could not count ledger records");
  }
  const auto result =
      static_cast<std::uint64_t>(sqlite3_column_int64(query.get(), 0));
  transaction.commit();
  return result;
}

bool SQLiteHostLedger::verify(std::string* reason) const {
  std::scoped_lock lock(implementation_->mutex);
  try {
    Transaction transaction(implementation_->database);
    const bool valid = implementation_->verify_unlocked(reason);
    transaction.commit();
    return valid;
  } catch (const std::exception& error) {
    if (reason != nullptr) *reason = error.what();
    return false;
  }
}

HostInventoryReceipt SQLiteHostLedger::inventory() const {
  std::scoped_lock lock(implementation_->mutex);
  Transaction transaction(implementation_->database);
  std::string reason;
  if (!implementation_->verify_unlocked(&reason)) {
    throw HostLedgerError("refusing inventory read: " + reason);
  }
  transaction.commit();
  return implementation_->inventory;
}

ResourceReleaseRequest seal_resource_release_request(
    ResourceReleaseRequest request) {
  if (request.api_version != kHostLedgerReleaseRequestApiVersion ||
      request.release_request_id.empty() || request.allocation_id.empty() ||
      !valid_digest(request.grant_digest) || request.journal_id.empty() ||
      request.run_id.empty() || request.logical_lease_id.empty() ||
      request.logical_fencing_token == 0U) {
    throw HostLedgerError("resource release request is malformed");
  }
  request.canonical_request_digest = sha256(
      "trainvm.host-resource-release-request/v1",
      release_request_digest_json(request).dump());
  return request;
}

nlohmann::json resource_release_request_json(
    const ResourceReleaseRequest& request) {
  if (seal_resource_release_request(request) != request) {
    throw HostLedgerError("resource release request digest is not canonical");
  }
  return encode_json(request);
}

ResourceReleaseRequest resource_release_request_from_json(
    const nlohmann::json& source) {
  ResourceReleaseRequest request =
      strict_decode<ResourceReleaseRequest>(source, "resource release request");
  (void)resource_release_request_json(request);
  return request;
}

nlohmann::json bundle_request_result_json(const BundleRequestResult& result) {
  if ((result.status != BundleRequestStatus::granted &&
       result.status != BundleRequestStatus::busy) ||
      !valid_digest(result.outcome_digest) ||
      (result.status == BundleRequestStatus::granted) != result.grant.has_value()) {
    throw HostLedgerError("bundle request result is invalid");
  }
  nlohmann::json grant = nullptr;
  std::string_view status = "busy";
  if (result.grant) {
    grant = resource_bundle_grant_json(*result.grant);
    status = "granted";
    if (result.outcome_digest != result.grant->receipt_digest) {
      throw HostLedgerError("bundle request result digest disagrees with grant");
    }
  }
  return {{"grant", std::move(grant)},
          {"outcome_digest", result.outcome_digest},
          {"replayed", result.replayed},
          {"status", status}};
}

BundleRequestResult bundle_request_result_from_json(
    const nlohmann::json& source) {
  if (!source.is_object() || source.size() != 4U ||
      !source.contains("grant") || !source.contains("outcome_digest") ||
      !source.contains("replayed") || !source.contains("status") ||
      !source.at("status").is_string() ||
      !source.at("outcome_digest").is_string() ||
      !source.at("replayed").is_boolean()) {
    throw HostLedgerError("bundle request result JSON shape is invalid");
  }
  const std::string status = source.at("status").get<std::string>();
  BundleRequestResult result{
      .status = status == "granted" ? BundleRequestStatus::granted
                                     : BundleRequestStatus::busy,
      .grant = std::nullopt,
      .outcome_digest = source.at("outcome_digest").get<std::string>(),
      .replayed = source.at("replayed").get<bool>(),
  };
  if (status != "granted" && status != "busy") {
    throw HostLedgerError("bundle request result status is invalid");
  }
  if (status == "granted") {
    if (!source.at("grant").is_object()) {
      throw HostLedgerError("granted bundle result has no grant");
    }
    result.grant = resource_bundle_grant_from_json(source.at("grant"));
  } else if (!source.at("grant").is_null()) {
    throw HostLedgerError("busy bundle result unexpectedly has a grant");
  }
  (void)bundle_request_result_json(result);
  return result;
}

nlohmann::json resource_bundle_grant_json(
    const ResourceBundleGrant& grant) {
  nlohmann::json value = grant_digest_json(grant);
  value["receipt_digest"] = grant.receipt_digest;
  const bool canonical_fences =
      std::ranges::is_sorted(grant.fences, [](const ResourceFence& left,
                                              const ResourceFence& right) {
        return canonical_resource_key(left.resource) <
               canonical_resource_key(right.resource);
      }) &&
      std::ranges::all_of(grant.fences, [](const ResourceFence& fence) {
        return fence.generation > 0U && valid_digest(fence.inventory_digest) &&
               valid_digest(fence.topology_digest);
      });
  bool nonconflicting_fences = true;
  for (std::size_t left = 0; left < grant.fences.size(); ++left) {
    for (std::size_t right = left + 1U; right < grant.fences.size(); ++right) {
      if (host_resources_conflict(grant.fences[left].resource,
                                  grant.fences[right].resource)) {
        nonconflicting_fences = false;
      }
    }
  }
  if (grant.api_version != kHostLedgerGrantApiVersion ||
      grant.allocation_id.empty() || grant.request_id.empty() ||
      !valid_digest(grant.request_digest) || grant.journal_id.empty() ||
      grant.run_id.empty() || grant.logical_lease_id.empty() ||
      grant.logical_fencing_token == 0U || grant.host_id.empty() ||
      grant.boot_id.empty() || grant.broker_epoch.empty() ||
      grant.fences.empty() || !canonical_fences || !nonconflicting_fences ||
      grant.granted_boottime_ns < 0 ||
      grant.granted_wall_time_ns < 0 ||
      !valid_digest(grant.previous_receipt_digest) ||
      grant.receipt_digest !=
          sha256("trainvm.host-resource-grant/v1",
                 grant_digest_json(grant).dump())) {
    throw HostLedgerError("resource bundle grant is invalid");
  }
  return value;
}

ResourceBundleGrant resource_bundle_grant_from_json(
    const nlohmann::json& source) {
  ResourceBundleGrant grant =
      strict_decode<ResourceBundleGrant>(source, "resource grant");
  (void)resource_bundle_grant_json(grant);
  return grant;
}

nlohmann::json resource_release_receipt_json(
    const ResourceReleaseReceipt& receipt) {
  nlohmann::json value = release_digest_json(receipt);
  value["receipt_digest"] = receipt.receipt_digest;
  if (receipt.api_version != kHostLedgerReleaseApiVersion ||
      receipt.release_request_id.empty() ||
      !valid_digest(receipt.release_request_digest) ||
      receipt.allocation_id.empty() || !valid_digest(receipt.grant_digest) ||
      receipt.host_id.empty() || receipt.boot_id.empty() ||
      receipt.broker_epoch.empty() || receipt.released_boottime_ns < 0 ||
      receipt.released_wall_time_ns < 0 ||
      !valid_digest(receipt.previous_receipt_digest) ||
      receipt.receipt_digest !=
          sha256("trainvm.host-resource-release/v1",
                 release_digest_json(receipt).dump())) {
    throw HostLedgerError("resource release receipt is invalid");
  }
  return value;
}

ResourceReleaseReceipt resource_release_receipt_from_json(
    const nlohmann::json& source) {
  ResourceReleaseReceipt receipt =
      strict_decode<ResourceReleaseReceipt>(source, "resource release receipt");
  (void)resource_release_receipt_json(receipt);
  return receipt;
}

nlohmann::json bundle_release_result_json(const BundleReleaseResult& result) {
  return {{"receipt", resource_release_receipt_json(result.receipt)},
          {"replayed", result.replayed}};
}

BundleReleaseResult bundle_release_result_from_json(
    const nlohmann::json& source) {
  if (!source.is_object() || source.size() != 2U ||
      !source.contains("receipt") || !source.contains("replayed") ||
      !source.at("receipt").is_object() ||
      !source.at("replayed").is_boolean()) {
    throw HostLedgerError("bundle release result JSON shape is invalid");
  }
  BundleReleaseResult result{
      .receipt = resource_release_receipt_from_json(source.at("receipt")),
      .replayed = source.at("replayed").get<bool>(),
  };
  (void)bundle_release_result_json(result);
  return result;
}

}  // namespace trainvm
