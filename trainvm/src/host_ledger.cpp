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
#include <ranges>
#include <set>
#include <sstream>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

#include "trainvm/reflection_json.hpp"

namespace trainvm {
namespace {

constexpr std::string_view kGenesisDigest =
    "sha256:0000000000000000000000000000000000000000000000000000000000000000";

constexpr std::string_view kSchema = R"sql(
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

const SchemaSnapshot& canonical_schema() {
  static const SchemaSnapshot snapshot = [] {
    sqlite3* database = nullptr;
    if (sqlite3_open(":memory:", &database) != SQLITE_OK) {
      throw HostLedgerError("could not construct canonical ledger schema");
    }
    try {
      execute(database, kSchema, "canonical schema creation failed");
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
  sqlite3* database{};
  std::shared_ptr<HostLedgerFilesystemAuthority> authority;
  int sqlite_database_fd{-1};
  HostInventoryReceipt inventory;
  IHostLedgerFaultInjector* faults{};
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
      WHERE singleton=1
    )sql");
    bind_integer(update.get(), 1, sequence);
    bind_text(update.get(), 2, chain);
    require_done(database, update.get(), "ledger chain update failed");
    if (sqlite3_changes(database) != 1) {
      throw HostLedgerError("ledger chain update lost its CAS row");
    }
    return record.at("receipt_digest").get<std::string>();
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
                       bool bind_instance_inventory = true) const {
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
      if (schema_snapshot(database) != canonical_schema()) {
        return fail("ledger schema does not exactly match v1");
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
          sqlite3_column_int64(version.get(), 0) != 1 ||
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
        if (record_type == "bundle.requested") {
          const ResourceBundleRequest request =
              resource_request_from_json(payload);
          if (!request_evidence.emplace(request.request_id, request).second) {
            return fail("duplicate bundle request authority evidence");
          }
        } else if (record_type == "bundle.granted") {
          const ResourceBundleGrant grant =
              resource_bundle_grant_from_json(payload);
          if (!grant_evidence.emplace(grant.allocation_id, grant).second) {
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
          if (!release_evidence.emplace(receipt.allocation_id, receipt).second) {
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
    IHostLedgerFaultInjector* fault_injector)
    : implementation_(std::make_unique<Implementation>()) {
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
    execute(implementation_->database, kSchema, "could not create host ledger v1");
    execute(implementation_->database,
            "PRAGMA application_id=0x5456484c; PRAGMA user_version=1;",
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
  validate_resource_request(request);
  if (!valid_time(now)) throw HostLedgerError("ledger time is invalid");
  std::scoped_lock lock(implementation_->mutex);
  Transaction transaction(implementation_->database);
  std::string reason;
  if (!implementation_->verify_unlocked(&reason)) {
    throw HostLedgerError("refusing bundle request: " + reason);
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

}  // namespace trainvm
