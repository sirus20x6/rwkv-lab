#include "trainvm/host_ledger.hpp"

#include <sqlite3.h>

#include <algorithm>
#include <atomic>
#include <fcntl.h>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

namespace {

using namespace trainvm;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Exception, typename Callable>
void require_throws(Callable&& callable, const std::string& message) {
  try {
    std::forward<Callable>(callable)();
  } catch (const Exception&) {
    return;
  }
  throw std::runtime_error(message);
}

HostResourceId mutex_id(std::string id) {
  if (!id.starts_with("host-mutex:")) id = "host-mutex:" + id;
  return {.kind = HostResourceKind::host_mutex,
          .vendor = std::nullopt,
          .stable_id = std::move(id),
          .parent_id = std::nullopt};
}

ObservedHostResource mutex_resource(std::string id) {
  return {.id = mutex_id(std::move(id)),
          .disposition = ResourceObservationDisposition::audited_eligible,
          .compute_contexts = ResourceContextDisposition::absent,
          .graphics_contexts = ResourceContextDisposition::absent,
          .pci_bdf = std::nullopt,
          .device_major = std::nullopt,
          .device_minor = std::nullopt,
          .numa_node = std::nullopt,
          .pcie_root_id = std::nullopt,
          .fabric_clique_id = std::nullopt,
          .total_memory_bytes = 0,
          .labels = {{"scope", "test"}}};
}

HostInventoryReceipt inventory(std::vector<std::string> resources,
                               std::string boot = "boot-001",
                               std::string broker = "broker-001") {
  HostKernelSnapshot snapshot{
      .api_version = std::string(kHostInventoryApiVersion),
      .host_id = "host-001",
      .boot_id = std::move(boot),
      .broker_epoch = std::move(broker),
      .begin_revision = "revision-001",
      .end_revision = "revision-001",
      .probes = {},
      .resources = {},
  };
  for (std::string& id : resources) {
    snapshot.resources.push_back(mutex_resource(std::move(id)));
  }
  FakeHostKernel kernel({{.snapshot = std::move(snapshot), .failure = std::nullopt}});
  return capture_host_inventory(kernel);
}

ResourceBundleRequest request(std::string id, std::uint32_t count = 1U) {
  return seal_resource_request({
      .api_version = std::string(kHostResourceRequestApiVersion),
      .request_id = std::move(id),
      .journal_id = "journal-001",
      .run_id = "run-001",
      .logical_lease_id = "lease-001",
      .logical_fencing_token = 7,
      .count = count,
      .access_mode = ResourceAccessMode::mutex_exclusive,
      .topology = TopologyPolicy::any,
      .selector = {},
      .canonical_request_digest = {},
  });
}

ResourceReleaseRequest release_request(const ResourceBundleGrant& grant,
                                       std::string id) {
  return seal_resource_release_request({
      .api_version = std::string(kHostLedgerReleaseRequestApiVersion),
      .release_request_id = std::move(id),
      .allocation_id = grant.allocation_id,
      .grant_digest = grant.receipt_digest,
      .journal_id = grant.journal_id,
      .run_id = grant.run_id,
      .logical_lease_id = grant.logical_lease_id,
      .logical_fencing_token = grant.logical_fencing_token,
      .canonical_request_digest = {},
  });
}

std::vector<std::filesystem::path>& test_directories() {
  static std::vector<std::filesystem::path> value;
  return value;
}

std::filesystem::path test_path(std::string suffix) {
  std::string pattern =
      (std::filesystem::temp_directory_path() /
       ("trainvm-host-ledger-" + std::move(suffix) + "-XXXXXX"))
          .string();
  std::vector<char> writable(pattern.begin(), pattern.end());
  writable.push_back('\0');
  const char* created = ::mkdtemp(writable.data());
  if (created == nullptr) {
    throw std::runtime_error("could not create host-ledger test directory");
  }
  const std::filesystem::path directory(created);
  test_directories().push_back(directory);
  return directory / "host-resource.db";
}

struct AuthorityRegistry final {
  std::mutex mutex;
  std::map<std::filesystem::path,
           std::shared_ptr<HostLedgerFilesystemAuthority>> authorities;

  ~AuthorityRegistry() {
    authorities.clear();
    for (const auto& directory : test_directories()) {
      std::error_code error;
      std::filesystem::remove_all(directory, error);
    }
  }
};

AuthorityRegistry& authority_registry() {
  static AuthorityRegistry value;
  return value;
}

std::shared_ptr<HostLedgerFilesystemAuthority> authority_for(
    const std::filesystem::path& path) {
  auto& registry = authority_registry();
  std::scoped_lock lock(registry.mutex);
  const auto found = registry.authorities.find(path);
  if (found != registry.authorities.end()) return found->second;
  auto authority = std::make_shared<HostLedgerFilesystemAuthority>(
      HostLedgerFilesystemAuthority::acquire({
          .api_version = std::string(kHostLedgerAuthorityApiVersion),
          .ledger_path = path,
          .expected_owner_uid = ::geteuid(),
          .expected_owner_gid = ::getegid(),
          .enforcement_grade = HostLedgerEnforcementGrade::cooperative_test,
      }));
  registry.authorities.emplace(path, authority);
  return authority;
}

void remove_database(const std::filesystem::path&) {
  // Authorities remain live until process teardown so tests cannot unlink a
  // database while a ledger still owns its host-global lock.
}

void raw_execute(const std::filesystem::path& path, const std::string& sql) {
  sqlite3* database = nullptr;
  require(sqlite3_open(path.c_str(), &database) == SQLITE_OK,
          "raw tamper connection opens");
  char* error = nullptr;
  const int status = sqlite3_exec(database, sql.c_str(), nullptr, nullptr, &error);
  const std::string detail = error ? error : "";
  sqlite3_free(error);
  sqlite3_close(database);
  require(status == SQLITE_OK, "raw tamper SQL succeeds: " + detail);
}

class OneShotFault final : public IHostLedgerFaultInjector {
 public:
  explicit OneShotFault(HostLedgerFaultPoint target) : target_(target) {}

  void hit(HostLedgerFaultPoint point) override {
    if (armed_ && point == target_) {
      armed_ = false;
      throw HostLedgerError("injected ledger fault");
    }
  }

  void arm() { armed_ = true; }

 private:
  HostLedgerFaultPoint target_;
  bool armed_{true};
};

void basic_replay_release_and_reopen() {
  const auto path = test_path("basic");
  const auto observed = inventory({"mutex-a"});
  ResourceBundleGrant second_grant;
  {
    SQLiteHostLedger ledger(authority_for(path), observed);
    require(ledger.record_count() == 1U && ledger.verify(),
            "new ledger has one valid inventory record");
    const auto first = ledger.request_bundle(request("request-1"), {10, 20});
    require(first.status == BundleRequestStatus::granted && first.grant &&
                first.grant->fences.size() == 1U &&
                first.grant->fences.front().generation == 1U &&
                ledger.generation(mutex_id("mutex-a")) == 1U &&
                ledger.occupancy().active_fences.size() == 1U,
            "first request atomically grants generation one");
    require(bundle_request_result_from_json(bundle_request_result_json(first)) ==
                first,
            "granted result has one strict canonical transport codec");
    const auto sealed_release = release_request(*first.grant, "release-codec");
    require(resource_release_request_from_json(
                resource_release_request_json(sealed_release)) == sealed_release,
            "release request has one strict canonical transport codec");
    auto forged_release = resource_release_request_json(sealed_release);
    forged_release["extra"] = true;
    require_throws<HostLedgerError>(
        [&] { (void)resource_release_request_from_json(forged_release); },
        "release request codec rejects unknown fields");
    const std::uint64_t after_grant = ledger.record_count();
    const auto replay = ledger.request_bundle(request("request-1"), {999, 999});
    require(replay.replayed && replay.grant == first.grant &&
                replay.outcome_digest == first.outcome_digest &&
                ledger.record_count() == after_grant &&
                ledger.generation(mutex_id("mutex-a")) == 1U,
            "exact request replay returns immutable grant without mutation");

    auto changed = request("request-1");
    changed.run_id = "run-002";
    changed = seal_resource_request(std::move(changed));
    require_throws<HostLedgerConflict>(
        [&] { (void)ledger.request_bundle(changed, {30, 40}); },
        "request ID reuse with changed content must conflict");

    const auto busy = ledger.request_bundle(request("request-2"), {30, 40});
    const std::uint64_t after_busy = ledger.record_count();
    const auto busy_replay =
        ledger.request_bundle(request("request-2"), {31, 41});
    require(busy.status == BundleRequestStatus::busy && !busy.grant &&
                busy_replay.status == BundleRequestStatus::busy &&
                busy_replay.replayed &&
                busy_replay.outcome_digest == busy.outcome_digest &&
                ledger.record_count() == after_busy,
            "busy decision is durable and exactly replayable");
    require(bundle_request_result_from_json(bundle_request_result_json(busy)) ==
                busy,
            "busy result has one strict canonical transport codec");
    auto forged_busy = bundle_request_result_json(busy);
    forged_busy["grant"] = resource_bundle_grant_json(*first.grant);
    require_throws<HostLedgerError>(
        [&] { (void)bundle_request_result_from_json(forged_busy); },
        "busy result codec rejects a contradictory grant");
    auto invalid_status = busy;
    invalid_status.status = static_cast<BundleRequestStatus>(99);
    require_throws<HostLedgerError>(
        [&] { (void)bundle_request_result_json(invalid_status); },
        "result codec rejects an invalid in-memory status enum");

    const auto release = ledger.release_bundle(
        release_request(*first.grant, "release-1"), {50, 60});
    const std::uint64_t after_release = ledger.record_count();
    const auto release_replay = ledger.release_bundle(
        release_request(*first.grant, "release-1"), {500, 600});
    require(!release.replayed && release_replay.replayed &&
                release_replay.receipt == release.receipt &&
                ledger.record_count() == after_release &&
                ledger.occupancy().active_fences.empty() &&
                ledger.generation(mutex_id("mutex-a")) == 1U,
            "release CAS is atomic, terminal, and exactly replayable");
    require(bundle_release_result_from_json(bundle_release_result_json(release)) ==
                release,
            "release result has one strict canonical transport codec");

    const auto second = ledger.request_bundle(request("request-3"), {70, 80});
    require(second.grant && second.grant->fences.front().generation == 2U,
            "regrant advances persistent generation");
    second_grant = *second.grant;
  }
  {
    SQLiteHostLedger reopened(authority_for(path), observed);
    require(reopened.verify() && reopened.generation(mutex_id("mutex-a")) == 2U &&
                reopened.occupancy().active_fences.size() == 1U &&
                reopened.occupancy().active_fences.front().generation == 2U,
            "reopen preserves chain, active grant, and generation");
    require_throws<HostLedgerConflict>(
        [&] {
          auto wrong = release_request(second_grant, "release-wrong");
          wrong.grant_digest = "sha256:" + std::string(64U, '0');
          wrong = seal_resource_release_request(std::move(wrong));
          (void)reopened.release_bundle(wrong, {90, 100});
        },
        "release with wrong grant identity cannot clear the bundle");
  }
  {
    const auto rebooted = inventory({"mutex-a"}, "boot-002", "broker-002");
    SQLiteHostLedger reopened(authority_for(path), rebooted);
    const auto blocked =
        reopened.request_bundle(request("request-after-reboot"), {10, 20});
    require(reopened.verify() &&
                reopened.generation(mutex_id("mutex-a")) == 2U &&
                reopened.occupancy().active_fences.size() == 1U &&
                blocked.status == BundleRequestStatus::busy,
            "reboot preserves generations and leaves prior-boot grants blocked");
    require_throws<HostLedgerConflict>(
        [&] {
          (void)reopened.release_bundle(
              release_request(second_grant, "prior-boot-release"), {30, 40});
        },
        "client release cannot clear a prior-boot host grant");
  }
  remove_database(path);
}

void concurrent_race() {
  const auto path = test_path("race");
  SQLiteHostLedger ledger(authority_for(path), inventory({"mutex-race"}));
  std::atomic<int> granted{0};
  std::atomic<int> busy{0};
  std::vector<std::thread> threads;
  for (int index = 0; index < 8; ++index) {
    threads.emplace_back([&, index] {
      const auto result = ledger.request_bundle(
          request("race-" + std::to_string(index)), {100 + index, 200 + index});
      if (result.status == BundleRequestStatus::granted) {
        ++granted;
      } else {
        ++busy;
      }
    });
  }
  for (auto& thread : threads) thread.join();
  require(granted == 1 && busy == 7 &&
              ledger.occupancy().active_fences.size() == 1U && ledger.verify(),
          "concurrent journals receive exactly one conflicting host grant");
  remove_database(path);
}

void independent_connection_race() {
  const auto path = test_path("independent-race");
  const auto observed = inventory({"mutex-independent"});
  SQLiteHostLedger first(authority_for(path), observed);
  SQLiteHostLedger second(authority_for(path), observed);
  std::atomic<int> granted{0};
  std::atomic<int> busy{0};
  std::thread left([&] {
    const auto result =
        first.request_bundle(request("independent-left"), {10, 20});
    result.status == BundleRequestStatus::granted ? ++granted : ++busy;
  });
  std::thread right([&] {
    const auto result =
        second.request_bundle(request("independent-right"), {11, 21});
    result.status == BundleRequestStatus::granted ? ++granted : ++busy;
  });
  left.join();
  right.join();
  require(granted == 1 && busy == 1 && first.verify() && second.verify() &&
              first.occupancy().active_fences.size() == 1U &&
              second.occupancy().active_fences.size() == 1U,
          "independent SQLite connections serialize one physical grant");
  remove_database(path);
}

void stale_inventory_instances_fail_closed() {
  const auto path = test_path("stale-inventory");
  const auto first_inventory =
      inventory({"mutex-stale"}, "boot-001", "broker-001");
  SQLiteHostLedger stale(authority_for(path), first_inventory);
  const auto current_inventory =
      inventory({"mutex-stale"}, "boot-002", "broker-002");
  SQLiteHostLedger current(authority_for(path), current_inventory);
  std::string reason;
  require(!stale.verify(&reason) && reason.find("stale") != std::string::npos &&
              current.verify(),
          "new inventory publication revokes stale live ledger instances");
  require_throws<HostLedgerError>([&] { (void)stale.occupancy(); },
                                  "stale occupancy read must fail closed");
  require_throws<HostLedgerError>(
      [&] { (void)stale.generation(mutex_id("mutex-stale")); },
      "stale generation read must fail closed");
  require_throws<HostLedgerError>([&] { (void)stale.record_count(); },
                                  "stale record read must fail closed");
  require_throws<HostLedgerError>([&] { (void)stale.inventory(); },
                                  "stale inventory read must fail closed");
  require_throws<HostLedgerError>(
      [&] {
        (void)stale.request_bundle(request("stale-request"), {10, 20});
      },
      "stale bundle mutation must fail closed");
  remove_database(path);
}

void request_and_release_rollback() {
  const auto path = test_path("rollback");
  const auto observed = inventory({"mutex-a", "mutex-b"});
  OneShotFault generation_fault(HostLedgerFaultPoint::after_generation_update);
  ResourceBundleGrant grant;
  {
    SQLiteHostLedger ledger(authority_for(path), observed, &generation_fault);
    require_throws<HostLedgerError>(
        [&] { (void)ledger.request_bundle(request("bundle", 2U), {10, 20}); },
        "generation fault must abort the request");
    require(ledger.record_count() == 1U &&
                ledger.generation(mutex_id("mutex-a")) == 0U &&
                ledger.generation(mutex_id("mutex-b")) == 0U &&
                ledger.occupancy().active_fences.empty() && ledger.verify(),
            "failed multi-resource grant rolls back records and all generations");
    const auto result =
        ledger.request_bundle(request("bundle", 2U), {30, 40});
    require(result.grant && result.grant->fences.size() == 2U &&
                std::ranges::all_of(result.grant->fences, [](const auto& fence) {
                  return fence.generation == 1U;
                }),
            "retry after rollback grants the whole bundle once");
    grant = *result.grant;
  }
  OneShotFault release_fault(HostLedgerFaultPoint::after_release_record);
  {
    SQLiteHostLedger ledger(authority_for(path), observed, &release_fault);
    const auto release = release_request(grant, "release-bundle");
    require_throws<HostLedgerError>(
        [&] { (void)ledger.release_bundle(release, {50, 60}); },
        "release fault must roll back terminal receipt");
    require(ledger.occupancy().active_fences.size() == 2U && ledger.verify(),
            "failed release leaves the exact full bundle blocked");
    const auto completed = ledger.release_bundle(release, {70, 80});
    require(!completed.replayed && ledger.occupancy().active_fences.empty() &&
                ledger.generation(mutex_id("mutex-a")) == 1U &&
                ledger.generation(mutex_id("mutex-b")) == 1U,
            "release retry commits once without resetting generations");
  }
  remove_database(path);
}

void tamper_is_fail_closed() {
  const auto path = test_path("tamper");
  const auto observed = inventory({"mutex-tamper"});
  {
    SQLiteHostLedger ledger(authority_for(path), observed);
    require(ledger.request_bundle(request("tamper-request"), {10, 20})
                .grant.has_value(),
            "tamper fixture grants a resource");
  }
  sqlite3* database = nullptr;
  require(sqlite3_open(path.c_str(), &database) == SQLITE_OK,
          "tamper fixture opens raw SQLite connection");
  require(sqlite3_exec(database,
                       "UPDATE resource_generations SET generation=99",
                       nullptr, nullptr, nullptr) == SQLITE_OK,
          "tamper fixture changes a rebuildable projection");
  sqlite3_close(database);
  require_throws<HostLedgerError>(
      [&] { SQLiteHostLedger rejected(authority_for(path), observed); },
      "projection tamper must reject ledger reopen");
  remove_database(path);

  const auto schema_path = test_path("schema-tamper");
  {
    SQLiteHostLedger ledger(authority_for(schema_path), observed);
  }
  require(sqlite3_open(schema_path.c_str(), &database) == SQLITE_OK,
          "schema tamper fixture opens raw connection");
  require(sqlite3_exec(database, "CREATE TABLE forged_authority(value TEXT)",
                       nullptr, nullptr, nullptr) == SQLITE_OK,
          "schema tamper fixture adds an authority table");
  sqlite3_close(database);
  require_throws<HostLedgerError>(
      [&] {
        SQLiteHostLedger rejected(authority_for(schema_path), observed);
      },
      "exact schema attestation must reject extra authority objects");
  remove_database(schema_path);

  const auto projection_tamper = [&](const std::string& name,
                                     const std::string& sql,
                                     bool release_first = false) {
    const auto tamper_path = test_path("projection-" + name);
    {
      SQLiteHostLedger ledger(authority_for(tamper_path), observed);
      const auto granted =
          ledger.request_bundle(request("projection-request"), {10, 20});
      require(granted.grant.has_value(),
              name + " fixture creates canonical grant evidence");
      if (release_first) {
        (void)ledger.release_bundle(
            release_request(*granted.grant, "projection-release"), {30, 40});
      }
    }
    raw_execute(tamper_path, "PRAGMA foreign_keys=OFF; " + sql);
    require_throws<HostLedgerError>(
        [&] {
          SQLiteHostLedger rejected(authority_for(tamper_path), observed);
        },
        name + " raw projection tamper must fail closed");
    remove_database(tamper_path);
  };

  projection_tamper("delete-active",
                    "DELETE FROM active_resource_grants");
  projection_tamper("delete-resource",
                    "DELETE FROM allocation_resources");
  projection_tamper("delete-allocation", "DELETE FROM allocations");
  projection_tamper("delete-request-outcome",
                    "DELETE FROM request_outcomes");
  projection_tamper("delete-release-outcome",
                    "DELETE FROM release_outcomes", true);
  projection_tamper(
      "released-status-spoof",
      "UPDATE allocations SET status='released', "
      "release_digest='sha256:0000000000000000000000000000000000000000000000000000000000000000'");
  projection_tamper(
      "active-release-digest-spoof",
      "UPDATE allocations SET "
      "release_digest='sha256:1111111111111111111111111111111111111111111111111111111111111111'");

  const std::vector<std::pair<std::string, std::string>> authority_mutations{
      {"allocation-id", "allocation_id='allocation-forged'"},
      {"request-id", "request_id='request-forged'"},
      {"request-digest",
       "request_digest='sha256:1111111111111111111111111111111111111111111111111111111111111111'"},
      {"grant-digest",
       "grant_digest='sha256:2222222222222222222222222222222222222222222222222222222222222222'"},
      {"journal-id", "journal_id='journal-forged'"},
      {"run-id", "run_id='run-forged'"},
      {"logical-lease-id", "logical_lease_id='lease-forged'"},
      {"logical-fencing-token", "logical_fencing_token=999"},
      {"host-id", "host_id='host-forged'"},
      {"grant-boot-id", "grant_boot_id='boot-forged'"},
      {"broker-epoch", "broker_epoch='broker-forged'"},
  };
  for (const auto& [name, assignment] : authority_mutations) {
    projection_tamper("authority-" + name,
                      "UPDATE allocations SET " + assignment);
  }
  const std::vector<std::pair<std::string, std::string>> outcome_mutations{
      {"request-id", "request_id='request-outcome-forged'"},
      {"request-digest",
       "request_digest='sha256:4444444444444444444444444444444444444444444444444444444444444444'"},
      {"status", "status='busy', allocation_id=NULL"},
      {"allocation-id", "allocation_id='allocation-outcome-forged'"},
      {"outcome-digest",
       "outcome_digest='sha256:5555555555555555555555555555555555555555555555555555555555555555'"},
      {"canonical-outcome", "canonical_outcome_json='{}'"},
  };
  for (const auto& [name, assignment] : outcome_mutations) {
    projection_tamper("outcome-" + name,
                      "UPDATE request_outcomes SET " + assignment);
  }
  const std::vector<std::pair<std::string, std::string>> resource_mutations{
      {"resource-key", "resource_key='host-mutex:forged'"},
      {"resource-json", "resource_json='{}'"},
      {"generation", "generation=99"},
      {"inventory-digest",
       "inventory_digest='sha256:6666666666666666666666666666666666666666666666666666666666666666'"},
      {"topology-digest",
       "topology_digest='sha256:7777777777777777777777777777777777777777777777777777777777777777'"},
  };
  for (const auto& [name, assignment] : resource_mutations) {
    projection_tamper("resource-" + name,
                      "UPDATE allocation_resources SET " + assignment);
  }
  const std::vector<std::pair<std::string, std::string>> active_mutations{
      {"resource-key", "resource_key='host-mutex:active-forged'"},
      {"allocation-id", "allocation_id='allocation-active-forged'"},
      {"generation", "generation=98"},
      {"grant-digest",
       "grant_digest='sha256:8888888888888888888888888888888888888888888888888888888888888888'"},
  };
  for (const auto& [name, assignment] : active_mutations) {
    projection_tamper("active-" + name,
                      "UPDATE active_resource_grants SET " + assignment);
  }
  projection_tamper("generation-last-allocation",
                    "UPDATE resource_generations SET "
                    "last_allocation_id='allocation-generation-forged'");
  projection_tamper(
      "generation-last-grant",
      "UPDATE resource_generations SET "
      "last_grant_digest='sha256:9999999999999999999999999999999999999999999999999999999999999999'");
  projection_tamper(
      "generation-extra-row",
      "INSERT INTO resource_generations(resource_key, generation) "
      "VALUES('host-mutex:unowned-extra', 0)");
  const std::vector<std::pair<std::string, std::string>> release_mutations{
      {"release-request-id", "release_request_id='release-forged'"},
      {"release-request-digest",
       "release_request_digest='sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'"},
      {"allocation-id", "allocation_id='allocation-release-forged'"},
      {"grant-digest",
       "grant_digest='sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'"},
      {"release-receipt-digest",
       "release_receipt_digest='sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc'"},
      {"canonical-release", "canonical_release_json='{}'"},
  };
  for (const auto& [name, assignment] : release_mutations) {
    projection_tamper("release-" + name,
                      "UPDATE release_outcomes SET " + assignment, true);
  }
  projection_tamper(
      "inventory-digest",
      "UPDATE current_inventory SET "
      "inventory_digest='sha256:3333333333333333333333333333333333333333333333333333333333333333'");
  projection_tamper(
      "inventory-json",
      "UPDATE current_inventory SET canonical_json='{}'");

  const auto chain_gap_path = test_path("chain-gap");
  {
    SQLiteHostLedger ledger(authority_for(chain_gap_path), observed);
    require(ledger.request_bundle(request("chain-gap-request"), {10, 20})
                .grant.has_value(),
            "chain gap fixture appends a multi-record history");
  }
  raw_execute(
      chain_gap_path,
      "DROP TRIGGER ledger_records_no_delete; "
      "DELETE FROM ledger_records WHERE ledger_sequence=2; "
      "CREATE TRIGGER ledger_records_no_delete BEFORE DELETE ON ledger_records "
      "BEGIN SELECT RAISE(ABORT, 'ledger records are immutable'); END;");
  require_throws<HostLedgerError>(
      [&] {
        SQLiteHostLedger rejected(authority_for(chain_gap_path), observed);
      },
      "gapped ledger sequence must fail closed");
  remove_database(chain_gap_path);

  const auto canonical_path = test_path("noncanonical-record");
  {
    SQLiteHostLedger ledger(authority_for(canonical_path), observed);
  }
  raw_execute(
      canonical_path,
      "DROP TRIGGER ledger_records_no_update; "
      "UPDATE ledger_records SET canonical_json=canonical_json || ' '; "
      "CREATE TRIGGER ledger_records_no_update BEFORE UPDATE ON ledger_records "
      "BEGIN SELECT RAISE(ABORT, 'ledger records are immutable'); END;");
  require_throws<HostLedgerError>(
      [&] {
        SQLiteHostLedger rejected(authority_for(canonical_path), observed);
      },
      "noncanonical record bytes must fail closed");
  remove_database(canonical_path);
}

void degraded_inventory_publication_rolls_back() {
  const auto path = test_path("degraded-inventory");
  const auto original = inventory({"mutex-owned"});
  SQLiteHostLedger live(authority_for(path), original);
  require(live.request_bundle(request("owned-request"), {10, 20})
              .grant.has_value(),
          "degradation fixture owns a physical resource");
  const auto missing =
      inventory({"mutex-other"}, "boot-002", "broker-002");
  require_throws<HostLedgerError>(
      [&] { SQLiteHostLedger rejected(authority_for(path), missing); },
      "inventory that loses an active resource must be rejected");
  require(live.verify() && live.inventory() == original &&
              live.occupancy().active_fences.size() == 1U,
          "failed degraded inventory publication rolls back atomically");
  remove_database(path);
}

void filesystem_authority_remains_bound() {
  const auto path = test_path("bound-authority");
  const auto observed = inventory({"mutex-bound"});
  SQLiteHostLedger ledger(authority_for(path), observed);
  require(ledger.verify(), "bound authority fixture begins valid");

  const auto displaced = path.string() + ".displaced";
  require(::rename(path.c_str(), displaced.c_str()) == 0,
          "replace bound database pathname");
  const int replacement =
      ::open(path.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
  require(replacement >= 0 && ::fchmod(replacement, 0600) == 0 &&
              ::close(replacement) == 0,
          "create protected-looking replacement database");
  std::string reason;
  require(!ledger.verify(&reason),
          "database inode replacement revokes the live SQLite ledger");
  require_throws<HostLedgerError>(
      [&] { (void)ledger.request_bundle(request("bound-regrant"), {10, 20}); },
      "revoked filesystem authority blocks further mutation");
}

}  // namespace

int main() {
  try {
    basic_replay_release_and_reopen();
    concurrent_race();
    independent_connection_race();
    stale_inventory_instances_fail_closed();
    request_and_release_rollback();
    tamper_is_fail_closed();
    degraded_inventory_publication_rolls_back();
    filesystem_authority_remains_bound();
    std::cout << "host ledger tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "host ledger test failure: " << error.what() << '\n';
    return 1;
  }
}
