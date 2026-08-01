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
          .device_nodes = {},
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

std::string test_digest(char digit) {
  return "sha256:" + std::string(64U, digit);
}

HostProcessLaunchRequest launch_request(const ResourceBundleGrant& grant,
                                        std::string launch_id) {
  return seal_host_process_launch_request({
      .api_version = std::string(kHostProcessLaunchRequestApiVersion),
      .launch_id = std::move(launch_id),
      .allocation_id = grant.allocation_id,
      .grant_digest = grant.receipt_digest,
      .journal_id = grant.journal_id,
      .run_id = grant.run_id,
      .logical_lease_id = grant.logical_lease_id,
      .logical_fencing_token = grant.logical_fencing_token,
      .resolved_launch_digest = test_digest('1'),
      .executable_path = "/usr/bin/trainvm-test-worker",
      .executable_digest = test_digest('2'),
      .cgroup_path = "/sys/fs/cgroup/trainvm/test-launch",
      .cgroup_device = 31,
      .cgroup_inode = 41,
      .canonical_request_digest = {},
  });
}

HostProcessSpawnRequest spawn_request(const HostProcessLaunchIntent& intent,
                                      std::int64_t pid = 4242) {
  return seal_host_process_spawn_request({
      .api_version = std::string(kHostProcessSpawnRequestApiVersion),
      .launch_id = intent.request.launch_id,
      .launch_intent_digest = intent.receipt_digest,
      .host_pid = pid,
      .process_starttime_ticks = 91234,
      .boot_id = intent.boot_id,
      .cgroup_path = intent.request.cgroup_path,
      .cgroup_device = intent.request.cgroup_device,
      .cgroup_inode = intent.request.cgroup_inode,
      .executable_digest = intent.request.executable_digest,
      .canonical_request_digest = {},
  });
}

HostProcessExitRequest exit_request(const HostProcessSpawnReceipt& spawn,
                                    std::string exit_request_id) {
  return seal_host_process_exit_request({
      .api_version = std::string(kHostProcessExitRequestApiVersion),
      .exit_request_id = std::move(exit_request_id),
      .launch_id = spawn.request.launch_id,
      .spawn_receipt_digest = spawn.receipt_digest,
      .host_pid = spawn.request.host_pid,
      .process_starttime_ticks = spawn.request.process_starttime_ticks,
      .wait_code = 1,
      .wait_status = 0,
      .cgroup_path = spawn.request.cgroup_path,
      .cgroup_device = spawn.request.cgroup_device,
      .cgroup_inode = spawn.request.cgroup_inode,
      .cgroup_empty = true,
      .accelerator_contexts_empty = true,
      .context_audit_digest = test_digest('4'),
      .canonical_request_digest = {},
  });
}

HostProcessRecoveryExitRequest recovery_exit_request(
    const HostProcessSpawnReceipt& spawn, std::string request_id) {
  return seal_host_process_recovery_exit_request({
      .api_version =
          std::string(kHostProcessRecoveryExitRequestApiVersion),
      .recovery_exit_request_id = std::move(request_id),
      .launch_id = spawn.request.launch_id,
      .spawn_receipt_digest = spawn.receipt_digest,
      .host_pid = spawn.request.host_pid,
      .process_starttime_ticks = spawn.request.process_starttime_ticks,
      .observation = HostProcessRecoveryExitObservation::pidfd_terminal,
      .observation_digest = test_digest('5'),
      .cgroup_path = spawn.request.cgroup_path,
      .cgroup_device = spawn.request.cgroup_device,
      .cgroup_inode = spawn.request.cgroup_inode,
      .cgroup_empty = true,
      .accelerator_contexts_empty = true,
      .context_audit_digest = test_digest('6'),
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

std::string raw_scalar(const std::filesystem::path& path,
                       const std::string& sql) {
  sqlite3* database = nullptr;
  require(sqlite3_open(path.c_str(), &database) == SQLITE_OK,
          "raw scalar connection opens");
  sqlite3_stmt* statement = nullptr;
  require(sqlite3_prepare_v2(database, sql.c_str(), -1, &statement, nullptr) ==
              SQLITE_OK,
          "raw scalar query prepares");
  require(sqlite3_step(statement) == SQLITE_ROW, "raw scalar query has a row");
  const auto* bytes = sqlite3_column_text(statement, 0);
  const std::string result =
      bytes == nullptr ? std::string{} : reinterpret_cast<const char*>(bytes);
  sqlite3_finalize(statement);
  sqlite3_close(database);
  return result;
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

void process_authority_is_durable_and_replay_safe() {
  const auto observed = inventory({"mutex-process"});
  const auto path = test_path("process-basic");
  {
    SQLiteHostLedger ledger(authority_for(path), observed);
    const auto bundle = ledger.request_bundle(request("process-grant"), {10, 20});
    require(bundle.grant.has_value(), "process fixture obtains an active grant");
    const auto launch = launch_request(*bundle.grant, "launch-basic");
    require(host_process_launch_request_from_json(
                host_process_launch_request_json(launch)) == launch,
            "launch request has a strict canonical codec");
    auto forged_launch = host_process_launch_request_json(launch);
    forged_launch["unknown"] = true;
    require_throws<HostLedgerError>(
        [&] { (void)host_process_launch_request_from_json(forged_launch); },
        "launch request codec rejects unknown fields");
    const auto intent = ledger.commit_process_launch_intent(launch, {30, 40});
    const auto intent_replay =
        ledger.commit_process_launch_intent(launch, {300, 400});
    require(!intent.replayed && intent_replay.replayed &&
                intent_replay.intent == intent.intent && ledger.verify(),
            "launch intent commits once and exactly replays");
    require(host_process_launch_intent_from_json(
                host_process_launch_intent_json(intent.intent)) == intent.intent,
            "launch intent has a strict canonical codec");
    auto recovery = ledger.active_process_recovery_records();
    require(recovery.size() == 1U && recovery.front().grant == *bundle.grant &&
                recovery.front().intent == intent.intent &&
                !recovery.front().spawn,
            "recovery view exposes an active intent without inventing a PID");
    require_throws<HostLedgerError>(
        [&] { (void)ledger.active_process_recovery_records(0U); },
        "recovery view rejects an unusable bound");
    auto changed_launch = launch;
    changed_launch.resolved_launch_digest = test_digest('3');
    changed_launch = seal_host_process_launch_request(std::move(changed_launch));
    require_throws<HostLedgerConflict>(
        [&] { (void)ledger.commit_process_launch_intent(changed_launch, {31, 41}); },
        "launch ID reuse with changed content conflicts");

    const auto spawn = spawn_request(intent.intent);
    require(host_process_spawn_request_from_json(
                host_process_spawn_request_json(spawn)) == spawn,
            "spawn observation has a strict canonical codec");
    const auto receipt = ledger.commit_process_spawn(spawn, {50, 60});
    const auto receipt_replay = ledger.commit_process_spawn(spawn, {500, 600});
    require(!receipt.replayed && receipt_replay.replayed &&
                receipt_replay.receipt == receipt.receipt && ledger.verify(),
            "spawn receipt commits once and exactly replays");
    require(host_process_spawn_receipt_from_json(
                host_process_spawn_receipt_json(receipt.receipt)) ==
                receipt.receipt,
            "spawn receipt has a strict canonical codec");
    recovery = ledger.active_process_recovery_records();
    require(recovery.size() == 1U &&
                recovery.front().spawn == receipt.receipt,
            "recovery view joins the exact unclosed spawn receipt");
    auto changed_spawn = spawn;
    changed_spawn.host_pid += 1;
    changed_spawn = seal_host_process_spawn_request(std::move(changed_spawn));
    require_throws<HostLedgerConflict>(
        [&] { (void)ledger.commit_process_spawn(changed_spawn, {51, 61}); },
        "spawn replay with a changed kernel identity conflicts");
    require_throws<HostLedgerConflict>(
        [&] {
          (void)ledger.release_bundle(
              release_request(*bundle.grant, "release-before-exit"),
              {65, 66});
        },
        "spawned allocation cannot release before terminal process evidence");
    const auto terminal = exit_request(receipt.receipt, "exit-basic");
    require(host_process_exit_request_from_json(
                host_process_exit_request_json(terminal)) == terminal,
            "terminal exit evidence has a strict canonical codec");
    const auto exited = ledger.commit_process_exit(terminal, {70, 80});
    const auto exit_replay = ledger.commit_process_exit(terminal, {700, 800});
    require(!exited.replayed && exit_replay.replayed &&
                exit_replay.receipt == exited.receipt && ledger.verify(),
            "terminal process exit commits once and exactly replays");
    require(ledger.active_process_recovery_records().empty(),
            "terminal process disappears from the recovery view");
    const auto terminal_release =
        ledger.active_terminal_process_release_records();
    require(terminal_release.size() == 1U &&
                terminal_release.front().grant == *bundle.grant &&
                terminal_release.front().spawn == receipt.receipt &&
                terminal_release.front().child_exit == exited.receipt &&
                !terminal_release.front().recovery_exit,
            "ordinary terminal process remains visible until bundle release");
    require_throws<HostLedgerError>(
        [&] { (void)ledger.active_terminal_process_release_records(0U); },
        "terminal release view rejects an unusable bound");
    require(!ledger.release_bundle(
                        release_request(*bundle.grant, "release-after-exit"),
                        {90, 100})
                 .replayed &&
                ledger.verify() &&
                ledger.active_terminal_process_release_records().empty(),
            "terminal process authority permits exact bundle release");
    auto incomplete = terminal;
    incomplete.cgroup_empty = false;
    require_throws<HostLedgerError>(
        [&] { (void)seal_host_process_exit_request(std::move(incomplete)); },
        "nonempty cgroup can never produce terminal exit authority");
  }

  const auto recovery_terminal_path =
      test_path("process-recovery-terminal-basic");
  {
    SQLiteHostLedger ledger(authority_for(recovery_terminal_path), observed);
    const auto bundle =
        ledger.request_bundle(request("recovery-terminal-grant"), {10, 20});
    const auto intent = ledger.commit_process_launch_intent(
        launch_request(*bundle.grant, "launch-recovery-terminal"), {30, 40});
    const auto spawn = ledger.commit_process_spawn(
        spawn_request(intent.intent, 4343), {50, 60});
    const auto terminal =
        recovery_exit_request(spawn.receipt, "recovery-exit-basic");
    require(host_process_recovery_exit_request_from_json(
                host_process_recovery_exit_request_json(terminal)) == terminal,
            "recovery terminal evidence has a strict canonical codec");
    const auto exited =
        ledger.commit_process_recovery_exit(terminal, {70, 80});
    const auto replayed =
        ledger.commit_process_recovery_exit(terminal, {700, 800});
    require(!exited.replayed && replayed.replayed &&
                exited.receipt == replayed.receipt && ledger.verify() &&
                ledger.active_process_recovery_records().empty(),
            "recovery terminal receipt commits once and closes recovery view");
    const auto terminal_release =
        ledger.active_terminal_process_release_records();
    require(terminal_release.size() == 1U &&
                !terminal_release.front().child_exit &&
                terminal_release.front().recovery_exit == exited.receipt,
            "recovery terminal process remains visible until bundle release");
    require(host_process_recovery_exit_receipt_from_json(
                host_process_recovery_exit_receipt_json(exited.receipt)) ==
                exited.receipt,
            "recovery terminal receipt has a strict canonical codec");
    require_throws<HostLedgerConflict>(
        [&] {
          (void)ledger.commit_process_exit(
              exit_request(spawn.receipt, "ordinary-after-recovery"),
              {81, 82});
        },
        "child-wait exit cannot coexist with a recovery exit");
    require(!ledger.release_bundle(
                        release_request(*bundle.grant,
                                        "release-after-recovery-exit"),
                        {90, 100})
                 .replayed &&
                ledger.verify() &&
                ledger.active_terminal_process_release_records().empty(),
            "recovery terminal authority permits exact bundle release");
    auto incomplete = terminal;
    incomplete.cgroup_empty = false;
    require_throws<HostLedgerError>(
        [&] {
          (void)seal_host_process_recovery_exit_request(
              std::move(incomplete));
        },
        "recovery exit cannot claim a nonempty allocation cgroup");
  }

  const auto multi_spawn_path = test_path("process-multiple-spawn-closure");
  {
    SQLiteHostLedger ledger(authority_for(multi_spawn_path), observed);
    const auto bundle =
        ledger.request_bundle(request("multi-spawn-grant"), {10, 20});
    const auto first_intent = ledger.commit_process_launch_intent(
        launch_request(*bundle.grant, "multi-spawn-first"), {30, 40});
    const auto first_spawn = ledger.commit_process_spawn(
        spawn_request(first_intent.intent, 5041), {50, 60});
    (void)ledger.commit_process_exit(
        exit_request(first_spawn.receipt, "multi-spawn-first-exit"), {70, 80});
    const auto second_intent = ledger.commit_process_launch_intent(
        launch_request(*bundle.grant, "multi-spawn-second"), {81, 82});
    const auto second_spawn = ledger.commit_process_spawn(
        spawn_request(second_intent.intent, 5042), {83, 84});
    require(ledger.active_terminal_process_release_records().size() == 1U &&
                ledger.active_process_recovery_records().size() == 1U,
            "restart views distinguish closed and unclosed sibling launches");
    require_throws<HostLedgerConflict>(
        [&] {
          (void)ledger.release_bundle(
              release_request(*bundle.grant, "multi-spawn-early-release"),
              {90, 91});
        },
        "one terminal launch cannot hide another unclosed spawn");
    (void)ledger.commit_process_recovery_exit(
        recovery_exit_request(second_spawn.receipt,
                              "multi-spawn-second-exit"),
        {92, 93});
    require(ledger.active_terminal_process_release_records().size() == 2U &&
                ledger.active_process_recovery_records().empty(),
            "every closed sibling remains available for cleanup replay");
    require(!ledger.release_bundle(
                        release_request(*bundle.grant,
                                        "multi-spawn-complete-release"),
                        {94, 95})
                 .replayed &&
                ledger.verify() &&
                ledger.active_terminal_process_release_records().empty(),
            "every spawned launch must close before allocation release");
  }

  const auto rollback_path = test_path("process-intent-rollback");
  OneShotFault intent_projection_fault(
      HostLedgerFaultPoint::after_process_intent_projection);
  {
    SQLiteHostLedger ledger(authority_for(rollback_path), observed,
                            &intent_projection_fault);
    const auto bundle =
        ledger.request_bundle(request("process-rollback-grant"), {10, 20});
    const auto launch = launch_request(*bundle.grant, "launch-rollback");
    require_throws<HostLedgerError>(
        [&] { (void)ledger.commit_process_launch_intent(launch, {30, 40}); },
        "pre-commit launch-intent fault rolls back record and projection");
    const auto retry = ledger.commit_process_launch_intent(launch, {31, 41});
    require(!retry.replayed && ledger.verify(),
            "rolled-back launch intent retries as a fresh mutation");
  }

  const auto lost_intent_path = test_path("process-intent-lost-reply");
  OneShotFault intent_commit_fault(
      HostLedgerFaultPoint::after_process_intent_commit);
  {
    SQLiteHostLedger ledger(authority_for(lost_intent_path), observed,
                            &intent_commit_fault);
    const auto bundle =
        ledger.request_bundle(request("process-lost-intent-grant"), {10, 20});
    const auto launch = launch_request(*bundle.grant, "launch-lost-intent");
    require_throws<HostLedgerError>(
        [&] { (void)ledger.commit_process_launch_intent(launch, {30, 40}); },
        "post-commit launch-intent fault simulates a lost reply");
    const auto retry = ledger.commit_process_launch_intent(launch, {31, 41});
    require(retry.replayed && ledger.verify(),
            "lost launch-intent reply resolves to exact durable replay");
  }

  const auto lost_spawn_path = test_path("process-spawn-lost-reply");
  OneShotFault spawn_commit_fault(
      HostLedgerFaultPoint::after_process_spawn_commit);
  {
    SQLiteHostLedger ledger(authority_for(lost_spawn_path), observed,
                            &spawn_commit_fault);
    const auto bundle =
        ledger.request_bundle(request("process-lost-spawn-grant"), {10, 20});
    const auto intent = ledger.commit_process_launch_intent(
        launch_request(*bundle.grant, "launch-lost-spawn"), {30, 40});
    const auto spawn = spawn_request(intent.intent, 4343);
    require_throws<HostLedgerError>(
        [&] { (void)ledger.commit_process_spawn(spawn, {50, 60}); },
        "post-commit spawn fault simulates a lost reply");
    const auto retry = ledger.commit_process_spawn(spawn, {51, 61});
    require(retry.replayed && ledger.verify(),
            "lost spawn reply resolves to exact durable replay");
  }

  const auto spawn_rollback_path = test_path("process-spawn-rollback");
  OneShotFault spawn_projection_fault(
      HostLedgerFaultPoint::after_process_spawn_projection);
  {
    SQLiteHostLedger ledger(authority_for(spawn_rollback_path), observed,
                            &spawn_projection_fault);
    const auto bundle =
        ledger.request_bundle(request("process-spawn-rollback-grant"), {10, 20});
    const auto intent = ledger.commit_process_launch_intent(
        launch_request(*bundle.grant, "launch-spawn-rollback"), {30, 40});
    const auto spawn = spawn_request(intent.intent, 4444);
    require_throws<HostLedgerError>(
        [&] { (void)ledger.commit_process_spawn(spawn, {50, 60}); },
        "pre-commit spawn fault rolls back record and projection");
    const auto retry = ledger.commit_process_spawn(spawn, {51, 61});
    require(!retry.replayed && ledger.verify(),
            "rolled-back spawn observation retries as a fresh mutation");
  }

  const auto released_path = test_path("process-released-before-spawn");
  {
    SQLiteHostLedger ledger(authority_for(released_path), observed);
    const auto bundle =
        ledger.request_bundle(request("process-released-grant"), {10, 20});
    const auto intent = ledger.commit_process_launch_intent(
        launch_request(*bundle.grant, "launch-released"), {30, 40});
    (void)ledger.release_bundle(
        release_request(*bundle.grant, "process-release-before-spawn"),
        {41, 42});
    require_throws<HostLedgerConflict>(
        [&] {
          (void)ledger.commit_process_spawn(spawn_request(intent.intent, 4545),
                                            {50, 60});
        },
        "a released allocation can never acquire a spawn receipt");
    require(ledger.verify(), "abandoned launch intent remains valid evidence");
  }

  const auto exit_rollback_path = test_path("process-exit-rollback");
  OneShotFault exit_projection_fault(
      HostLedgerFaultPoint::after_process_exit_projection);
  {
    SQLiteHostLedger ledger(authority_for(exit_rollback_path), observed,
                            &exit_projection_fault);
    const auto bundle =
        ledger.request_bundle(request("process-exit-rollback-grant"), {10, 20});
    const auto intent = ledger.commit_process_launch_intent(
        launch_request(*bundle.grant, "launch-exit-rollback"), {30, 40});
    const auto spawn = ledger.commit_process_spawn(
        spawn_request(intent.intent, 4646), {50, 60});
    const auto terminal = exit_request(spawn.receipt, "exit-rollback");
    require_throws<HostLedgerError>(
        [&] { (void)ledger.commit_process_exit(terminal, {70, 80}); },
        "pre-commit exit fault rolls back record and projection");
    const auto retry = ledger.commit_process_exit(terminal, {71, 81});
    require(!retry.replayed && ledger.verify(),
            "rolled-back exit evidence retries as a fresh mutation");
  }

  const auto lost_exit_path = test_path("process-exit-lost-reply");
  OneShotFault exit_commit_fault(
      HostLedgerFaultPoint::after_process_exit_commit);
  {
    SQLiteHostLedger ledger(authority_for(lost_exit_path), observed,
                            &exit_commit_fault);
    const auto bundle =
        ledger.request_bundle(request("process-lost-exit-grant"), {10, 20});
    const auto intent = ledger.commit_process_launch_intent(
        launch_request(*bundle.grant, "launch-lost-exit"), {30, 40});
    const auto spawn = ledger.commit_process_spawn(
        spawn_request(intent.intent, 4747), {50, 60});
    const auto terminal = exit_request(spawn.receipt, "exit-lost-reply");
    require_throws<HostLedgerError>(
        [&] { (void)ledger.commit_process_exit(terminal, {70, 80}); },
        "post-commit exit fault simulates a lost reply");
    const auto retry = ledger.commit_process_exit(terminal, {71, 81});
    require(retry.replayed && ledger.verify(),
            "lost exit reply resolves to exact durable replay");
  }

  const auto recovery_exit_rollback_path =
      test_path("process-recovery-exit-rollback");
  OneShotFault recovery_exit_projection_fault(
      HostLedgerFaultPoint::after_process_recovery_exit_projection);
  {
    SQLiteHostLedger ledger(authority_for(recovery_exit_rollback_path),
                            observed, &recovery_exit_projection_fault);
    const auto bundle = ledger.request_bundle(
        request("process-recovery-exit-rollback-grant"), {10, 20});
    const auto intent = ledger.commit_process_launch_intent(
        launch_request(*bundle.grant, "launch-recovery-exit-rollback"),
        {30, 40});
    const auto spawn = ledger.commit_process_spawn(
        spawn_request(intent.intent, 4848), {50, 60});
    const auto terminal =
        recovery_exit_request(spawn.receipt, "recovery-exit-rollback");
    require_throws<HostLedgerError>(
        [&] {
          (void)ledger.commit_process_recovery_exit(terminal, {70, 80});
        },
        "pre-commit recovery exit fault rolls back record and projection");
    const auto retry =
        ledger.commit_process_recovery_exit(terminal, {71, 81});
    require(!retry.replayed && ledger.verify(),
            "rolled-back recovery exit retries as a fresh mutation");
  }

  const auto lost_recovery_exit_path =
      test_path("process-recovery-exit-lost-reply");
  OneShotFault recovery_exit_commit_fault(
      HostLedgerFaultPoint::after_process_recovery_exit_commit);
  {
    SQLiteHostLedger ledger(authority_for(lost_recovery_exit_path), observed,
                            &recovery_exit_commit_fault);
    const auto bundle = ledger.request_bundle(
        request("process-lost-recovery-exit-grant"), {10, 20});
    const auto intent = ledger.commit_process_launch_intent(
        launch_request(*bundle.grant, "launch-lost-recovery-exit"), {30, 40});
    const auto spawn = ledger.commit_process_spawn(
        spawn_request(intent.intent, 4949), {50, 60});
    const auto terminal =
        recovery_exit_request(spawn.receipt, "recovery-exit-lost-reply");
    require_throws<HostLedgerError>(
        [&] {
          (void)ledger.commit_process_recovery_exit(terminal, {70, 80});
        },
        "post-commit recovery exit fault simulates a lost reply");
    const auto retry =
        ledger.commit_process_recovery_exit(terminal, {71, 81});
    require(retry.replayed && ledger.verify(),
            "lost recovery exit reply resolves to exact durable replay");
  }

  const auto tamper_path = test_path("process-chain-tamper");
  {
    SQLiteHostLedger ledger(authority_for(tamper_path), observed);
    const auto bundle =
        ledger.request_bundle(request("process-tamper-grant"), {10, 20});
    (void)ledger.commit_process_launch_intent(
        launch_request(*bundle.grant, "launch-tamper"), {30, 40});
  }
  raw_execute(tamper_path,
              "DROP TRIGGER process_records_no_update; "
              "UPDATE process_records SET canonical_json='{}'; "
              "CREATE TRIGGER process_records_no_update BEFORE UPDATE ON "
              "process_records BEGIN SELECT RAISE(ABORT, 'process records are "
              "immutable'); END;");
  require_throws<HostLedgerError>(
      [&] { SQLiteHostLedger rejected(authority_for(tamper_path), observed); },
      "process hash-chain tampering fails closed on reopen");

  const auto projection_path = test_path("process-projection-tamper");
  {
    SQLiteHostLedger ledger(authority_for(projection_path), observed);
    const auto bundle =
        ledger.request_bundle(request("process-projection-grant"), {10, 20});
    (void)ledger.commit_process_launch_intent(
        launch_request(*bundle.grant, "launch-projection"), {30, 40});
  }
  raw_execute(
      projection_path,
      "DROP TRIGGER process_launch_intents_no_update; "
      "UPDATE process_launch_intents SET "
      "request_digest='sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff'; "
      "CREATE TRIGGER process_launch_intents_no_update BEFORE UPDATE ON "
      "process_launch_intents BEGIN SELECT RAISE(ABORT, 'process launch "
      "intents are immutable'); END;");
  require_throws<HostLedgerError>(
      [&] {
        SQLiteHostLedger rejected(authority_for(projection_path), observed);
      },
      "process projection tampering fails closed on reopen");

  const auto v4_path = test_path("process-exit-v4-migration");
  {
    SQLiteHostLedger ledger(authority_for(v4_path), observed);
    require(ledger.verify(), "v4 migration fixture begins as valid v6");
  }
  raw_execute(v4_path,
              "DROP TRIGGER process_recovery_exits_no_update; "
              "DROP TRIGGER process_recovery_exits_no_delete; "
              "DROP TABLE process_recovery_exits; "
              "DROP TRIGGER process_exits_no_update; "
              "DROP TRIGGER process_exits_no_delete; "
              "DROP TABLE process_exits; "
              "DELETE FROM ledger_schema_extensions "
              "WHERE feature IN ('process_exit','process_recovery_exit'); "
              "PRAGMA user_version=4;");
  {
    SQLiteHostLedger migrated(authority_for(v4_path), observed);
    require(migrated.verify(),
            "empty v4 process authority migrates additively through v6");
  }

  const auto v5_path = test_path("process-recovery-exit-v5-migration");
  std::string v5_process_bytes;
  {
    SQLiteHostLedger ledger(authority_for(v5_path), observed);
    const auto bundle =
        ledger.request_bundle(request("v5-migration-grant"), {10, 20});
    const auto intent = ledger.commit_process_launch_intent(
        launch_request(*bundle.grant, "v5-migration-launch"), {30, 40});
    const auto spawn = ledger.commit_process_spawn(
        spawn_request(intent.intent, 5151), {50, 60});
    (void)ledger.commit_process_exit(
        exit_request(spawn.receipt, "v5-migration-exit"), {70, 80});
    require(!ledger.release_bundle(
                        release_request(*bundle.grant,
                                        "v5-migration-release"),
                        {90, 100})
                 .replayed,
            "v5 migration fixture has a released terminal process");
  }
  v5_process_bytes = raw_scalar(
      v5_path,
      "SELECT group_concat(canonical_json, char(10)) FROM process_records "
      "ORDER BY process_sequence");
  raw_execute(v5_path,
              "DROP TRIGGER process_recovery_exits_no_update; "
              "DROP TRIGGER process_recovery_exits_no_delete; "
              "DROP TABLE process_recovery_exits; "
              "DELETE FROM ledger_schema_extensions "
              "WHERE feature='process_recovery_exit'; PRAGMA user_version=5;");
  {
    SQLiteHostLedger migrated(authority_for(v5_path), observed);
    require(migrated.verify() &&
                raw_scalar(v5_path, "PRAGMA user_version") == "6" &&
                raw_scalar(
                    v5_path,
                    "SELECT group_concat(canonical_json, char(10)) FROM "
                    "process_records ORDER BY process_sequence") ==
                    v5_process_bytes,
            "v5 terminal history migrates to v6 without changing chain bytes");
  }
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
    process_authority_is_durable_and_replay_safe();
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
