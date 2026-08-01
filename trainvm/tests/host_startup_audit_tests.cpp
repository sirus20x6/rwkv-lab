#include "trainvm/host_ledger.hpp"
#include "trainvm/host_startup_audit.hpp"
#include "trainvm/reflection_json.hpp"

#include <sqlite3.h>

#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

std::vector<std::filesystem::path>& directories() {
  static std::vector<std::filesystem::path> value;
  return value;
}

std::filesystem::path test_path(std::string suffix) {
  std::string pattern =
      (std::filesystem::temp_directory_path() /
       ("trainvm-startup-audit-" + std::move(suffix) + "-XXXXXX"))
          .string();
  std::vector<char> writable(pattern.begin(), pattern.end());
  writable.push_back('\0');
  const char* created = ::mkdtemp(writable.data());
  if (created == nullptr) throw std::runtime_error("could not create test directory");
  directories().emplace_back(created);
  return directories().back() / "host-resource.db";
}

struct DirectoryCleanup final {
  ~DirectoryCleanup() {
    for (const auto& directory : directories()) {
      std::error_code error;
      std::filesystem::remove_all(directory, error);
    }
  }
};

std::shared_ptr<HostLedgerFilesystemAuthority> authority_for(
    const std::filesystem::path& path) {
  return std::make_shared<HostLedgerFilesystemAuthority>(
      HostLedgerFilesystemAuthority::acquire({
          .api_version = std::string(kHostLedgerAuthorityApiVersion),
          .ledger_path = path,
          .expected_owner_uid = ::geteuid(),
          .expected_owner_gid = ::getegid(),
          .enforcement_grade = HostLedgerEnforcementGrade::cooperative_test,
      }));
}

HostInventoryReceipt inventory(std::string broker_epoch = "epoch-audit") {
  const HostResourceId id{.kind = HostResourceKind::host_mutex,
                          .vendor = std::nullopt,
                          .stable_id = "host-mutex:audit",
                          .parent_id = std::nullopt};
  const ObservedHostResource resource{
      .id = id,
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
      .labels = {{"scope", "audit-test"}},
  };
  HostKernelSnapshot snapshot{
      .api_version = std::string(kHostInventoryApiVersion),
      .host_id = "host-audit",
      .boot_id = "boot-audit",
      .broker_epoch = std::move(broker_epoch),
      .begin_revision = "revision-audit",
      .end_revision = "revision-audit",
      .probes = {},
      .resources = {resource},
  };
  std::vector<FakeHostKernelStep> script;
  script.push_back(
      {.snapshot = std::move(snapshot), .failure = std::nullopt});
  FakeHostKernel kernel(std::move(script));
  return capture_host_inventory(kernel);
}

HostInventoryReceipt padded_inventory(
    std::size_t label_value_bytes,
    std::size_t labels_per_resource =
        HostResourceBounds::maximum_labels_per_resource) {
  HostKernelSnapshot snapshot{
      .api_version = std::string(kHostInventoryApiVersion),
      .host_id = "host-audit",
      .boot_id = "boot-audit",
      .broker_epoch = "epoch-audit",
      .begin_revision = "revision-padded",
      .end_revision = "revision-padded",
      .probes = {},
      .resources = {},
  };
  for (std::size_t resource_index = 0;
       resource_index < HostResourceBounds::maximum_resources;
       ++resource_index) {
    ObservedHostResource resource{
        .id = {.kind = HostResourceKind::host_mutex,
               .vendor = std::nullopt,
               .stable_id =
                   "host-mutex:padded-" + std::to_string(resource_index),
               .parent_id = std::nullopt},
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
        .labels = {},
    };
    for (std::size_t label_index = 0; label_index < labels_per_resource;
         ++label_index) {
      resource.labels.emplace("padding-label-" + std::to_string(label_index),
                              std::string(label_value_bytes, 'x'));
    }
    snapshot.resources.push_back(std::move(resource));
  }
  FakeHostKernel kernel(
      {{.snapshot = std::move(snapshot), .failure = std::nullopt}});
  return capture_host_inventory(kernel);
}

ResourceBundleRequest request(std::string request_id) {
  return seal_resource_request({
      .api_version = std::string(kHostResourceRequestApiVersion),
      .request_id = std::move(request_id),
      .journal_id = "journal-audit",
      .run_id = "run-audit",
      .logical_lease_id = "lease-audit",
      .logical_fencing_token = 1,
      .count = 1,
      .access_mode = ResourceAccessMode::mutex_exclusive,
      .topology = TopologyPolicy::any,
      .selector = {},
      .canonical_request_digest = {},
  });
}

ResourceReleaseRequest release_request(const ResourceBundleGrant& grant,
                                       std::string release_request_id) {
  return seal_resource_release_request({
      .api_version = std::string(kHostLedgerReleaseRequestApiVersion),
      .release_request_id = std::move(release_request_id),
      .allocation_id = grant.allocation_id,
      .grant_digest = grant.receipt_digest,
      .journal_id = grant.journal_id,
      .run_id = grant.run_id,
      .logical_lease_id = grant.logical_lease_id,
      .logical_fencing_token = grant.logical_fencing_token,
      .canonical_request_digest = {},
  });
}

HostStartupAuditPolicy trusted_policy() {
  return canonicalize_host_startup_audit_policy({
      .api_version = std::string(kHostStartupAuditPolicyApiVersion),
      .require_stable_occupancy = true,
      .fail_on_blocking_findings = true,
      .maximum_findings = 16U,
      .policy_digest = {},
  });
}

HostStartupAuditReport report_for(SQLiteHostLedger& ledger,
                                  std::string audit_id = "audit-001") {
  const auto observed = ledger.inventory();
  const auto before = ledger.chain_head();
  const auto occupancy = ledger.occupancy();
  return canonicalize_host_startup_audit_report({
      .api_version = std::string(kHostStartupAuditReportApiVersion),
      .audit_id = std::move(audit_id),
      .host_id = observed.host_id,
      .boot_id = observed.boot_id,
      .broker_epoch = observed.broker_epoch,
      .broker_instance_id = "broker-process-001",
      .inventory = observed,
      .pre_audit_occupancy = occupancy,
      .post_audit_occupancy = occupancy,
      .ledger_head_before = before,
      .ledger_head_after_observation = before,
      .policy = trusted_policy(),
      .findings = {},
      .disposition = HostStartupAuditDisposition::passed,
      .observed_begin_boottime_ns = 10,
      .observed_end_boottime_ns = 20,
      .findings_digest = {},
      .report_digest = {},
  });
}

void raw_execute(const std::filesystem::path& path, const std::string& sql) {
  sqlite3* database = nullptr;
  require(sqlite3_open(path.c_str(), &database) == SQLITE_OK,
          "raw SQLite opens");
  char* error = nullptr;
  const int status = sqlite3_exec(database, sql.c_str(), nullptr, nullptr, &error);
  const std::string detail = error == nullptr ? std::string{} : error;
  sqlite3_free(error);
  sqlite3_close(database);
  require(status == SQLITE_OK, "raw SQL succeeds: " + detail);
}

std::string raw_scalar(const std::filesystem::path& path,
                       const std::string& sql) {
  sqlite3* database = nullptr;
  require(sqlite3_open(path.c_str(), &database) == SQLITE_OK,
          "raw SQLite opens for scalar");
  sqlite3_stmt* statement = nullptr;
  require(sqlite3_prepare_v2(database, sql.c_str(), -1, &statement, nullptr) ==
              SQLITE_OK,
          "raw scalar prepares");
  require(sqlite3_step(statement) == SQLITE_ROW, "raw scalar has a row");
  const auto* bytes = sqlite3_column_text(statement, 0);
  const std::string result =
      bytes == nullptr ? std::string{} : reinterpret_cast<const char*>(bytes);
  sqlite3_finalize(statement);
  sqlite3_close(database);
  return result;
}

void downgrade_empty_v2_to_v1(const std::filesystem::path& path) {
  raw_execute(path, R"sql(
    DROP TRIGGER process_recovery_exits_no_update;
    DROP TRIGGER process_recovery_exits_no_delete;
    DROP TABLE process_recovery_exits;
    DROP TRIGGER process_exits_no_update;
    DROP TRIGGER process_exits_no_delete;
    DROP TABLE process_exits;
    DROP TRIGGER process_spawns_no_update;
    DROP TRIGGER process_spawns_no_delete;
    DROP TRIGGER process_launch_intents_no_update;
    DROP TRIGGER process_launch_intents_no_delete;
    DROP TRIGGER process_records_no_update;
    DROP TRIGGER process_records_no_delete;
    DROP TABLE process_spawns;
    DROP TABLE process_launch_intents;
    DROP TABLE process_records;
    DROP TABLE process_chain_head;
    DROP TRIGGER request_admission_epochs_no_update;
    DROP TRIGGER request_admission_epochs_no_delete;
    DROP TRIGGER request_admission_exemptions_no_update;
    DROP TRIGGER request_admission_exemptions_no_delete;
    DROP TRIGGER admission_epochs_no_update;
    DROP TRIGGER admission_epochs_no_delete;
    DROP TABLE request_admission_epochs;
    DROP TABLE active_admission_epoch;
    DROP TABLE admission_epochs;
    DROP TABLE request_admission_exemptions;
    DROP TRIGGER startup_audit_outcomes_no_update;
    DROP TRIGGER startup_audit_outcomes_no_delete;
    DROP TABLE startup_audit_outcomes;
    DROP TABLE ledger_schema_extensions;
    PRAGMA user_version=1;
  )sql");
}

class OneShotFault final : public IHostLedgerFaultInjector {
 public:
  explicit OneShotFault(HostLedgerFaultPoint target) : target_(target) {}

  void hit(HostLedgerFaultPoint point) override {
    if (armed_ && point == target_) {
      armed_ = false;
      throw HostLedgerError("injected startup-audit fault");
    }
  }

 private:
  HostLedgerFaultPoint target_;
  bool armed_{true};
};

class DataOnlyAuditor final : public IConfiguredHostStartupAuditorV2 {
 public:
  explicit DataOnlyAuditor(HostStartupAuditReport report)
      : report_(std::move(report)) {}

  HostStartupAuditReport audit() override {
    ++calls;
    return report_;
  }

  int calls{};

 private:
  HostStartupAuditReport report_;
};

void reflected_codecs_and_bounds() {
  const auto path = test_path("codecs");
  auto authority = authority_for(path);
  SQLiteHostLedger ledger(authority, inventory());
  const auto report = report_for(ledger);
  require_throws<HostLedgerError>(
      [&] { (void)ledger.commit_startup_audit(report, {30, 40}); },
      "ordinary ledger has no startup-audit commit authority");
  DataOnlyAuditor auditor(report);
  require(auditor.audit() == report && auditor.calls == 1,
          "auditor returns data-only typed evidence");
  const auto json = host_startup_audit_report_json(report);
  require(decode_untrusted_host_startup_audit_report(json) == report,
          "reflected v2 report codec round trips strictly");
  require(reflected_field_names<HostStartupAuditReport>().size() == 18U &&
              reflected_field_names<HostStartupAuditReceipt>().size() == 20U,
          "v2 report and receipt expose complete reflected schemas");

  auto unknown = json;
  unknown["future"] = true;
  require_throws<HostStartupAuditError>(
      [&] { (void)decode_untrusted_host_startup_audit_report(unknown); },
      "unknown report fields fail closed");
  auto oversized = report;
  require_throws<HostStartupAuditError>(
      [&] {
        oversized.findings.push_back(canonicalize_host_startup_audit_finding({
            .severity = HostStartupAuditFindingSeverity::warning,
            .code = "warning",
            .subject = "gpu",
            .detail = std::string(
                HostStartupAuditBounds::maximum_finding_detail_bytes + 1U,
                'x'),
            .evidence_digest = {},
        }));
      },
      "unbounded finding text is rejected");

  nlohmann::json too_deep = nullptr;
  for (std::size_t depth = 0;
       depth <= HostStartupAuditBounds::maximum_json_depth; ++depth) {
    too_deep = nlohmann::json::array({std::move(too_deep)});
  }
  require_throws<HostStartupAuditError>(
      [&] { (void)decode_untrusted_host_startup_audit_report(too_deep); },
      "structurally deep JSON fails before reflected decoding");

  nlohmann::json too_wide = nlohmann::json::array();
  for (std::size_t index = 0U;
       index <= HostStartupAuditBounds::maximum_json_container_width;
       ++index) {
    too_wide.push_back(nullptr);
  }
  require_throws<HostStartupAuditError>(
      [&] { (void)decode_untrusted_host_startup_audit_report(too_wide); },
      "wide JSON fails structural preflight before reflected decoding");

  nlohmann::json too_many_nodes = nlohmann::json::array();
  for (std::size_t row = 0U; row < 1'024U; ++row) {
    nlohmann::json values = nlohmann::json::array();
    for (std::size_t column = 0U; column < 64U; ++column) {
      values.push_back(nullptr);
    }
    too_many_nodes.push_back(std::move(values));
  }
  require_throws<HostStartupAuditError>(
      [&] {
        (void)decode_untrusted_host_startup_audit_report(too_many_nodes);
      },
      "excess JSON nodes fail structural preflight before reflection");

  auto invalid_policy = trusted_policy();
  invalid_policy.policy_digest = "sha256:" + std::string(64U, '0');
  const auto invalid_path = test_path("invalid-policy");
  require_throws<HostLedgerError>(
      [&] {
        SQLiteHostLedger rejected(authority_for(invalid_path), inventory(),
                                  nullptr, invalid_policy);
      },
      "ledger construction rejects noncanonical retained audit policy");
}

void near_limit_inventory_report_is_commit_readable() {
  const auto sizing_path = test_path("near-limit-sizing");
  SQLiteHostLedger sizing_ledger(authority_for(sizing_path), inventory());
  const auto base = report_for(sizing_ledger, "audit-near-limit");
  const auto candidate = [&](std::size_t value_bytes) {
    auto report = base;
    report.inventory = padded_inventory(value_bytes);
    report.pre_audit_occupancy = seal_resource_occupancy(
        report.inventory,
        {.api_version = std::string(kHostResourceOccupancyApiVersion),
         .host_id = report.inventory.host_id,
         .boot_id = report.inventory.boot_id,
         .inventory_digest = report.inventory.inventory_digest,
         .ledger_sequence = report.ledger_head_before.ledger_sequence,
         .active_fences = {},
         .occupancy_digest = {}});
    report.post_audit_occupancy = report.pre_audit_occupancy;
    return canonicalize_host_startup_audit_report(std::move(report));
  };

  std::size_t low = 0U;
  std::size_t high = HostResourceBounds::maximum_label_value_bytes + 1U;
  while (low + 1U < high) {
    const std::size_t middle = low + (high - low) / 2U;
    try {
      (void)candidate(middle);
      low = middle;
    } catch (const HostStartupAuditError&) {
      high = middle;
    }
  }
  require(high <= HostResourceBounds::maximum_label_value_bytes,
          "maximum inventory wrapper crosses the startup-audit byte bound");
  const auto sized_report = candidate(low);
  const auto sized_json = host_startup_audit_report_json(sized_report);
  const auto sized_bytes = sized_json.dump();
  require(sized_bytes.size() <= HostStartupAuditBounds::maximum_json_bytes &&
              HostStartupAuditBounds::maximum_json_bytes - sized_bytes.size() <
                  32U * 1024U &&
              decode_untrusted_host_startup_audit_report(sized_json) ==
                  sized_report,
          "largest accepted inventory wrapper is close to the canonical bound");
  require_throws<HostStartupAuditError>(
      [&] { (void)candidate(high); },
      "one larger inventory wrapper is rejected before persistence");

  const auto path = test_path("near-limit-commit");
  // The host-inventory codec additionally has an 8192-node bound. Eight
  // maximum-size labels per resource exercise a large persisted wrapper while
  // remaining valid for both the inventory and startup-audit codecs.
  auto large = padded_inventory(low, 8U);
  SQLiteHostLedger ledger(authority_for(path), large, nullptr,
                          trusted_policy());
  const auto report = report_for(ledger, "audit-near-limit-commit");
  const auto committed =
      ledger.commit_startup_audit(report, {30, 40});
  require(!committed.replayed && ledger.verify() &&
              decode_untrusted_host_startup_audit_report(
                  host_startup_audit_report_json(report)) == report,
          "near-limit nested report commits only when its persisted bytes are readable");
}

void commit_replay_cas_and_reopen() {
  const auto path = test_path("commit");
  auto authority = authority_for(path);
  HostStartupAuditReceipt receipt;
  {
    SQLiteHostLedger ledger(authority, inventory(), nullptr, trusted_policy());
    const auto report = report_for(ledger);
    const auto different_policy = canonicalize_host_startup_audit_policy({
        .api_version = std::string(kHostStartupAuditPolicyApiVersion),
        .require_stable_occupancy = true,
        .fail_on_blocking_findings = false,
        .maximum_findings = 16U,
        .policy_digest = {},
    });
    require_throws<HostLedgerConflict>(
        [&] {
          auto mismatched = report;
          mismatched.policy = different_policy;
          mismatched =
              canonicalize_host_startup_audit_report(std::move(mismatched));
          (void)ledger.commit_startup_audit(mismatched, {30, 40});
        },
        "self-canonicalized report cannot replace configured trusted policy");
    const std::uint64_t before_count = ledger.record_count();
    const auto committed =
        ledger.commit_startup_audit(report, {30, 40});
    require(!committed.replayed && ledger.verify() &&
                ledger.record_count() == before_count + 1U &&
                committed.receipt.report_digest == report.report_digest &&
                committed.receipt.committed_ledger_head == ledger.chain_head(),
            "ledger CAS-commits and re-reads the v2 startup audit");
    receipt = committed.receipt;
    const auto replayed =
        ledger.commit_startup_audit(report, {300, 400});
    require(replayed.replayed && replayed.receipt == committed.receipt &&
                ledger.record_count() == before_count + 1U,
            "exact audit replay is immutable and idempotent");

    auto changed = report;
    changed.broker_instance_id = "broker-process-changed";
    changed = canonicalize_host_startup_audit_report(std::move(changed));
    require_throws<HostLedgerConflict>(
        [&] {
          (void)ledger.commit_startup_audit(changed, {300, 400});
        },
        "audit ID reuse with different evidence conflicts");

    auto stale = report_for(ledger, "audit-stale");
    const auto intervening = report_for(ledger, "audit-intervening");
    (void)ledger.commit_startup_audit(intervening, {50, 60});
    require_throws<HostLedgerConflict>(
        [&] {
          (void)ledger.commit_startup_audit(stale, {70, 80});
        },
        "stale chain-head and occupancy evidence loses the audit CAS");
  }
  {
    SQLiteHostLedger reopened(authority, inventory());
    require(reopened.verify() &&
                receipt.committed_ledger_head.ledger_sequence <
                    reopened.chain_head().ledger_sequence,
            "strict replay verifies durable audit records after reopen");
  }
}

void transactional_faults_and_lost_reply() {
  for (const auto point : {HostLedgerFaultPoint::after_startup_audit_record,
                           HostLedgerFaultPoint::after_startup_audit_projection,
                           HostLedgerFaultPoint::before_commit}) {
    const auto path = test_path("rollback-" + std::to_string(
                                               static_cast<unsigned int>(point)));
    auto authority = authority_for(path);
    OneShotFault fault(point);
    SQLiteHostLedger ledger(authority, inventory(), &fault, trusted_policy());
    const auto report = report_for(ledger);
    const auto before = ledger.record_count();
    require_throws<HostLedgerError>(
        [&] {
          (void)ledger.commit_startup_audit(report, {30, 40});
        },
        "pre-commit audit fault is surfaced");
    require(ledger.verify() && ledger.record_count() == before,
            "pre-commit audit fault rolls back record and projection");
    const auto retry =
        ledger.commit_startup_audit(report, {31, 41});
    require(!retry.replayed && ledger.verify(),
            "rolled-back startup audit can be retried cleanly");
  }

  const auto path = test_path("lost-reply");
  auto authority = authority_for(path);
  OneShotFault fault(HostLedgerFaultPoint::after_startup_audit_commit);
  SQLiteHostLedger ledger(authority, inventory(), &fault, trusted_policy());
  const auto report = report_for(ledger);
  require_throws<HostLedgerError>(
      [&] {
        (void)ledger.commit_startup_audit(report, {30, 40});
      },
      "lost reply is injected after durable commit");
  const auto retry =
      ledger.commit_startup_audit(report, {300, 400});
  require(retry.replayed && ledger.verify(),
          "retry after a lost reply returns the one durable receipt");
}

void admission_epoch_seals_grants_and_replays_exactly() {
  const auto path = test_path("admission-epoch");
  auto authority = authority_for(path);
  SQLiteHostLedger ledger(authority, inventory(), nullptr, trusted_policy());
  require_throws<HostLedgerConflict>(
      [&] { (void)ledger.request_bundle(request("sealed-before-audit"),
                                        {10, 20}); },
      "policy ledger rejects grants before startup admission finalize");
  const auto report = report_for(ledger);
  const auto committed = ledger.commit_startup_audit(report, {30, 40});
  const auto finalized = ledger.finalize_startup_admission(
      report, committed.receipt, {31, 41});
  require(!finalized.replayed && ledger.verify(),
          "exact committed audit mints one opaque admission epoch");
  const auto granted = ledger.request_bundle(
      request("epoch-grant"), {50, 60}, finalized.epoch);
  require(granted.grant.has_value(),
          "active admission epoch authorizes an atomic grant");
  const auto grant_replay = ledger.request_bundle(
      request("epoch-grant"), {500, 600}, finalized.epoch);
  const auto finalize_replay = ledger.finalize_startup_admission(
      report, committed.receipt, {700, 800});
  require(grant_replay.replayed && grant_replay.grant == granted.grant &&
              finalize_replay.replayed && finalize_replay.epoch == finalized.epoch,
          "request and finalize lost replies replay their exact persisted epoch");

  const auto lost_path = test_path("admission-epoch-lost-reply");
  auto lost_authority = authority_for(lost_path);
  OneShotFault lost_fault(
      HostLedgerFaultPoint::after_admission_finalize_commit);
  SQLiteHostLedger lost(lost_authority, inventory(), &lost_fault,
                        trusted_policy());
  const auto lost_report = report_for(lost);
  const auto lost_commit = lost.commit_startup_audit(lost_report, {30, 40});
  require_throws<HostLedgerError>(
      [&] {
        (void)lost.finalize_startup_admission(lost_report,
                                              lost_commit.receipt, {31, 41});
      },
      "fault after admission finalize commit models a lost reply");
  const auto recovered = lost.finalize_startup_admission(
      lost_report, lost_commit.receipt, {50, 60});
  require(recovered.replayed && lost.verify(),
          "lost finalize reply recovers the exact durable epoch");
}

void prior_epoch_outcomes_reconcile_without_readmission() {
  const auto path = test_path("prior-epoch-reconciliation");
  auto authority = authority_for(path);
  const ResourceBundleRequest granted_request = request("prior-epoch-grant");
  const ResourceBundleRequest busy_request = request("prior-epoch-busy");
  std::optional<HostLedgerAdmissionEpoch> prior_epoch;
  BundleRequestResult granted;
  BundleRequestResult busy;
  {
    SQLiteHostLedger ledger(authority, inventory(), nullptr, trusted_policy());
    const auto report = report_for(ledger, "prior-epoch-audit");
    const auto committed = ledger.commit_startup_audit(report, {30, 40});
    prior_epoch = ledger
                      .finalize_startup_admission(report, committed.receipt,
                                                  {31, 41})
                      .epoch;
    granted = ledger.request_bundle(granted_request, {50, 60}, *prior_epoch);
    busy = ledger.request_bundle(busy_request, {51, 61}, *prior_epoch);
    require(granted.grant && busy.status == BundleRequestStatus::busy,
            "prior epoch commits exact grant and busy outcomes");
  }

  {
    SQLiteHostLedger restarted(authority, inventory(), nullptr,
                               trusted_policy());
    const auto records_before = restarted.record_count();
    const auto generation_before =
        restarted.generation(granted.grant->fences.front().resource);
    const auto occupancy_before = restarted.occupancy();
    const auto recovered_grant =
        restarted.reconcile_bundle_outcome(granted_request);
    const auto recovered_busy =
        restarted.reconcile_bundle_outcome(busy_request);
    require(recovered_grant && recovered_grant->replayed &&
                *recovered_grant == BundleRequestResult{
                                        .status = granted.status,
                                        .grant = granted.grant,
                                        .outcome_digest = granted.outcome_digest,
                                        .replayed = true} &&
                recovered_busy && recovered_busy->replayed &&
                recovered_busy->status == BundleRequestStatus::busy &&
                recovered_busy->outcome_digest == busy.outcome_digest,
            "restart recovers exact prior-epoch grant and busy outcomes");

    auto mismatched = granted_request;
    mismatched.run_id = "different-run";
    mismatched = seal_resource_request(std::move(mismatched));
    require_throws<HostLedgerConflict>(
        [&] { (void)restarted.reconcile_bundle_outcome(mismatched); },
        "same request ID with changed attribution/digest cannot reconcile");
    const auto absent =
        restarted.reconcile_bundle_outcome(request("never-admitted-request"));
    require(!absent && restarted.record_count() == records_before &&
                restarted.generation(
                    granted.grant->fences.front().resource) ==
                    generation_before &&
                restarted.occupancy() == occupancy_before,
            "missing reconciliation is read-only and cannot create a grant");
  }

  SQLiteHostLedger next_runtime(authority, inventory("epoch-audit-next"),
                                nullptr, trusted_policy());
  const auto next_report =
      report_for(next_runtime, "next-admission-audit");
  const auto next_commit =
      next_runtime.commit_startup_audit(next_report, {70, 80});
  const auto next_epoch = next_runtime.finalize_startup_admission(
      next_report, next_commit.receipt, {71, 81});
  require_throws<HostLedgerConflict>(
      [&] {
        (void)next_runtime.request_bundle(granted_request, {90, 100},
                                          *prior_epoch);
      },
      "stale prior epoch cannot replay through the mutating admission path");
  const auto after_new_epoch =
      next_runtime.reconcile_bundle_outcome(granted_request);
  require(after_new_epoch && after_new_epoch->grant == granted.grant &&
              after_new_epoch->outcome_digest == granted.outcome_digest,
          "new admission epoch does not hide immutable prior outcomes");
  const auto no_new_grant =
      next_runtime.reconcile_bundle_outcome(request("new-epoch-absent"));
  require(!no_new_grant && next_epoch.epoch != *prior_epoch,
          "reconciliation never borrows the new epoch to admit missing work");
}

void deleted_admission_authorization_fails_verify_and_reopen() {
  const auto path = test_path("deleted-admission-authorization");
  auto authority = authority_for(path);
  {
    SQLiteHostLedger ledger(authority, inventory(), nullptr, trusted_policy());
    const auto report = report_for(ledger, "audit-before-auth-deletion");
    const auto committed = ledger.commit_startup_audit(report, {30, 40});
    const auto finalized = ledger.finalize_startup_admission(
        report, committed.receipt, {31, 41});
    require(ledger
                .request_bundle(request("post-v3-authorized-request"),
                                {50, 60}, finalized.epoch)
                .grant.has_value() &&
                ledger.verify(),
            "post-v3 request begins with one valid epoch authorization");
    const std::string delete_trigger = raw_scalar(
        path,
        "SELECT sql FROM sqlite_master WHERE type='trigger' "
        "AND name='request_admission_epochs_no_delete'");
    raw_execute(path,
                "DROP TRIGGER request_admission_epochs_no_delete;"
                "DELETE FROM request_admission_epochs "
                "WHERE request_id='post-v3-authorized-request';" +
                    delete_trigger + ";");
    std::string reason;
    require(!ledger.verify(&reason) &&
                reason.find(
                    "missing admission authorization or durable exemption") !=
                    std::string::npos,
            "deleted post-v3 authorization is not reclassified as legacy");
    require_throws<HostLedgerError>(
        [&] {
          (void)ledger.reconcile_bundle_outcome(
              request("post-v3-authorized-request"));
        },
        "reconciliation refuses an outcome missing epoch authorization");
  }
  require_throws<HostLedgerError>(
      [&] {
        SQLiteHostLedger reopened(authority, inventory(), nullptr,
                                  trusted_policy());
      },
      "ledger with omitted post-v3 authorization fails closed on reopen");
}

void missing_outcome_projection_poison_reconciliation() {
  const auto path = test_path("missing-recovery-projection");
  auto authority = authority_for(path);
  SQLiteHostLedger ledger(authority, inventory(), nullptr, trusted_policy());
  const auto report = report_for(ledger, "projection-recovery-audit");
  const auto committed = ledger.commit_startup_audit(report, {30, 40});
  const auto epoch = ledger.finalize_startup_admission(
      report, committed.receipt, {31, 41});
  const auto exact = request("projection-recovery-request");
  require(ledger.request_bundle(exact, {50, 60}, epoch.epoch).grant.has_value(),
          "projection corruption fixture commits a grant");
  raw_execute(path,
              "DELETE FROM request_outcomes "
              "WHERE request_id='projection-recovery-request';");
  std::string reason;
  require(!ledger.verify(&reason),
          "missing request outcome projection breaks closure");
  require_throws<HostLedgerError>(
      [&] { (void)ledger.reconcile_bundle_outcome(exact); },
      "reconciliation poisons instead of treating a missing projection as absent");
}

void additive_migration_preserves_v1_evidence() {
  const auto path = test_path("migration");
  auto authority = authority_for(path);
  ResourceBundleRequest released_request = request("migration-released");
  ResourceReleaseRequest released_release;
  ResourceBundleRequest active_request = request("migration-active");
  ResourceBundleGrant active_grant;
  std::uint64_t v1_record_count = 0U;
  {
    SQLiteHostLedger initialized(authority, inventory());
    const auto released =
        initialized.request_bundle(released_request, {10, 20});
    require(released.grant.has_value(),
            "migration fixture creates released grant history");
    released_release = release_request(*released.grant, "migration-release");
    (void)initialized.release_bundle(released_release, {30, 40});
    const auto active = initialized.request_bundle(active_request, {50, 60});
    require(active.grant.has_value(),
            "migration fixture creates active grant history");
    active_grant = *active.grant;
    v1_record_count = initialized.record_count();
  }
  downgrade_empty_v2_to_v1(path);
  const std::string evidence_before = raw_scalar(
      path, "SELECT group_concat(canonical_json, char(10)) FROM "
            "ledger_records ORDER BY ledger_sequence");
  {
    SQLiteHostLedger migrated(authority, inventory(), nullptr,
                              trusted_policy());
    const auto occupancy = migrated.occupancy();
    require(migrated.verify() && migrated.record_count() == v1_record_count &&
                occupancy.active_fences == active_grant.fences &&
                migrated.release_bundle(released_release, {300, 400}).replayed,
            "v1 migration preserves active and released grant authority history");
    require_throws<HostLedgerConflict>(
        [&] { (void)migrated.request_bundle(active_request, {500, 600}); },
        "migrated policy ledger seals legacy request replay before finalize");
    const auto audit = report_for(migrated, "audit-after-v1-migration");
    const auto committed =
        migrated.commit_startup_audit(audit, {700, 800});
    const auto finalized = migrated.finalize_startup_admission(
        audit, committed.receipt, {701, 801});
    require_throws<HostLedgerConflict>(
        [&] {
          (void)migrated.request_bundle(active_request, {900, 1000},
                                        finalized.epoch);
        },
        "pre-migration request IDs cannot be rebound to a new epoch");
    require(raw_scalar(path, "SELECT COUNT(*) FROM request_admission_exemptions") ==
                "2" &&
                raw_scalar(
                    path,
                    "SELECT COUNT(*) FROM request_admission_exemptions AS exemption "
                    "JOIN request_outcomes AS outcome USING(request_id) "
                    "WHERE exemption.request_digest=outcome.request_digest "
                    "AND exemption.reason='pre_v3'") ==
                    "2",
            "v3 migration freezes the exact two pre-v3 outcomes as legacy");
    (void)migrated.release_bundle(
        release_request(active_grant, "migration-active-release"),
        {1100, 1200});
    require(migrated.verify(),
            "audit replay reconstructs the migrated sequence-1 occupancy evidence");
  }
  const std::string evidence_after = raw_scalar(
      path, "SELECT group_concat(canonical_json, char(10)) FROM "
            "ledger_records WHERE ledger_sequence<=" +
                std::to_string(v1_record_count) + " ORDER BY ledger_sequence");
  require(evidence_after == evidence_before &&
              raw_scalar(path, "PRAGMA user_version") == "6",
          "migration preserves every v1 evidence byte and marks v6");

  const auto rollback_path = test_path("migration-rollback");
  auto rollback_authority = authority_for(rollback_path);
  {
    SQLiteHostLedger initialized(rollback_authority, inventory());
  }
  downgrade_empty_v2_to_v1(rollback_path);
  OneShotFault migration_fault(
      HostLedgerFaultPoint::after_startup_audit_migration_schema);
  require_throws<HostLedgerError>(
      [&] {
        SQLiteHostLedger rejected(rollback_authority, inventory(),
                                  &migration_fault);
      },
      "migration fault fails closed");
  require(raw_scalar(rollback_path, "PRAGMA user_version") == "1" &&
              raw_scalar(rollback_path,
                         "SELECT COUNT(*) FROM sqlite_master WHERE "
                         "name='startup_audit_outcomes'") == "0",
          "failed migration rolls its schema changes back exactly");
  SQLiteHostLedger recovered(rollback_authority, inventory());
  require(recovered.verify(), "clean retry completes migration");
}

void projection_corruption_fails_closed() {
  const auto path = test_path("corruption");
  auto authority = authority_for(path);
  SQLiteHostLedger ledger(authority, inventory(), nullptr, trusted_policy());
  const auto report = report_for(ledger);
  (void)ledger.commit_startup_audit(report, {30, 40});
  raw_execute(path, R"sql(
    DROP TRIGGER startup_audit_outcomes_no_update;
    UPDATE startup_audit_outcomes
      SET canonical_report_json='{}';
    CREATE TRIGGER startup_audit_outcomes_no_update
    BEFORE UPDATE ON startup_audit_outcomes
    BEGIN SELECT RAISE(ABORT, 'startup audit outcomes are immutable'); END;
  )sql");
  std::string reason;
  require(!ledger.verify(&reason),
          "corrupt startup-audit projection fails strict verification");
  require_throws<HostLedgerError>(
      [&] { (void)ledger.chain_head(); },
      "corrupt audit evidence blocks later authoritative reads");
}

void predecessor_projection_corruption_fails_closed() {
  const auto path = test_path("predecessor-corruption");
  auto authority = authority_for(path);
  SQLiteHostLedger ledger(authority, inventory(), nullptr, trusted_policy());
  const auto report = report_for(ledger);
  (void)ledger.commit_startup_audit(report, {30, 40});
  raw_execute(path, R"sql(
    DROP TRIGGER startup_audit_outcomes_no_update;
    UPDATE startup_audit_outcomes
      SET record_previous_hash='sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
    CREATE TRIGGER startup_audit_outcomes_no_update
    BEFORE UPDATE ON startup_audit_outcomes
    BEGIN SELECT RAISE(ABORT, 'startup audit outcomes are immutable'); END;
  )sql");
  std::string reason;
  require(!ledger.verify(&reason),
          "projection cannot rewrite an audit record's actual predecessor");
}

}  // namespace

int main() {
  DirectoryCleanup cleanup;
  try {
    reflected_codecs_and_bounds();
    near_limit_inventory_report_is_commit_readable();
    commit_replay_cas_and_reopen();
    transactional_faults_and_lost_reply();
    admission_epoch_seals_grants_and_replays_exactly();
    prior_epoch_outcomes_reconcile_without_readmission();
    deleted_admission_authorization_fails_verify_and_reopen();
    missing_outcome_projection_poison_reconciliation();
    additive_migration_preserves_v1_evidence();
    projection_corruption_fails_closed();
    predecessor_projection_corruption_fails_closed();
    std::cout << "host startup audit tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "host startup audit test failure: " << error.what() << '\n';
    return 1;
  }
}
