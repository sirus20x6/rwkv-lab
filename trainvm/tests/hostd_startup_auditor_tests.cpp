#include "trainvm/hostd_startup_auditor.hpp"

#include "trainvm/hostd.hpp"
#include "trainvm/hostd_restart_process_recovery.hpp"
#include "trainvm/hostd_startup_controller.hpp"

#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

namespace {

using namespace trainvm;

void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

template <typename Exception, typename Callable>
void require_throws(Callable&& callable, std::string_view message) {
  try {
    std::forward<Callable>(callable)();
  } catch (const Exception&) {
    return;
  }
  throw std::runtime_error(std::string(message));
}

struct TemporaryDirectory final {
  std::filesystem::path path;

  TemporaryDirectory() {
    std::string pattern =
        (std::filesystem::temp_directory_path() /
         "trainvm-configured-auditor-XXXXXX")
            .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char* created = ::mkdtemp(writable.data());
    if (created == nullptr) throw std::runtime_error("mkdtemp failed");
    path = created;
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};

std::shared_ptr<HostLedgerFilesystemAuthority> authority(
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

HostInventoryReceipt inventory() {
  HostKernelSnapshot snapshot{
      .api_version = std::string(kHostInventoryApiVersion),
      .host_id = "host-configured-auditor",
      .boot_id = "12345678-1234-1234-1234-123456789abc",
      .broker_epoch = "broker-epoch-1",
      .begin_revision = "inventory-1",
      .end_revision = "inventory-1",
      .probes = {},
      .resources = {{
          .id = {.kind = HostResourceKind::host_mutex,
                 .vendor = std::nullopt,
                 .stable_id = "host-mutex:configured-auditor",
                 .parent_id = std::nullopt},
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
          .total_memory_bytes = 0U,
          .labels = {},
      }},
  };
  FakeHostKernel kernel({{.snapshot = std::move(snapshot), .failure = std::nullopt}});
  return capture_host_inventory(kernel);
}

HostStartupAuditPolicy policy() {
  return canonicalize_host_startup_audit_policy({
      .api_version = std::string(kHostStartupAuditPolicyApiVersion),
      .require_stable_occupancy = true,
      .fail_on_blocking_findings = true,
      .maximum_findings = 16U,
      .policy_digest = {},
  });
}

AuthorityClock clock(std::int64_t begin = 10) {
  auto next = std::make_shared<std::int64_t>(begin);
  return AuthorityClock([next] {
    const std::int64_t value = (*next)++;
    return AuthorityTimeSample{
        .wall = {.nanoseconds = value + 1'000},
        .boot = {.nanoseconds = value},
        .boot_id = "12345678-1234-1234-1234-123456789abc",
    };
  });
}

HostdConfiguredStartupAuditorConfig config() {
  return {
      .api_version = std::string(kHostdConfiguredStartupAuditorApiVersion),
      .broker_instance_id = "hostd-process-1",
      .policy = policy(),
  };
}

ResourceBundleRequest request() {
  return seal_resource_request({
      .api_version = std::string(kHostResourceRequestApiVersion),
      .request_id = "request-1",
      .journal_id = "journal-1",
      .run_id = "run-1",
      .logical_lease_id = "lease-1",
      .logical_fencing_token = 1U,
      .count = 1U,
      .access_mode = ResourceAccessMode::mutex_exclusive,
      .topology = TopologyPolicy::any,
      .selector = {},
      .canonical_request_digest = {},
  });
}

// Nothing durable survives, so recovery converges on its first step and the
// controller advances straight to the admission audit.
class ConvergedRecovery final : public IHostdRestartProcessRecovery {
 public:
  std::size_t steps{};
  HostdRestartProcessRecoverySummary step() override {
    ++steps;
    return {};
  }
};

HostdCoordinatorConfig coordinator_config() {
  const HostInventoryReceipt observed = inventory();
  return {.api_version = std::string(kHostdCoordinatorApiVersion),
          .host_id = observed.host_id,
          .boot_id = observed.boot_id,
          .broker_epoch = observed.broker_epoch,
          .maximum_live_sessions = 8U,
          .maximum_logical_scopes = 8U};
}

// Regression: the controller used to sample the startup-audit commit time
// before handing control to the admission authority, while the production
// auditor stamps its end of observation from a later sample. The ledger
// refuses a commit time older than the report it commits, so the real auditor
// could never be admitted. Every other test in this area drives a fixed-time
// fake auditor and cannot observe the ordering.
void production_auditor_reaches_admission_through_the_controller() {
  TemporaryDirectory temporary;
  auto ledger = std::make_shared<SQLiteHostLedger>(
      authority(temporary.path / "host.db"), inventory(), nullptr, policy());
  AuthorityClock authority_clock = clock();
  HostdConfiguredStartupAuditor auditor(*ledger, authority_clock, config());
  ConvergedRecovery recovery;
  HostGrantCoordinator coordinator(coordinator_config(), ledger);
  HostdCoordinatorStartupAdmission admission(coordinator);
  HostdStartupController controller(recovery, admission, auditor,
                                    authority_clock, {});

  const HostdStartupControllerStatus status = controller.advance();
  require(status.phase == HostdStartupPhase::admitting,
          "the production startup auditor must reach admission");
  require(recovery.steps == 1U && status.recovery_steps == 1U,
          "a converged recovery admits after exactly one bounded step");
  require(status.admission_receipt.has_value() &&
              !status.admission_receipt->audit_id.empty(),
          "admission returns an exact committed audit receipt");
  require(coordinator.status().lifecycle == HostdLifecycle::admitting,
          "the coordinator lifecycle reaches admitting");
  require(controller.advance().phase == HostdStartupPhase::admitting &&
              recovery.steps == 1U,
          "an already admitted controller neither re-audits nor re-recovers");
}

// The commit time is the coordinator's to sample, and it must be taken after
// the observation it commits.
void admission_commit_time_is_sampled_after_the_audit() {
  TemporaryDirectory temporary;
  auto ledger = std::make_shared<SQLiteHostLedger>(
      authority(temporary.path / "host.db"), inventory(), nullptr, policy());
  AuthorityClock authority_clock = clock();
  HostdConfiguredStartupAuditor auditor(*ledger, authority_clock, config());
  HostGrantCoordinator coordinator(coordinator_config(), ledger);
  AuthorityClockStartupAuditCommitTime commit_time(authority_clock);
  const HostStartupAuditReceipt receipt =
      coordinator.run_startup_audit(auditor, commit_time);
  require(!receipt.audit_id.empty(),
          "the sampling overload commits an exact audit receipt");

  TemporaryDirectory stale_directory;
  auto stale_ledger = std::make_shared<SQLiteHostLedger>(
      authority(stale_directory.path / "host.db"), inventory(), nullptr,
      policy());
  AuthorityClock stale_clock = clock();
  HostdConfiguredStartupAuditor stale_auditor(*stale_ledger, stale_clock,
                                              config());
  HostGrantCoordinator stale_coordinator(coordinator_config(), stale_ledger);
  require_throws<HostdStateError>(
      [&] {
        // A time sampled before the audit runs is exactly the defect this
        // interface removes; the fixed-time overload must still refuse it.
        (void)stale_coordinator.run_startup_audit(stale_auditor,
                                                  HostLedgerTime{0, 0});
      },
      "a commit time older than the report it commits must be refused");
}

void clean_ledger_produces_exact_passing_report() {
  TemporaryDirectory temporary;
  SQLiteHostLedger ledger(authority(temporary.path / "host.db"), inventory(),
                          nullptr, policy());
  AuthorityClock time = clock();
  HostdConfiguredStartupAuditor auditor(ledger, time, config());
  const HostStartupAuditReport report = auditor.audit();
  validate_host_startup_audit_report(report);
  require(report.disposition == HostStartupAuditDisposition::passed,
          "clean startup passes");
  require(report.findings.empty(), "clean startup has no findings");
  require(auditor.process_recovery().initialized() &&
              auditor.process_recovery().summary().records == 0U &&
              auditor.terminal_process_releases().empty(),
          "clean startup freezes an empty one-shot recovery set");
  require(report.pre_audit_occupancy == report.post_audit_occupancy &&
              report.ledger_head_before ==
                  report.ledger_head_after_observation,
          "report binds stable ledger evidence");
  require(report.observed_begin_boottime_ns == 10 &&
              report.observed_end_boottime_ns == 11,
          "report uses authority boottime");
}

void retained_fence_blocks_until_process_adoption_exists() {
  TemporaryDirectory temporary;
  const auto path = temporary.path / "host.db";
  {
    SQLiteHostLedger legacy(authority(path), inventory());
    require(legacy.request_bundle(request(), {.boottime_ns = 1,
                                               .wall_time_ns = 1})
                    .status == BundleRequestStatus::granted,
            "fixture creates active fence");
  }
  SQLiteHostLedger ledger(authority(path), inventory(), nullptr, policy());
  AuthorityClock time = clock(20);
  HostdConfiguredStartupAuditor auditor(ledger, time, config());
  const HostStartupAuditReport report = auditor.audit();
  require(report.disposition == HostStartupAuditDisposition::failed,
          "unadopted fence blocks startup");
  require(report.findings.size() == 1U &&
              report.findings.front().code == "process-adoption-required" &&
              report.findings.front().severity ==
                  HostStartupAuditFindingSeverity::blocking &&
              report.findings.front().detail.contains("intent_only=0"),
          "blocking finding identifies adoption gap");
  require(auditor.process_recovery().initialized() &&
              auditor.process_recovery().summary().records == 0U &&
              auditor.terminal_process_releases().empty(),
          "active grant without launch evidence does not invent a process");
}

void invalid_configuration_fails_closed() {
  TemporaryDirectory temporary;
  SQLiteHostLedger ledger(authority(temporary.path / "host.db"), inventory(),
                          nullptr, policy());
  AuthorityClock time = clock();
  auto invalid = config();
  invalid.policy.policy_digest.clear();
  require_throws<HostStartupAuditError>(
      [&] { HostdConfiguredStartupAuditor auditor(ledger, time, invalid); },
      "noncanonical policy rejected");
  invalid = config();
  invalid.broker_instance_id = "contains a newline\n";
  require_throws<HostStartupAuditError>(
      [&] { HostdConfiguredStartupAuditor auditor(ledger, time, invalid); },
      "invalid instance identity rejected");
}

}  // namespace

int main() {
  try {
    clean_ledger_produces_exact_passing_report();
    production_auditor_reaches_admission_through_the_controller();
    admission_commit_time_is_sampled_after_the_audit();
    retained_fence_blocks_until_process_adoption_exists();
    invalid_configuration_fails_closed();
    std::cout << "configured hostd startup auditor tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "configured hostd startup auditor test failure: " << error.what()
              << '\n';
    return 1;
  }
}
