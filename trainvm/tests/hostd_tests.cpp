#include "trainvm/hostd.hpp"

#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace {

using namespace trainvm;

constexpr std::string_view kDigest =
    "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <typename Exception, typename Callable>
void require_throws(Callable &&callable, const std::string &message) {
  try {
    std::forward<Callable>(callable)();
  } catch (const Exception &) {
    return;
  }
  throw std::runtime_error(message);
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    std::string pattern = "/tmp/trainvm-hostd-XXXXXX";
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const char *created = ::mkdtemp(writable.data());
    if (created == nullptr)
      throw std::runtime_error("mkdtemp failed");
    path_ = created;
    require(::chmod(path_.c_str(), 0700) == 0,
            "hostd fixture directory is protected");
  }

  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  TemporaryDirectory(const TemporaryDirectory &) = delete;
  TemporaryDirectory &operator=(const TemporaryDirectory &) = delete;

  [[nodiscard]] const std::filesystem::path &path() const { return path_; }

private:
  std::filesystem::path path_;
};

HostResourceId mutex_id(std::string id) {
  if (!id.starts_with("host-mutex:"))
    id = "host-mutex:" + id;
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
          .labels = {{"scope", "hostd-test"}}};
}

HostInventoryReceipt inventory(std::vector<std::string> resources) {
  HostKernelSnapshot snapshot{
      .api_version = std::string(kHostInventoryApiVersion),
      .host_id = "host-001",
      .boot_id = "boot-001",
      .broker_epoch = "broker-001",
      .begin_revision = "revision-001",
      .end_revision = "revision-001",
      .probes = {},
      .resources = {},
  };
  for (std::string &id : resources) {
    snapshot.resources.push_back(mutex_resource(std::move(id)));
  }
  FakeHostKernel kernel(
      {{.snapshot = std::move(snapshot), .failure = std::nullopt}});
  return capture_host_inventory(kernel);
}

HostdSessionAttribution attribution(std::string journal = "journal-001",
                                    std::string run = "run-001",
                                    std::string lease = "lease-001",
                                    std::uint64_t fence = 7U,
                                    std::string concurrency = "gpu:0") {
  return {.journal_id = std::move(journal),
          .run_id = std::move(run),
          .concurrency_key = std::move(concurrency),
          .logical_lease_id = std::move(lease),
          .logical_fencing_token = fence};
}

ResourceBundleRequest request_for(const HostdSessionAttribution &scope,
                                  std::string id) {
  return seal_resource_request({
      .api_version = std::string(kHostResourceRequestApiVersion),
      .request_id = std::move(id),
      .journal_id = scope.journal_id,
      .run_id = scope.run_id,
      .logical_lease_id = scope.logical_lease_id,
      .logical_fencing_token = scope.logical_fencing_token,
      .count = 1U,
      .access_mode = ResourceAccessMode::mutex_exclusive,
      .topology = TopologyPolicy::any,
      .selector = {},
      .canonical_request_digest = {},
  });
}

ResourceReleaseRequest release_for(const ResourceBundleGrant &grant,
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

class FakePeer final : public IHostdPeerEvidenceSource {
public:
  explicit FakePeer(HostdSessionAccess access,
                    HostdPeerEnforcementGrade grade =
                        HostdPeerEnforcementGrade::service_identity_enforced)
      : evidence_{.api_version = std::string(kHostdPeerEvidenceApiVersion),
                  .peer_uid = ::geteuid(),
                  .peer_gid = ::getegid(),
                  .peer_pid = ::getpid(),
                  .service_identity = "test.hostd-client.service",
                  .enforcement_grade = grade,
                  .access = access,
                  .evidence_digest = std::string(kDigest)} {}

  HostdPeerEvidence observe() override {
    if (!alive)
      throw std::runtime_error("fake peer disconnected");
    ++observations;
    if (during_observe)
      during_observe();
    return evidence_;
  }

  HostdPeerEvidence evidence_;
  bool alive{true};
  std::size_t observations{};
  std::function<void()> during_observe;
};

class FakeLogicalFenceAuthority final
    : public IHostdLogicalFenceEvidenceSource {
public:
  void set_live(const HostdSessionAttribution &scope) {
    current_[key(scope)] = scope.logical_fencing_token;
  }

  void authorize_cleanup(const HostdSessionAttribution &scope,
                         const ResourceBundleGrant &grant,
                         const ResourceReleaseRequest &release) {
    cleanup_[cleanup_key(scope)] = {
        .allocation_id = grant.allocation_id,
        .grant_digest = grant.receipt_digest,
        .release_request_digest = release.canonical_request_digest,
    };
  }

  HostdLogicalFenceEvidence
  attest(const HostdSessionAttribution &scope) override {
    ++attestations;
    if (during_attest)
      during_attest();
    const auto found = current_.find(key(scope));
    const auto cleanup = cleanup_.find(cleanup_key(scope));
    return {
        .api_version = std::string(kHostdLogicalFenceEvidenceApiVersion),
        .attribution = scope,
        .live = found != current_.end() &&
                found->second == scope.logical_fencing_token,
        .cleanup_authorized = cleanup != cleanup_.end(),
        .cleanup_allocation_id = cleanup == cleanup_.end()
                                     ? std::string{}
                                     : cleanup->second.allocation_id,
        .cleanup_grant_digest = cleanup == cleanup_.end()
                                    ? std::string{}
                                    : cleanup->second.grant_digest,
        .cleanup_release_request_digest =
            cleanup == cleanup_.end()
                ? std::string{}
                : cleanup->second.release_request_digest,
        .evidence_digest = std::string(kDigest),
    };
  }

  std::size_t attestations{};
  std::function<void()> during_attest;

private:
  static std::string key(const HostdSessionAttribution &scope) {
    return scope.journal_id + "\n" + scope.run_id + "\n" +
           scope.concurrency_key + "\n" + scope.logical_lease_id;
  }

  static std::string cleanup_key(const HostdSessionAttribution &scope) {
    return key(scope) + "\n" +
           std::to_string(scope.logical_fencing_token);
  }

  struct CleanupEvidence final {
    std::string allocation_id;
    std::string grant_digest;
    std::string release_request_digest;
  };

  std::map<std::string, std::uint64_t> current_;
  std::map<std::string, CleanupEvidence> cleanup_;
};

class AdversarialLedgerBoundary final : public IHostdLedgerBoundary {
public:
  enum class GrantFault {
    none,
    api_version,
    request_id,
    request_digest,
    journal_id,
    logical_fence,
    fence,
    receipt_digest,
  };
  enum class ReleaseFault {
    none,
    api_version,
    release_request_id,
    release_request_digest,
    allocation_id,
    grant_digest,
    receipt_digest,
  };

  explicit AdversarialLedgerBoundary(std::shared_ptr<SQLiteHostLedger> ledger)
      : ledger_(std::move(ledger)) {}

  bool verify() const override {
    if (during_verify)
      during_verify();
    return ledger_->verify();
  }

  HostInventoryReceipt inventory() const override {
    if (during_inventory)
      during_inventory();
    return ledger_->inventory();
  }

  HostLedgerChainHead chain_head() const override {
    return ledger_->chain_head();
  }

  ResourceOccupancySnapshot occupancy() const override {
    return ledger_->occupancy();
  }

  HostStartupAuditLedgerCommitResult
  commit_startup_audit(const HostStartupAuditReport &report,
                       const HostLedgerTime &now) override {
    if (during_audit_commit)
      during_audit_commit();
    return ledger_->commit_startup_audit(report, now);
  }

  HostLedgerAdmissionFinalizeResult
  finalize_startup_admission(const HostStartupAuditReport &report,
                             const HostStartupAuditReceipt &receipt,
                             const HostLedgerTime &now) override {
    if (before_admission_finalize)
      before_admission_finalize();
    auto result = ledger_->finalize_startup_admission(report, receipt, now);
    if (after_admission_finalize)
      after_admission_finalize(result);
    return result;
  }

  BundleRequestResult request_bundle(const ResourceBundleRequest &request,
                                     const HostLedgerTime &now,
                                     const HostLedgerAdmissionEpoch
                                         &admission_epoch) override {
    {
      std::unique_lock lock(block_mutex);
      request_entered = true;
      block_condition.notify_all();
      block_condition.wait(lock, [&] { return !block_request; });
    }
    BundleRequestResult result =
        ledger_->request_bundle(request, now, admission_epoch);
    if (!result.grant)
      return result;
    switch (grant_fault) {
    case GrantFault::none:
      break;
    case GrantFault::api_version:
      result.grant->api_version = "trainvm.host-resource-grant/v999";
      break;
    case GrantFault::request_id:
      result.grant->request_id = "wrong-request";
      break;
    case GrantFault::request_digest:
      result.grant->request_digest = std::string(kDigest);
      break;
    case GrantFault::journal_id:
      result.grant->journal_id = "wrong-journal";
      break;
    case GrantFault::logical_fence:
      ++result.grant->logical_fencing_token;
      break;
    case GrantFault::fence:
      require(!result.grant->fences.empty(), "grant fault needs a fence");
      ++result.grant->fences.front().generation;
      break;
    case GrantFault::receipt_digest:
      result.grant->receipt_digest = std::string(kDigest);
      break;
    }
    return result;
  }

  std::optional<BundleRequestResult>
  reconcile_bundle_outcome(
      const ResourceBundleRequest &request) const override {
    return ledger_->reconcile_bundle_outcome(request);
  }

  BundleReleaseResult
  release_bundle(const ResourceReleaseRequest &request,
                 const HostLedgerTime &now) override {
    BundleReleaseResult result = ledger_->release_bundle(request, now);
    switch (release_fault) {
    case ReleaseFault::none:
      break;
    case ReleaseFault::api_version:
      result.receipt.api_version = "trainvm.host-resource-release/v999";
      break;
    case ReleaseFault::release_request_id:
      result.receipt.release_request_id = "wrong-release";
      break;
    case ReleaseFault::release_request_digest:
      result.receipt.release_request_digest = std::string(kDigest);
      break;
    case ReleaseFault::allocation_id:
      result.receipt.allocation_id = "wrong-allocation";
      break;
    case ReleaseFault::grant_digest:
      result.receipt.grant_digest = std::string(kDigest);
      break;
    case ReleaseFault::receipt_digest:
      result.receipt.receipt_digest = std::string(kDigest);
      break;
    }
    return result;
  }

  void wait_for_request() {
    std::unique_lock lock(block_mutex);
    block_condition.wait(lock, [&] { return request_entered; });
  }

  void unblock_request() {
    std::scoped_lock lock(block_mutex);
    block_request = false;
    block_condition.notify_all();
  }

  mutable std::function<void()> during_verify;
  mutable std::function<void()> during_inventory;
  std::function<void()> during_audit_commit;
  std::function<void()> before_admission_finalize;
  std::function<void(const HostLedgerAdmissionFinalizeResult &)>
      after_admission_finalize;
  GrantFault grant_fault{GrantFault::none};
  ReleaseFault release_fault{ReleaseFault::none};
  bool block_request{};

private:
  std::shared_ptr<SQLiteHostLedger> ledger_;
  std::mutex block_mutex;
  std::condition_variable block_condition;
  bool request_entered{};
};

HostStartupAuditPolicy startup_audit_policy() {
  return canonicalize_host_startup_audit_policy({
      .api_version = std::string(kHostStartupAuditPolicyApiVersion),
      .require_stable_occupancy = true,
      .fail_on_blocking_findings = true,
      .maximum_findings = 16U,
      .policy_digest = {},
  });
}

HostStartupAuditReport audit_report(
    SQLiteHostLedger &ledger,
    HostStartupAuditDisposition disposition =
        HostStartupAuditDisposition::passed) {
  static std::atomic<std::uint64_t> next_audit_id{1U};
  const auto observed = ledger.inventory();
  const auto head = ledger.chain_head();
  const auto occupancy = ledger.occupancy();
  std::vector<HostStartupAuditFinding> findings;
  if (disposition == HostStartupAuditDisposition::failed) {
    findings.push_back(canonicalize_host_startup_audit_finding({
        .severity = HostStartupAuditFindingSeverity::blocking,
        .code = "hostd.test.blocking",
        .subject = "startup",
        .detail = "deterministic blocking startup evidence",
        .evidence_digest = {},
    }));
  }
  return canonicalize_host_startup_audit_report({
      .api_version = std::string(kHostStartupAuditReportApiVersion),
      .audit_id =
          "hostd-audit-" + std::to_string(next_audit_id.fetch_add(1U)),
      .host_id = observed.host_id,
      .boot_id = observed.boot_id,
      .broker_epoch = observed.broker_epoch,
      .broker_instance_id = "hostd-test-instance",
      .inventory = observed,
      .pre_audit_occupancy = occupancy,
      .post_audit_occupancy = occupancy,
      .ledger_head_before = head,
      .ledger_head_after_observation = head,
      .policy = startup_audit_policy(),
      .findings = std::move(findings),
      .disposition = disposition,
      .observed_begin_boottime_ns = 10,
      .observed_end_boottime_ns = 20,
      .findings_digest = {},
      .report_digest = {},
  });
}

class FakeAuditor final : public IConfiguredHostStartupAuditorV2 {
public:
  explicit FakeAuditor(HostStartupAuditReport report)
      : report_(std::move(report)) {}

  HostStartupAuditReport audit() override {
    ++calls;
    if (during_audit)
      during_audit();
    return report_;
  }

  HostStartupAuditReport report_;
  std::function<void()> during_audit;
  std::size_t calls{};
};

class OneShotFault final : public IHostLedgerFaultInjector {
public:
  explicit OneShotFault(HostLedgerFaultPoint target) : target_(target) {}

  void hit(HostLedgerFaultPoint point) override {
    if (armed_ && point == target_) {
      armed_ = false;
      throw HostLedgerError("injected hostd ledger fault");
    }
  }

private:
  HostLedgerFaultPoint target_;
  bool armed_{true};
};

struct Fixture final {
  Fixture()
      : path(directory.path() / "host-ledger.db"),
        observed(inventory({"mutex-a"})),
        authority(std::make_shared<HostLedgerFilesystemAuthority>(
            HostLedgerFilesystemAuthority::acquire({
                .api_version = std::string(kHostLedgerAuthorityApiVersion),
                .ledger_path = path,
                .expected_owner_uid = ::geteuid(),
                .expected_owner_gid = ::getegid(),
                .enforcement_grade =
                    HostLedgerEnforcementGrade::cooperative_test,
            }))),
        ledger(std::make_shared<SQLiteHostLedger>(
            authority, observed, nullptr, startup_audit_policy())) {}

  [[nodiscard]] HostdCoordinatorConfig config() const {
    return {.api_version = std::string(kHostdCoordinatorApiVersion),
            .host_id = observed.host_id,
            .boot_id = observed.boot_id,
            .broker_epoch = observed.broker_epoch,
            .maximum_live_sessions = 32U,
            .maximum_logical_scopes = 64U};
  }

  [[nodiscard]] std::unique_ptr<HostGrantCoordinator>
  coordinator(HostdCoordinatorConfig value) {
    return std::make_unique<HostGrantCoordinator>(std::move(value), ledger,
                                                  logical_fences);
  }

  [[nodiscard]] std::unique_ptr<HostGrantCoordinator>
  coordinator_without_logical_authority() {
    return std::make_unique<HostGrantCoordinator>(config(), ledger);
  }

  [[nodiscard]] std::unique_ptr<HostGrantCoordinator> coordinator() {
    return coordinator(config());
  }

  [[nodiscard]] std::unique_ptr<HostGrantCoordinator>
  coordinator(std::shared_ptr<IHostdLedgerBoundary> boundary) {
    return std::make_unique<HostGrantCoordinator>(config(),
                                                  std::move(boundary),
                                                  logical_fences);
  }

  TemporaryDirectory directory;
  std::filesystem::path path;
  HostInventoryReceipt observed;
  std::shared_ptr<HostLedgerFilesystemAuthority> authority;
  std::shared_ptr<SQLiteHostLedger> ledger;
  std::shared_ptr<FakeLogicalFenceAuthority> logical_fences =
      std::make_shared<FakeLogicalFenceAuthority>();
};

HostdConnectedSession connect_admission(HostGrantCoordinator &coordinator,
                                        const HostdSessionAttribution &scope,
                                        FakeLogicalFenceAuthority &authority) {
  authority.set_live(scope);
  auto peer = std::make_shared<FakePeer>(HostdSessionAccess::grant_release);
  return coordinator.connect({.attribution = scope}, std::move(peer));
}

void admit(HostGrantCoordinator &coordinator, SQLiteHostLedger &ledger) {
  FakeAuditor auditor(audit_report(ledger));
  const auto receipt = coordinator.run_startup_audit(auditor, {30, 40});
  require(receipt.report_digest == auditor.report_.report_digest &&
              auditor.calls == 1U,
          "startup audit commits and returns exact report evidence");
}

void lifecycle_and_pre_audit_gate() {
  Fixture fixture;
  auto coordinator = fixture.coordinator();
  require(coordinator->status().lifecycle == HostdLifecycle::sealed,
          "coordinator begins sealed");

  auto admission_peer =
      std::make_shared<FakePeer>(HostdSessionAccess::grant_release);
  require_throws<HostdStateError>(
      [&] {
        (void)coordinator->connect({.attribution = attribution()},
                                   admission_peer);
      },
      "mutation-capable sessions cannot exist before startup audit");
  auto observer = std::make_shared<FakePeer>(
      HostdSessionAccess::read_only,
      HostdPeerEnforcementGrade::observed_only);
  const auto read_only =
      coordinator->connect({.attribution = std::nullopt}, observer);
  require(coordinator->status(read_only.session_id).lifecycle ==
              HostdLifecycle::sealed,
          "read-only diagnostics remain available while sealed");
  require_throws<HostdUnauthorized>(
      [&] {
        (void)coordinator->request_bundle(
            read_only.session_id,
            request_for(attribution(), "pre-audit-request"), {10, 20});
      },
      "no connected pre-audit peer can obtain a grant");
  require(fixture.ledger->occupancy().active_fences.empty(),
          "pre-audit attempt leaves the physical ledger unallocated");

  FakeAuditor auditor(audit_report(*fixture.ledger));
  auditor.during_audit = [&] {
    require(coordinator->status().lifecycle == HostdLifecycle::startup_auditing,
            "audit callback observes explicit startup_auditing state");
  };
  (void)coordinator->run_startup_audit(auditor, {30, 40});
  require(coordinator->status().lifecycle == HostdLifecycle::admitting,
          "exact passed audit enables admission");
  const auto scope = attribution();
  const auto session =
      connect_admission(*coordinator, scope, *fixture.logical_fences);
  const auto granted = coordinator->request_bundle(
      session.session_id, request_for(scope, "lifecycle-request"), {10, 20});
  require(granted.grant.has_value(),
          "audited mutation-capable session can grant");
}

void failed_policy_mismatch_and_malformed_audits_fail_closed() {
  {
    Fixture fixture;
    auto coordinator = fixture.coordinator();
    FakeAuditor auditor(
        audit_report(*fixture.ledger, HostStartupAuditDisposition::failed));
    require_throws<HostdStateError>(
        [&] { (void)coordinator->run_startup_audit(auditor, {30, 40}); },
        "committed failed audit must block admission");
    const auto status = coordinator->status();
    require(status.lifecycle == HostdLifecycle::startup_blocked &&
                status.startup_audit.has_value() &&
                status.startup_audit->report_digest ==
                    auditor.report_.report_digest,
            "failed evidence remains inspectable in an explicit blocked state");
    auto observer = std::make_shared<FakePeer>(
        HostdSessionAccess::read_only,
        HostdPeerEnforcementGrade::observed_only);
    const auto diagnostic = coordinator->connect({}, observer);
    require(coordinator->status(diagnostic.session_id).lifecycle ==
                HostdLifecycle::startup_blocked,
            "read-only diagnostics remain available while startup is blocked");
  }

  {
    Fixture fixture;
    auto coordinator = fixture.coordinator();
    auto mismatched = audit_report(*fixture.ledger);
    mismatched.policy = canonicalize_host_startup_audit_policy({
        .api_version = std::string(kHostStartupAuditPolicyApiVersion),
        .require_stable_occupancy = true,
        .fail_on_blocking_findings = false,
        .maximum_findings = 16U,
        .policy_digest = {},
    });
    mismatched = canonicalize_host_startup_audit_report(std::move(mismatched));
    FakeAuditor auditor(std::move(mismatched));
    require_throws<HostdStateError>(
        [&] { (void)coordinator->run_startup_audit(auditor, {30, 40}); },
        "auditor cannot substitute its own startup policy");
    require(coordinator->status().lifecycle == HostdLifecycle::poisoned &&
                !coordinator->status().startup_audit,
            "trusted-policy mismatch poisons without retaining a receipt");
  }

  {
    Fixture fixture;
    auto coordinator = fixture.coordinator();
    auto malformed_report = audit_report(*fixture.ledger);
    malformed_report.audit_id.assign(1024U * 1024U, 'x');
    FakeAuditor auditor(std::move(malformed_report));
    require_throws<HostdStateError>(
        [&] { (void)coordinator->run_startup_audit(auditor, {30, 40}); },
        "unbounded auditor report must fail");
    const auto status = coordinator->status();
    require(status.lifecycle == HostdLifecycle::poisoned &&
                !status.startup_audit && status.poison_reason.size() <= 512U,
            "malformed auditor output is never retained in status");
  }
}

void concurrent_startup_audits_are_single_flight() {
  Fixture fixture;
  auto coordinator = fixture.coordinator();
  FakeAuditor first(audit_report(*fixture.ledger));
  FakeAuditor second(audit_report(*fixture.ledger));
  std::mutex gate_mutex;
  std::condition_variable gate_condition;
  bool entered = false;
  bool release = false;
  first.during_audit = [&] {
    require(coordinator->status().lifecycle == HostdLifecycle::startup_auditing,
            "auditor callback can reenter status outside the mutex");
    std::unique_lock lock(gate_mutex);
    entered = true;
    gate_condition.notify_all();
    gate_condition.wait(lock, [&] { return release; });
  };

  std::atomic<bool> first_passed{false};
  std::thread first_thread([&] {
    try {
      (void)coordinator->run_startup_audit(first, {30, 40});
      first_passed = true;
    } catch (...) {
    }
  });
  {
    std::unique_lock lock(gate_mutex);
    gate_condition.wait(lock, [&] { return entered; });
  }
  require_throws<HostdStateError>(
      [&] { (void)coordinator->run_startup_audit(second, {30, 40}); },
      "a concurrent startup audit cannot overtake the single flight");
  require(coordinator->status().lifecycle == HostdLifecycle::startup_auditing &&
              second.calls == 0U,
          "rejected concurrent audit neither runs nor poisons the active one");
  {
    std::scoped_lock lock(gate_mutex);
    release = true;
    gate_condition.notify_all();
  }
  first_thread.join();
  require(first_passed &&
              coordinator->status().lifecycle == HostdLifecycle::admitting,
          "the original exact audit remains able to admit");
}

void stale_startup_report_is_rejected_by_ledger_cas() {
  Fixture fixture;
  auto coordinator = fixture.coordinator();
  FakeAuditor stale(audit_report(*fixture.ledger));
  auto intervening = audit_report(*fixture.ledger);
  intervening.audit_id = "intervening-startup-audit";
  intervening = canonicalize_host_startup_audit_report(std::move(intervening));
  (void)fixture.ledger->commit_startup_audit(intervening, {30, 40});
  require_throws<HostdStateError>(
      [&] { (void)coordinator->run_startup_audit(stale, {50, 60}); },
      "stale head evidence cannot be committed");
  const auto status = coordinator->status();
  require(status.lifecycle == HostdLifecycle::poisoned &&
              !status.startup_audit,
          "stale report leaves no inspectable committed receipt");
}

void crash_after_commit_retries_by_exact_ledger_replay() {
  TemporaryDirectory directory;
  const auto path = directory.path() / "host-ledger.db";
  const auto observed = inventory({"mutex-a"});
  auto authority = std::make_shared<HostLedgerFilesystemAuthority>(
      HostLedgerFilesystemAuthority::acquire({
          .api_version = std::string(kHostLedgerAuthorityApiVersion),
          .ledger_path = path,
          .expected_owner_uid = ::geteuid(),
          .expected_owner_gid = ::getegid(),
          .enforcement_grade =
              HostLedgerEnforcementGrade::cooperative_test,
      }));
  const HostdCoordinatorConfig config{
      .api_version = std::string(kHostdCoordinatorApiVersion),
      .host_id = observed.host_id,
      .boot_id = observed.boot_id,
      .broker_epoch = observed.broker_epoch,
      .maximum_live_sessions = 8U,
      .maximum_logical_scopes = 8U,
  };
  HostStartupAuditReport report;
  {
    OneShotFault fault(HostLedgerFaultPoint::after_startup_audit_commit);
    auto ledger = std::make_shared<SQLiteHostLedger>(
        authority, observed, &fault, startup_audit_policy());
    report = audit_report(*ledger);
    HostGrantCoordinator coordinator(config, ledger);
    FakeAuditor auditor(report);
    require_throws<HostdStateError>(
        [&] { (void)coordinator.run_startup_audit(auditor, {30, 40}); },
        "lost reply after durable commit fails the first coordinator");
    require(coordinator.status().lifecycle == HostdLifecycle::poisoned &&
                !coordinator.status().startup_audit,
            "first coordinator does not fabricate a receipt after lost reply");
  }

  auto reopened = std::make_shared<SQLiteHostLedger>(
      authority, observed, nullptr, startup_audit_policy());
  HostGrantCoordinator restarted(config, reopened);
  FakeAuditor retry(report);
  const auto receipt = restarted.run_startup_audit(retry, {50, 60});
  require(receipt.report_digest == report.report_digest &&
              restarted.status().startup_audit == receipt &&
              restarted.status().lifecycle == HostdLifecycle::admitting,
          "exact retry rereads the durable receipt and admits after restart");

  HostGrantCoordinator stale_restart(config, reopened);
  FakeAuditor stale_retry(report);
  const auto replayed = stale_restart.run_startup_audit(stale_retry, {90, 100});
  require(replayed == receipt &&
              stale_restart.status().lifecycle == HostdLifecycle::admitting,
          "exact audit/finalize replay recovers the same opaque admission epoch");
}

void blocked_startup_allows_only_exact_release_cleanup() {
  TemporaryDirectory directory;
  const auto path = directory.path() / "host-ledger.db";
  const auto observed = inventory({"mutex-a"});
  auto authority = std::make_shared<HostLedgerFilesystemAuthority>(
      HostLedgerFilesystemAuthority::acquire({
          .api_version = std::string(kHostLedgerAuthorityApiVersion),
          .ledger_path = path,
          .expected_owner_uid = ::geteuid(),
          .expected_owner_gid = ::getegid(),
          .enforcement_grade = HostLedgerEnforcementGrade::cooperative_test,
      }));
  const auto scope = attribution("journal-cleanup", "run-cleanup",
                                 "lease-cleanup", 1U);
  ResourceBundleGrant legacy_grant;
  {
    SQLiteHostLedger legacy(authority, observed);
    const auto granted = legacy.request_bundle(
        request_for(scope, "legacy-cleanup-grant"), {10, 20});
    require(granted.grant.has_value(),
            "cleanup fixture has a pre-policy active allocation");
    legacy_grant = *granted.grant;
  }
  auto ledger = std::make_shared<SQLiteHostLedger>(
      authority, observed, nullptr, startup_audit_policy());
  auto logical = std::make_shared<FakeLogicalFenceAuthority>();
  HostGrantCoordinator coordinator(
      {.api_version = std::string(kHostdCoordinatorApiVersion),
       .host_id = observed.host_id,
       .boot_id = observed.boot_id,
       .broker_epoch = observed.broker_epoch,
       .maximum_live_sessions = 8U,
       .maximum_logical_scopes = 8U},
      ledger, logical);
  FakeAuditor failed(
      audit_report(*ledger, HostStartupAuditDisposition::failed));
  require_throws<HostdStateError>(
      [&] { (void)coordinator.run_startup_audit(failed, {30, 40}); },
      "failed audit blocks new admission while retaining cleanup state");
  const auto release = release_for(legacy_grant, "blocked-cleanup-release");
  logical->authorize_cleanup(scope, legacy_grant, release);
  auto peer = std::make_shared<FakePeer>(HostdSessionAccess::release_only);
  const auto session =
      coordinator.connect({.attribution = scope}, std::move(peer));
  const auto cleaned =
      coordinator.release_bundle(session.session_id, release, {50, 60});
  require(cleaned.receipt.allocation_id == legacy_grant.allocation_id &&
              ledger->occupancy().active_fences.empty() &&
              coordinator.status().lifecycle == HostdLifecycle::startup_blocked,
          "blocked hostd permits exact release-only occupancy reduction");
  require_throws<HostdStateError>(
      [&] {
        (void)coordinator.request_bundle(
            session.session_id, request_for(scope, "blocked-grant"),
            {70, 80});
      },
      "blocked release-only cleanup capability cannot request grants");
}

void blocked_startup_recovers_stale_exact_outcomes_read_only() {
  Fixture fixture;
  const auto scope = attribution("journal-recovery", "run-recovery",
                                 "lease-recovery", 4U);
  const auto exact_request = request_for(scope, "restart-recovery-grant");
  BundleRequestResult original;
  {
    auto first = fixture.coordinator();
    admit(*first, *fixture.ledger);
    const auto session =
        connect_admission(*first, scope, *fixture.logical_fences);
    original = first->request_bundle(session.session_id, exact_request,
                                     {50, 60});
    require(original.grant.has_value(),
            "recovery fixture commits an exact grant before restart");
  }

  auto restarted = fixture.coordinator();
  FakeAuditor failed(
      audit_report(*fixture.ledger, HostStartupAuditDisposition::failed));
  require_throws<HostdStateError>(
      [&] { (void)restarted->run_startup_audit(failed, {70, 80}); },
      "restart enters startup-blocked after committing failed evidence");

  auto superseding = scope;
  ++superseding.logical_fencing_token;
  fixture.logical_fences->set_live(superseding);
  auto recovery_peer =
      std::make_shared<FakePeer>(HostdSessionAccess::read_only);
  const auto recovery_session = restarted->connect(
      {.attribution = scope}, recovery_peer);
  const auto records_before = fixture.ledger->record_count();
  const auto generation_before = fixture.ledger->generation(
      original.grant->fences.front().resource);
  const auto occupancy_before = fixture.ledger->occupancy();
  const auto recovered = restarted->reconcile_bundle_outcome(
      recovery_session.session_id, exact_request);
  require(recovered && recovered->replayed &&
              recovered->grant == original.grant &&
              recovered->outcome_digest == original.outcome_digest,
          "attributed service session recovers stale exact lost-reply grant while blocked");

  const auto absent_request = request_for(scope, "blocked-recovery-absent");
  const auto absent = restarted->reconcile_bundle_outcome(
      recovery_session.session_id, absent_request);
  require(!absent && fixture.ledger->record_count() == records_before &&
              fixture.ledger->generation(
                  original.grant->fences.front().resource) ==
                  generation_before &&
              fixture.ledger->occupancy() == occupancy_before,
          "blocked recovery cannot admit a missing request or mutate projections");
  require_throws<HostdUnauthorized>(
      [&] {
        (void)restarted->request_bundle(recovery_session.session_id,
                                        absent_request, {90, 100});
      },
      "read-only recovery session cannot use the mutating grant path");

  recovery_peer->evidence_.access = HostdSessionAccess::denied;
  require_throws<HostdUnauthorized>(
      [&] {
        (void)restarted->reconcile_bundle_outcome(
            recovery_session.session_id, exact_request);
      },
      "a peer re-observed as denied cannot recover a receipt");
}

void admission_finalize_boundary_is_atomic_and_recoverable() {
  {
    Fixture fixture;
    auto boundary =
        std::make_shared<AdversarialLedgerBoundary>(fixture.ledger);
    auto coordinator = fixture.coordinator(boundary);
    boundary->before_admission_finalize = [&] {
      auto intervening = audit_report(*fixture.ledger);
      intervening.audit_id = "finalize-boundary-intervening-audit";
      intervening =
          canonicalize_host_startup_audit_report(std::move(intervening));
      (void)fixture.ledger->commit_startup_audit(intervening, {31, 41});
    };
    FakeAuditor auditor(audit_report(*fixture.ledger));
    require_throws<HostdStateError>(
        [&] { (void)coordinator->run_startup_audit(auditor, {30, 40}); },
        "mutation between audit commit and finalize loses the ledger CAS");
    require(coordinator->status().lifecycle == HostdLifecycle::poisoned,
            "pre-finalize mutation cannot be admitted by a local latch");
  }

  {
    Fixture fixture;
    auto boundary =
        std::make_shared<AdversarialLedgerBoundary>(fixture.ledger);
    auto coordinator = fixture.coordinator(boundary);
    std::optional<BundleRequestResult> boundary_request;
    boundary->after_admission_finalize = [&](const auto &finalized) {
      require(coordinator->status().lifecycle ==
                  HostdLifecycle::startup_auditing,
              "finalize runs outside the coordinator mutex before local latch");
      boundary_request = fixture.ledger->request_bundle(
          request_for(attribution(), "authorized-finalize-boundary-request"),
          {31, 41}, finalized.epoch);
    };
    FakeAuditor auditor(audit_report(*fixture.ledger));
    (void)coordinator->run_startup_audit(auditor, {30, 40});
    require(boundary_request && boundary_request->grant &&
                coordinator->status().lifecycle == HostdLifecycle::admitting,
            "epoch-authorized mutation after finalize survives local latch lag");
  }
}

void admission_finalize_lost_reply_and_async_poison_recover() {
  {
    TemporaryDirectory directory;
    const auto path = directory.path() / "host-ledger.db";
    const auto observed = inventory({"mutex-a"});
    auto authority = std::make_shared<HostLedgerFilesystemAuthority>(
        HostLedgerFilesystemAuthority::acquire({
            .api_version = std::string(kHostLedgerAuthorityApiVersion),
            .ledger_path = path,
            .expected_owner_uid = ::geteuid(),
            .expected_owner_gid = ::getegid(),
            .enforcement_grade =
                HostLedgerEnforcementGrade::cooperative_test,
        }));
    const HostdCoordinatorConfig config{
        .api_version = std::string(kHostdCoordinatorApiVersion),
        .host_id = observed.host_id,
        .boot_id = observed.boot_id,
        .broker_epoch = observed.broker_epoch,
        .maximum_live_sessions = 8U,
        .maximum_logical_scopes = 8U,
    };
    HostStartupAuditReport report;
    {
      OneShotFault fault(HostLedgerFaultPoint::after_admission_finalize_commit);
      auto ledger = std::make_shared<SQLiteHostLedger>(
          authority, observed, &fault, startup_audit_policy());
      report = audit_report(*ledger);
      HostGrantCoordinator coordinator(config, ledger);
      FakeAuditor auditor(report);
      require_throws<HostdStateError>(
          [&] { (void)coordinator.run_startup_audit(auditor, {30, 40}); },
          "lost finalize reply fails the first local coordinator");
      require(coordinator.status().lifecycle == HostdLifecycle::poisoned,
              "lost finalize reply never fabricates a local admission latch");
    }
    auto reopened = std::make_shared<SQLiteHostLedger>(
        authority, observed, nullptr, startup_audit_policy());
    HostGrantCoordinator restarted(config, reopened);
    FakeAuditor retry(report);
    (void)restarted.run_startup_audit(retry, {50, 60});
    require(restarted.status().lifecycle == HostdLifecycle::admitting,
            "exact finalize replay recovers after a lost durable reply");
  }

  {
    Fixture fixture;
    auto boundary =
        std::make_shared<AdversarialLedgerBoundary>(fixture.ledger);
    auto coordinator = fixture.coordinator(boundary);
    const auto report = audit_report(*fixture.ledger);
    boundary->after_admission_finalize = [&](const auto &) {
      require(::chmod(fixture.path.c_str(), 0666) == 0,
              "test makes authority unsafe after durable finalize");
      require(coordinator->status().lifecycle == HostdLifecycle::poisoned,
              "asynchronous observation poisons before local admission latch");
      require(::chmod(fixture.path.c_str(), 0600) == 0,
              "test restores authority after finalize poison");
    };
    FakeAuditor auditor(report);
    require_throws<HostdStateError>(
        [&] { (void)coordinator->run_startup_audit(auditor, {30, 40}); },
        "asynchronous poison wins over finalized local latch");
    auto restarted = fixture.coordinator();
    FakeAuditor retry(report);
    (void)restarted->run_startup_audit(retry, {50, 60});
    require(restarted->status().lifecycle == HostdLifecycle::admitting,
            "new coordinator recovers exact durable epoch after async poison");
  }
}

void unauthorized_and_read_only_peers() {
  Fixture fixture;
  auto coordinator = fixture.coordinator();
  admit(*coordinator, *fixture.ledger);

  auto denied = std::make_shared<FakePeer>(HostdSessionAccess::denied);
  require_throws<HostdUnauthorized>(
      [&] { (void)coordinator->connect({}, denied); },
      "denied peer cannot establish a session");
  auto unenforced = std::make_shared<FakePeer>(
      HostdSessionAccess::grant_release,
      HostdPeerEnforcementGrade::observed_only);
  require_throws<HostdUnauthorized>(
      [&] {
        (void)coordinator->connect({.attribution = attribution()}, unenforced);
      },
      "observed-only identity cannot acquire mutation authority");

  auto observer = std::make_shared<FakePeer>(
      HostdSessionAccess::read_only,
      HostdPeerEnforcementGrade::observed_only);
  const auto session = coordinator->connect({}, observer);
  require(coordinator->status(session.session_id).lifecycle ==
              HostdLifecycle::admitting,
          "read-only peer can inspect status");
  require_throws<HostdUnauthorized>(
      [&] {
        (void)coordinator->request_bundle(
            session.session_id, request_for(attribution(), "observer-request"),
            {10, 20});
      },
      "read-only peer cannot request a grant");
  require_throws<HostdUnauthorized>(
      [&] { (void)coordinator->status("hostd-session-forged"); },
      "unknown peer cannot inspect session-scoped status");
}

void disconnected_sessions_cannot_replay() {
  Fixture fixture;
  auto coordinator = fixture.coordinator();
  admit(*coordinator, *fixture.ledger);
  const auto scope = attribution();
  const auto first =
      connect_admission(*coordinator, scope, *fixture.logical_fences);
  coordinator->disconnect(first.session_id);
  require_throws<HostdUnauthorized>(
      [&] { (void)coordinator->status(first.session_id); },
      "disconnected session cannot replay status access");
  require_throws<HostdUnauthorized>(
      [&] {
        (void)coordinator->request_bundle(
            first.session_id, request_for(scope, "disconnected-request"),
            {10, 20});
      },
      "disconnected session cannot replay mutation access");
  require_throws<HostdUnauthorized>(
      [&] { coordinator->disconnect(first.session_id); },
      "disconnect is exact and non-replayable");

  const auto replacement =
      connect_admission(*coordinator, scope, *fixture.logical_fences);
  require(replacement.session_id != first.session_id,
          "reconnection receives a new server-generated identity");
}

void concurrency_keys_are_independent_logical_scopes() {
  Fixture fixture;
  auto coordinator = fixture.coordinator();
  admit(*coordinator, *fixture.ledger);
  const auto left = attribution("journal-shared", "run-shared",
                                "lease-shared", 7U, "gpu:0");
  const auto right = attribution("journal-shared", "run-shared",
                                 "lease-shared", 7U, "gpu:1");
  const auto left_session =
      connect_admission(*coordinator, left, *fixture.logical_fences);
  const auto right_session =
      connect_admission(*coordinator, right, *fixture.logical_fences);
  require(left_session.effective_access == HostdSessionAccess::grant_release &&
              right_session.effective_access ==
                  HostdSessionAccess::grant_release,
          "equal journal/run/lease/fence identities remain distinct by concurrency key");

  auto advanced_left = left;
  ++advanced_left.logical_fencing_token;
  const auto advanced_left_session = connect_admission(
      *coordinator, advanced_left, *fixture.logical_fences);
  require(advanced_left_session.effective_access ==
              HostdSessionAccess::grant_release,
          "one concurrency scope advances independently");
  require_throws<HostdUnauthorized>(
      [&] {
        auto peer =
            std::make_shared<FakePeer>(HostdSessionAccess::grant_release);
        (void)coordinator->connect({.attribution = left}, peer);
      },
      "the prior fence is stale within its exact concurrency scope");
  auto right_peer =
      std::make_shared<FakePeer>(HostdSessionAccess::grant_release);
  const auto right_reconnected =
      coordinator->connect({.attribution = right}, right_peer);
  require(right_reconnected.effective_access ==
              HostdSessionAccess::grant_release,
          "advancing one concurrency scope does not stale another");

  auto malformed = right;
  malformed.concurrency_key.clear();
  require_throws<HostdUnauthorized>(
      [&] {
        auto peer =
            std::make_shared<FakePeer>(HostdSessionAccess::grant_release);
        (void)coordinator->connect({.attribution = malformed}, peer);
      },
      "missing concurrency scope attribution fails closed");
}

void stale_fences_and_attribution_mismatch_fail() {
  Fixture fixture;
  auto coordinator = fixture.coordinator();
  admit(*coordinator, *fixture.ledger);
  const auto old_scope =
      attribution("journal-scope", "run-scope", "lease-scope", 7U);
  const auto old_session =
      connect_admission(*coordinator, old_scope, *fixture.logical_fences);
  const auto current_scope =
      attribution("journal-scope", "run-scope", "lease-scope", 8U);
  const auto current_session =
      connect_admission(*coordinator, current_scope, *fixture.logical_fences);
  require_throws<HostdUnauthorized>(
      [&] {
        (void)coordinator->request_bundle(
            old_session.session_id, request_for(old_scope, "stale-request"),
            {10, 20});
      },
      "new logical fence revokes an older live session");
  require_throws<HostdUnauthorized>(
      [&] {
        auto stale_peer =
            std::make_shared<FakePeer>(HostdSessionAccess::grant_release);
        (void)coordinator->connect({.attribution = old_scope}, stale_peer);
      },
      "stale logical fence cannot reconnect");

  for (std::size_t field = 0U; field < 4U; ++field) {
    auto wrong = current_scope;
    if (field == 0U)
      wrong.journal_id = "journal-wrong";
    if (field == 1U)
      wrong.run_id = "run-wrong";
    if (field == 2U)
      wrong.logical_lease_id = "lease-wrong";
    if (field == 3U)
      wrong.logical_fencing_token = 9U;
    require_throws<HostdUnauthorized>(
        [&] {
          (void)coordinator->request_bundle(
              current_session.session_id,
              request_for(wrong, "mismatch-" + std::to_string(field)),
              {10, 20});
        },
        "request attribution must match every session field");
  }
  require(coordinator
              ->request_bundle(current_session.session_id,
                               request_for(current_scope, "current-request"),
                               {10, 20})
              .grant.has_value(),
          "current exact logical fence admits a request");
}

void wrong_host_boot_and_broker_poison() {
  const std::vector<std::string> fields{"host", "boot", "broker"};
  for (const std::string &field : fields) {
    Fixture fixture;
    auto config = fixture.config();
    if (field == "host")
      config.host_id = "host-wrong";
    if (field == "boot")
      config.boot_id = "boot-wrong";
    if (field == "broker")
      config.broker_epoch = "broker-wrong";
    auto coordinator = fixture.coordinator(config);
    FakeAuditor auditor(audit_report(*fixture.ledger));
    require_throws<HostdStateError>(
        [&] { (void)coordinator->run_startup_audit(auditor, {30, 40}); },
        "coordinator identity must match host ledger identity");
    require(coordinator->status().lifecycle == HostdLifecycle::poisoned,
            "wrong host/boot/broker identity poisons coordinator");
  }
}

void ledger_request_and_release_replay_are_exact() {
  Fixture fixture;
  auto coordinator = fixture.coordinator();
  admit(*coordinator, *fixture.ledger);
  const auto scope = attribution();
  const auto session =
      connect_admission(*coordinator, scope, *fixture.logical_fences);
  const auto request = request_for(scope, "replay-request");
  const auto first =
      coordinator->request_bundle(session.session_id, request, {10, 20});
  const auto replay =
      coordinator->request_bundle(session.session_id, request, {100, 200});
  require(first.grant && replay.replayed && replay.grant == first.grant &&
              replay.outcome_digest == first.outcome_digest,
          "coordinator preserves exact host-ledger request replay");

  const auto release = release_for(*first.grant, "replay-release");
  const auto released =
      coordinator->release_bundle(session.session_id, release, {30, 40});
  const auto release_replay =
      coordinator->release_bundle(session.session_id, release, {300, 400});
  require(!released.replayed && release_replay.replayed &&
              release_replay.receipt == released.receipt,
          "coordinator preserves exact host-ledger release replay");

  auto spoofed = release;
  spoofed.run_id = "run-spoofed";
  spoofed = seal_resource_release_request(std::move(spoofed));
  require_throws<HostdUnauthorized>(
      [&] {
        (void)coordinator->release_bundle(session.session_id, spoofed,
                                          {500, 600});
      },
      "release attribution spoof is rejected before ledger replay");
}

void concurrent_sessions_and_journals_share_one_ledger() {
  Fixture fixture;
  auto coordinator = fixture.coordinator();
  admit(*coordinator, *fixture.ledger);
  const auto left_scope =
      attribution("journal-left", "run-left", "lease-left", 1U);
  const auto right_scope =
      attribution("journal-right", "run-right", "lease-right", 1U);
  const auto left =
      connect_admission(*coordinator, left_scope, *fixture.logical_fences);
  const auto right =
      connect_admission(*coordinator, right_scope, *fixture.logical_fences);
  std::atomic<int> granted{0};
  std::atomic<int> busy{0};
  std::thread left_thread([&] {
    const auto result = coordinator->request_bundle(
        left.session_id, request_for(left_scope, "concurrent-left"), {10, 20});
    result.status == BundleRequestStatus::granted ? ++granted : ++busy;
  });
  std::thread right_thread([&] {
    const auto result = coordinator->request_bundle(
        right.session_id, request_for(right_scope, "concurrent-right"),
        {11, 21});
    result.status == BundleRequestStatus::granted ? ++granted : ++busy;
  });
  left_thread.join();
  right_thread.join();
  const auto status = coordinator->status();
  require(
      granted == 1 && busy == 1 && status.live_sessions == 2U &&
          status.admission_sessions == 2U && fixture.ledger->verify(),
      "concurrent journals receive one physical grant on the shared ledger");
}

void durable_fence_authority_and_cleanup_reconnect() {
  Fixture missing;
  auto fail_closed = missing.coordinator_without_logical_authority();
  admit(*fail_closed, *missing.ledger);
  auto peer =
      std::make_shared<FakePeer>(HostdSessionAccess::grant_release);
  require_throws<HostdStateError>(
      [&] {
        (void)fail_closed->connect({.attribution = attribution()}, peer);
      },
      "mutation admission fails closed without durable logical evidence");

  Fixture fixture;
  auto coordinator = fixture.coordinator();
  admit(*coordinator, *fixture.ledger);
  const auto scope7 =
      attribution("journal-cleanup", "run-cleanup", "lease-cleanup", 7U);
  const auto session7 =
      connect_admission(*coordinator, scope7, *fixture.logical_fences);
  const auto granted7 = coordinator->request_bundle(
      session7.session_id, request_for(scope7, "cleanup-grant-7"), {10, 20});
  require(granted7.grant.has_value(), "token 7 obtains its exact allocation");
  coordinator->disconnect(session7.session_id);

  auto scope8 = scope7;
  scope8.logical_fencing_token = 8U;
  fixture.logical_fences->set_live(scope8);
  const auto release7 = release_for(*granted7.grant, "cleanup-release-7");
  fixture.logical_fences->authorize_cleanup(scope7, *granted7.grant,
                                             release7);
  auto cleanup_peer =
      std::make_shared<FakePeer>(HostdSessionAccess::grant_release);
  const auto cleanup =
      coordinator->connect({.attribution = scope7}, cleanup_peer);
  require(cleanup.effective_access == HostdSessionAccess::release_only,
          "stale durable cleanup evidence is downgraded to release-only");
  require_throws<HostdUnauthorized>(
      [&] {
        (void)coordinator->request_bundle(
            cleanup.session_id, request_for(scope7, "cleanup-denied-grant"),
            {30, 40});
      },
      "release-only cleanup session can never obtain a new grant");
  require_throws<HostdUnauthorized>(
      [&] {
        (void)coordinator->release_bundle(
            cleanup.session_id,
            release_for(*granted7.grant, "unauthorized-cleanup-release"),
            {40, 50});
      },
      "cleanup evidence authorizes only its exact canonical release request");
  require(!coordinator
               ->release_bundle(cleanup.session_id, release7, {50, 60})
               .replayed,
          "reconnected token 7 cleanup session releases its exact allocation");

  const auto session8 =
      connect_admission(*coordinator, scope8, *fixture.logical_fences);
  require(coordinator
              ->request_bundle(session8.session_id,
                               request_for(scope8, "cleanup-grant-8"),
                               {70, 80})
              .grant.has_value(),
          "token 8 can grant after token 7 exact cleanup");
  const auto status = coordinator->status();
  require(status.admission_sessions == 1U &&
              status.release_only_sessions == 1U,
          "status separates current grant and release-only cleanup sessions");
}

void advancing_fence_preserves_exact_connected_cleanup() {
  Fixture fixture;
  auto coordinator = fixture.coordinator();
  admit(*coordinator, *fixture.ledger);
  const auto scope7 =
      attribution("journal-live-cleanup", "run-live-cleanup",
                  "lease-live-cleanup", 7U);
  const auto session7 =
      connect_admission(*coordinator, scope7, *fixture.logical_fences);
  const auto grant7 = coordinator->request_bundle(
      session7.session_id, request_for(scope7, "live-cleanup-grant-7"),
      {10, 20});
  require(grant7.grant.has_value(), "token 7 has an allocation to clean up");
  auto scope8 = scope7;
  scope8.logical_fencing_token = 8U;
  const auto session8 =
      connect_admission(*coordinator, scope8, *fixture.logical_fences);
  const auto status = coordinator->status();
  require(status.admission_sessions == 1U &&
              status.stale_admission_sessions == 1U,
          "status excludes the stale token 7 session from admission");
  require_throws<HostdUnauthorized>(
      [&] {
        (void)coordinator->request_bundle(
            session7.session_id,
            request_for(scope7, "live-cleanup-stale-grant"), {30, 40});
      },
      "advanced token prevents the old connected session from granting");
  require(!coordinator
               ->release_bundle(
                   session7.session_id,
                   release_for(*grant7.grant, "live-cleanup-release-7"),
                   {50, 60})
               .replayed,
          "advanced token leaves the old exact session able to release");
  require(coordinator
              ->request_bundle(
                  session8.session_id,
                  request_for(scope8, "live-cleanup-grant-8"), {70, 80})
              .grant.has_value(),
          "new token grants after old exact cleanup");
}

void restart_peer_liveness_capacity_and_status_poison() {
  Fixture fixture;
  const auto old_scope =
      attribution("journal-restart", "run-restart", "lease-restart", 3U);
  fixture.logical_fences->set_live(old_scope);
  const auto startup_report = audit_report(*fixture.ledger);
  {
    auto first = fixture.coordinator();
    FakeAuditor auditor(startup_report);
    (void)first->run_startup_audit(auditor, {30, 40});
    auto first_peer =
        std::make_shared<FakePeer>(HostdSessionAccess::grant_release);
    (void)first->connect({.attribution = old_scope}, first_peer);
  }
  auto current_scope = old_scope;
  current_scope.logical_fencing_token = 4U;
  fixture.logical_fences->set_live(current_scope);
  auto restarted = fixture.coordinator();
  FakeAuditor retry(startup_report);
  (void)restarted->run_startup_audit(retry, {50, 60});
  auto stale_peer =
      std::make_shared<FakePeer>(HostdSessionAccess::grant_release);
  require_throws<HostdUnauthorized>(
      [&] { (void)restarted->connect({.attribution = old_scope}, stale_peer); },
      "restart never reconstructs logical authority from caller high-water");
  auto live_peer =
      std::make_shared<FakePeer>(HostdSessionAccess::grant_release);
  const auto live =
      restarted->connect({.attribution = current_scope}, live_peer);
  live_peer->alive = false;
  require_throws<HostdUnauthorized>(
      [&] { (void)restarted->status(live.session_id); },
      "session use re-observes simulated peer liveness");
  require_throws<HostdUnauthorized>(
      [&] {
        (void)restarted->request_bundle(
            live.session_id, request_for(current_scope, "dead-peer-request"),
            {10, 20});
      },
      "disappeared peer cannot exercise a retained session claim");
  require_throws<HostdUnauthorized>(
      [&] { (void)restarted->status(std::string(1024U * 1024U, 'x')); },
      "unbounded session claims fail shape validation before lookup");

  Fixture capacity_fixture;
  auto config = capacity_fixture.config();
  config.maximum_live_sessions = 2U;
  config.maximum_logical_scopes = 1U;
  auto bounded = capacity_fixture.coordinator(config);
  admit(*bounded, *capacity_fixture.ledger);
  auto observer1 = std::make_shared<FakePeer>(HostdSessionAccess::read_only);
  auto observer2 = std::make_shared<FakePeer>(HostdSessionAccess::read_only);
  const auto retained_observer = bounded->connect({}, observer1);
  require_throws<HostdStateError>(
      [&] { (void)bounded->connect({}, observer2); },
      "observer sessions cannot consume the reserved mutation slot");
  const auto scope_a =
      attribution("journal-capacity-a", "run-a", "lease-a", 1U);
  const auto admitted_a = connect_admission(
      *bounded, scope_a, *capacity_fixture.logical_fences);
  bounded->disconnect(admitted_a.session_id);
  bounded->disconnect(retained_observer.session_id);
  const auto scope_b =
      attribution("journal-capacity-b", "run-b", "lease-b", 1U);
  require(connect_admission(*bounded, scope_b,
                            *capacity_fixture.logical_fences)
              .effective_access == HostdSessionAccess::grant_release,
          "disconnect reclaims bounded logical-scope and session capacity");

  Fixture poisoned_fixture;
  auto poison_visible = poisoned_fixture.coordinator();
  admit(*poison_visible, *poisoned_fixture.ledger);
  require(::chmod(poisoned_fixture.path.c_str(), 0666) == 0,
          "test makes ledger authority permissions unsafe");
  const auto poisoned_status = poison_visible->status();
  require(poisoned_status.lifecycle == HostdLifecycle::poisoned &&
              !poisoned_status.poison_reason.empty(),
          "status reattests ledger identity and returns latched poison details");
  require(::chmod(poisoned_fixture.path.c_str(), 0600) == 0,
          "test restores ledger file permissions");
}

void asynchronous_startup_poison_is_never_overwritten() {
  Fixture fixture;
  auto coordinator = fixture.coordinator();
  FakeAuditor auditor(audit_report(*fixture.ledger));
  std::mutex gate_mutex;
  std::condition_variable gate_condition;
  bool audit_entered = false;
  bool audit_may_return = false;
  auditor.during_audit = [&] {
    std::unique_lock lock(gate_mutex);
    audit_entered = true;
    gate_condition.notify_all();
    gate_condition.wait(lock, [&] { return audit_may_return; });
  };
  std::atomic<bool> audit_rejected{false};
  std::thread audit_thread([&] {
    try {
      (void)coordinator->run_startup_audit(auditor, {30, 40});
    } catch (const HostdStateError &) {
      audit_rejected = true;
    }
  });
  {
    std::unique_lock lock(gate_mutex);
    gate_condition.wait(lock, [&] { return audit_entered; });
  }
  require(::chmod(fixture.path.c_str(), 0666) == 0,
          "test makes the ledger unsafe while startup audit is in flight");
  const auto poisoned = coordinator->status();
  require(poisoned.lifecycle == HostdLifecycle::poisoned &&
              !poisoned.poison_reason.empty(),
          "asynchronous identity observation latches startup poison");
  {
    std::scoped_lock lock(gate_mutex);
    audit_may_return = true;
    gate_condition.notify_all();
  }
  audit_thread.join();
  const auto final_status = coordinator->status();
  require(audit_rejected && final_status.lifecycle == HostdLifecycle::poisoned &&
              final_status.poison_reason == poisoned.poison_reason &&
              !final_status.startup_audit,
          "successful audit return cannot overwrite the first poison latch");
  require(::chmod(fixture.path.c_str(), 0600) == 0,
          "test restores ledger permissions after poison race");
}

void callbacks_reenter_without_coordinator_mutex() {
  Fixture fixture;
  auto boundary =
      std::make_shared<AdversarialLedgerBoundary>(fixture.ledger);
  auto coordinator = fixture.coordinator(boundary);
  boundary->during_audit_commit = [&] {
    require(coordinator->status().lifecycle == HostdLifecycle::startup_auditing,
            "ledger audit commit callback reenters without coordinator mutex");
  };
  admit(*coordinator, *fixture.ledger);
  boundary->during_audit_commit = {};
  const auto scope = attribution("journal-reentrant", "run-reentrant",
                                 "lease-reentrant", 1U);
  fixture.logical_fences->set_live(scope);
  auto peer = std::make_shared<FakePeer>(HostdSessionAccess::grant_release);
  const auto session = coordinator->connect({.attribution = scope}, peer);
  std::atomic<std::size_t> reentries{0U};
  const auto reenter = [&] {
    require_throws<HostdUnauthorized>(
        [&] { coordinator->disconnect("not-a-session"); },
        "reentrant callback acquires coordinator mutex without deadlock");
    ++reentries;
  };
  peer->during_observe = reenter;
  fixture.logical_fences->during_attest = reenter;
  boundary->during_verify = reenter;
  boundary->during_inventory = reenter;
  require(coordinator
              ->request_bundle(session.session_id,
                               request_for(scope, "reentrant-request"),
                               {10, 20})
              .grant.has_value(),
          "two-phase callback authorization still admits an exact grant");
  require(reentries >= 4U,
          "peer, logical fence, verify, and inventory callbacks all reentered");
  peer->during_observe = {};
  fixture.logical_fences->during_attest = {};
  boundary->during_verify = {};
  boundary->during_inventory = {};
}

void dead_session_reaping_and_inflight_fence_reservation() {
  Fixture capacity_fixture;
  auto capacity_config = capacity_fixture.config();
  capacity_config.maximum_live_sessions = 2U;
  capacity_config.maximum_logical_scopes = 1U;
  auto capacity = capacity_fixture.coordinator(capacity_config);
  admit(*capacity, *capacity_fixture.ledger);
  auto dead_observer = std::make_shared<FakePeer>(HostdSessionAccess::read_only);
  const auto dead_session = capacity->connect({}, dead_observer);
  dead_observer->alive = false;
  auto replacement = std::make_shared<FakePeer>(HostdSessionAccess::read_only);
  const auto replacement_session = capacity->connect({}, replacement);
  require(capacity->status().live_sessions == 1U &&
              !replacement_session.session_id.empty(),
          "capacity admission performs a bounded dead-session sweep");
  require_throws<HostdUnauthorized>(
      [&] { (void)capacity->status(dead_session.session_id); },
      "reaped session identity cannot be replayed");
  capacity->disconnect(replacement_session.session_id);
  const auto scope_a = attribution("journal-reap-a", "run-reap-a",
                                   "lease-reap-a", 1U);
  capacity_fixture.logical_fences->set_live(scope_a);
  auto dead_grant =
      std::make_shared<FakePeer>(HostdSessionAccess::grant_release);
  (void)capacity->connect({.attribution = scope_a}, dead_grant);
  dead_grant->alive = false;
  const auto scope_b = attribution("journal-reap-b", "run-reap-b",
                                   "lease-reap-b", 1U);
  capacity_fixture.logical_fences->set_live(scope_b);
  auto live_grant =
      std::make_shared<FakePeer>(HostdSessionAccess::grant_release);
  require(capacity->connect({.attribution = scope_b}, live_grant)
              .effective_access == HostdSessionAccess::grant_release,
          "bounded sweep also reclaims dead logical-scope capacity");

  Fixture fixture;
  auto boundary =
      std::make_shared<AdversarialLedgerBoundary>(fixture.ledger);
  boundary->block_request = true;
  auto coordinator = fixture.coordinator(boundary);
  admit(*coordinator, *fixture.ledger);
  const auto scope7 = attribution("journal-inflight", "run-inflight",
                                  "lease-inflight", 7U);
  fixture.logical_fences->set_live(scope7);
  auto peer7 = std::make_shared<FakePeer>(HostdSessionAccess::grant_release);
  const auto session7 = coordinator->connect({.attribution = scope7}, peer7);
  std::atomic<bool> grant_finished{false};
  std::thread grant_thread([&] {
    const auto result = coordinator->request_bundle(
        session7.session_id, request_for(scope7, "inflight-request"),
        {10, 20});
    grant_finished = result.grant.has_value();
  });
  boundary->wait_for_request();
  peer7->alive = false;
  require(coordinator->reap_dead_sessions(64U) == 0U,
          "reaper never removes a session with an in-flight mutation");
  auto scope8 = scope7;
  scope8.logical_fencing_token = 8U;
  fixture.logical_fences->set_live(scope8);
  auto peer8 = std::make_shared<FakePeer>(HostdSessionAccess::grant_release);
  require_throws<HostdStateError>(
      [&] { (void)coordinator->connect({.attribution = scope8}, peer8); },
      "higher fence cannot overtake an already reserved grant callback");
  boundary->unblock_request();
  grant_thread.join();
  require(grant_finished && coordinator->reap_dead_sessions(64U) == 1U,
          "completed dead mutation session is safely reclaimed");
  require(coordinator->connect({.attribution = scope8}, peer8)
              .effective_access == HostdSessionAccess::grant_release,
          "higher fence admits after the older grant reservation completes");
}

void malformed_ledger_outputs_poison_exactly() {
  const std::vector<AdversarialLedgerBoundary::GrantFault> grant_faults{
      AdversarialLedgerBoundary::GrantFault::api_version,
      AdversarialLedgerBoundary::GrantFault::request_id,
      AdversarialLedgerBoundary::GrantFault::request_digest,
      AdversarialLedgerBoundary::GrantFault::journal_id,
      AdversarialLedgerBoundary::GrantFault::logical_fence,
      AdversarialLedgerBoundary::GrantFault::fence,
      AdversarialLedgerBoundary::GrantFault::receipt_digest,
  };
  for (const auto fault : grant_faults) {
    Fixture fixture;
    auto boundary =
        std::make_shared<AdversarialLedgerBoundary>(fixture.ledger);
    auto coordinator = fixture.coordinator(boundary);
    admit(*coordinator, *fixture.ledger);
    const auto scope = attribution("journal-bad-grant", "run-bad-grant",
                                   "lease-bad-grant", 1U);
    const auto session =
        connect_admission(*coordinator, scope, *fixture.logical_fences);
    boundary->grant_fault = fault;
    require_throws<HostdStateError>(
        [&] {
          (void)coordinator->request_bundle(
              session.session_id, request_for(scope, "bad-grant"), {10, 20});
        },
        "every malformed grant field poisons the coordinator boundary");
    require(coordinator->status().lifecycle == HostdLifecycle::poisoned,
            "malformed grant leaves a visible poison latch");
  }

  const std::vector<AdversarialLedgerBoundary::ReleaseFault> release_faults{
      AdversarialLedgerBoundary::ReleaseFault::api_version,
      AdversarialLedgerBoundary::ReleaseFault::release_request_id,
      AdversarialLedgerBoundary::ReleaseFault::release_request_digest,
      AdversarialLedgerBoundary::ReleaseFault::allocation_id,
      AdversarialLedgerBoundary::ReleaseFault::grant_digest,
      AdversarialLedgerBoundary::ReleaseFault::receipt_digest,
  };
  for (const auto fault : release_faults) {
    Fixture fixture;
    auto boundary =
        std::make_shared<AdversarialLedgerBoundary>(fixture.ledger);
    auto coordinator = fixture.coordinator(boundary);
    admit(*coordinator, *fixture.ledger);
    const auto scope = attribution("journal-bad-release", "run-bad-release",
                                   "lease-bad-release", 1U);
    const auto session =
        connect_admission(*coordinator, scope, *fixture.logical_fences);
    const auto grant = coordinator->request_bundle(
        session.session_id, request_for(scope, "release-source"), {10, 20});
    require(grant.grant.has_value(), "release fault test obtains an allocation");
    boundary->release_fault = fault;
    require_throws<HostdStateError>(
        [&] {
          (void)coordinator->release_bundle(
              session.session_id,
              release_for(*grant.grant, "bad-release"), {30, 40});
        },
        "every malformed release field poisons the coordinator boundary");
    require(coordinator->status().lifecycle == HostdLifecycle::poisoned,
            "malformed release leaves a visible poison latch");
  }
}

} // namespace

int main() {
  try {
    lifecycle_and_pre_audit_gate();
    failed_policy_mismatch_and_malformed_audits_fail_closed();
    concurrent_startup_audits_are_single_flight();
    stale_startup_report_is_rejected_by_ledger_cas();
    crash_after_commit_retries_by_exact_ledger_replay();
    blocked_startup_allows_only_exact_release_cleanup();
    blocked_startup_recovers_stale_exact_outcomes_read_only();
    admission_finalize_boundary_is_atomic_and_recoverable();
    admission_finalize_lost_reply_and_async_poison_recover();
    unauthorized_and_read_only_peers();
    disconnected_sessions_cannot_replay();
    concurrency_keys_are_independent_logical_scopes();
    stale_fences_and_attribution_mismatch_fail();
    wrong_host_boot_and_broker_poison();
    ledger_request_and_release_replay_are_exact();
    concurrent_sessions_and_journals_share_one_ledger();
    durable_fence_authority_and_cleanup_reconnect();
    advancing_fence_preserves_exact_connected_cleanup();
    restart_peer_liveness_capacity_and_status_poison();
    asynchronous_startup_poison_is_never_overwritten();
    callbacks_reenter_without_coordinator_mutex();
    dead_session_reaping_and_inflight_fence_reservation();
    malformed_ledger_outputs_poison_exactly();
    std::cout << "hostd tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "hostd test failure: " << error.what() << '\n';
    return 1;
  }
}
