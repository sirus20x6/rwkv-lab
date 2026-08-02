#include "trainvm/hostd_transport.hpp"
#include "trainvm/document.hpp"
#include "trainvm/hostd_process_client.hpp"
#include "trainvm/hostd_resource_client.hpp"
#include "trainvm/worker_bootstrap.hpp"

#include <fcntl.h>
#include <openssl/sha.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

using namespace trainvm;

// Shared with authority_fuzz_tests so one knob widens every generated sweep.
std::uint64_t fuzz_rounds() {
  const char *value = std::getenv("TRAINVM_FUZZ_ROUNDS");
  if (value == nullptr || *value == '\0') return 1U;
  try {
    return std::max<std::uint64_t>(1U, std::stoull(value));
  } catch (const std::exception &) {
    return 1U;
  }
}

void require(bool condition, std::string_view message) {
  if (!condition)
    throw std::runtime_error(std::string(message));
}

template <typename Exception, typename Callable>
void require_throws(Callable &&callable, std::string_view message) {
  try {
    std::forward<Callable>(callable)();
  } catch (const Exception &) {
    return;
  }
  throw std::runtime_error(std::string(message));
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    std::array<char, 64U> pattern{};
    constexpr std::string_view source = "/tmp/trainvm-hostd-transport-XXXXXX";
    std::ranges::copy(source, pattern.begin());
    char *created = ::mkdtemp(pattern.data());
    require(created != nullptr, "create transport temporary directory");
    path_ = created;
    require(::chmod(path_.c_str(), 0700) == 0,
            "protect transport temporary directory");
    parent_fd_ =
        ::open(path_.c_str(), O_PATH | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    require(parent_fd_ >= 0, "pin transport temporary directory");
  }

  ~TemporaryDirectory() {
    if (parent_fd_ >= 0)
      (void)::close(parent_fd_);
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  TemporaryDirectory(const TemporaryDirectory &) = delete;
  TemporaryDirectory &operator=(const TemporaryDirectory &) = delete;

  [[nodiscard]] const std::filesystem::path &path() const { return path_; }
  [[nodiscard]] int parent_fd() const { return parent_fd_; }

private:
  std::filesystem::path path_;
  int parent_fd_{-1};
};

class HeldToken final : public IHostdSingletonToken {
public:
  bool attest_held() const override { return held; }
  bool held{true};
};

class ThrowAtBindCheckpoint final : public IHostdSocketBindFaultInjector {
public:
  explicit ThrowAtBindCheckpoint(HostdSocketBindCheckpoint selected)
      : selected_(selected) {}

  void checkpoint(HostdSocketBindCheckpoint checkpoint) override {
    if (checkpoint == selected_)
      throw std::runtime_error("injected bind checkpoint failure");
  }

private:
  HostdSocketBindCheckpoint selected_;
};

class ObserveBlockedSignals final : public IHostdSocketBindFaultInjector {
public:
  void checkpoint(HostdSocketBindCheckpoint checkpoint) override {
    if (checkpoint != HostdSocketBindCheckpoint::signals_blocked)
      return;
    sigset_t current{};
    if (::pthread_sigmask(SIG_SETMASK, nullptr, &current) != 0)
      throw std::runtime_error("could not observe checkpoint signal mask");
    observed = ::sigismember(&current, SIGUSR1) == 1 &&
               ::sigismember(&current, SIGTERM) == 1;
  }

  bool observed{};
};

HostdSocketAuthorityConfig socket_config(const TemporaryDirectory &directory,
                                         std::string name = "hostd.sock") {
  return {.api_version = std::string(kHostdStatusTransportApiVersion),
          .socket_path = directory.path() / std::move(name),
          .expected_owner_uid = ::geteuid(),
          .expected_owner_gid = ::getegid(),
          .expected_parent_mode = 0700U,
          .expected_socket_mode = 0600U,
          .listen_backlog = 8U,
          .enforcement_grade =
              HostdSocketEnforcementGrade::cooperative_test,
          .fault_injector = nullptr};
}

HostResourceId mutex_id() {
  return {.kind = HostResourceKind::host_mutex,
          .vendor = std::nullopt,
          .stable_id = "host-mutex:transport",
          .parent_id = std::nullopt};
}

HostInventoryReceipt inventory() {
  HostKernelSnapshot snapshot{
      .api_version = std::string(kHostInventoryApiVersion),
      .host_id = "sha256:" + std::string(64U, 'a'),
      .boot_id = "11111111-2222-4333-8444-555555555555",
      .broker_epoch = "broker-transport",
      .begin_revision = "revision-transport",
      .end_revision = "revision-transport",
      .probes = {},
      .resources = {{.id = mutex_id(),
                     .disposition =
                         ResourceObservationDisposition::audited_eligible,
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
                     .labels = {{"scope", "transport-test"}}}},
  };
  FakeHostKernel kernel(
      {{.snapshot = std::move(snapshot), .failure = std::nullopt}});
  return capture_host_inventory(kernel);
}

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
        .code = "transport.test.blocking",
        .subject = "startup",
        .detail = "deterministic blocking transport evidence",
        .evidence_digest = {},
    }));
  }
  return canonicalize_host_startup_audit_report({
      .api_version = std::string(kHostStartupAuditReportApiVersion),
      .audit_id =
          "transport-audit-" + std::to_string(next_audit_id.fetch_add(1U)),
      .host_id = observed.host_id,
      .boot_id = observed.boot_id,
      .broker_epoch = observed.broker_epoch,
      .broker_instance_id = "transport-test-instance",
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

class Auditor final : public IConfiguredHostStartupAuditorV2 {
public:
  explicit Auditor(HostStartupAuditReport value) : report(std::move(value)) {}

  HostStartupAuditReport audit() override { return report; }
  HostStartupAuditReport report;
};

class BlockingAuditor final : public IConfiguredHostStartupAuditorV2 {
public:
  explicit BlockingAuditor(HostStartupAuditReport report)
      : report_(std::move(report)) {}

  HostStartupAuditReport audit() override {
    std::unique_lock lock(mutex_);
    entered_ = true;
    changed_.notify_all();
    changed_.wait(lock, [&] { return released_; });
    return report_;
  }

  void wait_until_entered() {
    std::unique_lock lock(mutex_);
    changed_.wait(lock, [&] { return entered_; });
  }

  void release() {
    std::scoped_lock lock(mutex_);
    released_ = true;
    changed_.notify_all();
  }

private:
  HostStartupAuditReport report_;
  std::mutex mutex_;
  std::condition_variable changed_;
  bool entered_{};
  bool released_{};
};

class InvalidTextAuditor final : public IConfiguredHostStartupAuditorV2 {
public:
  HostStartupAuditReport audit() override {
    throw std::runtime_error(std::string("invalid-") + char(0x01));
  }
};

struct CoordinatorFixture final {
  explicit CoordinatorFixture(
      const TemporaryDirectory &directory,
      std::shared_ptr<IHostdLogicalFenceEvidenceSource> logical = nullptr)
      : observed(inventory()),
        authority(std::make_shared<HostLedgerFilesystemAuthority>(
            HostLedgerFilesystemAuthority::acquire(
                {.api_version = std::string(kHostLedgerAuthorityApiVersion),
                 .ledger_path = directory.path() / "ledger.db",
                 .expected_owner_uid = ::geteuid(),
                 .expected_owner_gid = ::getegid(),
                 .enforcement_grade =
                     HostLedgerEnforcementGrade::cooperative_test}))),
        ledger(std::make_shared<SQLiteHostLedger>(
            authority, observed, nullptr, startup_audit_policy())),
        coordinator(std::make_shared<HostGrantCoordinator>(
            HostdCoordinatorConfig{
                .api_version = std::string(kHostdCoordinatorApiVersion),
                .host_id = observed.host_id,
                .boot_id = observed.boot_id,
                .broker_epoch = observed.broker_epoch,
                .maximum_live_sessions = 8U,
                .maximum_logical_scopes = 8U},
            ledger, std::move(logical))) {}

  void admit() {
    Auditor auditor(audit_report(*ledger));
    (void)coordinator->run_startup_audit(auditor, {30, 40});
  }

  HostInventoryReceipt observed;
  std::shared_ptr<HostLedgerFilesystemAuthority> authority;
  std::shared_ptr<SQLiteHostLedger> ledger;
  std::shared_ptr<HostGrantCoordinator> coordinator;
};

constexpr std::string_view kMutationEvidenceDigest =
    "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

HostdSessionChallengeClaim mutation_claim() {
  return {
      .journal = {.directory_path = "/var/lib/trainvm/journals",
                  .journal_name = "transport.db",
                  .authority_name = "transport.lock",
                  .journal_id = "journal-transport",
                  .directory_device = 11U,
                  .directory_inode = 12U,
                  .journal_device = 11U,
                  .journal_inode = 13U,
                  .authority_device = 11U,
                  .authority_inode = 14U,
                  .owner_uid = static_cast<std::uint64_t>(::geteuid())},
      .controller = {.run_id = "run-transport",
                     .concurrency_key = "gpu:transport",
                     .controller_id = "controller-transport",
                     .controller_generation = 1U,
                     .logical_lease_id = "lease-transport",
                     .logical_fencing_token = 1U},
  };
}

HostdSessionAttribution mutation_attribution() {
  const auto value = mutation_claim();
  return {.journal_id = value.journal.journal_id,
          .run_id = value.controller.run_id,
          .concurrency_key = value.controller.concurrency_key,
          .logical_lease_id = value.controller.logical_lease_id,
          .logical_fencing_token = value.controller.logical_fencing_token};
}

class MutationLogicalFence final : public IHostdLogicalFenceEvidenceSource {
public:
  HostdLogicalFenceEvidence
  attest(const HostdSessionAttribution &attribution) override {
    ++calls;
    return {.api_version = std::string(kHostdLogicalFenceEvidenceApiVersion),
            .attribution = attribution,
            .live = attribution == mutation_attribution() && live,
            .cleanup_authorized = false,
            .cleanup_allocation_id = {},
            .cleanup_grant_digest = {},
            .cleanup_release_request_digest = {},
            .evidence_digest = std::string(kMutationEvidenceDigest)};
  }

  bool live{true};
  std::size_t calls{};
};

class MutationNonce final : public IHostdSessionChallengeNonceSource {
public:
  std::string next_hex_256(std::string_view) override {
    constexpr std::string_view digits = "0123456789abcdef";
    ++counter;
    return std::string(64U, digits.at(counter % digits.size()));
  }
  std::size_t counter{};
};

class MutationChallengeTime final : public IHostdSessionChallengeTimeSource {
public:
  HostdSessionChallengeTime now() override {
    return {.boot_id = "11111111-2222-4333-8444-555555555555",
            .boottime_ns = now_ns};
  }
  std::int64_t now_ns{1'000'000'000LL};
};

class MutationJournalAttestor final : public IHostdJournalFenceAttestor {
public:
  HostdJournalFenceEvidence
  attest(const HostdJournalFenceQuery &query) override {
    ++calls;
    return hostd_seal_journal_fence_evidence({
        .api_version = std::string(kHostdJournalFenceEvidenceApiVersion),
        .challenge_id = query.challenge_id,
        .session_nonce = query.session_nonce,
        .host_id = query.host_id,
        .boot_id = query.boot_id,
        .broker_epoch = query.broker_epoch,
        .observed_boottime_ns = 1'000'000'000LL,
        .journal = query.claim.journal,
        .controller = query.claim.controller,
        .live = true,
        .evidence_digest = {},
    });
  }
  std::size_t calls{};
};

class MutationLinuxKernel final : public IHostdLinuxSessionKernel {
public:
  HostdLinuxSessionEnforcementGrade enforcement_grade() const override {
    return HostdLinuxSessionEnforcementGrade::
        cooperative_namespace_observation;
  }
  HostdLinuxRandomRead getrandom_bytes(void *, std::size_t) override {
    return {.count = -1, .error_number = ENOSYS};
  }
  HostdLinuxClockRead clock_boottime() override {
    return {.success = true,
            .value = {.tv_sec = 1, .tv_nsec = 0},
            .error_number = 0};
  }
  HostdLinuxBootIdRead read_boot_id() override {
    return {.success = true,
            .value = "11111111-2222-4333-8444-555555555555",
            .error_number = 0};
  }
  HostdLinuxPeerKernelObservation observe_process(pid_t pid) override {
    ++observations;
    return {.pid = pid,
            .effective_uid_before = ::geteuid(),
            .effective_gid_before = ::getegid(),
            .effective_uid_after = ::geteuid(),
            .effective_gid_after = ::getegid(),
            .process_starttime_ticks_before = 12345U,
            .process_starttime_ticks_after = 12345U,
            .process_directory_device_before = 21U,
            .process_directory_inode_before = 22U,
            .process_directory_device_after = 21U,
            .process_directory_inode_after = 22U,
            .pidfd_available = false,
            .pidfd_alive_before = true,
            .pidfd_alive_after = true,
            .complete = true,
            .error_number = 0};
  }
  std::size_t observations{};
};

class MutationServiceAuthority final
    : public IHostdMutationServiceIdentityAuthority {
public:
  HostdMutationServiceAuthorization
  authorize(const HostdSocketPeerInstance &peer) override {
    ++calls;
    last_peer = peer;
    return {.service_identity = "trainvm.transport.test.service",
            .access = access,
            .service_identity_enforced = enforced};
  }
  HostdSessionAccess access{HostdSessionAccess::grant_release};
  bool enforced{true};
  std::size_t calls{};
  HostdSocketPeerInstance last_peer;
};

class MutationLedgerTime final : public IHostdLedgerTimeSource {
public:
  HostLedgerTime now() override {
    ++calls;
    return {.boottime_ns = 100 + static_cast<std::int64_t>(calls) * 10,
            .wall_time_ns = 200 + static_cast<std::int64_t>(calls) * 10};
  }
  std::size_t calls{};
};

class MutationFaultInjector final
    : public IHostdMutationTransportFaultInjector {
public:
  void checkpoint(HostdMutationTransportCheckpoint checkpoint) override {
    ++visits;
    if (selected && *selected == checkpoint) {
      selected.reset();
      throw std::runtime_error("injected mutation transport interruption");
    }
  }
  std::optional<HostdMutationTransportCheckpoint> selected;
  std::size_t visits{};
};

class LedgerProcessSupervisor final : public IHostdProcessSupervisor {
 public:
  explicit LedgerProcessSupervisor(SQLiteHostLedger& ledger)
      : ledger_(ledger) {}

  HostdProcessPreparedResult prepare(
      const HostdProcessPrepareRequest& request, int executable_fd,
      std::optional<int> code_fd, int working_directory_fd,
      int worker_bootstrap_fd) override {
    require(executable_fd >= 0 && !code_fd && working_directory_fd >= 0 &&
                worker_bootstrap_fd >= 0 &&
                ::fcntl(executable_fd, F_GETFD) >= 0 &&
                ::fcntl(working_directory_fd, F_GETFD) >= 0 &&
                worker_bootstrap_from_sealed_fd(
                    worker_bootstrap_fd,
                    request.worker_bootstrap_digest).run_id ==
                    request.launch.identity.run_id,
            "process supervisor receives three live role-ordered descriptors");
    ++prepare_calls;
    const auto intended = ledger_.commit_process_launch_intent(
        seal_host_process_launch_request({
            .api_version =
                std::string(kHostProcessLaunchRequestApiVersion),
            .launch_id = request.launch.identity.launch_event_id,
            .allocation_id = request.grant.allocation_id,
            .grant_digest = request.grant.receipt_digest,
            .journal_id = request.grant.journal_id,
            .run_id = request.grant.run_id,
            .logical_lease_id = request.grant.logical_lease_id,
            .logical_fencing_token =
                request.grant.logical_fencing_token,
            .resolved_launch_digest = hostd_bound_process_launch_digest(
                request.launch, request.worker_bootstrap_digest,
                request.process_policy),
            .executable_path = request.launch.identity.executable.source_path,
            .executable_digest =
                request.launch.identity.executable.sealed_sha256,
            .cgroup_path = "/trainvm/test-allocation",
            .cgroup_device = 91U,
            .cgroup_inode = 92U,
            .worker_credentials = std::nullopt,
            .device_policy = std::nullopt,
            .process_policy = std::nullopt,
            .canonical_request_digest = {},
        }),
        {500, 600});
    const auto spawned = ledger_.commit_process_spawn(
        seal_host_process_spawn_request({
            .api_version =
                std::string(kHostProcessSpawnRequestApiVersion),
            .launch_id = request.launch.identity.launch_event_id,
            .launch_intent_digest = intended.intent.receipt_digest,
            .host_pid = ::getpid(),
            .process_starttime_ticks = 12345U,
            .boot_id = request.grant.boot_id,
            .cgroup_path = "/trainvm/test-allocation",
            .cgroup_device = 91U,
            .cgroup_inode = 92U,
            .executable_digest =
                request.launch.identity.executable.sealed_sha256,
            .worker_credentials = std::nullopt,
            .device_policy = std::nullopt,
            .process_policy = std::nullopt,
            .canonical_request_digest = {},
        }),
        {510, 610});
    prepared = HostdProcessPreparedResult{
        .api_version = std::string(kHostdProcessPreparedApiVersion),
        .intent = intended.intent,
        .spawn = spawned.receipt,
        .replayed = intended.replayed || spawned.replayed,
    };
    return *prepared;
  }

  HostdProcessCommittedResult commit(
      const HostdProcessCommitRequest& request) override {
    require(prepared &&
                request.launch_id == prepared->spawn.request.launch_id &&
                request.spawn_receipt_digest == prepared->spawn.receipt_digest,
            "process commit binds the prepared spawn receipt");
    ++commit_calls;
    const bool replayed = committed;
    committed = true;
    return {.api_version =
                std::string(kHostdProcessCommittedApiVersion),
            .launch_id = request.launch_id,
            .spawn_receipt_digest = request.spawn_receipt_digest,
            .released_to_exec = true,
            .replayed = replayed};
  }

  HostProcessExitResult finalize(
      const HostdProcessExitCommand& command) override {
    require(prepared && committed &&
                command.launch_id == prepared->spawn.request.launch_id &&
                command.spawn_receipt_digest == prepared->spawn.receipt_digest,
            "process exit binds the committed spawn receipt");
    ++exit_calls;
    return ledger_.commit_process_exit(
        seal_host_process_exit_request({
            .api_version = std::string(kHostProcessExitRequestApiVersion),
            .exit_request_id = command.exit_request_id,
            .launch_id = command.launch_id,
            .spawn_receipt_digest = command.spawn_receipt_digest,
            .host_pid = prepared->spawn.request.host_pid,
            .process_starttime_ticks =
                prepared->spawn.request.process_starttime_ticks,
            .wait_code = CLD_EXITED,
            .wait_status = 0,
            .cgroup_path = prepared->spawn.request.cgroup_path,
            .cgroup_device = prepared->spawn.request.cgroup_device,
            .cgroup_inode = prepared->spawn.request.cgroup_inode,
            .cgroup_empty = true,
            .accelerator_contexts_empty = true,
            .context_audit_digest = std::string(kMutationEvidenceDigest),
            .canonical_request_digest = {},
        }),
        {520, 620});
  }

  SQLiteHostLedger& ledger_;
  std::optional<HostdProcessPreparedResult> prepared;
  bool committed{};
  std::size_t prepare_calls{};
  std::size_t commit_calls{};
  std::size_t exit_calls{};
};

ResolvedLaunchSpec process_launch_for(const ResourceBundleGrant& grant) {
  const std::string executable_digest =
      "sha256:" + std::string(64U, 'e');
  ResolvedLaunchSpec spec{
      .identity = {
          .api_version = "trainvm.resolved-launch/v4",
          .launch_event_id = grant.run_id + ":worker-launch:node:attempt",
          .run_id = grant.run_id,
          .node_id = "node",
          .attempt_id = "attempt",
          .launch_nonce = "nonce",
          .adapter_key = {.adapter = "transport-native",
                          .version = "1",
                          .runtime = ComponentRuntime::native_worker,
                          .operation = "train",
                          .contract = "trainvm.worker/v1"},
          .code_fingerprint = executable_digest,
          .bootstrap_runtime_closure_fingerprint =
              "sha256:" + std::string(64U, 'd'),
          .required_capabilities = {},
          .provided_capabilities = {},
          .host_registry_digest =
              "sha256:" + std::string(64U, 'b'),
          .host_profile_digest =
              "sha256:" + std::string(64U, 'c'),
          .concurrency_key = mutation_claim().controller.concurrency_key,
          .lease_id = grant.logical_lease_id,
          .fencing_token = grant.logical_fencing_token,
          .host_grant = HostLaunchGrantClaim{
              .request_id = grant.request_id,
              .grant_digest = grant.receipt_digest,
              .fences = grant.fences},
          .host = {.host_id = grant.host_id, .boot_id = grant.boot_id},
          .executable = {.source_path = "/sealed/trainvm-worker",
                         .source_device = 1U,
                         .source_inode = 2U,
                         .source_size = 1U,
                         .source_mode =
                             static_cast<std::uint32_t>(S_IFREG | 0500),
                         .source_uid = 0U,
                         .source_gid = 0U,
                         .sealed_sha256 = executable_digest},
          .code = std::nullopt,
          .public_arguments = {"--transport-test"},
          .working_directory = {
              .source_path = "/var/lib/trainvm",
              .device = 3U,
              .inode = 4U,
              .mode = static_cast<std::uint32_t>(S_IFDIR | 0700),
              .uid = 0U,
              .gid = 0U}},
      .spec_digest = {},
  };
  spec.spec_digest = "sha256:" +
                     sha256_hex(
                         resolved_launch_identity_json(spec.identity).dump());
  return resolved_launch_spec_from_json(resolved_launch_spec_json(spec));
}

HostdStatusClientConfig client_config(HostdSocketAuthority &authority) {
  return {.socket_path = authority.socket_path(),
          .expected_endpoint = authority.reattest(),
          .expected_server_uid = ::geteuid(),
          .expected_server_gid = ::getegid(),
          .maximum_payload_bytes = kHostdStatusMaximumPayloadBytes};
}

std::int64_t deadline(std::int64_t delta_ns = 2'000'000'000LL) {
  return hostd_monotonic_now_ns() + delta_ns;
}

HostdStatusPeerPolicy peer_policy() {
  return {.allowed_uid = ::geteuid(), .allowed_gid = ::getegid()};
}

int create_raw_listener(const HostdSocketAuthorityConfig &config) {
  const int descriptor =
      ::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
  require(descriptor >= 0, "create raw replacement listener");
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  const std::string path = config.socket_path.string();
  require(path.size() < sizeof(address.sun_path), "raw listener path fits");
  std::memcpy(address.sun_path, path.data(), path.size());
  address.sun_path[path.size()] = '\0';
  const socklen_t length = static_cast<socklen_t>(
      offsetof(sockaddr_un, sun_path) + path.size() + 1U);
  require(::bind(descriptor, reinterpret_cast<const sockaddr *>(&address),
                 length) == 0,
          "bind raw replacement listener");
  require(::chmod(path.c_str(), config.expected_socket_mode) == 0,
          "protect raw listener path");
  require(::listen(descriptor, static_cast<int>(config.listen_backlog)) == 0,
          "listen on raw replacement listener");
  return descriptor;
}

void put_u16(std::vector<std::byte> &bytes, std::size_t offset,
             std::uint16_t value) {
  bytes.at(offset) = std::byte((value >> 8U) & 0xffU);
  bytes.at(offset + 1U) = std::byte(value & 0xffU);
}

void put_u32(std::vector<std::byte> &bytes, std::size_t offset,
             std::uint32_t value) {
  for (std::size_t index = 0U; index < 4U; ++index)
    bytes.at(offset + index) =
        std::byte((value >> (24U - static_cast<unsigned int>(index) * 8U)) &
                  0xffU);
}

void put_u64(std::vector<std::byte> &bytes, std::size_t offset,
             std::uint64_t value) {
  for (std::size_t index = 0U; index < 8U; ++index)
    bytes.at(offset + index) =
        std::byte((value >> (56U - static_cast<unsigned int>(index) * 8U)) &
                  0xffU);
}

std::vector<std::byte> packet_with_payload(std::uint16_t opcode,
                                           std::uint64_t correlation,
                                           nlohmann::json payload) {
  const std::string canonical = payload.dump();
  std::vector<std::byte> bytes(kHostdStatusWireHeaderBytes + canonical.size());
  bytes[0] = std::byte{'T'};
  bytes[1] = std::byte{'V'};
  bytes[2] = std::byte{'H'};
  bytes[3] = std::byte{'D'};
  put_u16(bytes, 4U, kHostdStatusWireVersion);
  put_u16(bytes, 6U,
          static_cast<std::uint16_t>(kHostdStatusWireHeaderBytes));
  put_u16(bytes, 8U, opcode);
  put_u16(bytes, 10U, 0U);
  put_u32(bytes, 12U, static_cast<std::uint32_t>(canonical.size()));
  put_u64(bytes, 16U, correlation);
  std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
  (void)::SHA256(reinterpret_cast<const unsigned char *>(canonical.data()),
                 canonical.size(), digest.data());
  for (std::size_t index = 0U; index < digest.size(); ++index)
    bytes[24U + index] = std::byte(digest[index]);
  std::memcpy(bytes.data() + kHostdStatusWireHeaderBytes, canonical.data(),
              canonical.size());
  return bytes;
}

nlohmann::json status_payload(const HostdCoordinatorStatus &status) {
  nlohmann::json audit = nullptr;
  if (status.startup_audit) {
    const auto &value = *status.startup_audit;
    const auto head = [](const HostLedgerChainHead &ledger_head) {
      return nlohmann::json{{"chain_hash", ledger_head.chain_hash},
                            {"ledger_sequence",
                             ledger_head.ledger_sequence}};
    };
    audit = {{"api_version", value.api_version},
             {"audit_id", value.audit_id},
             {"boot_id", value.boot_id},
             {"broker_epoch", value.broker_epoch},
             {"broker_instance_id", value.broker_instance_id},
             {"commit_record_digest", value.commit_record_digest},
             {"committed_boottime_ns", value.committed_boottime_ns},
             {"committed_ledger_head", head(value.committed_ledger_head)},
             {"committed_wall_time_ns", value.committed_wall_time_ns},
             {"disposition",
              value.disposition == HostStartupAuditDisposition::passed
                  ? "passed"
                  : "failed"},
             {"findings_digest", value.findings_digest},
             {"host_id", value.host_id},
             {"inventory_digest", value.inventory_digest},
             {"ledger_head_before", head(value.ledger_head_before)},
             {"policy_digest", value.policy_digest},
             {"post_occupancy_digest", value.post_occupancy_digest},
             {"pre_occupancy_digest", value.pre_occupancy_digest},
             {"receipt_digest", value.receipt_digest},
             {"report_digest", value.report_digest},
             {"topology_digest", value.topology_digest}};
  }
  const auto lifecycle = [&] {
    switch (status.lifecycle) {
    case HostdLifecycle::sealed:
      return "sealed";
    case HostdLifecycle::startup_auditing:
      return "startup_auditing";
    case HostdLifecycle::startup_blocked:
      return "startup_blocked";
    case HostdLifecycle::admitting:
      return "admitting";
    case HostdLifecycle::poisoned:
      return "poisoned";
    }
    return "invalid";
  }();
  return {{"admission_counts_are_cached_evidence",
           status.admission_counts_are_cached_evidence},
          {"admission_sessions", status.admission_sessions},
          {"api_version", status.api_version},
          {"boot_id", status.boot_id},
          {"broker_epoch", status.broker_epoch},
          {"host_id", status.host_id},
          {"inventory_digest", status.inventory_digest},
          {"lifecycle", lifecycle},
          {"live_sessions", status.live_sessions},
          {"poison_reason", status.poison_reason},
          {"release_only_sessions", status.release_only_sessions},
          {"stale_admission_sessions", status.stale_admission_sessions},
          {"startup_audit", std::move(audit)}};
}

void serve_fake_response(const HostdSocketAuthority &authority,
                         const std::vector<std::byte> &response,
                         std::optional<int> passed_fd = std::nullopt) {
  pollfd readiness{.fd = authority.listener_fd(),
                   .events = POLLIN,
                   .revents = 0};
  require(::poll(&readiness, 1U, 2000) == 1 &&
              (readiness.revents & POLLIN) != 0,
          "fake server observes client connection");
  const int accepted = ::accept4(authority.listener_fd(), nullptr, nullptr,
                                 SOCK_CLOEXEC);
  require(accepted >= 0, "fake server accepts client");
  std::array<std::byte, 4096U> request{};
  require(::recv(accepted, request.data(), request.size(), 0) > 0,
          "fake server receives request");
  iovec vector{.iov_base = const_cast<std::byte *>(response.data()),
               .iov_len = response.size()};
  alignas(cmsghdr)
      std::array<std::byte,
                 CMSG_SPACE(sizeof(ucred)) + CMSG_SPACE(sizeof(int))>
          control{};
  msghdr message{};
  message.msg_iov = &vector;
  message.msg_iovlen = 1U;
  message.msg_control = control.data();
  message.msg_controllen = passed_fd ? control.size()
                                     : CMSG_SPACE(sizeof(ucred));
  cmsghdr *credentials_header = CMSG_FIRSTHDR(&message);
  require(credentials_header != nullptr, "construct response credentials");
  credentials_header->cmsg_level = SOL_SOCKET;
  credentials_header->cmsg_type = SCM_CREDENTIALS;
  credentials_header->cmsg_len = CMSG_LEN(sizeof(ucred));
  const ucred credentials{.pid = ::getpid(),
                          .uid = ::geteuid(),
                          .gid = ::getegid()};
  std::memcpy(CMSG_DATA(credentials_header), &credentials,
              sizeof(credentials));
  if (passed_fd) {
    cmsghdr *rights = CMSG_NXTHDR(&message, credentials_header);
    require(rights != nullptr, "construct response rights");
    rights->cmsg_level = SOL_SOCKET;
    rights->cmsg_type = SCM_RIGHTS;
    rights->cmsg_len = CMSG_LEN(sizeof(int));
    std::memcpy(CMSG_DATA(rights), &*passed_fd, sizeof(int));
  }
  require(::sendmsg(accepted, &message, MSG_NOSIGNAL) ==
              static_cast<ssize_t>(response.size()),
          "fake server sends response");
  require(::close(accepted) == 0, "close fake accepted socket");
}

int connect_raw(const std::filesystem::path &path) {
  const int descriptor =
      ::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
  require(descriptor >= 0, "create raw transport client");
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  const std::string native = path.string();
  std::memcpy(address.sun_path, native.data(), native.size());
  address.sun_path[native.size()] = '\0';
  const socklen_t length = static_cast<socklen_t>(
      offsetof(sockaddr_un, sun_path) + native.size() + 1U);
  require(::connect(descriptor, reinterpret_cast<const sockaddr *>(&address),
                    length) == 0,
          "connect raw transport client");
  return descriptor;
}

HostdServeResult send_raw(HostdStatusServer &server,
                          const std::filesystem::path &path,
                          const std::vector<std::byte> &packet,
                          std::optional<int> passed_fd = std::nullopt,
                          bool expect_silence = false) {
  HostdServeResult outcome = HostdServeResult::timed_out;
  std::jthread serving([&] { outcome = server.serve_one(deadline()); });
  const int client = connect_raw(path);
  alignas(cmsghdr)
      std::array<std::byte,
                 CMSG_SPACE(sizeof(ucred)) + CMSG_SPACE(sizeof(int))>
          control{};
  iovec vector{.iov_base = const_cast<std::byte *>(packet.data()),
               .iov_len = packet.size()};
  msghdr message{};
  message.msg_iov = &vector;
  message.msg_iovlen = 1U;
  message.msg_control = control.data();
  message.msg_controllen = passed_fd ? control.size()
                                     : CMSG_SPACE(sizeof(ucred));
  cmsghdr *header = CMSG_FIRSTHDR(&message);
  require(header != nullptr, "construct SCM_CREDENTIALS header");
  header->cmsg_level = SOL_SOCKET;
  header->cmsg_type = SCM_CREDENTIALS;
  header->cmsg_len = CMSG_LEN(sizeof(ucred));
  const ucred credentials{.pid = ::getpid(),
                          .uid = ::geteuid(),
                          .gid = ::getegid()};
  std::memcpy(CMSG_DATA(header), &credentials, sizeof(credentials));
  if (passed_fd) {
    cmsghdr *rights = CMSG_NXTHDR(&message, header);
    require(rights != nullptr, "construct SCM_RIGHTS header");
    rights->cmsg_level = SOL_SOCKET;
    rights->cmsg_type = SCM_RIGHTS;
    rights->cmsg_len = CMSG_LEN(sizeof(int));
    std::memcpy(CMSG_DATA(rights), &*passed_fd, sizeof(int));
  }
  require(::sendmsg(client, &message, MSG_NOSIGNAL) ==
              static_cast<ssize_t>(packet.size()),
          "send raw transport packet");
  serving.join();
  if (expect_silence) {
    std::array<std::byte, 256U> response{};
    const ssize_t received =
        ::recv(client, response.data(), response.size(), MSG_DONTWAIT);
    require(received == 0 ||
                (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)),
            "configured response bound suppresses an oversized typed error");
  }
  require(::close(client) == 0, "close raw transport client");
  return outcome;
}

std::size_t open_fd_count() {
  std::size_t count = 0U;
  for (const auto &entry : std::filesystem::directory_iterator("/proc/self/fd")) {
    (void)entry;
    ++count;
  }
  return count;
}

void authority_requires_external_singleton_and_pins_path() {
  TemporaryDirectory directory;
  auto missing = std::make_shared<HeldToken>();
  missing->held = false;
  require_throws<HostdTransportError>(
      [&] {
        (void)HostdSocketAuthority::self_bind(
            socket_config(directory), directory.parent_fd(), missing);
      },
      "self-bind refuses a missing external singleton token");

  auto held = std::make_shared<HeldToken>();
  auto authority = HostdSocketAuthority::self_bind(
      socket_config(directory), directory.parent_fd(), held);
  const auto identity = authority.reattest();
  require(identity.path_inode != 0U && !authority.poisoned(),
          "self-bound listener retains pinned parent/path identity");
  const int fd_flags = ::fcntl(authority.listener_fd(), F_GETFD);
  const int status_flags = ::fcntl(authority.listener_fd(), F_GETFL);
  require((fd_flags & FD_CLOEXEC) != 0 &&
              (status_flags & O_NONBLOCK) != 0,
          "listener is atomically CLOEXEC and nonblocking");
  require_throws<HostdTransportError>(
      [&] {
        (void)HostdSocketAuthority::self_bind(
            socket_config(directory), directory.parent_fd(), held);
      },
      "second live bind never unlinks the first socket pathname");
  require(authority.reattest() == identity,
          "failed second bind leaves original authority intact");
  held->held = false;
  require_throws<HostdTransportError>(
      [&] { (void)authority.reattest(); },
      "authority poisons immediately when its retained singleton is lost");
  require(authority.poisoned(),
          "singleton loss remains observable through authority status");
}

void startup_faults_rollback_and_restore_process_state() {
  TemporaryDirectory directory;
  struct stat cwd_before{};
  require(::stat(".", &cwd_before) == 0, "capture cwd before bind faults");
  sigset_t mask_before{};
  require(::pthread_sigmask(SIG_SETMASK, nullptr, &mask_before) == 0,
          "capture signal mask before bind faults");

  const std::array checkpoints{
      HostdSocketBindCheckpoint::identity_captured,
      HostdSocketBindCheckpoint::before_socket_protection,
      HostdSocketBindCheckpoint::before_listen};
  std::size_t sequence = 0U;
  for (const auto checkpoint : checkpoints) {
    const std::string name = "fault-" + std::to_string(sequence++) + ".sock";
    auto config = socket_config(directory, name);
    config.fault_injector =
        std::make_shared<ThrowAtBindCheckpoint>(checkpoint);
    require_throws<HostdTransportError>(
        [&] {
          (void)HostdSocketAuthority::self_bind(
              config, directory.parent_fd(), std::make_shared<HeldToken>());
        },
        "post-bind checkpoint failure is normalized");
    require(!std::filesystem::exists(config.socket_path),
            "captured post-bind failure exact-unlinks its own pathname");
  }

  struct stat cwd_after{};
  require(::stat(".", &cwd_after) == 0 &&
              cwd_after.st_dev == cwd_before.st_dev &&
              cwd_after.st_ino == cwd_before.st_ino,
          "post-bind failure restores the exact process cwd");
  sigset_t mask_after{};
  require(::pthread_sigmask(SIG_SETMASK, nullptr, &mask_after) == 0,
          "capture signal mask after bind faults");
  for (int signal_number = 1; signal_number < NSIG; ++signal_number)
    require(::sigismember(&mask_before, signal_number) ==
                ::sigismember(&mask_after, signal_number),
            "post-bind failure restores the process signal mask");

  auto mask_observer = std::make_shared<ObserveBlockedSignals>();
  auto mask_config = socket_config(directory, "mask-checkpoint.sock");
  mask_config.fault_injector = mask_observer;
  {
    auto mask_authority = HostdSocketAuthority::self_bind(
        mask_config, directory.parent_fd(), std::make_shared<HeldToken>());
    require(mask_observer->observed,
            "signals are blocked before startup task enumeration");
    (void)mask_authority.reattest();
  }
  sigset_t mask_after_success{};
  require(::pthread_sigmask(SIG_SETMASK, nullptr, &mask_after_success) == 0,
          "capture signal mask after successful bind");
  for (int signal_number = 1; signal_number < NSIG; ++signal_number)
    require(::sigismember(&mask_before, signal_number) ==
                ::sigismember(&mask_after_success, signal_number),
            "successful bind restores the process signal mask");

  std::mutex thread_mutex;
  std::condition_variable thread_changed;
  bool thread_ready = false;
  bool thread_stop = false;
  std::jthread extra_thread([&] {
    std::unique_lock lock(thread_mutex);
    thread_ready = true;
    thread_changed.notify_all();
    thread_changed.wait(lock, [&] { return thread_stop; });
  });
  {
    std::unique_lock lock(thread_mutex);
    thread_changed.wait(lock, [&] { return thread_ready; });
  }
  const auto threaded_config = socket_config(directory, "threaded.sock");
  require_throws<HostdTransportError>(
      [&] {
        (void)HostdSocketAuthority::self_bind(
            threaded_config, directory.parent_fd(),
            std::make_shared<HeldToken>());
      },
      "self-bind refuses to change cwd after another thread exists");
  require(!std::filesystem::exists(threaded_config.socket_path),
          "multi-thread rejection occurs before creating a pathname");
  sigset_t mask_after_thread_rejection{};
  require(::pthread_sigmask(SIG_SETMASK, nullptr,
                            &mask_after_thread_rejection) == 0,
          "capture signal mask after multi-thread rejection");
  for (int signal_number = 1; signal_number < NSIG; ++signal_number)
    require(::sigismember(&mask_before, signal_number) ==
                ::sigismember(&mask_after_thread_rejection, signal_number),
            "multi-thread rejection restores the process signal mask");
  {
    std::scoped_lock lock(thread_mutex);
    thread_stop = true;
    thread_changed.notify_all();
  }
  extra_thread.join();

  auto capture_failstop_config =
      socket_config(directory, "capture-failstop.sock");
  capture_failstop_config.fault_injector =
      std::make_shared<ThrowAtBindCheckpoint>(
          HostdSocketBindCheckpoint::before_identity_capture);
  const pid_t capture_child = ::fork();
  require(capture_child >= 0, "fork capture fail-stop test child");
  if (capture_child == 0) {
    std::set_terminate([] { ::_exit(85); });
    (void)HostdSocketAuthority::self_bind(
        capture_failstop_config, directory.parent_fd(),
        std::make_shared<HeldToken>());
    ::_exit(87);
  }
  int capture_child_status = 0;
  require(::waitpid(capture_child, &capture_child_status, 0) == capture_child &&
              WIFEXITED(capture_child_status) &&
              WEXITSTATUS(capture_child_status) == 85,
          "pre-capture failure terminates instead of losing rollback identity");
  require(::unlink(capture_failstop_config.socket_path.c_str()) == 0,
          "remove capture fail-stop child's intentionally stale test path");

  auto failstop_config = socket_config(directory, "cwd-failstop.sock");
  failstop_config.fault_injector =
      std::make_shared<ThrowAtBindCheckpoint>(
          HostdSocketBindCheckpoint::before_cwd_restore);
  const pid_t child = ::fork();
  require(child >= 0, "fork cwd fail-stop test child");
  if (child == 0) {
    std::set_terminate([] { ::_exit(86); });
    (void)HostdSocketAuthority::self_bind(
        failstop_config, directory.parent_fd(),
        std::make_shared<HeldToken>());
    ::_exit(87);
  }
  int child_status = 0;
  require(::waitpid(child, &child_status, 0) == child &&
              WIFEXITED(child_status) && WEXITSTATUS(child_status) == 86,
          "cwd restoration failure terminates instead of continuing unsafely");
  require(::unlink(failstop_config.socket_path.c_str()) == 0,
          "remove fail-stop child's intentionally stale test pathname");
}

void path_replacement_poison_and_guarded_move_cleanup() {
  TemporaryDirectory directory;
  const auto config = socket_config(directory);
  auto held = std::make_shared<HeldToken>();
  auto authority = HostdSocketAuthority::self_bind(
      config, directory.parent_fd(), held);
  const auto original = authority.reattest();
  const auto displaced = directory.path() / "displaced.sock";
  require(::rename(config.socket_path.c_str(), displaced.c_str()) == 0,
          "replace visible socket pathname");
  const int replacement = create_raw_listener(config);
  require_throws<HostdTransportError>(
      [&] { (void)authority.reattest(); },
      "listener/path replacement poisons the authority");
  require(authority.poisoned() && !authority.poison_reason().empty(),
          "path replacement poison remains observable");
  require(::close(replacement) == 0, "close replacement listener");
  require(original.path_inode != 0U, "original endpoint was pinned");

  require(std::filesystem::exists(config.socket_path),
          "poisoned owner does not remove a replacement pathname");

  TemporaryDirectory move_directory;
  const auto first_path = move_directory.path() / "first.sock";
  const auto second_path = move_directory.path() / "second.sock";
  auto first = HostdSocketAuthority::self_bind(
      socket_config(move_directory, "first.sock"),
      move_directory.parent_fd(), std::make_shared<HeldToken>());
  auto second = HostdSocketAuthority::self_bind(
      socket_config(move_directory, "second.sock"),
      move_directory.parent_fd(), std::make_shared<HeldToken>());
  second = std::move(first);
  require(std::filesystem::exists(first_path) &&
              !std::filesystem::exists(second_path),
          "move assignment guarded-unlinks only the destination's old path");
  (void)second.reattest();
}

void status_only_lifecycle_and_endpoint_identity() {
  TemporaryDirectory directory;
  CoordinatorFixture fixture(directory);
  auto held = std::make_shared<HeldToken>();
  auto authority = std::make_shared<HostdSocketAuthority>(
      HostdSocketAuthority::self_bind(socket_config(directory),
                                      directory.parent_fd(), held));
  HostdStatusServer server(authority, fixture.coordinator, peer_policy());
  const auto client = client_config(*authority);

  HostdServeResult sealed_result = HostdServeResult::timed_out;
  std::jthread sealed_server(
      [&] { sealed_result = server.serve_one(deadline()); });
  const auto sealed = hostd_request_status(client, 11U, deadline());
  sealed_server.join();
  require(sealed_result == HostdServeResult::served &&
              sealed.kind == HostdStatusReplyKind::status && sealed.status &&
              sealed.status->lifecycle == HostdLifecycle::sealed,
          "status transport truthfully exposes the sealed lifecycle");

  BlockingAuditor auditor(audit_report(*fixture.ledger));
  std::jthread auditing(
      [&] { (void)fixture.coordinator->run_startup_audit(auditor, {30, 40}); });
  auditor.wait_until_entered();
  HostdServeResult auditing_result = HostdServeResult::timed_out;
  std::jthread auditing_server(
      [&] { auditing_result = server.serve_one(deadline()); });
  const auto during_audit = hostd_request_status(client, 12U, deadline());
  auditing_server.join();
  require(auditing_result == HostdServeResult::served &&
              during_audit.status &&
              during_audit.status->lifecycle ==
                  HostdLifecycle::startup_auditing,
          "status transport truthfully exposes startup auditing");
  auditor.release();
  auditing.join();
  const auto committed_status = fixture.coordinator->status();
  const auto ledger_records_before_status = fixture.ledger->record_count();
  HostdServeResult admitted_result = HostdServeResult::timed_out;
  std::jthread admitted_server(
      [&] { admitted_result = server.serve_one(deadline()); });
  const auto admitted = hostd_request_status(client, 13U, deadline());
  admitted_server.join();
  require(admitted_result == HostdServeResult::served &&
              admitted.kind == HostdStatusReplyKind::status &&
              admitted.status &&
              admitted.status->lifecycle == HostdLifecycle::admitting &&
              admitted.status->startup_audit ==
                  committed_status.startup_audit &&
              admitted.correlation_id == 13U,
          "admitting coordinator returns the exact committed inspection receipt");
  require(fixture.ledger->record_count() == ledger_records_before_status,
          "status inspection does not mutate the durable ledger");

  HostdStatusServer unauthorized_server(
      authority, fixture.coordinator,
      {.allowed_uid = static_cast<uid_t>(::geteuid() + 1U),
       .allowed_gid = ::getegid()});
  HostdServeResult unauthorized_result = HostdServeResult::served;
  std::jthread unauthorized_thread([&] {
    unauthorized_result = unauthorized_server.serve_one(deadline());
  });
  require_throws<HostdTransportError>(
      [&] { (void)hostd_request_status(client, 14U, deadline()); },
      "status server closes a peer outside explicit UID/GID policy");
  unauthorized_thread.join();
  require(unauthorized_result == HostdServeResult::rejected,
          "wrong peer credentials are rejected before packet authority");

  auto wrong_endpoint = client;
  ++wrong_endpoint.expected_endpoint.path_inode;
  require_throws<HostdTransportError>(
      [&] { (void)hostd_request_status(wrong_endpoint, 15U, deadline()); },
      "client refuses a path whose pinned endpoint identity is inexact");

  TemporaryDirectory poisoned_directory;
  CoordinatorFixture poisoned_fixture(poisoned_directory);
  auto poisoned_authority = std::make_shared<HostdSocketAuthority>(
      HostdSocketAuthority::self_bind(socket_config(poisoned_directory),
                                      poisoned_directory.parent_fd(),
                                      std::make_shared<HeldToken>()));
  Auditor failed_auditor(audit_report(
      *poisoned_fixture.ledger, HostStartupAuditDisposition::failed));
  require_throws<HostdStateError>(
      [&] {
        (void)poisoned_fixture.coordinator->run_startup_audit(failed_auditor,
                                                              {30, 40});
      },
      "failed startup audit blocks coordinator");
  HostdStatusServer poisoned_server(poisoned_authority,
                                    poisoned_fixture.coordinator,
                                    peer_policy());
  HostdServeResult poisoned_result = HostdServeResult::timed_out;
  std::jthread poisoned_thread(
      [&] { poisoned_result = poisoned_server.serve_one(deadline()); });
  const auto poisoned = hostd_request_status(
      client_config(*poisoned_authority), 16U, deadline());
  poisoned_thread.join();
  require(poisoned_result == HostdServeResult::served && poisoned.status &&
              poisoned.status->lifecycle == HostdLifecycle::startup_blocked &&
              poisoned.status->startup_audit.has_value() &&
              poisoned.status->startup_audit->report_digest ==
                  failed_auditor.report.report_digest &&
              !poisoned.status->poison_reason.empty(),
          "status transport truthfully exposes committed blocked evidence");

  TemporaryDirectory invalid_text_directory;
  CoordinatorFixture invalid_text_fixture(invalid_text_directory);
  auto invalid_text_authority = std::make_shared<HostdSocketAuthority>(
      HostdSocketAuthority::self_bind(socket_config(invalid_text_directory),
                                      invalid_text_directory.parent_fd(),
                                      std::make_shared<HeldToken>()));
  InvalidTextAuditor invalid_text_auditor;
  require_throws<HostdStateError>(
      [&] {
        (void)invalid_text_fixture.coordinator->run_startup_audit(
            invalid_text_auditor, {30, 40});
      },
      "invalid-text auditor poisons coordinator");
  HostdStatusServer invalid_text_server(invalid_text_authority,
                                        invalid_text_fixture.coordinator,
                                        peer_policy());
  HostdServeResult invalid_text_result = HostdServeResult::served;
  std::jthread invalid_text_thread(
      [&] { invalid_text_result = invalid_text_server.serve_one(deadline()); });
  require_throws<HostdTransportError>(
      [&] {
        (void)hostd_request_status(client_config(*invalid_text_authority), 17U,
                                   deadline());
      },
      "server refuses unsafe status text before JSON serialization");
  invalid_text_thread.join();
  require(invalid_text_result == HostdServeResult::rejected,
          "server normalizes invalid status serialization state to rejection");
}

void malformed_packets_rights_and_deadlines_are_bounded() {
  TemporaryDirectory directory;
  CoordinatorFixture fixture(directory);
  fixture.admit();
  auto held = std::make_shared<HeldToken>();
  auto authority = std::make_shared<HostdSocketAuthority>(
      HostdSocketAuthority::self_bind(socket_config(directory),
                                      directory.parent_fd(), held));
  HostdStatusServer server(
      authority, fixture.coordinator, peer_policy(),
      {.maximum_payload_bytes = 1024U,
       .per_session_timeout_ns = 30'000'000LL});

  std::vector<std::vector<std::byte>> malformed;
  auto bad_version = hostd_encode_status_request(21U);
  put_u16(bad_version, 4U, 999U);
  malformed.push_back(std::move(bad_version));
  auto bad_opcode = hostd_encode_status_request(22U);
  put_u16(bad_opcode, 8U, 999U);
  malformed.push_back(std::move(bad_opcode));
  auto bad_flags = hostd_encode_status_request(23U);
  put_u16(bad_flags, 10U, 1U);
  malformed.push_back(std::move(bad_flags));
  auto bad_digest = hostd_encode_status_request(24U);
  bad_digest[24U] ^= std::byte{1U};
  malformed.push_back(std::move(bad_digest));
  auto trailing = hostd_encode_status_request(25U);
  trailing.push_back(std::byte{0U});
  malformed.push_back(std::move(trailing));
  malformed.push_back(packet_with_payload(
      1U, 26U,
      {{"api_version", kHostdStatusTransportApiVersion}, {"extra", true}}));
  for (const auto &packet : malformed)
    require(send_raw(server, authority->socket_path(), packet) ==
                HostdServeResult::rejected,
            "version/op/flags/fields/digest/trailing packet is rejected");

  HostdStatusServer tiny_response_server(
      authority, fixture.coordinator, peer_policy(),
      {.maximum_payload_bytes = 1U,
       .per_session_timeout_ns = 30'000'000LL});
  require(send_raw(tiny_response_server, authority->socket_path(),
                   hostd_encode_status_request(29U), std::nullopt, true) ==
              HostdServeResult::rejected,
          "typed errors never exceed the configured response payload bound");

  auto oversized = hostd_encode_status_request(27U);
  oversized.resize(kHostdStatusWireHeaderBytes + 2048U, std::byte{0U});
  require(send_raw(server, authority->socket_path(), oversized) ==
              HostdServeResult::rejected,
          "MSG_TRUNC oversized packet is rejected without partial decode");

  const int harmless = ::open("/dev/null", O_RDONLY | O_CLOEXEC);
  require(harmless >= 0, "open harmless descriptor for SCM_RIGHTS test");
  const std::size_t before = open_fd_count();
  require(send_raw(server, authority->socket_path(),
                   hostd_encode_status_request(28U), harmless) ==
              HostdServeResult::rejected,
          "status-only transport rejects every SCM_RIGHTS packet");
  require(open_fd_count() == before,
          "rejected SCM_RIGHTS packet leaks no received descriptor");
  require(::close(harmless) == 0, "close harmless sender descriptor");

  require(server.serve_one(deadline(15'000'000LL)) ==
              HostdServeResult::timed_out,
          "absolute monotonic accept deadline is bounded");
  require(server.serve_one(std::numeric_limits<std::int64_t>::min()) ==
              HostdServeResult::timed_out,
          "an extreme expired deadline cannot underflow into a long wait");
  require_throws<HostdTransportError>(
      [&] {
        HostdStatusServer invalid(
            authority, fixture.coordinator, peer_policy(),
            {.maximum_payload_bytes = 1024U,
             .per_session_timeout_ns =
                 std::numeric_limits<std::int64_t>::max()});
        (void)invalid;
      },
      "an extreme session timeout is rejected before deadline arithmetic");
  HostdServeResult idle = HostdServeResult::served;
  std::jthread idle_server([&] { idle = server.serve_one(deadline()); });
  const int idle_client = connect_raw(authority->socket_path());
  idle_server.join();
  require(idle == HostdServeResult::rejected,
          "per-session receive deadline rejects an idle peer");
  require(::close(idle_client) == 0, "close idle client");

  std::vector<int> backlog_fillers;
  sockaddr_un saturated_address{};
  saturated_address.sun_family = AF_UNIX;
  const std::string saturated_path = authority->socket_path().string();
  std::memcpy(saturated_address.sun_path, saturated_path.data(),
              saturated_path.size());
  saturated_address.sun_path[saturated_path.size()] = '\0';
  const socklen_t saturated_length = static_cast<socklen_t>(
      offsetof(sockaddr_un, sun_path) + saturated_path.size() + 1U);
  bool observed_eagain = false;
  for (std::size_t attempt = 0U; attempt < 128U; ++attempt) {
    const int filler =
        ::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
    require(filler >= 0, "create backlog saturation client");
    if (::connect(filler,
                  reinterpret_cast<const sockaddr *>(&saturated_address),
                  saturated_length) == 0 || errno == EINPROGRESS ||
        errno == EALREADY) {
      backlog_fillers.push_back(filler);
      continue;
    }
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      observed_eagain = true;
      require(::close(filler) == 0, "close EAGAIN saturation client");
      break;
    }
    require(false, "unexpected backlog saturation connect outcome");
  }
  require(observed_eagain,
          "AF_UNIX listener backlog reaches deterministic EAGAIN");
  require_throws<HostdTransportError>(
      [&] {
        (void)hostd_request_status(client_config(*authority), 30U,
                                   deadline(30'000'000LL));
      },
      "AF_UNIX EAGAIN connect retries stop at the absolute deadline");
  for (const int filler : backlog_fillers)
    require(::close(filler) == 0, "close backlog saturation client");
}

void client_rejects_corruption_delegation_and_no_children_exist() {
  TemporaryDirectory directory;
  CoordinatorFixture fixture(directory);
  fixture.admit();
  auto held = std::make_shared<HeldToken>();
  auto authority = std::make_shared<HostdSocketAuthority>(
      HostdSocketAuthority::self_bind(socket_config(directory),
                                      directory.parent_fd(), held));
  const auto client = client_config(*authority);
  std::jthread fake_server([&] {
    pollfd readiness{.fd = authority->listener_fd(),
                     .events = POLLIN,
                     .revents = 0};
    require(::poll(&readiness, 1U, 2000) == 1 &&
                (readiness.revents & POLLIN) != 0,
            "fake server observes correlation-test connection");
    const int accepted = ::accept4(authority->listener_fd(), nullptr, nullptr,
                                   SOCK_CLOEXEC);
    require(accepted >= 0, "fake server accepts correlation test");
    std::array<std::byte, 4096U> request{};
    require(::recv(accepted, request.data(), request.size(), 0) > 0,
            "fake server receives status request");
    auto mismatched = hostd_encode_status_request(999U);
    put_u16(mismatched, 8U, 2U);
    require(::send(accepted, mismatched.data(), mismatched.size(), MSG_NOSIGNAL) ==
                static_cast<ssize_t>(mismatched.size()),
            "fake server sends mismatched correlation");
    require(::close(accepted) == 0, "close fake accepted socket");
  });
  require_throws<HostdTransportError>(
      [&] { (void)hostd_request_status(client, 31U, deadline()); },
      "client rejects a response with the wrong correlation ID");
  fake_server.join();

  auto invalid_status = status_payload(fixture.coordinator->status());
  invalid_status["startup_audit"]["host_id"] = "different-host";
  invalid_status["live_sessions"] = 0U;
  invalid_status["admission_sessions"] = 1U;
  const auto semantic_response = packet_with_payload(
      2U, 32U,
      {{"api_version", kHostdStatusTransportApiVersion},
       {"status", invalid_status}});
  std::jthread semantic_server(
      [&] { serve_fake_response(*authority, semantic_response); });
  require_throws<HostdTransportError>(
      [&] { (void)hostd_request_status(client, 32U, deadline()); },
      "client rejects semantically contradictory status and audit evidence");
  semantic_server.join();

  auto corrupted_receipt = status_payload(fixture.coordinator->status());
  corrupted_receipt["startup_audit"]["receipt_digest"] =
      "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
  const auto corrupted_receipt_response = packet_with_payload(
      2U, 36U,
      {{"api_version", kHostdStatusTransportApiVersion},
       {"status", std::move(corrupted_receipt)}});
  std::jthread corrupted_receipt_server(
      [&] { serve_fake_response(*authority, corrupted_receipt_response); });
  require_throws<HostdTransportError>(
      [&] { (void)hostd_request_status(client, 36U, deadline()); },
      "client rejects self-inconsistent startup receipt inspection data");
  corrupted_receipt_server.join();

  std::vector<nlohmann::json> malicious_numbers;
  auto fractional_count = status_payload(fixture.coordinator->status());
  fractional_count["live_sessions"] = 1.5;
  malicious_numbers.push_back(std::move(fractional_count));
  auto negative_count = status_payload(fixture.coordinator->status());
  negative_count["live_sessions"] = -1;
  malicious_numbers.push_back(std::move(negative_count));
  auto negative_sequence = status_payload(fixture.coordinator->status());
  negative_sequence["startup_audit"]["ledger_head_before"]
                   ["ledger_sequence"] = -1;
  malicious_numbers.push_back(std::move(negative_sequence));
  auto fractional_time = status_payload(fixture.coordinator->status());
  fractional_time["startup_audit"]["committed_boottime_ns"] = 30.5;
  malicious_numbers.push_back(std::move(fractional_time));
  auto negative_time = status_payload(fixture.coordinator->status());
  negative_time["startup_audit"]["committed_wall_time_ns"] = -1;
  malicious_numbers.push_back(std::move(negative_time));
  auto huge_signed_time = status_payload(fixture.coordinator->status());
  huge_signed_time["startup_audit"]["committed_wall_time_ns"] =
      std::numeric_limits<std::uint64_t>::max();
  malicious_numbers.push_back(std::move(huge_signed_time));
  auto huge_count = status_payload(fixture.coordinator->status());
  huge_count["live_sessions"] = nlohmann::json::parse(
      "184467440737095516160000000000000000000");
  malicious_numbers.push_back(std::move(huge_count));
  for (std::size_t index = 0U; index < malicious_numbers.size(); ++index) {
    const std::uint64_t correlation = 40U + index;
    const auto response = packet_with_payload(
        2U, correlation,
        {{"api_version", kHostdStatusTransportApiVersion},
         {"status", malicious_numbers[index]}});
    std::jthread malicious_server(
        [&] { serve_fake_response(*authority, response); });
    require_throws<HostdTransportError>(
        [&] { (void)hostd_request_status(client, correlation, deadline()); },
        "client rejects fractional, negative, or out-of-range status numbers");
    malicious_server.join();
  }

  const auto wrong_type_response = packet_with_payload(
      2U, 34U,
      {{"api_version", 7},
       {"status", status_payload(fixture.coordinator->status())}});
  std::jthread wrong_type_server(
      [&] { serve_fake_response(*authority, wrong_type_response); });
  require_throws<HostdTransportError>(
      [&] { (void)hostd_request_status(client, 34U, deadline()); },
      "client normalizes untrusted JSON type exceptions");
  wrong_type_server.join();

  const auto invalid_error_response = packet_with_payload(
      3U, 35U,
      {{"api_version", kHostdStatusTransportApiVersion},
       {"code", "bad_error"},
       {"message", std::string("control\x01text", 12U)}});
  std::jthread invalid_error_server(
      [&] { serve_fake_response(*authority, invalid_error_response); });
  require_throws<HostdTransportError>(
      [&] { (void)hostd_request_status(client, 35U, deadline()); },
      "client rejects typed-error strings outside the text policy");
  invalid_error_server.join();

  const auto valid_response = packet_with_payload(
      2U, 33U,
      {{"api_version", kHostdStatusTransportApiVersion},
       {"status", status_payload(fixture.coordinator->status())}});
  const int harmless = ::open("/dev/null", O_RDONLY | O_CLOEXEC);
  require(harmless >= 0, "open delegated response descriptor");
  const std::size_t before = open_fd_count();
  std::jthread rights_server(
      [&] { serve_fake_response(*authority, valid_response, harmless); });
  require_throws<HostdTransportError>(
      [&] { (void)hostd_request_status(client, 33U, deadline()); },
      "client rejects SCM_RIGHTS delegation from an otherwise valid server");
  rights_server.join();
  require(open_fd_count() == before,
          "client closes every rejected response-side descriptor");
  require(::close(harmless) == 0, "close delegated response descriptor");

  int status = 0;
  errno = 0;
  require(::waitpid(-1, &status, WNOHANG) == -1 && errno == ECHILD,
          "status transport creates no child process");
}

ResourceBundleRequest mutation_request(std::string id) {
  const auto scope = mutation_attribution();
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

void mutation_transport_dispatches_replays_and_disconnects() {
  TemporaryDirectory directory;
  auto logical = std::make_shared<MutationLogicalFence>();
  CoordinatorFixture fixture(directory, logical);
  fixture.admit();
  auto held = std::make_shared<HeldToken>();
  auto authority = std::make_shared<HostdSocketAuthority>(
      HostdSocketAuthority::self_bind(socket_config(directory),
                                      directory.parent_fd(), held));
  auto nonce = std::make_shared<MutationNonce>();
  auto challenge_time = std::make_shared<MutationChallengeTime>();
  auto journal = std::make_shared<MutationJournalAttestor>();
  auto verifier = std::make_shared<HostdSessionChallengeVerifier>(
      HostdSessionChallengeVerifierConfig{
          .api_version = std::string(kHostdSessionChallengeApiVersion),
          .host_id = fixture.observed.host_id,
          .boot_id = fixture.observed.boot_id,
          .broker_epoch = fixture.observed.broker_epoch,
          .challenge_ttl_ns = 2'000'000'000LL,
          .maximum_outstanding_challenges = 16U,
          .maximum_outstanding_challenges_per_peer = 4U},
      nonce, challenge_time, journal);
  auto kernel = std::make_shared<MutationLinuxKernel>();
  auto service = std::make_shared<MutationServiceAuthority>();
  auto ledger_time = std::make_shared<MutationLedgerTime>();
  auto fault = std::make_shared<MutationFaultInjector>();
  auto process_supervisor =
      std::make_shared<LedgerProcessSupervisor>(*fixture.ledger);
  HostdStatusServer status_server(
      authority, fixture.coordinator,
      {.allowed_uid = ::geteuid(), .allowed_gid = ::getegid()});
  HostdMutationServer server(
      authority, fixture.coordinator, verifier, kernel, service, ledger_time,
      {.api_version = std::string(kHostdMutationTransportApiVersion),
       .allowed_uid = ::geteuid(),
       .allowed_gid = ::getegid(),
       .socket_peer_grade = HostdLinuxSessionEnforcementGrade::
           cooperative_namespace_observation,
       .enforcement_grade =
           HostdMutationTransportEnforcementGrade::cooperative_test,
       .maximum_payload_bytes = kHostdStatusMaximumPayloadBytes,
       .per_session_timeout_ns = 2'000'000'000LL,
       .fault_injector = fault},
      process_supervisor);
  HostdUnifiedServer unified(authority, status_server, server);
  const HostdMutationClientConfig client{
      .socket_path = authority->socket_path(),
      .expected_endpoint = authority->reattest(),
      .expected_server_uid = ::geteuid(),
      .expected_server_gid = ::getegid(),
      .maximum_payload_bytes = kHostdStatusMaximumPayloadBytes};
  const HostdMutationOpen open{
      .api_version = std::string(kHostdMutationProtocolApiVersion),
      .claim = mutation_claim()};

  std::optional<HostdStatusReply> routed_status;
  std::exception_ptr status_error;
  std::jthread status_thread([&] {
    try {
      routed_status =
          hostd_request_status(client_config(*authority), 100U, deadline());
    } catch (...) {
      status_error = std::current_exception();
    }
  });
  const HostdServeResult status_served = unified.serve_one(deadline());
  status_thread.join();
  if (status_error) std::rethrow_exception(status_error);
  require(status_served == HostdServeResult::served && routed_status &&
              routed_status->status &&
              routed_status->status->lifecycle == HostdLifecycle::admitting,
          "unified listener routes a status request without consuming it");

  const auto exchange = [&](HostdMutationRequest request,
                            std::uint64_t correlation) {
    std::optional<HostdMutationReply> reply;
    std::exception_ptr client_error;
    std::jthread client_thread([&] {
      try {
        reply = hostd_request_mutation(client, request, correlation, deadline());
      } catch (...) {
        client_error = std::current_exception();
      }
    });
    const HostdServeResult served = unified.serve_one(deadline());
    client_thread.join();
    if (client_error)
      std::rethrow_exception(client_error);
    require(served == HostdServeResult::served && reply.has_value(),
            "mutation request is served with one bound reply");
    require(fixture.coordinator->status().live_sessions == 0U,
            "mutation connection always disconnects coordinator session");
    return *reply;
  };

  const ResourceBundleRequest request = mutation_request("request-transport");
  const HostdMutationRequest grant_request{
      .open = open,
      .mutation = HostdMutationKind::request_bundle,
      .bundle_request = request,
      .release_request = std::nullopt};
  const auto granted = exchange(grant_request, 101U);
  require(granted.kind == HostdMutationReplyKind::bundle_outcome &&
              granted.bundle_result && granted.bundle_result->grant &&
              !granted.bundle_result->replayed,
          "first socket mutation returns a newly committed grant");
  const auto replayed = exchange(grant_request, 102U);
  require(replayed.bundle_result && replayed.bundle_result->replayed &&
              replayed.bundle_result->outcome_digest ==
                  granted.bundle_result->outcome_digest,
          "duplicate socket request returns the exact durable replay");

  std::size_t resource_open_calls = 0U;
  HostdResourceClient resource_client({
      .channel = {.transport = client,
                  .monotonic_now = [] { return hostd_monotonic_now_ns(); },
                  .request_timeout_ns = 2'000'000'000LL,
                  .exchange = {}},
      .open_for_request = [&](std::string_view) {
        ++resource_open_calls;
        return open;
      },
  });
  const auto call_resource_client = [&](auto&& operation) {
    using Result = std::invoke_result_t<decltype(operation)>;
    std::optional<Result> result;
    std::exception_ptr client_error;
    std::jthread client_thread([&] {
      try {
        result = operation();
      } catch (...) {
        client_error = std::current_exception();
      }
    });
    const HostdServeResult served = server.serve_one(deadline());
    client_thread.join();
    if (client_error) std::rethrow_exception(client_error);
    require(served == HostdServeResult::served && result,
            "typed hostd resource client receives one authenticated reply");
    return *result;
  };
  const auto typed_grant_replay = call_resource_client(
      [&] { return resource_client.request_bundle(request); });
  require(typed_grant_replay.replayed &&
              typed_grant_replay.outcome_digest ==
                  replayed.bundle_result->outcome_digest,
          "typed hostd resource client replays the exact durable grant");

  const HostdMutationRequest reconcile_request{
      .open = open,
      .mutation = HostdMutationKind::reconcile_bundle_outcome,
      .bundle_request = request,
      .release_request = std::nullopt};
  const auto reconciled = exchange(reconcile_request, 103U);
  require(reconciled.kind == HostdMutationReplyKind::bundle_outcome &&
              reconciled.bundle_result && reconciled.bundle_result->replayed &&
              reconciled.bundle_result->outcome_digest ==
                  granted.bundle_result->outcome_digest,
          "read-only reconciliation recovers the exact durable outcome");

  const auto &grant = *granted.bundle_result->grant;
  const ResolvedLaunchSpec process_launch = process_launch_for(grant);
  auto process_bootstrap = create_sealed_worker_bootstrap({
      .api_version = std::string(kWorkerBootstrapApiVersion),
      .controller_target = "unix:/run/trainvm/test.sock",
      .run_id = process_launch.identity.run_id,
      .node_id = process_launch.identity.node_id,
      .attempt_id = process_launch.identity.attempt_id,
      .launch_nonce = process_launch.identity.launch_nonce,
      .adapter = process_launch.identity.adapter_key.adapter,
      .adapter_version = process_launch.identity.adapter_key.version,
      .code_fingerprint = process_launch.identity.code_fingerprint,
      .capabilities = process_launch.identity.provided_capabilities,
      .last_acked_controller_sequence = 0U,
      .concurrency_key = process_launch.identity.concurrency_key,
      .lease_id = process_launch.identity.lease_id,
      .fencing_token = process_launch.identity.fencing_token,
      .bootstrap_digest = {},
  });
  const int delegated_bootstrap = process_bootstrap.duplicate_fd();
  const HostdProcessPrepareRequest process_prepare{
      .api_version = std::string(kHostdProcessPrepareApiVersion),
      .launch = process_launch,
      .grant = grant,
      .worker_bootstrap_digest = process_bootstrap.spec().bootstrap_digest,
      .process_policy = compile_linux_process_policy(std::nullopt),
      .descriptor_roles = {"executable", "working_directory",
                           "worker_bootstrap"},
  };
  const int delegated_executable =
      ::open("/dev/null", O_RDONLY | O_CLOEXEC);
  require(delegated_executable >= 0,
          "open delegated process executable test descriptor");
  const HostdMutationRequest prepare_request{
      .open = open,
      .mutation = HostdMutationKind::prepare_process,
      .process_prepare = process_prepare,
      .delegated_launch =
          HostdMutationRequest::DelegatedLaunchDescriptors{
              .executable_fd = delegated_executable,
              .code_fd = std::nullopt,
              .working_directory_fd = directory.parent_fd(),
              .worker_bootstrap_fd = delegated_bootstrap},
  };
  const auto prepared = exchange(prepare_request, 120U);
  require(prepared.kind == HostdMutationReplyKind::process_prepared &&
              prepared.process_prepared &&
              !prepared.process_prepared->replayed &&
              process_supervisor->prepare_calls == 1U,
          "process prepare transfers exact descriptors and durable receipts");
  const auto prepare_replay = exchange(prepare_request, 121U);
  require(prepare_replay.process_prepared &&
              prepare_replay.process_prepared->replayed &&
              process_supervisor->prepare_calls == 2U,
          "lost process-prepare reply has an exact durable replay");
  const auto& spawn = prepare_replay.process_prepared->spawn;
  const HostdProcessCommitRequest process_commit{
      .api_version = std::string(kHostdProcessCommitApiVersion),
      .launch_id = process_launch.identity.launch_event_id,
      .allocation_id = grant.allocation_id,
      .grant_digest = grant.receipt_digest,
      .journal_id = grant.journal_id,
      .run_id = grant.run_id,
      .logical_lease_id = grant.logical_lease_id,
      .logical_fencing_token = grant.logical_fencing_token,
      .spawn_receipt_digest = spawn.receipt_digest,
  };
  const HostdMutationRequest commit_request{
      .open = open,
      .mutation = HostdMutationKind::commit_process,
      .process_commit = process_commit,
  };
  const auto committed = exchange(commit_request, 122U);
  const auto commit_replay = exchange(commit_request, 123U);
  require(committed.process_committed &&
              !committed.process_committed->replayed &&
              commit_replay.process_committed &&
              commit_replay.process_committed->replayed &&
              process_supervisor->commit_calls == 2U,
          "process pre-exec release is exact and replay-safe");

  const int client_executable =
      ::fcntl(delegated_executable, F_DUPFD_CLOEXEC, 5);
  const int client_working_directory =
      ::fcntl(directory.parent_fd(), F_DUPFD_CLOEXEC, 5);
  require(client_executable >= 0 && client_working_directory >= 0,
          "duplicate descriptors for typed hostd client");
  ResolvedLaunch client_launch(process_launch, client_executable,
                               std::nullopt, client_working_directory);
  std::size_t rejected_exchange_calls = 0U;
  HostdProcessClient misattributed_client({
      .channel = {.transport = client,
                  .monotonic_now = [] { return hostd_monotonic_now_ns(); },
                  .request_timeout_ns = 2'000'000'000LL,
                  .exchange = [&](const HostdMutationClientConfig&,
                                  const HostdMutationRequest&, std::uint64_t,
                                  std::int64_t) -> HostdMutationReply {
                    ++rejected_exchange_calls;
                    throw std::runtime_error(
                        "misattributed request reached transport");
                  }},
      .open_for_launch = [&](std::string_view) {
        auto wrong = open;
        wrong.claim.controller.run_id = "wrong-run";
        return wrong;
      },
  });
  bool misattributed_rejected = false;
  try {
    (void)misattributed_client.prepare_process(
        process_prepare, client_launch, process_bootstrap);
  } catch (const OperationPreconditionError&) {
    misattributed_rejected = true;
  }
  require(misattributed_rejected && rejected_exchange_calls == 0U,
          "typed hostd client rejects a journal/controller claim mismatch before descriptor transfer");
  std::size_t open_calls = 0U;
  HostdProcessClient process_client({
      .channel = {.transport = client,
                  .monotonic_now = [] { return hostd_monotonic_now_ns(); },
                  .request_timeout_ns = 2'000'000'000LL,
                  .exchange = {}},
      .open_for_launch = [&](std::string_view launch_id) {
        ++open_calls;
        require(launch_id == process_launch.identity.launch_event_id,
                "typed hostd client requests authority for the exact launch");
        return open;
      },
  });
  const auto typed_exchange = [&](auto&& operation) {
    using Result = std::invoke_result_t<decltype(operation)>;
    std::optional<Result> result;
    std::exception_ptr client_error;
    std::jthread client_thread([&] {
      try {
        result = operation();
      } catch (...) {
        client_error = std::current_exception();
      }
    });
    const HostdServeResult served = server.serve_one(deadline());
    client_thread.join();
    if (client_error) std::rethrow_exception(client_error);
    require(served == HostdServeResult::served && result,
            "typed hostd client receives one authenticated reply");
    return *result;
  };
  const auto typed_prepared = typed_exchange([&] {
    return process_client.prepare_process(process_prepare, client_launch,
                                          process_bootstrap);
  });
  const auto typed_committed = typed_exchange([&] {
    return process_client.commit_process(process_commit);
  });
  require(typed_prepared.replayed && typed_committed.replayed &&
              process_supervisor->prepare_calls == 3U &&
              process_supervisor->commit_calls == 3U && open_calls == 2U,
          "typed hostd client delegates sealed descriptors and replays prepare/commit over the real mutation transport");

  const HostdProcessExitCommand process_exit{
      .api_version = std::string(kHostdProcessExitApiVersion),
      .exit_request_id = "exit-transport",
      .launch_id = process_launch.identity.launch_event_id,
      .allocation_id = grant.allocation_id,
      .grant_digest = grant.receipt_digest,
      .journal_id = grant.journal_id,
      .run_id = grant.run_id,
      .logical_lease_id = grant.logical_lease_id,
      .logical_fencing_token = grant.logical_fencing_token,
      .spawn_receipt_digest = spawn.receipt_digest,
      .request_termination = false,
  };
  const HostdMutationRequest exit_request{
      .open = open,
      .mutation = HostdMutationKind::finalize_process,
      .process_exit = process_exit,
  };
  const auto exited = exchange(exit_request, 124U);
  const auto exit_replay = exchange(exit_request, 125U);
  require(exited.process_exit && !exited.process_exit->replayed &&
              exit_replay.process_exit && exit_replay.process_exit->replayed &&
              process_supervisor->exit_calls == 2U,
          "process exit is durable, terminal, and replay-safe");
  const auto typed_exited = typed_exchange(
      [&] { return process_client.finalize_process(process_exit); });
  require(typed_exited.replayed && process_supervisor->exit_calls == 3U &&
              open_calls == 3U,
          "typed hostd client finalizes through the authenticated mutation transport");
  require(::close(delegated_executable) == 0,
          "close delegated process executable test descriptor");
  require(::close(delegated_bootstrap) == 0,
          "close delegated process bootstrap test descriptor");
  const ResourceReleaseRequest release = seal_resource_release_request({
      .api_version = std::string(kHostLedgerReleaseRequestApiVersion),
      .release_request_id = "release-transport",
      .allocation_id = grant.allocation_id,
      .grant_digest = grant.receipt_digest,
      .journal_id = grant.journal_id,
      .run_id = grant.run_id,
      .logical_lease_id = grant.logical_lease_id,
      .logical_fencing_token = grant.logical_fencing_token,
      .canonical_request_digest = {},
  });
  const auto released = exchange(
      {.open = open,
       .mutation = HostdMutationKind::release_bundle,
       .bundle_request = std::nullopt,
       .release_request = release},
      104U);
  require(released.kind == HostdMutationReplyKind::release_outcome &&
              released.release_result && !released.release_result->replayed &&
              released.release_result->receipt.allocation_id ==
                  grant.allocation_id,
          "socket release returns the exact committed release receipt");
  const auto typed_release_replay = call_resource_client(
      [&] { return resource_client.release_bundle(release); });
  require(typed_release_replay.replayed &&
              typed_release_replay.receipt == released.release_result->receipt &&
              resource_open_calls == 2U,
          "typed hostd resource client replays the exact durable release");

  const auto rejected_exchange = [&](HostdMutationRequest mutation,
                                     std::uint64_t correlation) {
    std::exception_ptr client_error;
    std::jthread client_thread([&] {
      try {
        (void)hostd_request_mutation(client, mutation, correlation, deadline());
      } catch (...) {
        client_error = std::current_exception();
      }
    });
    const HostdServeResult served = server.serve_one(deadline());
    client_thread.join();
    require(served == HostdServeResult::rejected && client_error != nullptr &&
                fixture.coordinator->status().live_sessions == 0U,
            "rejected mutation closes transport and coordinator session");
  };

  auto cross_scope = mutation_request("request-cross-scope");
  cross_scope.run_id = "run-other";
  cross_scope = seal_resource_request(std::move(cross_scope));
  rejected_exchange(
      {.open = open,
       .mutation = HostdMutationKind::request_bundle,
       .bundle_request = cross_scope,
       .release_request = std::nullopt},
      105U);
  require(verifier->outstanding_challenges() == 0U,
          "client-side cross-scope rejection discards its issued challenge");

  logical->live = false;
  rejected_exchange(
      {.open = open,
       .mutation = HostdMutationKind::request_bundle,
       .bundle_request = mutation_request("request-stale-fence"),
       .release_request = std::nullopt},
      106U);
  require(ledger_time->calls == 5U,
          "stale logical fence is rejected before host time or ledger mutation");

  logical->live = true;
  const std::array precommit_checkpoints{
      HostdMutationTransportCheckpoint::after_challenge_sent,
      HostdMutationTransportCheckpoint::after_command_received,
      HostdMutationTransportCheckpoint::after_challenge_verified,
      HostdMutationTransportCheckpoint::after_coordinator_connected,
  };
  for (std::size_t index = 0U; index < precommit_checkpoints.size(); ++index) {
    const auto interrupted =
        mutation_request("request-precommit-" + std::to_string(index));
    fault->selected = precommit_checkpoints[index];
    rejected_exchange(
        {.open = open,
         .mutation = HostdMutationKind::request_bundle,
         .bundle_request = interrupted,
         .release_request = std::nullopt},
        110U + index);
    require(!fault->selected &&
                !fixture.ledger->reconcile_bundle_outcome(interrupted) &&
                ledger_time->calls == 5U &&
                verifier->outstanding_challenges() == 0U,
            "every pre-dispatch interruption leaves no durable outcome or challenge");
  }

  const auto lost_reply_request = mutation_request("request-lost-reply");
  fault->selected =
      HostdMutationTransportCheckpoint::after_dispatch_committed;
  rejected_exchange(
      {.open = open,
       .mutation = HostdMutationKind::request_bundle,
       .bundle_request = lost_reply_request,
       .release_request = std::nullopt},
      107U);
  require(!fault->selected && fixture.coordinator->status().live_sessions == 0U,
          "post-commit interruption still tears down the scoped session");
  const auto recovered = exchange(
      {.open = open,
       .mutation = HostdMutationKind::request_bundle,
       .bundle_request = lost_reply_request,
       .release_request = std::nullopt},
      108U);
  require(recovered.bundle_result && recovered.bundle_result->replayed &&
              recovered.bundle_result->grant,
          "retry after post-commit lost reply returns the durable replay");
  const auto &recovered_grant = *recovered.bundle_result->grant;
  const auto recovered_release = seal_resource_release_request({
      .api_version = std::string(kHostLedgerReleaseRequestApiVersion),
      .release_request_id = "release-lost-reply",
      .allocation_id = recovered_grant.allocation_id,
      .grant_digest = recovered_grant.receipt_digest,
      .journal_id = recovered_grant.journal_id,
      .run_id = recovered_grant.run_id,
      .logical_lease_id = recovered_grant.logical_lease_id,
      .logical_fencing_token = recovered_grant.logical_fencing_token,
      .canonical_request_digest = {},
  });
  const auto recovered_released = exchange(
      {.open = open,
       .mutation = HostdMutationKind::release_bundle,
       .bundle_request = std::nullopt,
       .release_request = recovered_release},
      109U);
  require(recovered_released.release_result &&
              recovered_released.release_result->receipt.allocation_id ==
                  recovered_grant.allocation_id,
          "lost-reply replay remains exactly releasable");
  require(ledger_time->calls == 8U,
          "only dispatched grants and releases sample host ledger time");
  require(journal->calls == 21U && service->calls >= 27U &&
              logical->calls >= 25U &&
              verifier->outstanding_challenges() == 0U,
          "every connection consumes a journal challenge and reattests peer and fence");
}

// Every declared transport interruption point, crossed with every mutation
// kind the surface accepts, asserted against the whole durable fingerprint
// rather than a single reconciliation lookup.
//
// The hand-written coverage above interrupts four pre-dispatch checkpoints and
// one post-dispatch checkpoint, for request_bundle only, and checks that no
// bundle outcome exists. That leaves before_reply_send and after_reply_send
// unexercised entirely, leaves release_bundle and reconcile_bundle_outcome
// uninterrupted at every point, and would not notice an interruption that left
// the chain head, occupancy digest, or a resource generation moved while the
// bundle outcome stayed absent.
//
// TRAINVM_FUZZ_SEED and TRAINVM_FUZZ_ROUNDS match the authority fuzz suite so
// one knob widens every generated sweep.
void mutation_transport_interruptions_preserve_durable_state() {
  TemporaryDirectory directory;
  auto logical = std::make_shared<MutationLogicalFence>();
  CoordinatorFixture fixture(directory, logical);
  fixture.admit();
  auto held = std::make_shared<HeldToken>();
  auto authority = std::make_shared<HostdSocketAuthority>(
      HostdSocketAuthority::self_bind(socket_config(directory),
                                      directory.parent_fd(), held));
  auto nonce = std::make_shared<MutationNonce>();
  auto challenge_time = std::make_shared<MutationChallengeTime>();
  auto journal = std::make_shared<MutationJournalAttestor>();
  auto verifier = std::make_shared<HostdSessionChallengeVerifier>(
      HostdSessionChallengeVerifierConfig{
          .api_version = std::string(kHostdSessionChallengeApiVersion),
          .host_id = fixture.observed.host_id,
          .boot_id = fixture.observed.boot_id,
          .broker_epoch = fixture.observed.broker_epoch,
          .challenge_ttl_ns = 2'000'000'000LL,
          .maximum_outstanding_challenges = 16U,
          .maximum_outstanding_challenges_per_peer = 4U},
      nonce, challenge_time, journal);
  auto kernel = std::make_shared<MutationLinuxKernel>();
  auto service = std::make_shared<MutationServiceAuthority>();
  auto ledger_time = std::make_shared<MutationLedgerTime>();
  auto fault = std::make_shared<MutationFaultInjector>();
  auto process_supervisor =
      std::make_shared<LedgerProcessSupervisor>(*fixture.ledger);
  HostdStatusServer status_server(
      authority, fixture.coordinator,
      {.allowed_uid = ::geteuid(), .allowed_gid = ::getegid()});
  HostdMutationServer server(
      authority, fixture.coordinator, verifier, kernel, service, ledger_time,
      {.api_version = std::string(kHostdMutationTransportApiVersion),
       .allowed_uid = ::geteuid(),
       .allowed_gid = ::getegid(),
       .socket_peer_grade = HostdLinuxSessionEnforcementGrade::
           cooperative_namespace_observation,
       .enforcement_grade =
           HostdMutationTransportEnforcementGrade::cooperative_test,
       .maximum_payload_bytes = kHostdStatusMaximumPayloadBytes,
       .per_session_timeout_ns = 2'000'000'000LL,
       .fault_injector = fault},
      process_supervisor);
  const HostdMutationClientConfig client{
      .socket_path = authority->socket_path(),
      .expected_endpoint = authority->reattest(),
      .expected_server_uid = ::geteuid(),
      .expected_server_gid = ::getegid(),
      .maximum_payload_bytes = kHostdStatusMaximumPayloadBytes};
  const HostdMutationOpen open{
      .api_version = std::string(kHostdMutationProtocolApiVersion),
      .claim = mutation_claim()};

  // Byte-exact view of everything durable the ledger exposes. Any interruption
  // that moves one field of this has forked durable history.
  struct DurableFingerprint final {
    std::uint64_t record_count{};
    std::string chain_hash;
    std::uint64_t ledger_sequence{};
    std::string occupancy_digest;
    std::size_t active_fences{};
    std::size_t recovery_records{};
    std::size_t terminal_records{};
    bool operator==(const DurableFingerprint &) const = default;
  };
  const auto fingerprint = [&fixture]() {
    const auto head = fixture.ledger->chain_head();
    const auto occupancy = fixture.ledger->occupancy();
    return DurableFingerprint{
        .record_count = fixture.ledger->record_count(),
        .chain_hash = head.chain_hash,
        .ledger_sequence = head.ledger_sequence,
        .occupancy_digest = occupancy.occupancy_digest,
        .active_fences = occupancy.active_fences.size(),
        .recovery_records =
            fixture.ledger->active_process_recovery_records().size(),
        .terminal_records =
            fixture.ledger->active_terminal_process_release_records().size(),
    };
  };

  const auto attempt = [&](HostdMutationRequest mutation,
                           std::uint64_t correlation) {
    std::exception_ptr client_error;
    std::optional<HostdMutationReply> reply;
    std::jthread client_thread([&] {
      try {
        reply = hostd_request_mutation(client, mutation, correlation,
                                       deadline());
      } catch (...) {
        client_error = std::current_exception();
      }
    });
    const HostdServeResult served = server.serve_one(deadline());
    client_thread.join();
    require(fixture.coordinator->status().live_sessions == 0U,
            "every interrupted mutation tears down its coordinator session");
    return std::pair{served, std::move(reply)};
  };

  const std::array checkpoints{
      HostdMutationTransportCheckpoint::after_challenge_sent,
      HostdMutationTransportCheckpoint::after_command_received,
      HostdMutationTransportCheckpoint::after_challenge_verified,
      HostdMutationTransportCheckpoint::after_coordinator_connected,
      HostdMutationTransportCheckpoint::after_dispatch_committed,
      HostdMutationTransportCheckpoint::before_reply_send,
      HostdMutationTransportCheckpoint::after_reply_send,
  };
  // A checkpoint at or after the durable dispatch may leave an outcome; every
  // earlier one must leave nothing at all.
  const auto is_post_dispatch = [](HostdMutationTransportCheckpoint value) {
    return value == HostdMutationTransportCheckpoint::after_dispatch_committed ||
           value == HostdMutationTransportCheckpoint::before_reply_send ||
           value == HostdMutationTransportCheckpoint::after_reply_send;
  };

  const std::uint64_t rounds = fuzz_rounds();
  std::uint64_t correlation = 500U;
  std::size_t interruptions = 0U;
  for (std::uint64_t round = 0U; round < rounds; ++round) {
    for (std::size_t index = 0U; index < checkpoints.size(); ++index) {
      const auto checkpoint = checkpoints[index];
      const std::string id = "sweep-" + std::to_string(round) + "-" +
                             std::to_string(index);
      const auto request = mutation_request("request-" + id);

      const DurableFingerprint before = fingerprint();
      const std::size_t visits_before = fault->visits;
      fault->selected = checkpoint;
      const auto [served, reply] = attempt(
          {.open = open,
           .mutation = HostdMutationKind::request_bundle,
           .bundle_request = request,
           .release_request = std::nullopt},
          ++correlation);
      (void)served;
      (void)reply;
      require(!fault->selected,
              "the transport never reached the armed interruption point");
      require(fault->visits > visits_before,
              "the armed checkpoint was not visited by the transport at all");
      ++interruptions;

      std::string reason;
      require(fixture.ledger->verify(&reason),
              "interruption at a transport checkpoint broke the ledger chain: " +
                  reason);
      require(verifier->outstanding_challenges() == 0U,
              "an interrupted exchange leaks no outstanding challenge");

      const DurableFingerprint after = fingerprint();
      const auto outcome = fixture.ledger->reconcile_bundle_outcome(request);
      if (!is_post_dispatch(checkpoint)) {
        require(after == before,
                "a pre-dispatch interruption moved durable state");
        require(!outcome,
                "a pre-dispatch interruption left a durable outcome");
        continue;
      }

      // Post-dispatch: the outcome is durable exactly once, and the retry the
      // client is expected to make must replay it without moving anything.
      require(outcome.has_value(),
              "a post-dispatch interruption lost its durable outcome");
      const DurableFingerprint settled = fingerprint();
      fault->selected.reset();
      const auto [retried, retry_reply] = attempt(
          {.open = open,
           .mutation = HostdMutationKind::request_bundle,
           .bundle_request = request,
           .release_request = std::nullopt},
          ++correlation);
      require(retried == HostdServeResult::served && retry_reply &&
                  retry_reply->bundle_result &&
                  retry_reply->bundle_result->replayed,
              "retry after a post-dispatch interruption must replay, not redo");
      require(fingerprint() == settled,
              "the retry after a post-dispatch interruption appended history");

      // Release it so the next round starts from an unoccupied resource.
      if (retry_reply->bundle_result->grant) {
        const auto &grant = *retry_reply->bundle_result->grant;
        const auto release = seal_resource_release_request({
            .api_version = std::string(kHostLedgerReleaseRequestApiVersion),
            .release_request_id = "release-" + id,
            .allocation_id = grant.allocation_id,
            .grant_digest = grant.receipt_digest,
            .journal_id = grant.journal_id,
            .run_id = grant.run_id,
            .logical_lease_id = grant.logical_lease_id,
            .logical_fencing_token = grant.logical_fencing_token,
            .canonical_request_digest = {},
        });
        const auto [released, release_reply] = attempt(
            {.open = open,
             .mutation = HostdMutationKind::release_bundle,
             .bundle_request = std::nullopt,
             .release_request = release},
            ++correlation);
        require(released == HostdServeResult::served && release_reply,
                "the swept grant is released before the next round");
      }
    }
  }
  require(interruptions == rounds * checkpoints.size(),
          "every declared checkpoint was interrupted in every round");
  std::string reason;
  require(fixture.ledger->verify(&reason),
          "the ledger chain survives the whole interruption sweep: " + reason);
}

} // namespace

int main() {
  const std::vector<std::pair<std::string_view, void (*)()>> tests{
      {"authority", authority_requires_external_singleton_and_pins_path},
      {"startup-faults", startup_faults_rollback_and_restore_process_state},
      {"replacement", path_replacement_poison_and_guarded_move_cleanup},
      {"status", status_only_lifecycle_and_endpoint_identity},
      {"framing", malformed_packets_rights_and_deadlines_are_bounded},
      {"client-hardening",
       client_rejects_corruption_delegation_and_no_children_exist},
      {"mutation-dispatch",
       mutation_transport_dispatches_replays_and_disconnects},
      {"mutation-interruption-sweep",
       mutation_transport_interruptions_preserve_durable_state},
  };
  try {
    for (const auto &[name, test] : tests) {
      test();
      std::cout << "PASS " << name << '\n';
    }
    std::cout << "hostd transport tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "hostd transport test failure: " << error.what() << '\n';
    return 1;
  }
}
