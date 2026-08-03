#include "trainvm/hostd_crash_qualification.hpp"

#include <fcntl.h>
#include <linux/magic.h>
#include <openssl/evp.h>
#include <poll.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/syscall.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "trainvm/authority_time.hpp"
#include "trainvm/host_launch.hpp"
#include "trainvm/host_launch_registry.hpp"
#include "trainvm/host_ledger.hpp"
#include "trainvm/host_ledger_authority.hpp"
#include "trainvm/host_resources.hpp"
#include "trainvm/host_startup_audit.hpp"
#include "trainvm/hostd.hpp"
#include "trainvm/hostd_ledger_singleton_token.hpp"
#include "trainvm/hostd_linux_cgroup_authority.hpp"
#include "trainvm/hostd_linux_device_kernel.hpp"
#include "trainvm/hostd_linux_process_authority.hpp"
#include "trainvm/hostd_linux_process_policy_kernel.hpp"
#include "trainvm/hostd_linux_process_recovery.hpp"
#include "trainvm/hostd_linux_stopped_launcher.hpp"
#include "trainvm/hostd_restart_process_recovery.hpp"
#include "trainvm/hostd_startup_auditor.hpp"
#include "trainvm/hostd_startup_controller.hpp"
#include "trainvm/hostd_terminal_release_recovery.hpp"
#include "trainvm/hostd_transport.hpp"
#include "trainvm/reflection_json.hpp"
#include "trainvm/worker_bootstrap.hpp"

namespace trainvm {
namespace {

constexpr std::string_view kQualificationHostId =
    "sha256:5155414c494649434154494f4e2d484f53542d4944454e544954592d30303031";
constexpr std::string_view kQualificationBrokerEpoch =
    "broker-crash-qualification";
constexpr std::string_view kQualificationResource = "crash-qualification";

void require(bool condition, std::string_view message) {
  if (!condition)
    throw HostdCrashQualificationError(std::string(message));
}

std::string digest_of(std::string_view material) {
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
      EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1 ||
      EVP_DigestUpdate(context.get(), material.data(), material.size()) != 1)
    throw HostdCrashQualificationError("qualification digest failed");
  std::array<unsigned char, EVP_MAX_MD_SIZE> value{};
  unsigned int size = 0U;
  if (EVP_DigestFinal_ex(context.get(), value.data(), &size) != 1 || size != 32U)
    throw HostdCrashQualificationError("qualification digest final failed");
  constexpr std::string_view digits = "0123456789abcdef";
  std::string result = "sha256:";
  result.reserve(71U);
  for (unsigned int index = 0U; index < size; ++index) {
    result.push_back(digits[value[index] >> 4U]);
    result.push_back(digits[value[index] & 0x0fU]);
  }
  return result;
}

// The delegated cgroup subtree the qualification is allowed to create inside.
// An empty configuration selects the caller's own systemd user manager scope.
std::filesystem::path resolve_cgroup_parent(
    const std::filesystem::path& configured) {
  if (!configured.empty())
    return configured;
  const std::string uid = std::to_string(::geteuid());
  return std::filesystem::path("/sys/fs/cgroup/user.slice") /
         ("user-" + uid + ".slice") / ("user@" + uid + ".service");
}

std::string fixed_digest(char digit) {
  return "sha256:" + std::string(64U, digit);
}

bool canonical_sha256_digest(std::string_view value) {
  return value.size() == 71U && value.starts_with("sha256:") &&
         std::ranges::all_of(value.substr(7U), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

// The startup auditor cross-checks its clock against the inventory boot
// identity, so a disposable inventory still has to carry the live boot ID.
const std::string& live_boot_id() {
  static const std::string value = [] {
    AuthorityClock clock;
    return clock.sample().boot_id;
  }();
  return value;
}

HostResourceId qualification_resource_id() {
  return {.kind = HostResourceKind::host_mutex,
          .vendor = std::nullopt,
          .stable_id = "host-mutex:" + std::string(kQualificationResource),
          .parent_id = std::nullopt};
}

HostResourceId qualification_accelerator_id() {
  return {.kind = HostResourceKind::accelerator,
          .vendor = HostAcceleratorVendor::nvidia,
          .stable_id = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
          .parent_id = std::nullopt};
}

// A disposable synthetic inventory. The accelerator-shaped entry maps only
// to /dev/null: it exercises the production device-policy compiler and kernel
// attachment without reserving or opening a real accelerator.
HostInventoryReceipt qualification_inventory() {
  HostKernelSnapshot snapshot{
      .api_version = std::string(kHostInventoryApiVersion),
      .host_id = std::string(kQualificationHostId),
      .boot_id = live_boot_id(),
      .broker_epoch = std::string(kQualificationBrokerEpoch),
      .begin_revision = "revision-001",
      .end_revision = "revision-001",
      .probes = {{.vendor = HostAcceleratorVendor::nvidia,
                  .disposition = ProbeDisposition::complete,
                  .context_details_complete = true,
                  .detail = "synthetic /dev/null qualification capability"}},
      .resources = {},
  };
  snapshot.resources.push_back(ObservedHostResource{
      .id = qualification_accelerator_id(),
      .disposition = ResourceObservationDisposition::audited_eligible,
      .compute_contexts = ResourceContextDisposition::absent,
      .graphics_contexts = ResourceContextDisposition::absent,
      .pci_bdf = std::nullopt,
      .device_major = 1U,
      .device_minor = 3U,
      .device_nodes = {{.type = HostDeviceNodeType::character,
                        .purpose =
                            HostDeviceNodePurpose::assigned_accelerator,
                        .major = 1U,
                        .minor = 3U,
                        .read = true,
                        .write = true}},
      .numa_node = 0,
      .pcie_root_id = std::nullopt,
      .fabric_clique_id = std::nullopt,
      .total_memory_bytes = 1U,
      .labels = {{"scope", "synthetic-device-policy-qualification"}}});
  snapshot.resources.push_back(ObservedHostResource{
      .id = qualification_resource_id(),
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
      .labels = {{"scope", "crash-qualification"}}});
  FakeHostKernel kernel(
      {{.snapshot = std::move(snapshot), .failure = std::nullopt}});
  return capture_host_inventory(kernel);
}

HostStartupAuditPolicy qualification_audit_policy() {
  return canonicalize_host_startup_audit_policy({
      .api_version = std::string(kHostStartupAuditPolicyApiVersion),
      .require_stable_occupancy = true,
      .fail_on_blocking_findings = true,
      .maximum_findings = 16U,
      .policy_digest = {},
  });
}

std::shared_ptr<HostLedgerFilesystemAuthority> ledger_authority_for(
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

ResourceBundleRequest qualification_request(std::string id) {
  return seal_resource_request({
      .api_version = std::string(kHostResourceRequestApiVersion),
      .request_id = std::move(id),
      .journal_id = "journal-crash-qualification",
      .run_id = "run-crash-qualification",
      .logical_lease_id = "lease-crash-qualification",
      .logical_fencing_token = 7,
      .count = 1U,
      .access_mode = ResourceAccessMode::mutex_exclusive,
      .topology = TopologyPolicy::exact_resources,
      .selector = {.exact_resources = {qualification_resource_id()}},
      .canonical_request_digest = {},
  });
}

ResourceBundleRequest qualification_device_request(std::string id) {
  return seal_resource_request({
      .api_version = std::string(kHostResourceRequestApiVersion),
      .request_id = std::move(id),
      .journal_id = "journal-crash-qualification",
      .run_id = "run-crash-qualification",
      .logical_lease_id = "lease-crash-qualification",
      .logical_fencing_token = 7,
      .count = 1U,
      .access_mode = ResourceAccessMode::exclusive_device,
      .topology = TopologyPolicy::exact_resources,
      .selector = {.exact_resources = {qualification_accelerator_id()}},
      .canonical_request_digest = {},
  });
}

ResourceReleaseRequest qualification_release_request(
    const ResourceBundleGrant& grant, std::string id) {
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

HostProcessLaunchRequest qualification_launch_request(
    const ResourceBundleGrant& grant, const std::string& launch_id,
    const LinuxAllocationCgroupIdentity& cgroup,
    const std::string& executable_digest) {
  return seal_host_process_launch_request({
      .api_version = std::string(kHostProcessLaunchRequestApiVersion),
      .launch_id = launch_id,
      .allocation_id = grant.allocation_id,
      .grant_digest = grant.receipt_digest,
      .journal_id = grant.journal_id,
      .run_id = grant.run_id,
      .logical_lease_id = grant.logical_lease_id,
      .logical_fencing_token = grant.logical_fencing_token,
      .resolved_launch_digest = fixed_digest('1'),
      .executable_path = "/proc/self/exe",
      .executable_digest = executable_digest,
      .cgroup_path = cgroup.unified_path,
      .cgroup_device = cgroup.device,
      .cgroup_inode = cgroup.inode,
      .worker_credentials = std::nullopt,
      .device_policy = std::nullopt,
      .process_policy = std::nullopt,
      .canonical_request_digest = {},
  });
}

HostProcessSpawnRequest qualification_spawn_request(
    const HostProcessLaunchIntent& intent, std::int64_t pid,
    std::uint64_t starttime, const std::string& executable_digest) {
  return seal_host_process_spawn_request({
      .api_version = std::string(kHostProcessSpawnRequestApiVersion),
      .launch_id = intent.request.launch_id,
      .launch_intent_digest = intent.receipt_digest,
      .host_pid = pid,
      .process_starttime_ticks = starttime,
      .boot_id = intent.boot_id,
      .cgroup_path = intent.request.cgroup_path,
      .cgroup_device = intent.request.cgroup_device,
      .cgroup_inode = intent.request.cgroup_inode,
      .executable_digest = executable_digest,
      .worker_credentials = std::nullopt,
      .device_policy = std::nullopt,
      .process_policy = std::nullopt,
      .canonical_request_digest = {},
  });
}

HostProcessExitRequest qualification_exit_request(
    const HostProcessSpawnReceipt& spawn, std::string id) {
  return seal_host_process_exit_request({
      .api_version = std::string(kHostProcessExitRequestApiVersion),
      .exit_request_id = std::move(id),
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
      .context_audit_digest = fixed_digest('4'),
      .canonical_request_digest = {},
  });
}

// Turns a declared ledger boundary into a real process death. Nothing is
// unwound: no destructor runs, no SQLite handle is closed.
class SigkillFault final : public IHostLedgerFaultInjector {
 public:
  explicit SigkillFault(HostLedgerFaultPoint target) : target_(target) {}

  // Fixture setup runs through the same ledger, so the window stays shut
  // until the saga actually reaches the stage under qualification.
  void arm() noexcept { armed_ = true; }

  void hit(HostLedgerFaultPoint point) override {
    if (!armed_ || point != target_)
      return;
    (void)::raise(SIGKILL);
    // A blocked or ignored SIGKILL is impossible; reaching here means the
    // qualification lost its crash and must not report a crash window.
    ::_exit(97);
  }

 private:
  HostLedgerFaultPoint target_;
  bool armed_{};
};

// The leave_and_block startup policy must never reach recovered-process
// mutation. Any call is a contract violation, not a test detail.
class NoAdoptionSupervisor final : public IHostdRecoveredProcessSupervisor {
 public:
  std::size_t adopt_exact_recovered_processes(
      LinuxProcessRecoverySet&) override {
    throw HostdCrashQualificationError(
        "leave_and_block recovery attempted process adoption");
  }
  std::size_t finalize_observed_nonlive_processes(
      const LinuxProcessRecoverySet&) override {
    throw HostdCrashQualificationError(
        "leave_and_block recovery attempted nonlive finalization");
  }
  HostdRecoveredProcessProgress progress_recovered_terminations() override {
    throw HostdCrashQualificationError(
        "leave_and_block recovery attempted termination progress");
  }
};

std::string read_small_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw HostdCrashQualificationError("could not read " + path.string());
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

std::string self_executable_digest(std::int64_t pid) {
  const std::string path = "/proc/" + std::to_string(pid) + "/exe";
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw HostdCrashQualificationError("could not open observed executable");
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
      EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1)
    throw HostdCrashQualificationError("executable digest init failed");
  std::array<char, 1U << 16U> bytes{};
  while (input) {
    input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    const std::streamsize count = input.gcount();
    if (count > 0 &&
        EVP_DigestUpdate(context.get(), bytes.data(),
                         static_cast<std::size_t>(count)) != 1)
      throw HostdCrashQualificationError("executable digest update failed");
  }
  std::array<unsigned char, EVP_MAX_MD_SIZE> value{};
  unsigned int size = 0U;
  if (EVP_DigestFinal_ex(context.get(), value.data(), &size) != 1 || size != 32U)
    throw HostdCrashQualificationError("executable digest final failed");
  constexpr std::string_view digits = "0123456789abcdef";
  std::string result = "sha256:";
  for (unsigned int index = 0U; index < size; ++index) {
    result.push_back(digits[value[index] >> 4U]);
    result.push_back(digits[value[index] & 0x0fU]);
  }
  return result;
}

class QualificationContextAuditor final
    : public ILinuxProcessContextAuditor {
 public:
  LinuxProcessContextAudit audit(
      const ResourceBundleGrant& grant,
      const HostProcessSpawnReceipt& spawn) override {
    require(grant.fences.size() == 1U,
            "qualification context audit requires exactly one fence");
    require(std::ranges::all_of(
                grant.fences, [](const ResourceFence& fence) {
                  return fence.resource == qualification_resource_id() ||
                         fence.resource == qualification_accelerator_id();
                }),
            "qualification context audit received non-synthetic authority");
    require(spawn.request.launch_id.size() > 0U,
            "qualification context audit received no process identity");
    std::string evidence("trainvm.synthetic-context-audit/v1");
    evidence.push_back('\0');
    evidence += grant.receipt_digest;
    evidence.push_back('\0');
    evidence += spawn.receipt_digest;
    return {
        .complete = true,
        .accelerator_contexts_empty = true,
        .evidence_digest = digest_of(evidence),
    };
  }
};

std::filesystem::path install_qualification_worker(
    const std::filesystem::path& workspace) {
  const auto worker = workspace / "qualification-worker";
  std::error_code failure;
  std::filesystem::copy_file(
      "/proc/self/exe", worker,
      std::filesystem::copy_options::overwrite_existing, failure);
  if (failure || ::chmod(worker.c_str(), 0500) != 0)
    throw HostdCrashQualificationError(
        "could not install the immutable qualification worker");
  return worker;
}

struct QualificationWorkerLaunch final {
  ResolvedLaunch resolved;
  SealedWorkerBootstrap bootstrap;
  LinuxProcessPolicy process_policy;
};

QualificationWorkerLaunch resolve_qualification_worker(
    const std::filesystem::path& workspace,
    const std::filesystem::path& worker_path,
    const ResourceBundleGrant& grant, std::string attempt_id) {
  const std::string executable_digest = self_executable_digest(::getpid());
  const AdapterKey key{
      .adapter = "trainvm.hostd-crash-qualification",
      .version = "1.0.0",
      .runtime = ComponentRuntime::native_worker,
      .operation = "qualify",
      .contract = "trainvm.hostd-crash-qualification-worker/v1",
  };
  const std::vector<std::string> capabilities{
      "qualification.hostd-crash"};
  HostLaunchRegistry registry({
      .api_version = "trainvm.host-launches/v4",
      .trusted_roots = {workspace.string()},
      .profiles = {{.key = key,
                    .code_fingerprint = executable_digest,
                    .bootstrap_runtime_closure_fingerprint = fixed_digest('b'),
                    .provided_capabilities = capabilities,
                    .executable_path = worker_path.string(),
                    .executable_fingerprint = executable_digest,
                    .code_path = std::nullopt,
                    .code_argument_index = 0U,
                    .public_arguments = {"--qualification-worker"},
                    .working_directory = workspace.string()}},
      .profiler_executables = std::nullopt,
  });
  const WorkerLaunchTicket ticket{
      .run_id = grant.run_id,
      .node_id = "privileged-worker",
      .attempt_id = std::move(attempt_id),
      .launch_nonce = "hostd-crash-qualification-nonce",
      .adapter = key.adapter,
      .adapter_version = key.version,
      .code_fingerprint = executable_digest,
      .required_capabilities = capabilities,
      .concurrency_key = "hostd-crash-qualification",
      .lease_id = grant.logical_lease_id,
      .fencing_token = grant.logical_fencing_token,
      .host_grant = HostLaunchGrantClaim{
          .request_id = grant.request_id,
          .grant_digest = grant.receipt_digest,
          .fences = grant.fences},
  };
  HostLaunchResolver resolver(
      registry, {.host_id = grant.host_id, .boot_id = grant.boot_id});
  ResolvedLaunch resolved = resolver.resolve(ticket, key);
  const auto& identity = resolved.spec().identity;
  SealedWorkerBootstrap bootstrap = create_sealed_worker_bootstrap({
      .api_version = std::string(kWorkerBootstrapApiVersion),
      .controller_target = "unix:/run/trainvm/qualification.sock",
      .run_id = identity.run_id,
      .node_id = identity.node_id,
      .attempt_id = identity.attempt_id,
      .launch_nonce = identity.launch_nonce,
      .adapter = identity.adapter_key.adapter,
      .adapter_version = identity.adapter_key.version,
      .code_fingerprint = identity.code_fingerprint,
      .capabilities = identity.provided_capabilities,
      .last_acked_controller_sequence = 0U,
      .concurrency_key = identity.concurrency_key,
      .lease_id = identity.lease_id,
      .fencing_token = identity.fencing_token,
      .bootstrap_digest = {},
  });
  return {.resolved = std::move(resolved),
          .bootstrap = std::move(bootstrap),
          .process_policy = compile_linux_process_policy(std::nullopt)};
}

void block_worker_ready_signal() {
  sigset_t signals;
  if (::sigemptyset(&signals) != 0 || ::sigaddset(&signals, SIGUSR1) != 0 ||
      ::sigprocmask(SIG_BLOCK, &signals, nullptr) != 0) {
    throw HostdCrashQualificationError(
        "could not block the qualification worker readiness signal");
  }
}

void await_worker_ready_signal() {
  sigset_t signals;
  if (::sigemptyset(&signals) != 0 || ::sigaddset(&signals, SIGUSR1) != 0)
    throw HostdCrashQualificationError(
        "could not construct the worker readiness signal set");
  struct timespec timeout {
    .tv_sec = 5,
    .tv_nsec = 0,
  };
  int result = -1;
  do {
    result = ::sigtimedwait(&signals, nullptr, &timeout);
  } while (result < 0 && errno == EINTR);
  require(result == SIGUSR1,
          "qualification worker did not attest a successful exec");
}

std::uint64_t observed_starttime(std::int64_t pid) {
  const std::string stat =
      read_small_file("/proc/" + std::to_string(pid) + "/stat");
  return hostd_linux_process_recovery_test_seam::parse_proc_starttime(stat);
}

std::string observed_cgroup(std::int64_t pid) {
  const std::string value =
      read_small_file("/proc/" + std::to_string(pid) + "/cgroup");
  return hostd_linux_process_recovery_test_seam::parse_unified_cgroup(value);
}

LinuxAllocationCgroupIdentity observed_cgroup_identity(std::int64_t pid) {
  LinuxAllocationCgroupIdentity identity;
  identity.unified_path = observed_cgroup(pid);
  const std::filesystem::path path =
      std::filesystem::path("/sys/fs/cgroup") /
      std::filesystem::path(identity.unified_path).relative_path();
  struct stat status{};
  if (::stat(path.c_str(), &status) != 0)
    throw HostdCrashQualificationError("observed cgroup is unavailable");
  identity.device = static_cast<std::uint64_t>(status.st_dev);
  identity.inode = static_cast<std::uint64_t>(status.st_ino);
  return identity;
}

struct CrashOutcome final {
  bool signalled{};
  bool timed_out{};
  int signal_number{};
  int exit_status{};
  std::int64_t pid{};
  // Diagnostic text the child reported before dying of anything but SIGKILL.
  std::string failure;
};

int open_process_pidfd(pid_t pid) noexcept {
#ifdef SYS_pidfd_open
  return static_cast<int>(::syscall(SYS_pidfd_open, pid, 0U));
#else
  (void)pid;
  errno = ENOSYS;
  return -1;
#endif
}

int signal_process_pidfd(int pidfd, int signal_number) noexcept {
#ifdef SYS_pidfd_send_signal
  return static_cast<int>(
      ::syscall(SYS_pidfd_send_signal, pidfd, signal_number, nullptr, 0U));
#else
  (void)pidfd;
  (void)signal_number;
  errno = ENOSYS;
  return -1;
#endif
}

CrashOutcome run_crashing_child(const std::function<void()>& body,
                                std::int64_t timeout_ms) {
  std::array<int, 2U> channel{-1, -1};
  if (::pipe2(channel.data(), O_CLOEXEC | O_NONBLOCK) != 0)
    throw HostdCrashQualificationError("qualification diagnostic pipe failed");
  const pid_t child = ::fork();
  if (child < 0) {
    (void)::close(channel[0]);
    (void)::close(channel[1]);
    throw HostdCrashQualificationError("qualification fork failed");
  }
  if (child == 0) {
    (void)::close(channel[0]);
    try {
      body();
    } catch (const std::exception& error) {
      const std::string_view text(error.what());
      (void)::write(channel[1], text.data(), text.size());
      ::_exit(96);
    } catch (...) {
      constexpr std::string_view text = "unknown child failure";
      (void)::write(channel[1], text.data(), text.size());
      ::_exit(96);
    }
    ::_exit(95);
  }
  (void)::close(channel[1]);
  CrashOutcome outcome;
  const int raw_pidfd = open_process_pidfd(child);
  if (raw_pidfd < 0) {
    (void)::kill(child, SIGKILL);
    int ignored = 0;
    while (::waitpid(child, &ignored, 0) < 0 && errno == EINTR) {
    }
    (void)::close(channel[0]);
    throw HostdCrashQualificationError(
        "qualification could not pin its child pidfd");
  }
  Descriptor pidfd(raw_pidfd);
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(timeout_ms);
  // Wait on the exact direct child, not diagnostic-pipe closure. A stopped
  // pre-exec worker legitimately inherits the pipe until its launch gate is
  // resolved and must not make a successfully killed daemon look wedged.
  pollfd process{
      .fd = pidfd.get(),
      .events = POLLIN,
      .revents = 0,
  };
  int ready = 0;
  while (true) {
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - std::chrono::steady_clock::now());
    if (remaining.count() <= 0) {
      outcome.timed_out = true;
      break;
    }
    const auto bounded = std::min<std::int64_t>(
        remaining.count(), std::numeric_limits<int>::max());
    ready = ::poll(&process, 1U, static_cast<int>(bounded));
    if (ready < 0 && errno == EINTR) continue;
    if (ready == 0) {
      outcome.timed_out = true;
      break;
    }
    if (ready < 0) {
      (void)signal_process_pidfd(pidfd.get(), SIGKILL);
      int ignored = 0;
      while (::waitpid(child, &ignored, 0) < 0 && errno == EINTR) {
      }
      (void)::close(channel[0]);
      throw HostdCrashQualificationError(
          "qualification child pidfd poll failed");
    }
    if ((process.revents & (POLLIN | POLLHUP)) == 0) {
      (void)signal_process_pidfd(pidfd.get(), SIGKILL);
      int ignored = 0;
      while (::waitpid(child, &ignored, 0) < 0 && errno == EINTR) {
      }
      (void)::close(channel[0]);
      throw HostdCrashQualificationError(
          "qualification child pidfd reported an invalid poll state");
    }
    break;
  }
  if (outcome.timed_out) {
    if (signal_process_pidfd(pidfd.get(), SIGKILL) != 0) {
      // The unreaped direct child still owns its PID, so this fallback cannot
      // signal a recycled process identity.
      (void)::kill(child, SIGKILL);
    }
    outcome.failure = "qualification case exceeded its wall-clock bound";
  }
  std::array<char, 512U> bytes{};
  ssize_t count = 0;
  if (!outcome.timed_out) {
    do {
      count = ::read(channel[0], bytes.data(), bytes.size());
    } while (count < 0 && errno == EINTR);
    if (count > 0)
      outcome.failure.assign(bytes.data(), static_cast<std::size_t>(count));
  }
  (void)::close(channel[0]);
  int status = 0;
  while (::waitpid(child, &status, 0) < 0) {
    if (errno != EINTR)
      throw HostdCrashQualificationError("qualification waitpid failed");
  }
  outcome.pid = child;
  outcome.signalled = WIFSIGNALED(status);
  outcome.signal_number = outcome.signalled ? WTERMSIG(status) : 0;
  outcome.exit_status = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  return outcome;
}

// A cgroup subtree created for one case and removed with it. Nothing outside
// the subtree is ever touched.
class DisposableCgroupRoot final {
 public:
  DisposableCgroupRoot(const std::filesystem::path& parent, std::string name)
      : path_(parent / name) {
    std::error_code failure;
    std::filesystem::remove_all(path_, failure);
    if (!std::filesystem::create_directory(path_, failure) || failure)
      throw HostdCrashQualificationError(
          "could not create the disposable qualification cgroup root");
    const std::filesystem::path root("/sys/fs/cgroup");
    unified_path_ =
        "/" + std::filesystem::relative(path_, root, failure).string();
    if (failure || unified_path_.size() < 2U)
      throw HostdCrashQualificationError(
          "disposable qualification cgroup root is outside cgroup v2");
  }

  ~DisposableCgroupRoot() {
    std::error_code failure;
    for (const auto& entry :
         std::filesystem::directory_iterator(path_, failure)) {
      if (entry.is_directory(failure))
        std::filesystem::remove(entry.path(), failure);
    }
    std::filesystem::remove(path_, failure);
  }

  DisposableCgroupRoot(const DisposableCgroupRoot&) = delete;
  DisposableCgroupRoot& operator=(const DisposableCgroupRoot&) = delete;

  [[nodiscard]] const std::filesystem::path& path() const noexcept {
    return path_;
  }
  [[nodiscard]] const std::string& unified_path() const noexcept {
    return unified_path_;
  }
  [[nodiscard]] LinuxCgroupAuthorityConfig authority_config() const {
    return {.root_path = path_,
            .root_unified_path = unified_path_,
            .expected_owner_uid = ::geteuid(),
            .expected_owner_gid = ::getegid()};
  }

 private:
  std::filesystem::path path_;
  std::string unified_path_;
};

HostdCoordinatorConfig qualification_coordinator_config() {
  const HostInventoryReceipt observed = qualification_inventory();
  return {.api_version = std::string(kHostdCoordinatorApiVersion),
          .host_id = observed.host_id,
          .boot_id = observed.boot_id,
          .broker_epoch = observed.broker_epoch,
          .maximum_live_sessions = 32U,
          .maximum_logical_scopes = 64U};
}

// How far the crashing child drives the saga, and where it dies.
struct DurablePlan final {
  int stages{};
  std::optional<HostLedgerFaultPoint> fault;
  bool kill_after_completion{};
};

DurablePlan durable_plan(HostdCrashPoint point) {
  switch (point) {
    case HostdCrashPoint::intent_prepare_window:
      return {2, HostLedgerFaultPoint::after_process_intent_record, false};
    case HostdCrashPoint::intent_commit_window:
      return {2, HostLedgerFaultPoint::after_process_intent_commit, false};
    case HostdCrashPoint::spawn_prepare_window:
      return {3, HostLedgerFaultPoint::after_process_spawn_record, false};
    case HostdCrashPoint::spawn_commit_window:
      return {3, HostLedgerFaultPoint::after_process_spawn_commit, false};
    case HostdCrashPoint::exit_prepare_window:
      return {4, HostLedgerFaultPoint::after_process_exit_record, false};
    case HostdCrashPoint::release_prepare_window:
      return {5, HostLedgerFaultPoint::after_release_record, false};
    case HostdCrashPoint::release_reply_lost:
      return {5, std::nullopt, true};
    default:
      throw HostdCrashQualificationError(
          "crash point has no durable ledger plan");
  }
}

// What a correct restart must observe once the child is dead.
struct DurableExpectation final {
  std::size_t unclosed_records{};
  std::size_t terminal_release_records{};
  // Whether restart reconciliation is expected to release the physical grant.
  // A spawned worker that outlived its daemon is never closed by the
  // non-mutating policy: it must keep its fence and keep admission shut.
  bool releases_grant{};
};

DurableExpectation durable_expectation(HostdCrashPoint point) {
  switch (point) {
    case HostdCrashPoint::intent_prepare_window:
      // The prepare boundary rolled back; the physical grant survives alone
      // and must stay held rather than becoming silently reusable.
      return {0U, 0U, false};
    case HostdCrashPoint::intent_commit_window:
    case HostdCrashPoint::spawn_prepare_window:
      return {1U, 0U, true};
    case HostdCrashPoint::spawn_commit_window:
    case HostdCrashPoint::exit_prepare_window:
      return {1U, 0U, false};
    case HostdCrashPoint::release_prepare_window:
      return {0U, 1U, true};
    case HostdCrashPoint::release_reply_lost:
      return {0U, 0U, true};
    default:
      throw HostdCrashQualificationError(
          "crash point has no durable ledger expectation");
  }
}

HostLedgerTime ledger_now(AuthorityClock& clock) {
  const AuthorityTimeSample sample = clock.sample();
  return {.boottime_ns = sample.boot.nanoseconds,
          .wall_time_ns = sample.wall.nanoseconds};
}

// The real one-shot admission path. A sealed ledger refuses every bundle
// request until an exact startup audit is committed and finalized, so the
// crashing child has to reach admission the way hostd does.
HostLedgerAdmissionEpoch open_admission(SQLiteHostLedger& ledger,
                                        AuthorityClock& clock) {
  HostdConfiguredStartupAuditor auditor(
      ledger, clock,
      {.api_version = std::string(kHostdConfiguredStartupAuditorApiVersion),
       .broker_instance_id = "hostd-crash-qualification",
       .policy = qualification_audit_policy()});
  const HostStartupAuditReport report = auditor.audit();
  if (report.disposition != HostStartupAuditDisposition::passed)
    throw HostdCrashQualificationError(
        "disposable host did not pass its own startup audit");
  const HostLedgerTime now = ledger_now(clock);
  const auto committed = ledger.commit_startup_audit(report, now);
  return ledger
      .finalize_startup_admission(report, committed.receipt, ledger_now(clock))
      .epoch;
}

void durable_child_body(const std::filesystem::path& ledger_path,
                        const LinuxCgroupAuthorityConfig& cgroup_config,
                        const DurablePlan& plan) {
  LinuxCgroupAuthority cgroups(cgroup_config);
  AuthorityClock clock;
  SigkillFault fault(
      plan.fault.value_or(HostLedgerFaultPoint::after_startup_audit_record));
  // The injector is attached from the start but stays disarmed until the
  // fixture is in place, so a shared fault point can never destroy the child
  // before its declared window.
  SQLiteHostLedger armed(ledger_authority_for(ledger_path),
                         qualification_inventory(),
                         plan.fault ? &fault : nullptr,
                         qualification_audit_policy());
  const HostLedgerAdmissionEpoch epoch = open_admission(armed, clock);
  const auto bundle = armed.request_bundle(qualification_request("request-1"),
                                           ledger_now(clock), epoch);
  if (!bundle.grant)
    throw HostdCrashQualificationError(
        "disposable host did not grant its qualification bundle");
  auto cgroup = cgroups.open_or_create(bundle.grant->allocation_id, "launch-1");
  cgroup.retain_for_durable_intent();
  const std::int64_t pid = ::getpid();
  const std::string executable = self_executable_digest(pid);
  const std::uint64_t starttime = observed_starttime(pid);
  fault.arm();

  const auto intent = armed.commit_process_launch_intent(
      qualification_launch_request(*bundle.grant, "launch-1", cgroup.identity(),
                                   executable),
      ledger_now(clock));
  if (plan.stages < 3)
    return;
  const auto spawn = armed.commit_process_spawn(
      qualification_spawn_request(intent.intent, pid, starttime, executable),
      ledger_now(clock));
  if (plan.stages < 4)
    return;
  (void)armed.commit_process_exit(
      qualification_exit_request(spawn.receipt, "exit-1"), ledger_now(clock));
  if (plan.stages < 5)
    return;
  (void)armed.release_bundle(
      qualification_release_request(*bundle.grant, "release-1"),
      ledger_now(clock));
  if (plan.kill_after_completion) {
    (void)::raise(SIGKILL);
    ::_exit(97);
  }
}

struct RestartObservation final {
  std::size_t unclosed_records{};
  std::size_t terminal_release_records{};
  HostdTerminalReleaseRecoverySummary first_cleanup;
  HostdTerminalReleaseRecoverySummary second_cleanup;
  std::size_t recovery_steps{};
  bool admitted{};
  std::string admission_detail;
  std::size_t active_fences{};
  std::uint64_t generation{};
  bool chain_verified{};
  // Set when a durable launch/spawn record survived the crash and its exact
  // request replayed instead of creating a second identity.
  bool launch_replay_checked{};
  bool spawn_replay_checked{};
  std::uint64_t records_before_replay{};
  std::uint64_t records_after_replay{};
};

// The real restart path: the landed terminal-release/restart recovery driven
// by the landed wake-driven startup controller and the real coordinator
// admission authority. Nothing here is a stand-in.
RestartObservation observe_restart(const std::filesystem::path& ledger_path,
                                   const LinuxCgroupAuthorityConfig& cgroups,
                                   const HostResourceId& resource) {
  RestartObservation observed;
  auto ledger = std::make_shared<SQLiteHostLedger>(
      ledger_authority_for(ledger_path), qualification_inventory(), nullptr,
      qualification_audit_policy());
  AuthorityClock clock;
  SQLiteHostdTerminalReleaseAuthority records(*ledger, clock);
  observed.unclosed_records = records.unclosed_records().size();
  observed.terminal_release_records = records.terminal_records().size();

  // A restarted service retries the exact request it lost the reply to. That
  // retry must resolve to the already durable identity, never a second launch.
  const std::vector<HostProcessRecoveryRecord> surviving =
      records.unclosed_records();
  if (!surviving.empty()) {
    observed.records_before_replay = ledger->record_count();
    const auto replayed_intent = ledger->commit_process_launch_intent(
        surviving.front().intent.request, ledger_now(clock));
    if (!replayed_intent.replayed ||
        replayed_intent.intent != surviving.front().intent)
      throw HostdCrashQualificationError(
          "an exact launch retry after restart did not replay its intent");
    observed.launch_replay_checked = true;
    if (surviving.front().spawn) {
      const auto replayed_spawn = ledger->commit_process_spawn(
          surviving.front().spawn->request, ledger_now(clock));
      if (!replayed_spawn.replayed ||
          replayed_spawn.receipt != *surviving.front().spawn)
        throw HostdCrashQualificationError(
            "an exact spawn retry after restart did not replay its receipt");
      observed.spawn_replay_checked = true;
    }
    observed.records_after_replay = ledger->record_count();
    if (observed.records_after_replay != observed.records_before_replay)
      throw HostdCrashQualificationError(
          "an exact retry after restart appended new durable records");
  }

  LinuxCgroupAuthority cgroup_authority(cgroups);
  LinuxHostdTerminalCgroupCleaner cleaner(cgroup_authority);
  HostdTerminalReleaseRecovery cleanup(records, cleaner);
  NoAdoptionSupervisor supervisor;
  HostdRestartProcessRecovery recovery(
      records, cleanup, supervisor,
      {.exact_live_policy = HostdExactRecoveredProcessPolicy::leave_and_block,
       .reconcile_observed_nonlive = false});
  HostdConfiguredStartupAuditor auditor(
      *ledger, clock,
      {.api_version =
           std::string(kHostdConfiguredStartupAuditorApiVersion),
       .broker_instance_id = "hostd-crash-qualification",
       .policy = qualification_audit_policy()});
  HostGrantCoordinator coordinator(qualification_coordinator_config(), ledger);
  HostdCoordinatorStartupAdmission admission(coordinator);
  HostdStartupController controller(recovery, admission, auditor, clock, {});

  constexpr std::size_t kMaximumAdvances = 8U;
  for (std::size_t step = 0U; step < kMaximumAdvances; ++step) {
    try {
      const auto status = controller.advance();
      observed.recovery_steps = status.recovery_steps;
      if (status.phase == HostdStartupPhase::admitting) {
        observed.admitted = true;
        observed.admission_detail = "startup admission committed";
        break;
      }
      if (status.last_recovery && step == 0U)
        observed.first_cleanup = status.last_recovery->cleanup_before;
      observed.admission_detail =
          "startup controller is still reconciling durable records";
    } catch (const std::exception& error) {
      observed.admitted = false;
      observed.admission_detail = error.what();
      break;
    }
  }

  // Recovery must be replayable: a second pass over the converged ledger may
  // observe records, but it may not mutate anything.
  observed.second_cleanup = cleanup.recover();
  observed.active_fences =
      static_cast<std::size_t>(ledger->occupancy().active_fences.size());
  observed.generation = ledger->generation(resource);
  observed.chain_verified = ledger->verify();
  return observed;
}

HostdCrashCaseReceipt unqualified_case(HostdCrashPoint point,
                                       HostdCrashUnqualifiedReason reason,
                                       std::string detail) {
  return {.crash_point = point,
          .executor = hostd_crash_point_executor(point),
          .status = HostdCrashCaseStatus::unqualified,
          .unqualified_reason = reason,
          .crash_delivered = false,
          .crashed_pid = 0,
          .detail = std::move(detail),
          .invariants = {},
          .evidence = {}};
}

void record_finding(std::vector<HostdCrashQualificationFinding>& findings,
                    HostdCrashQualificationFinding finding) {
  const auto found = std::ranges::find_if(
      findings, [&finding](const HostdCrashQualificationFinding& value) {
        return value.code == finding.code && value.subject == finding.subject;
      });
  if (found == findings.end())
    findings.push_back(std::move(finding));
}

HostdCrashCaseReceipt run_durable_case(
    HostdCrashPoint point, const HostdCrashQualificationConfig& config,
    const std::filesystem::path& cgroup_parent,
    std::vector<HostdCrashQualificationFinding>& findings) {
  const std::string name =
      "case-" + std::to_string(static_cast<int>(point));
  const std::filesystem::path workspace = config.workspace / name;
  std::filesystem::create_directories(workspace);
  if (::chmod(workspace.c_str(), 0700) != 0)
    throw HostdCrashQualificationError(
        "could not protect the disposable case workspace");
  DisposableCgroupRoot cgroup_root(cgroup_parent, name);
  const std::filesystem::path ledger_path = workspace / "host-ledger.sqlite3";
  const DurablePlan plan = durable_plan(point);
  const DurableExpectation expectation = durable_expectation(point);
  const LinuxCgroupAuthorityConfig cgroup_config =
      cgroup_root.authority_config();

  const CrashOutcome crash = run_crashing_child([&] {
    durable_child_body(ledger_path, cgroup_config, plan);
    // Reaching here without a crash means the declared window never opened.
    ::_exit(93);
  }, config.case_timeout_ms);
  require(!crash.timed_out && crash.signalled &&
              crash.signal_number == SIGKILL,
          "declared crash window did not destroy the process (exit status " +
              std::to_string(crash.exit_status) + ", signal " +
              std::to_string(crash.signal_number) + ", detail: " +
              crash.failure + ")");

  const RestartObservation observed = observe_restart(
      ledger_path, cgroup_config, qualification_resource_id());
  require(observed.chain_verified,
          "restart could not verify the ledger hash chain");
  require(observed.unclosed_records == expectation.unclosed_records,
          "restart observed an unexpected unclosed process record count");
  require(observed.terminal_release_records ==
              expectation.terminal_release_records,
          "restart observed an unexpected terminal release record count");
  require(observed.second_cleanup.allocations_released == 0U &&
              observed.second_cleanup.cgroups_removed == 0U &&
              observed.second_cleanup.intent_cgroups_removed == 0U,
          "repeated restart recovery mutated already converged state");
  if (expectation.releases_grant) {
    require(observed.active_fences == 0U,
            "converged restart recovery did not release the physical grant");
    if (!observed.admitted) {
      // Recovery converged, so admission is the remaining blocker. That is a
      // defect in the startup stack rather than a property of this window.
      const bool epoch_exhausted =
          observed.admission_detail.find("another admission epoch") !=
          std::string::npos;
      record_finding(
          findings,
          {.code = epoch_exhausted
                       ? "startup-admission-epoch-not-renewable-after-restart"
                       : "startup-admission-blocked-after-convergence",
           .subject = epoch_exhausted
                          ? "trainvm::SQLiteHostLedger::"
                            "finalize_startup_admission"
                          : "trainvm::HostdStartupController",
           .detail =
               "restart recovery converged with no durable record remaining, "
               "but the restarted daemon could not commit its admission "
               "audit: " +
               observed.admission_detail});
    }
  } else {
    require(observed.active_fences == 1U,
            "restart recovery released a grant it could not prove terminal");
    require(!observed.admitted,
            "hostd admitted while a physical grant remained unreconciled");
  }
  require(observed.generation == 1U,
          "resource generation is not exactly one after a single grant");

  std::vector<HostdCrashInvariant> invariants{
      HostdCrashInvariant::ledger_chain_intact,
      HostdCrashInvariant::no_leaked_physical_grant,
      HostdCrashInvariant::monotonic_generations,
      HostdCrashInvariant::recovery_is_idempotent,
  };
  if (expectation.releases_grant)
    invariants.push_back(HostdCrashInvariant::no_double_release);
  else
    invariants.push_back(
        HostdCrashInvariant::admission_withheld_until_converged);
  if (observed.launch_replay_checked)
    invariants.push_back(HostdCrashInvariant::no_double_launch);

  return {.crash_point = point,
          .executor = HostdCrashExecutor::durable_ledger,
          .status = HostdCrashCaseStatus::qualified,
          .unqualified_reason = HostdCrashUnqualifiedReason::none,
          .crash_delivered = true,
          .crashed_pid = crash.pid,
          .detail = observed.admission_detail,
          .invariants = std::move(invariants),
          .evidence = {
              {"unclosed_records", std::to_string(observed.unclosed_records)},
              {"terminal_release_records",
               std::to_string(observed.terminal_release_records)},
              {"recovery_steps", std::to_string(observed.recovery_steps)},
              {"active_fences_after",
               std::to_string(observed.active_fences)},
              {"resource_generation", std::to_string(observed.generation)},
              {"admitted", observed.admitted ? "true" : "false"},
              {"launch_retry_replayed",
               observed.launch_replay_checked ? "true" : "not_applicable"},
              {"spawn_retry_replayed",
               observed.spawn_replay_checked ? "true" : "not_applicable"},
          }};
}

// A live worker that outlives its daemon, observed through the production
// pidfd recovery probe. The daemon is a real forked process that is SIGKILLed
// while the worker keeps running.
struct SurvivingWorker final {
  ~SurvivingWorker() {
    if (pid > 0) {
      (void)::kill(static_cast<pid_t>(pid), SIGKILL);
      int status = 0;
      while (::waitpid(static_cast<pid_t>(pid), &status, 0) < 0 &&
             errno == EINTR) {
      }
    }
  }
  SurvivingWorker() = default;
  SurvivingWorker(const SurvivingWorker&) = delete;
  SurvivingWorker& operator=(const SurvivingWorker&) = delete;

  std::int64_t pid{};
};

std::int64_t spawn_surviving_worker() {
  const pid_t child = ::fork();
  if (child < 0)
    throw HostdCrashQualificationError("qualification worker fork failed");
  if (child == 0) {
    // A worker with no business of its own: it exists to be observed.
    for (;;)
      (void)::pause();
  }
  return child;
}

HostProcessSpawnRequest surviving_worker_identity(std::int64_t pid) {
  HostProcessLaunchIntent intent;
  intent.request.launch_id = "launch-surviving-worker";
  intent.boot_id = live_boot_id();
  intent.receipt_digest = fixed_digest('7');
  const LinuxAllocationCgroupIdentity cgroup = observed_cgroup_identity(pid);
  intent.request.cgroup_path = cgroup.unified_path;
  intent.request.cgroup_device = cgroup.device;
  intent.request.cgroup_inode = cgroup.inode;
  return qualification_spawn_request(intent, pid, observed_starttime(pid),
                                     self_executable_digest(pid));
}

HostdCrashCaseReceipt run_process_case(
    HostdCrashPoint point, const HostdCrashQualificationConfig& config) {
  SurvivingWorker worker;
  worker.pid = spawn_surviving_worker();
  // The observation probe reads /proc; give the child a moment to be visible
  // with a stable start time before the daemon dies.
  (void)observed_starttime(worker.pid);

  const CrashOutcome crash = run_crashing_child([] {
    (void)::raise(SIGKILL);
    ::_exit(97);
  }, config.case_timeout_ms);
  require(!crash.timed_out && crash.signalled &&
              crash.signal_number == SIGKILL,
          "daemon crash was not delivered before restart observation");

  const LinuxProcessRecoveryProbe probe;
  const HostProcessSpawnRequest identity =
      surviving_worker_identity(worker.pid);
  std::map<std::string, std::string> evidence{
      {"worker_pid", std::to_string(worker.pid)},
      {"daemon_pid", std::to_string(crash.pid)},
  };
  std::vector<HostdCrashInvariant> invariants;
  std::string detail;

  switch (point) {
    case HostdCrashPoint::live_worker_exact_adoption: {
      LinuxProcessRecoverySet set;
      HostProcessRecoveryRecord record;
      record.intent.request.launch_id = identity.launch_id;
      record.intent.receipt_digest = identity.launch_intent_digest;
      HostProcessSpawnReceipt receipt;
      receipt.request = identity;
      record.spawn = receipt;
      std::vector<HostProcessRecoveryRecord> records;
      records.push_back(std::move(record));
      set.recover(std::move(records), probe);
      require(set.summary().exact_live == 1U,
              "restart did not recover the exact live worker identity");
      require(set.exact_live_process(identity.launch_id) != nullptr,
              "exact live recovery did not pin a pidfd");
      auto adopted = set.take_exact_live_process_for_adoption(identity.launch_id);
      require(adopted.has_value() && adopted->alive(),
              "adoption transfer did not yield the live pinned pidfd");
      require(!set.take_exact_live_process_for_adoption(identity.launch_id)
                   .has_value(),
              "adoption authority was transferred more than once");
      invariants = {HostdCrashInvariant::single_adoption_transfer,
                    HostdCrashInvariant::no_unauthorized_adoption};
      detail = "exact identity adopted once through a pinned pidfd";
      break;
    }
    case HostdCrashPoint::live_worker_orphan_termination: {
      LinuxProcessRecoveryResult result = probe.observe(identity);
      require(result.disposition ==
                      LinuxProcessRecoveryDisposition::exact_live_process &&
                  result.process.has_value(),
              "orphan termination requires an exact live observation");
      const auto terminated = result.process->request_termination();
      require(terminated.disposition ==
                  LinuxRecoveredTerminationDisposition::delivered,
              "SIGKILL was not delivered through the pinned pidfd");
      int status = 0;
      while (::waitpid(static_cast<pid_t>(worker.pid), &status, 0) < 0 &&
             errno == EINTR) {
      }
      require(WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL,
              "orphan worker did not terminate from the pidfd signal");
      worker.pid = 0;
      const LinuxProcessRecoveryResult after = probe.observe(identity);
      require(after.disposition == LinuxProcessRecoveryDisposition::already_gone &&
                  !after.process.has_value(),
              "terminated orphan still exposes recovery authority");
      invariants = {HostdCrashInvariant::termination_only_through_pinned_pidfd,
                    HostdCrashInvariant::no_unauthorized_adoption};
      detail = "orphan terminated through its pinned pidfd and reobserved gone";
      break;
    }
    case HostdCrashPoint::mismatched_identity_refuses_adoption: {
      HostProcessSpawnRequest forged = identity;
      forged.process_starttime_ticks += 1U;
      forged.canonical_request_digest.clear();
      forged = seal_host_process_spawn_request(std::move(forged));
      const LinuxProcessRecoveryResult result = probe.observe(forged);
      require(result.disposition ==
                  LinuxProcessRecoveryDisposition::identity_mismatch,
              "a mismatched start time was not refused");
      require(!result.process.has_value(),
              "a refused identity still retained process authority");
      HostProcessSpawnRequest wrong_executable = identity;
      wrong_executable.executable_digest = fixed_digest('8');
      wrong_executable.canonical_request_digest.clear();
      wrong_executable =
          seal_host_process_spawn_request(std::move(wrong_executable));
      const LinuxProcessRecoveryResult executable_result =
          probe.observe(wrong_executable);
      require(executable_result.disposition ==
                      LinuxProcessRecoveryDisposition::identity_mismatch &&
                  !executable_result.process.has_value(),
              "a mismatched executable digest was not refused");
      invariants = {HostdCrashInvariant::no_unauthorized_adoption};
      detail = "start-time and executable mismatches both refuse adoption";
      break;
    }
    case HostdCrashPoint::absent_pid_refuses_adoption: {
      const CrashOutcome gone = run_crashing_child([] {
        (void)::raise(SIGKILL);
        ::_exit(97);
      }, config.case_timeout_ms);
      require(!gone.timed_out && gone.signalled,
              "absent-pid fixture did not terminate");
      HostProcessSpawnRequest absent = identity;
      absent.host_pid = gone.pid;
      absent.canonical_request_digest.clear();
      absent = seal_host_process_spawn_request(std::move(absent));
      const LinuxProcessRecoveryResult result = probe.observe(absent);
      require(result.disposition ==
                      LinuxProcessRecoveryDisposition::already_gone &&
                  !result.process.has_value(),
              "a reaped pid was not classified as already gone");
      evidence["absent_pid"] = std::to_string(gone.pid);
      invariants = {HostdCrashInvariant::no_unauthorized_adoption};
      detail = "a reaped pid yields no adoption authority";
      break;
    }
    default:
      throw HostdCrashQualificationError(
          "crash point has no real-process executor");
  }

  return {.crash_point = point,
          .executor = HostdCrashExecutor::real_process,
          .status = HostdCrashCaseStatus::qualified,
          .unqualified_reason = HostdCrashUnqualifiedReason::none,
          .crash_delivered = true,
          .crashed_pid = crash.pid,
          .detail = std::move(detail),
          .invariants = std::move(invariants),
          .evidence = std::move(evidence)};
}

HostdCrashCaseReceipt run_cgroup_case(
    HostdCrashPoint point, const HostdCrashQualificationConfig& config,
    const std::filesystem::path& cgroup_parent) {
  const std::string name = "cgroup-" + std::to_string(static_cast<int>(point));
  const std::filesystem::path workspace = config.workspace / name;
  std::filesystem::create_directories(workspace);
  DisposableCgroupRoot root(cgroup_parent, name);
  LinuxCgroupAuthority cgroups(root.authority_config());

  const std::string allocation = "allocation-cgroup-qualification";
  const std::string launch = "launch-cgroup-qualification";
  LinuxAllocationCgroupIdentity identity;
  {
    auto cgroup = cgroups.open_or_create(allocation, launch);
    cgroup.retain_for_durable_intent();
    identity = cgroup.identity();
  }
  require(std::filesystem::exists(
              root.path() /
              hostd_linux_cgroup_authority_test_seam::allocation_cgroup_name(
                  allocation, launch)),
          "retained allocation cgroup did not survive its launch attempt");

  std::vector<HostdCrashInvariant> invariants;
  std::string detail;
  std::map<std::string, std::string> evidence{
      {"cgroup_unified_path", identity.unified_path},
      {"cgroup_inode", std::to_string(identity.inode)},
  };

  if (point == HostdCrashPoint::intent_cgroup_termination) {
    const auto first =
        cgroups.terminate_intent_or_confirm_absent(allocation, launch, identity);
    require(first == LinuxTerminalCgroupCleanupDisposition::removed,
            "intent-only cgroup recovery did not remove the exact directory");
    const auto second =
        cgroups.terminate_intent_or_confirm_absent(allocation, launch, identity);
    require(second == LinuxTerminalCgroupCleanupDisposition::already_absent,
            "repeated intent cgroup recovery was not idempotent");
    invariants = {HostdCrashInvariant::recovery_is_idempotent};
    detail = "abandoned intent cgroup removed once and then already absent";
  } else if (point == HostdCrashPoint::terminal_cgroup_cleanup) {
    const auto first = cgroups.cleanup_terminal_or_confirm_absent(
        allocation, launch, identity);
    require(first == LinuxTerminalCgroupCleanupDisposition::removed,
            "terminal cgroup cleanup did not remove the exact directory");
    const auto second = cgroups.cleanup_terminal_or_confirm_absent(
        allocation, launch, identity);
    require(second == LinuxTerminalCgroupCleanupDisposition::already_absent,
            "repeated terminal cgroup cleanup was not idempotent");
    invariants = {HostdCrashInvariant::recovery_is_idempotent};
    detail = "terminal cgroup removed once and then already absent";
  } else {
    throw HostdCrashQualificationError("crash point has no cgroup executor");
  }

  return {.crash_point = point,
          .executor = HostdCrashExecutor::real_cgroup,
          .status = HostdCrashCaseStatus::qualified,
          .unqualified_reason = HostdCrashUnqualifiedReason::none,
          .crash_delivered = false,
          .crashed_pid = 0,
          .detail = std::move(detail),
          .invariants = std::move(invariants),
          .evidence = std::move(evidence)};
}

ResourceBundleGrant open_qualification_grant(SQLiteHostLedger& ledger,
                                             AuthorityClock& clock,
                                             ResourceBundleRequest request) {
  const HostLedgerAdmissionEpoch epoch = open_admission(ledger, clock);
  const BundleRequestResult result = ledger.request_bundle(
      request, ledger_now(clock), epoch);
  if (!result.grant)
    throw HostdCrashQualificationError(
        "privileged qualification did not acquire its synthetic resource");
  return *result.grant;
}

void cleanup_privileged_workers_on_failure(
    const std::filesystem::path& ledger_path,
    const LinuxCgroupAuthorityConfig& cgroup_config) noexcept;

void run_stopped_child_crashing_daemon(
    const std::filesystem::path& ledger_path,
    const LinuxCgroupAuthorityConfig& cgroup_config,
    const std::filesystem::path& workspace,
    const std::filesystem::path& worker_path) {
  AuthorityClock clock;
  SigkillFault fault(HostLedgerFaultPoint::after_process_spawn_record);
  SQLiteHostLedger ledger(ledger_authority_for(ledger_path),
                          qualification_inventory(), &fault,
                          qualification_audit_policy());
  const ResourceBundleGrant grant =
      open_qualification_grant(
          ledger, clock,
          qualification_device_request("request-privileged-stopped"));
  QualificationWorkerLaunch worker = resolve_qualification_worker(
      workspace, worker_path, grant, "stopped-before-spawn-commit");
  LinuxCgroupAuthority cgroups(cgroup_config);
  LinuxCgroupDeviceKernel device_kernel;
  LinuxDevicePolicyInstaller device_policies(device_kernel);
  LinuxCgroupProcessPolicyKernel process_kernel;
  LinuxProcessPolicyInstaller process_policies(process_kernel);
  LinuxStoppedLauncherKernel launcher;
  QualificationContextAuditor contexts;
  LinuxProcessAuthority processes(
      ledger, clock, cgroups, device_policies, process_policies, launcher,
      {.uid = static_cast<uid_t>(65534U),
       .gid = static_cast<gid_t>(65534U),
       .no_new_privileges = true},
      contexts);
  Descriptor bootstrap(worker.bootstrap.duplicate_fd());
  fault.arm();
  (void)processes.prepare(worker.resolved, grant, bootstrap.get(),
                          worker.bootstrap.spec().bootstrap_digest,
                          worker.process_policy);
  throw HostdCrashQualificationError(
      "spawn-record crash injection returned without SIGKILL");
}

HostdCrashCaseReceipt run_privileged_stopped_child_case(
    const HostdCrashQualificationConfig& config,
    const std::filesystem::path& cgroup_parent) {
  constexpr std::string_view name =
      "case-privileged-stopped-child-before-spawn-commit";
  const auto workspace = config.workspace / std::string(name);
  std::filesystem::create_directories(workspace);
  if (::chmod(workspace.c_str(), 0700) != 0)
    throw HostdCrashQualificationError(
        "could not protect the stopped-child workspace");
  DisposableCgroupRoot cgroup_root(cgroup_parent, std::string(name));
  const auto ledger_path = workspace / "host-ledger.sqlite3";
  const auto worker_path = install_qualification_worker(workspace);
  const auto cgroup_config = cgroup_root.authority_config();

  const CrashOutcome crash = run_crashing_child([&] {
    run_stopped_child_crashing_daemon(ledger_path, cgroup_config, workspace,
                                      worker_path);
  }, config.case_timeout_ms);
  if (crash.timed_out || !crash.signalled ||
      crash.signal_number != SIGKILL) {
    cleanup_privileged_workers_on_failure(ledger_path, cgroup_config);
  }
  require(!crash.timed_out && crash.signalled &&
              crash.signal_number == SIGKILL,
          "stopped-child spawn window did not SIGKILL hostd (exit status " +
              std::to_string(crash.exit_status) + ", detail: " +
              crash.failure + ")");

  const RestartObservation observed = observe_restart(
      ledger_path, cgroup_config, qualification_accelerator_id());
  require(observed.chain_verified,
          "stopped-child recovery could not verify the ledger chain");
  require(observed.unclosed_records == 1U &&
              observed.terminal_release_records == 0U,
          "stopped-child recovery did not retain exactly the durable intent");
  require(observed.launch_replay_checked &&
              !observed.spawn_replay_checked &&
              observed.records_before_replay == observed.records_after_replay,
          "stopped-child recovery did not replay only the durable intent");
  require(observed.active_fences == 0U && observed.generation == 1U,
          "stopped-child recovery leaked or regenerated its physical grant");
  require(observed.admitted,
          "stopped-child recovery did not converge before admission");
  require(observed.second_cleanup.allocations_released == 0U &&
              observed.second_cleanup.cgroups_removed == 0U &&
              observed.second_cleanup.intent_cgroups_removed == 0U,
          "stopped-child recovery was not idempotent");

  return {
      .crash_point =
          HostdCrashPoint::privileged_stopped_child_before_spawn_commit,
      .executor = HostdCrashExecutor::privileged_launch,
      .status = HostdCrashCaseStatus::qualified,
      .unqualified_reason = HostdCrashUnqualifiedReason::none,
      .crash_delivered = true,
      .crashed_pid = crash.pid,
      .detail =
          "real clone3 stopped-child and policy installation completed; "
          "SIGKILL rolled back the spawn row and restart reconciled only the "
          "durable intent",
      .invariants = {HostdCrashInvariant::no_double_launch,
                     HostdCrashInvariant::no_double_release,
                     HostdCrashInvariant::no_leaked_physical_grant,
                     HostdCrashInvariant::monotonic_generations,
                     HostdCrashInvariant::ledger_chain_intact,
                     HostdCrashInvariant::recovery_is_idempotent},
      .evidence =
          {{"unclosed_records_before", "1"},
           {"spawn_receipts_before", "0"},
           {"launch_retry_replayed", "true"},
           {"active_fences_after", "0"},
           {"resource_generation", "1"},
           {"admitted_after_convergence", "true"}},
  };
}

void run_device_policy_crashing_daemon(
    const std::filesystem::path& ledger_path,
    const LinuxCgroupAuthorityConfig& cgroup_config,
    const std::filesystem::path& workspace,
    const std::filesystem::path& worker_path) {
  block_worker_ready_signal();
  AuthorityClock clock;
  SQLiteHostLedger ledger(ledger_authority_for(ledger_path),
                          qualification_inventory(), nullptr,
                          qualification_audit_policy());
  const ResourceBundleGrant grant =
      open_qualification_grant(
          ledger, clock,
          qualification_device_request("request-privileged-device"));
  QualificationWorkerLaunch worker = resolve_qualification_worker(
      workspace, worker_path, grant, "device-policy-recovery");
  LinuxCgroupAuthority cgroups(cgroup_config);
  LinuxCgroupDeviceKernel device_kernel;
  LinuxDevicePolicyInstaller device_policies(device_kernel);
  LinuxCgroupProcessPolicyKernel process_kernel;
  LinuxProcessPolicyInstaller process_policies(process_kernel);
  LinuxStoppedLauncherKernel launcher;
  QualificationContextAuditor contexts;
  LinuxProcessAuthority processes(
      ledger, clock, cgroups, device_policies, process_policies, launcher,
      {.uid = static_cast<uid_t>(65534U),
       .gid = static_cast<gid_t>(65534U),
       .no_new_privileges = true},
      contexts);
  Descriptor bootstrap(worker.bootstrap.duplicate_fd());
  LinuxPreparedLaunch prepared = processes.prepare(
      worker.resolved, grant, bootstrap.get(),
      worker.bootstrap.spec().bootstrap_digest, worker.process_policy);
  processes.release_to_exec(prepared);
  await_worker_ready_signal();
  (void)::raise(SIGKILL);
  ::_exit(97);
}

void terminate_qualification_worker_on_failure(
    const HostProcessSpawnRequest& identity) noexcept {
  try {
    LinuxProcessRecoveryProbe probe;
    LinuxProcessRecoveryResult recovered = probe.observe(identity);
    if (recovered.disposition !=
            LinuxProcessRecoveryDisposition::exact_live_process ||
        !recovered.process) {
      return;
    }
    (void)recovered.process->request_termination();
    for (std::size_t attempt = 0U; attempt < 2000U; ++attempt) {
      if (recovered.process->state() != LinuxPidfdState::live)
        return;
      (void)::usleep(1000U);
    }
  } catch (...) {
  }
}

void cleanup_privileged_workers_on_failure(
    const std::filesystem::path& ledger_path,
    const LinuxCgroupAuthorityConfig& cgroup_config) noexcept {
  try {
    auto ledger = std::make_shared<SQLiteHostLedger>(
        ledger_authority_for(ledger_path), qualification_inventory(), nullptr,
        qualification_audit_policy());
    const auto records = ledger->active_process_recovery_records();
    LinuxCgroupAuthority cgroups(cgroup_config);
    for (const HostProcessRecoveryRecord& record : records) {
      if (record.spawn)
        terminate_qualification_worker_on_failure(record.spawn->request);
      const auto& request = record.intent.request;
      for (std::size_t attempt = 0U; attempt < 2000U; ++attempt) {
        const auto disposition = cgroups.terminate_intent_or_confirm_absent(
            record.grant.allocation_id, request.launch_id,
            {.unified_path = request.cgroup_path,
             .device = request.cgroup_device,
             .inode = request.cgroup_inode});
        if (disposition !=
            LinuxTerminalCgroupCleanupDisposition::termination_pending) {
          break;
        }
        (void)::usleep(1000U);
      }
    }
  } catch (...) {
  }
}

HostdCrashCaseReceipt run_privileged_device_policy_case(
    const HostdCrashQualificationConfig& config,
    const std::filesystem::path& cgroup_parent) {
  constexpr std::string_view name =
      "case-privileged-device-policy-recovery";
  const auto workspace = config.workspace / std::string(name);
  std::filesystem::create_directories(workspace);
  if (::chmod(workspace.c_str(), 0700) != 0)
    throw HostdCrashQualificationError(
        "could not protect the device-policy workspace");
  DisposableCgroupRoot cgroup_root(cgroup_parent, std::string(name));
  const auto ledger_path = workspace / "host-ledger.sqlite3";
  const auto worker_path = install_qualification_worker(workspace);
  const auto cgroup_config = cgroup_root.authority_config();

  const CrashOutcome crash = run_crashing_child([&] {
    run_device_policy_crashing_daemon(ledger_path, cgroup_config, workspace,
                                      worker_path);
  }, config.case_timeout_ms);
  if (crash.timed_out || !crash.signalled ||
      crash.signal_number != SIGKILL) {
    cleanup_privileged_workers_on_failure(ledger_path, cgroup_config);
  }
  require(!crash.timed_out && crash.signalled &&
              crash.signal_number == SIGKILL,
          "device-policy daemon did not SIGKILL after worker exec (exit status " +
              std::to_string(crash.exit_status) + ", detail: " +
              crash.failure + ")");

  auto ledger = std::make_shared<SQLiteHostLedger>(
      ledger_authority_for(ledger_path), qualification_inventory(), nullptr,
      qualification_audit_policy());
  AuthorityClock clock;
  SQLiteHostdTerminalReleaseAuthority records(*ledger, clock);
  const auto before = records.unclosed_records();
  require(before.size() == 1U && before.front().spawn &&
              before.front().spawn->request.device_policy &&
              before.front().spawn->request.process_policy &&
              before.front().spawn->request.worker_credentials,
          "device-policy crash did not preserve one fully bound spawn receipt");
  const HostProcessSpawnRequest worker_identity = before.front().spawn->request;
  const auto device_binding = *worker_identity.device_policy;
  require(device_binding.program_id != 0U &&
              device_binding.installation_digest.starts_with("sha256:"),
          "spawn receipt did not bind the installed device program");

  try {
    LinuxCgroupAuthority cgroups(cgroup_config);
    LinuxCgroupDeviceKernel device_kernel;
    LinuxDevicePolicyInstaller device_policies(device_kernel);
    LinuxCgroupProcessPolicyKernel process_kernel;
    LinuxProcessPolicyInstaller process_policies(process_kernel);
    LinuxStoppedLauncherKernel launcher;
    QualificationContextAuditor contexts;
    LinuxProcessAuthority processes(
        *ledger, clock, cgroups, device_policies, process_policies, launcher,
        {.uid = static_cast<uid_t>(65534U),
         .gid = static_cast<gid_t>(65534U),
         .no_new_privileges = true},
        contexts);
    HostdLinuxProcessSupervisor supervisor(processes);
    LinuxHostdTerminalCgroupCleaner cleaner(cgroups);
    HostdTerminalReleaseRecovery cleanup(records, cleaner);
    HostdRestartProcessRecovery recovery(
        records, cleanup, supervisor,
        {.exact_live_policy =
             HostdExactRecoveredProcessPolicy::terminate_and_reconcile,
         .reconcile_observed_nonlive = true});

    std::size_t recovery_steps = 0U;
    std::size_t exact_live_observations = 0U;
    std::size_t adopted = 0U;
    std::size_t finalized = 0U;
    bool blocked_release_observed = false;
    bool converged = false;
    for (; recovery_steps < 2000U; ++recovery_steps) {
      const HostdRestartProcessRecoverySummary step = recovery.step();
      exact_live_observations += step.observed.exact_live;
      adopted += step.newly_adopted;
      finalized += step.progress.finalized;
      blocked_release_observed =
          blocked_release_observed ||
          step.cleanup_before.allocations_blocked_by_unclosed_process > 0U;
      if (step.remaining_unclosed_records == 0U &&
          step.remaining_terminal_release_records == 0U) {
        converged = true;
        ++recovery_steps;
        break;
      }
      (void)::usleep(1000U);
    }
    require(converged && exact_live_observations >= 1U && adopted == 1U &&
                finalized == 1U && blocked_release_observed,
            "device-policy restart did not adopt, terminate, and converge the "
            "exact live worker once");
    const HostdRestartProcessRecoverySummary replay = recovery.step();
    require(replay.newly_adopted == 0U && replay.nonlive_finalized == 0U &&
                replay.progress.retained_before == 0U &&
                replay.cleanup_before.allocations_released == 0U &&
                replay.cleanup_after.allocations_released == 0U &&
                replay.remaining_unclosed_records == 0U &&
                replay.remaining_terminal_release_records == 0U,
            "device-policy recovery mutated state after convergence");
    require(ledger->occupancy().active_fences.empty() &&
                ledger->generation(qualification_accelerator_id()) == 1U,
            "device-policy recovery leaked or regenerated the host grant");
    std::string ledger_failure;
    require(ledger->verify(&ledger_failure),
            "device-policy recovery could not verify the ledger chain");

    return {
        .crash_point = HostdCrashPoint::privileged_device_policy_recovery,
        .executor = HostdCrashExecutor::privileged_launch,
        .status = HostdCrashCaseStatus::qualified,
        .unqualified_reason = HostdCrashUnqualifiedReason::none,
        .crash_delivered = true,
        .crashed_pid = crash.pid,
        .detail =
            "restart adopted the exact worker through a pinned pidfd, "
            "reattested its durable cgroup-device BPF and process policy, "
            "then terminated and released it once",
        .invariants = {
            HostdCrashInvariant::no_double_release,
            HostdCrashInvariant::no_leaked_physical_grant,
            HostdCrashInvariant::no_unauthorized_adoption,
            HostdCrashInvariant::single_adoption_transfer,
            HostdCrashInvariant::monotonic_generations,
            HostdCrashInvariant::ledger_chain_intact,
            HostdCrashInvariant::recovery_is_idempotent,
            HostdCrashInvariant::termination_only_through_pinned_pidfd},
        .evidence =
            {{"worker_pid", std::to_string(worker_identity.host_pid)},
             {"device_program_id", std::to_string(device_binding.program_id)},
             {"device_program_tag", device_binding.program_tag},
             {"device_installation_digest",
              device_binding.installation_digest},
             {"exact_live_observations",
              std::to_string(exact_live_observations)},
             {"adoption_transfers", std::to_string(adopted)},
             {"terminal_finalizations", std::to_string(finalized)},
             {"recovery_steps", std::to_string(recovery_steps)},
             {"active_fences_after", "0"},
             {"resource_generation", "1"}},
    };
  } catch (...) {
    terminate_qualification_worker_on_failure(worker_identity);
    throw;
  }
}

HostdSocketAuthorityConfig qualification_socket_config(
    const std::filesystem::path& socket_path) {
  return {
      .api_version = std::string(kHostdStatusTransportApiVersion),
      .socket_path = socket_path,
      .expected_owner_uid = ::geteuid(),
      .expected_owner_gid = ::getegid(),
      .expected_parent_mode = 0700U,
      .expected_socket_mode = 0600U,
      .listen_backlog = 8U,
      .enforcement_grade = HostdSocketEnforcementGrade::cooperative_test,
      .fault_injector = nullptr,
  };
}

HostdCrashCaseReceipt run_privileged_socket_restart_case(
    const HostdCrashQualificationConfig& config) {
  const auto workspace =
      config.workspace / "case-privileged-daemon-socket-restart";
  std::filesystem::create_directories(workspace);
  if (::chmod(workspace.c_str(), 0700) != 0)
    throw HostdCrashQualificationError(
        "could not protect the socket-restart workspace");
  const auto ledger_path = workspace / "host-ledger.sqlite3";
  const auto socket_path = workspace / "hostd.sock";
  const auto socket_config = qualification_socket_config(socket_path);

  const CrashOutcome crash = run_crashing_child([&] {
    auto ledger_authority = ledger_authority_for(ledger_path);
    auto singleton =
        std::make_shared<HostdLedgerSingletonToken>(ledger_authority);
    const int parent = ::open(workspace.c_str(),
                              O_PATH | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (parent < 0)
      throw HostdCrashQualificationError(
          "could not open the socket-restart parent");
    Descriptor parent_descriptor(parent);
    auto socket = HostdSocketAuthority::self_bind(
        socket_config, parent_descriptor.get(), singleton);
    (void)socket.reattest();
    (void)::raise(SIGKILL);
    ::_exit(97);
  }, config.case_timeout_ms);
  require(!crash.timed_out && crash.signalled &&
              crash.signal_number == SIGKILL,
          "socket-owning daemon did not die by SIGKILL");

  struct stat stale{};
  require(::lstat(socket_path.c_str(), &stale) == 0 &&
              S_ISSOCK(stale.st_mode),
          "daemon crash did not leave its socket inode for restart recovery");
  auto ledger_authority = ledger_authority_for(ledger_path);
  auto singleton =
      std::make_shared<HostdLedgerSingletonToken>(ledger_authority);
  SQLiteHostLedger ledger(ledger_authority, qualification_inventory());
  std::string ledger_failure;
  require(ledger.verify(&ledger_failure),
          "socket restart ledger verification failed");
  const int parent = ::open(workspace.c_str(),
                            O_PATH | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (parent < 0)
    throw HostdCrashQualificationError(
        "could not reopen the socket-restart parent");
  Descriptor parent_descriptor(parent);
  auto restarted = HostdSocketAuthority::self_bind(
      socket_config, parent_descriptor.get(), singleton);
  const HostdSocketIdentity identity = restarted.reattest();
  require(identity.path_inode != static_cast<std::uint64_t>(stale.st_ino),
          "restart retained the dead daemon socket inode");
  bool second_live_rejected = false;
  try {
    (void)HostdSocketAuthority::self_bind(
        socket_config, parent_descriptor.get(), singleton);
  } catch (const HostdTransportError&) {
    second_live_rejected = true;
  }
  require(second_live_rejected && restarted.reattest() == identity,
          "socket restart did not preserve singleton listener authority");

  return {
      .crash_point = HostdCrashPoint::privileged_daemon_socket_restart,
      .executor = HostdCrashExecutor::privileged_launch,
      .status = HostdCrashCaseStatus::qualified,
      .unqualified_reason = HostdCrashUnqualifiedReason::none,
      .crash_delivered = true,
      .crashed_pid = crash.pid,
      .detail =
          "a SIGKILL-stale socket was replaced only after reacquiring the "
          "durable singleton; a second live listener remained protected",
      .invariants = {HostdCrashInvariant::no_double_launch,
                     HostdCrashInvariant::ledger_chain_intact},
      .evidence =
          {{"stale_socket_inode", std::to_string(stale.st_ino)},
           {"restarted_socket_inode", std::to_string(identity.path_inode)},
           {"socket_owner_uid", std::to_string(identity.owner_uid)},
           {"socket_owner_gid", std::to_string(identity.owner_gid)},
           {"second_live_bind_rejected",
            second_live_rejected ? "true" : "false"}},
  };
}

}  // namespace

std::vector<HostdCrashPoint> declared_hostd_crash_points() {
  return {
      HostdCrashPoint::intent_prepare_window,
      HostdCrashPoint::intent_commit_window,
      HostdCrashPoint::spawn_prepare_window,
      HostdCrashPoint::spawn_commit_window,
      HostdCrashPoint::exit_prepare_window,
      HostdCrashPoint::release_prepare_window,
      HostdCrashPoint::release_reply_lost,
      HostdCrashPoint::live_worker_exact_adoption,
      HostdCrashPoint::live_worker_orphan_termination,
      HostdCrashPoint::mismatched_identity_refuses_adoption,
      HostdCrashPoint::absent_pid_refuses_adoption,
      HostdCrashPoint::intent_cgroup_termination,
      HostdCrashPoint::terminal_cgroup_cleanup,
      HostdCrashPoint::privileged_stopped_child_before_spawn_commit,
      HostdCrashPoint::privileged_device_policy_recovery,
      HostdCrashPoint::privileged_daemon_socket_restart,
  };
}

HostdCrashExecutor hostd_crash_point_executor(HostdCrashPoint point) {
  switch (point) {
    case HostdCrashPoint::intent_prepare_window:
    case HostdCrashPoint::intent_commit_window:
    case HostdCrashPoint::spawn_prepare_window:
    case HostdCrashPoint::spawn_commit_window:
    case HostdCrashPoint::exit_prepare_window:
    case HostdCrashPoint::release_prepare_window:
    case HostdCrashPoint::release_reply_lost:
      return HostdCrashExecutor::durable_ledger;
    case HostdCrashPoint::live_worker_exact_adoption:
    case HostdCrashPoint::live_worker_orphan_termination:
    case HostdCrashPoint::mismatched_identity_refuses_adoption:
    case HostdCrashPoint::absent_pid_refuses_adoption:
      return HostdCrashExecutor::real_process;
    case HostdCrashPoint::intent_cgroup_termination:
    case HostdCrashPoint::terminal_cgroup_cleanup:
      return HostdCrashExecutor::real_cgroup;
    case HostdCrashPoint::privileged_stopped_child_before_spawn_commit:
    case HostdCrashPoint::privileged_device_policy_recovery:
    case HostdCrashPoint::privileged_daemon_socket_restart:
      return HostdCrashExecutor::privileged_launch;
  }
  throw HostdCrashQualificationError("unknown hostd crash point");
}

nlohmann::json hostd_crash_qualification_receipt_json(
    const HostdCrashQualificationReceipt& receipt) {
  return encode_json(receipt);
}

std::string hostd_crash_qualification_receipt_digest(
    const HostdCrashQualificationReceipt& receipt) {
  HostdCrashQualificationReceipt material = receipt;
  material.receipt_digest.clear();
  return digest_of(hostd_crash_qualification_receipt_json(material).dump());
}

void validate_hostd_crash_qualification_receipt(
    const HostdCrashQualificationReceipt& receipt) {
  require(receipt.api_version == kHostdCrashQualificationApiVersion,
          "crash qualification receipt api version is unknown");
  require(canonical_sha256_digest(receipt.qualification_binary_digest),
          "crash qualification receipt has no exact binary identity");
  const auto declared = declared_hostd_crash_points();
  require(receipt.cases.size() == declared.size(),
          "crash qualification receipt does not cover every declared point");
  require(receipt.declared_points == declared.size(),
          "crash qualification declared count disagrees with the contract");
  std::size_t qualified = 0U;
  std::vector<HostdCrashPoint> blocking;
  for (std::size_t index = 0U; index < declared.size(); ++index) {
    const HostdCrashCaseReceipt& value = receipt.cases[index];
    require(value.crash_point == declared[index],
            "crash qualification cases are not in declared order");
    require(value.executor == hostd_crash_point_executor(value.crash_point),
            "crash qualification case uses an undeclared executor");
    if (value.status == HostdCrashCaseStatus::qualified) {
      require(value.unqualified_reason == HostdCrashUnqualifiedReason::none,
              "a qualified case cannot carry an unqualified reason");
      require(!value.invariants.empty(),
              "a qualified case must name the invariants it proved");
      ++qualified;
    } else {
      require(value.unqualified_reason != HostdCrashUnqualifiedReason::none,
              "an unqualified case must name its reason");
      require(value.invariants.empty(),
              "an unqualified case cannot claim proven invariants");
      blocking.push_back(value.crash_point);
    }
  }
  require(receipt.qualified_points == qualified &&
              receipt.unqualified_points == declared.size() - qualified,
          "crash qualification totals disagree with its cases");
  require(receipt.blocking_points == blocking,
          "crash qualification blocking points disagree with its cases");
  require(receipt.gate_open == (blocking.empty() && receipt.findings.empty()),
          "crash qualification gate disagrees with its blocking points or "
          "recovery-stack findings");
  for (const HostdCrashQualificationFinding& finding : receipt.findings)
    require(!finding.code.empty() && !finding.subject.empty() &&
                !finding.detail.empty(),
            "a crash qualification finding must be fully attributed");
  require(receipt.receipt_digest ==
              hostd_crash_qualification_receipt_digest(receipt),
          "crash qualification receipt digest does not bind its content");
}

HostdCrashQualificationHost probe_hostd_crash_qualification_host(
    const HostdCrashQualificationConfig& config) {
  HostdCrashQualificationHost host;
  host.effective_uid = static_cast<std::uint32_t>(::geteuid());
  host.effective_gid = static_cast<std::uint32_t>(::getegid());
  host.root_authority = ::geteuid() == 0U;
  struct statfs filesystem{};
  host.cgroup_v2 = ::statfs("/sys/fs/cgroup", &filesystem) == 0 &&
                   filesystem.f_type == CGROUP2_SUPER_MAGIC;
  struct utsname name{};
  host.kernel_release = ::uname(&name) == 0 ? name.release : "";

  const std::filesystem::path parent =
      resolve_cgroup_parent(config.cgroup_parent);
  std::error_code failure;
  const std::filesystem::path probe =
      parent / ".trainvm-crash-qualification-probe";
  if (host.cgroup_v2 && std::filesystem::exists(parent, failure) &&
      std::filesystem::create_directory(probe, failure) && !failure) {
    host.cgroup_delegation = true;
    std::filesystem::remove(probe, failure);
    const std::filesystem::path root("/sys/fs/cgroup");
    host.cgroup_root_unified_path =
        "/" + std::filesystem::relative(parent, root, failure).string();
    if (failure)
      host.cgroup_root_unified_path.clear();
  }
  return host;
}

HostdCrashQualificationReceipt qualify_hostd_crash_recovery(
    const HostdCrashQualificationConfig& config) {
  require(config.workspace.is_absolute(),
          "the disposable qualification workspace must be an absolute path");
  require(config.case_timeout_ms >= 1000 &&
              config.case_timeout_ms <= 60'000,
          "the destructive case timeout must be between 1 and 60 seconds");
  std::error_code failure;
  std::filesystem::create_directories(config.workspace, failure);
  require(!failure, "could not create the disposable qualification workspace");

  HostdCrashQualificationReceipt receipt;
  receipt.qualification_binary_digest = self_executable_digest(::getpid());
  receipt.host = probe_hostd_crash_qualification_host(config);
  const std::filesystem::path cgroup_parent =
      resolve_cgroup_parent(config.cgroup_parent);

  for (const HostdCrashPoint point : declared_hostd_crash_points()) {
    const HostdCrashExecutor executor = hostd_crash_point_executor(point);
    if (executor == HostdCrashExecutor::privileged_launch &&
        !receipt.host.root_authority) {
      receipt.cases.push_back(unqualified_case(
          point, HostdCrashUnqualifiedReason::privilege_unavailable,
          "a root host authority with a distinct non-root worker identity is "
          "required to execute this window"));
      continue;
    }
    if ((executor == HostdCrashExecutor::durable_ledger ||
         executor == HostdCrashExecutor::real_cgroup ||
         (executor == HostdCrashExecutor::privileged_launch &&
          point != HostdCrashPoint::privileged_daemon_socket_restart)) &&
        !receipt.host.cgroup_delegation) {
      receipt.cases.push_back(unqualified_case(
          point, HostdCrashUnqualifiedReason::cgroup_delegation_unavailable,
          "no writable delegated cgroup v2 subtree is available for a "
          "disposable allocation cgroup"));
      continue;
    }
    try {
      switch (executor) {
        case HostdCrashExecutor::durable_ledger:
          receipt.cases.push_back(
              run_durable_case(point, config, cgroup_parent, receipt.findings));
          break;
        case HostdCrashExecutor::real_process:
          receipt.cases.push_back(run_process_case(point, config));
          break;
        case HostdCrashExecutor::real_cgroup:
          receipt.cases.push_back(
              run_cgroup_case(point, config, cgroup_parent));
          break;
        case HostdCrashExecutor::privileged_launch:
          switch (point) {
            case HostdCrashPoint::privileged_stopped_child_before_spawn_commit:
              receipt.cases.push_back(run_privileged_stopped_child_case(
                  config, cgroup_parent));
              break;
            case HostdCrashPoint::privileged_device_policy_recovery:
              receipt.cases.push_back(run_privileged_device_policy_case(
                  config, cgroup_parent));
              break;
            case HostdCrashPoint::privileged_daemon_socket_restart:
              receipt.cases.push_back(
                  run_privileged_socket_restart_case(config));
              break;
            default:
              throw HostdCrashQualificationError(
                  "crash point has no privileged launch executor");
          }
          break;
      }
    } catch (const HostdCrashQualificationError& error) {
      receipt.cases.push_back(unqualified_case(
          point, HostdCrashUnqualifiedReason::invariant_violated,
          error.what()));
    } catch (const std::exception& error) {
      receipt.cases.push_back(unqualified_case(
          point, HostdCrashUnqualifiedReason::executor_error, error.what()));
    }
  }

  receipt.declared_points = receipt.cases.size();
  for (const HostdCrashCaseReceipt& value : receipt.cases) {
    if (value.status == HostdCrashCaseStatus::qualified) {
      ++receipt.qualified_points;
    } else {
      ++receipt.unqualified_points;
      receipt.blocking_points.push_back(value.crash_point);
    }
  }
  receipt.gate_open =
      receipt.blocking_points.empty() && receipt.findings.empty();
  receipt.receipt_digest = hostd_crash_qualification_receipt_digest(receipt);
  validate_hostd_crash_qualification_receipt(receipt);
  return receipt;
}

}  // namespace trainvm
