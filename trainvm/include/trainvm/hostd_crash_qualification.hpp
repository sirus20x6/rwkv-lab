#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "trainvm/json.hpp"

namespace trainvm {

inline constexpr std::string_view kHostdCrashQualificationApiVersion =
    "trainvm.hostd-crash-qualification/v1";

class HostdCrashQualificationError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Every declared destructive crash window. The enumeration is the contract:
// a receipt that omits a point is invalid, and a point that cannot be executed
// on the current host is reported as unqualified rather than silently dropped.
enum class HostdCrashPoint {
  // Durable ledger prepare/commit windows. A real child process is SIGKILLed
  // from inside the ledger transaction at the named fault boundary.
  intent_prepare_window,
  intent_commit_window,
  spawn_prepare_window,
  spawn_commit_window,
  exit_prepare_window,
  release_prepare_window,
  release_reply_lost,
  // Real-process restart observation of a worker that outlived its daemon.
  live_worker_exact_adoption,
  live_worker_orphan_termination,
  mismatched_identity_refuses_adoption,
  absent_pid_refuses_adoption,
  // Real cgroup recovery on a disposable delegated subtree.
  intent_cgroup_termination,
  terminal_cgroup_cleanup,
  // Privileged launch authority. These require a root host authority with a
  // distinct non-root worker identity and cgroup device enforcement.
  privileged_stopped_child_before_spawn_commit,
  privileged_device_policy_recovery,
  privileged_daemon_socket_restart,
};

enum class HostdCrashExecutor {
  durable_ledger,
  real_process,
  real_cgroup,
  privileged_launch,
};

enum class HostdCrashCaseStatus {
  qualified,
  unqualified,
};

// Named convergence properties. A case lists only the invariants it actually
// observed; an unqualified case lists none.
enum class HostdCrashInvariant {
  no_double_launch,
  no_double_release,
  no_leaked_physical_grant,
  no_unauthorized_adoption,
  single_adoption_transfer,
  monotonic_generations,
  ledger_chain_intact,
  recovery_is_idempotent,
  admission_withheld_until_converged,
  termination_only_through_pinned_pidfd,
};

enum class HostdCrashUnqualifiedReason {
  none,
  privilege_unavailable,
  cgroup_delegation_unavailable,
  executor_error,
  invariant_violated,
};

struct HostdCrashCaseReceipt final {
  HostdCrashPoint crash_point{};
  HostdCrashExecutor executor{};
  HostdCrashCaseStatus status{};
  HostdCrashUnqualifiedReason unqualified_reason{
      HostdCrashUnqualifiedReason::none};
  // True only when a real process was destroyed by SIGKILL for this case.
  bool crash_delivered{};
  std::int64_t crashed_pid{};
  std::string detail;
  std::vector<HostdCrashInvariant> invariants;
  std::map<std::string, std::string> evidence;

  bool operator==(const HostdCrashCaseReceipt&) const = default;
};

struct HostdCrashQualificationHost final {
  std::uint32_t effective_uid{};
  std::uint32_t effective_gid{};
  bool root_authority{};
  bool cgroup_v2{};
  bool cgroup_delegation{};
  std::string cgroup_root_unified_path;
  std::string kernel_release;

  bool operator==(const HostdCrashQualificationHost&) const = default;
};

// A defect the destructive matrix observed in the recovery stack itself,
// independent of any single crash window. A finding keeps the gate closed.
struct HostdCrashQualificationFinding final {
  std::string code;
  std::string subject;
  std::string detail;

  bool operator==(const HostdCrashQualificationFinding&) const = default;
};

struct HostdCrashQualificationReceipt final {
  std::string api_version{std::string(kHostdCrashQualificationApiVersion)};
  HostdCrashQualificationHost host;
  std::vector<HostdCrashCaseReceipt> cases;
  std::vector<HostdCrashQualificationFinding> findings;
  std::size_t declared_points{};
  std::size_t qualified_points{};
  std::size_t unqualified_points{};
  // The deployment gate. It is open only when every declared crash point is
  // qualified on this host and no finding was raised; either keeps it closed.
  bool gate_open{};
  std::vector<HostdCrashPoint> blocking_points;
  std::string receipt_digest;

  bool operator==(const HostdCrashQualificationReceipt&) const = default;
};

struct HostdCrashQualificationConfig final {
  // Disposable host root. Everything the qualification creates lives beneath
  // it and is removed afterwards; nothing touches a live deployment.
  std::filesystem::path workspace;
  // Parent of the disposable cgroup subtree. Empty selects the caller's
  // delegated systemd user scope when one is writable.
  std::filesystem::path cgroup_parent;
  // Wall-clock bound for a single destructive case.
  std::int64_t case_timeout_ms{15'000};

  bool operator==(const HostdCrashQualificationConfig&) const = default;
};

[[nodiscard]] std::vector<HostdCrashPoint> declared_hostd_crash_points();
[[nodiscard]] HostdCrashExecutor hostd_crash_point_executor(
    HostdCrashPoint point);

[[nodiscard]] HostdCrashQualificationHost probe_hostd_crash_qualification_host(
    const HostdCrashQualificationConfig& config);

// Runs the destructive matrix. Each case forks and kills real processes; it
// must only be pointed at a disposable host.
[[nodiscard]] HostdCrashQualificationReceipt qualify_hostd_crash_recovery(
    const HostdCrashQualificationConfig& config);

[[nodiscard]] nlohmann::json hostd_crash_qualification_receipt_json(
    const HostdCrashQualificationReceipt& receipt);
[[nodiscard]] std::string hostd_crash_qualification_receipt_digest(
    const HostdCrashQualificationReceipt& receipt);
// Rejects a receipt that omits a declared point, duplicates one, claims
// invariants on an unqualified case, or disagrees with its own gate.
void validate_hostd_crash_qualification_receipt(
    const HostdCrashQualificationReceipt& receipt);

}  // namespace trainvm
