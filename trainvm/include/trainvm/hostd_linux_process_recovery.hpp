#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "trainvm/host_ledger.hpp"

namespace trainvm {

enum class LinuxProcessRecoveryDisposition {
  intent_only,
  exact_live_process,
  already_gone,
  identity_mismatch,
  observation_failed,
};

enum class LinuxPidfdState {
  live,
  terminal,
  observation_failed,
};

enum class LinuxRecoveredTerminationDisposition {
  delivered,
  already_terminal,
  observation_failed,
};

struct LinuxRecoveredTerminationResult final {
  LinuxRecoveredTerminationDisposition disposition{};
  std::string detail;

  bool operator==(const LinuxRecoveredTerminationResult&) const = default;
};

class LinuxRecoveredProcess final {
 public:
  LinuxRecoveredProcess(LinuxRecoveredProcess&& other) noexcept;
  LinuxRecoveredProcess& operator=(LinuxRecoveredProcess&& other) noexcept;
  ~LinuxRecoveredProcess();

  LinuxRecoveredProcess(const LinuxRecoveredProcess&) = delete;
  LinuxRecoveredProcess& operator=(const LinuxRecoveredProcess&) = delete;

  [[nodiscard]] const HostProcessSpawnRequest& identity() const noexcept;
  [[nodiscard]] LinuxPidfdState state() const noexcept;
  [[nodiscard]] bool alive() const noexcept;
  [[nodiscard]] std::optional<std::string> terminal_observation_digest()
      const;
  // Sends SIGKILL only through the already-pinned exact pidfd. It never falls
  // back to kill(numeric_pid), so PID reuse cannot redirect the signal.
  [[nodiscard]] LinuxRecoveredTerminationResult request_termination() noexcept;

 private:
  friend class LinuxProcessRecoveryProbe;
  LinuxRecoveredProcess(HostProcessSpawnRequest identity, int pidfd) noexcept;

  HostProcessSpawnRequest identity_;
  int pidfd_{-1};
};

struct LinuxProcessRecoveryResult final {
  LinuxProcessRecoveryDisposition disposition{};
  std::string detail;
  std::string evidence_digest;
  std::optional<LinuxRecoveredProcess> process;

  LinuxProcessRecoveryResult(
      LinuxProcessRecoveryDisposition disposition_value,
      std::string detail_value,
      std::string evidence_digest_value,
      std::optional<LinuxRecoveredProcess> process_value = std::nullopt)
      : disposition(disposition_value), detail(std::move(detail_value)),
        evidence_digest(std::move(evidence_digest_value)),
        process(std::move(process_value)) {}

  LinuxProcessRecoveryResult(LinuxProcessRecoveryResult&&) noexcept = default;
  LinuxProcessRecoveryResult& operator=(LinuxProcessRecoveryResult&&) noexcept =
      default;
  LinuxProcessRecoveryResult(const LinuxProcessRecoveryResult&) = delete;
  LinuxProcessRecoveryResult& operator=(const LinuxProcessRecoveryResult&) =
      delete;
};

// Read-only startup primitive. It opens the recorded PID through pidfd_open,
// double-samples proc starttime/cgroup identity around executable hashing, and
// pins the exact cgroup inode before returning a live handle. It never signals
// or adopts ownership merely because a numeric PID exists.
class LinuxProcessRecoveryProbe final {
 public:
  [[nodiscard]] LinuxProcessRecoveryResult observe(
      const HostProcessSpawnRequest& expected) const;
};

struct LinuxProcessRecoverySummary final {
  std::size_t records{};
  std::size_t exact_live{};
  std::size_t already_gone{};
  std::size_t identity_mismatch{};
  std::size_t observation_failed{};
  std::size_t intent_only{};

  bool operator==(const LinuxProcessRecoverySummary&) const = default;
};

struct LinuxProcessRecoveryEntry final {
  HostProcessRecoveryRecord record;
  LinuxProcessRecoveryDisposition disposition{};
  std::string detail;
  std::string evidence_digest;
  std::optional<LinuxRecoveredProcess> process;

  LinuxProcessRecoveryEntry(
      HostProcessRecoveryRecord record_value,
      LinuxProcessRecoveryDisposition disposition_value,
      std::string detail_value,
      std::string evidence_digest_value,
      std::optional<LinuxRecoveredProcess> process_value = std::nullopt)
      : record(std::move(record_value)),
        disposition(disposition_value),
        detail(std::move(detail_value)),
        evidence_digest(std::move(evidence_digest_value)),
        process(std::move(process_value)) {}
  LinuxProcessRecoveryEntry(LinuxProcessRecoveryEntry&&) noexcept = default;
  LinuxProcessRecoveryEntry& operator=(LinuxProcessRecoveryEntry&&) noexcept =
      default;
  LinuxProcessRecoveryEntry(const LinuxProcessRecoveryEntry&) = delete;
  LinuxProcessRecoveryEntry& operator=(const LinuxProcessRecoveryEntry&) =
      delete;
};

// One-shot recovery boundary. Durable records are classified exactly once and
// exact live identities remain pinned by pidfd until a future supervisor takes
// the handle for adoption or this set is destroyed. Re-reading a numeric PID
// after this boundary is deliberately not an adoption mechanism.
class LinuxProcessRecoverySet final {
 public:
  void recover(std::vector<HostProcessRecoveryRecord> records,
               const LinuxProcessRecoveryProbe& probe);

  [[nodiscard]] bool initialized() const noexcept;
  [[nodiscard]] const LinuxProcessRecoverySummary& summary() const noexcept;
  [[nodiscard]] const std::vector<LinuxProcessRecoveryEntry>& entries()
      const noexcept;
  [[nodiscard]] const LinuxRecoveredProcess* exact_live_process(
      std::string_view launch_id) const noexcept;
  [[nodiscard]] std::optional<LinuxRecoveredProcess>
  take_exact_live_process_for_adoption(std::string_view launch_id);

 private:
  bool initialized_{};
  LinuxProcessRecoverySummary summary_;
  std::vector<LinuxProcessRecoveryEntry> entries_;
};

namespace hostd_linux_process_recovery_test_seam {

[[nodiscard]] std::uint64_t parse_proc_starttime(std::string_view value);
[[nodiscard]] std::int32_t parse_proc_nice(std::string_view value);
[[nodiscard]] std::string parse_unified_cgroup(std::string_view value);

}  // namespace hostd_linux_process_recovery_test_seam

}  // namespace trainvm
