#pragma once

#include <optional>
#include <string>
#include <utility>

#include "trainvm/host_ledger.hpp"

namespace trainvm {

enum class LinuxProcessRecoveryDisposition {
  exact_live_process,
  already_gone,
  identity_mismatch,
  observation_failed,
};

class LinuxRecoveredProcess final {
 public:
  LinuxRecoveredProcess(LinuxRecoveredProcess&& other) noexcept;
  LinuxRecoveredProcess& operator=(LinuxRecoveredProcess&& other) noexcept;
  ~LinuxRecoveredProcess();

  LinuxRecoveredProcess(const LinuxRecoveredProcess&) = delete;
  LinuxRecoveredProcess& operator=(const LinuxRecoveredProcess&) = delete;

  [[nodiscard]] const HostProcessSpawnRequest& identity() const noexcept;
  [[nodiscard]] bool alive() const noexcept;

 private:
  friend class LinuxProcessRecoveryProbe;
  LinuxRecoveredProcess(HostProcessSpawnRequest identity, int pidfd) noexcept;

  HostProcessSpawnRequest identity_;
  int pidfd_{-1};
};

struct LinuxProcessRecoveryResult final {
  LinuxProcessRecoveryDisposition disposition{};
  std::string detail;
  std::optional<LinuxRecoveredProcess> process;

  LinuxProcessRecoveryResult(
      LinuxProcessRecoveryDisposition disposition_value,
      std::string detail_value,
      std::optional<LinuxRecoveredProcess> process_value = std::nullopt)
      : disposition(disposition_value), detail(std::move(detail_value)),
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

namespace hostd_linux_process_recovery_test_seam {

[[nodiscard]] std::uint64_t parse_proc_starttime(std::string_view value);
[[nodiscard]] std::string parse_unified_cgroup(std::string_view value);

}  // namespace hostd_linux_process_recovery_test_seam

}  // namespace trainvm
