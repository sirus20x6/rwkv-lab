#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <sys/types.h>

namespace trainvm {

inline constexpr int kLinuxWorkerCodeDescriptor = 3;
inline constexpr int kLinuxWorkerBootstrapDescriptor = 4;

struct LinuxWorkerCredentialSpec final {
  uid_t uid{};
  gid_t gid{};
  bool no_new_privileges{};

  bool operator==(const LinuxWorkerCredentialSpec&) const = default;
};

struct LinuxStoppedLaunchSpec final {
  std::string launch_id;
  int cgroup_fd{-1};
  std::string expected_cgroup_path;
  std::uint64_t expected_cgroup_device{};
  std::uint64_t expected_cgroup_inode{};
  int executable_fd{-1};
  std::optional<int> code_fd;
  int worker_bootstrap_fd{-1};
  std::string executable_name;
  std::string executable_digest;
  int working_directory_fd{-1};
  LinuxWorkerCredentialSpec credentials;
  std::optional<std::int32_t> nice;
  std::uint16_t code_argument_index{};
  std::vector<std::string> arguments;

  bool operator==(const LinuxStoppedLaunchSpec&) const = default;
};

struct LinuxStoppedChildIdentity final {
  pid_t host_pid{};
  std::uint64_t process_starttime_ticks{};
  std::string cgroup_path;
  std::uint64_t cgroup_device{};
  std::uint64_t cgroup_inode{};
  std::string executable_digest;
  uid_t uid{};
  gid_t gid{};
  bool no_new_privileges{};
  std::int32_t nice{};

  bool operator==(const LinuxStoppedChildIdentity&) const = default;
};

struct LinuxChildExitObservation final {
  std::int32_t wait_code{};
  std::int32_t wait_status{};

  bool operator==(const LinuxChildExitObservation&) const = default;
};

class LinuxStoppedLauncherError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Owns the pidfd and the private pre-exec gate. Destruction fails safe by
// closing the gate, signalling the exact pidfd process, and reaping it. The
// numeric PID is never used for signalling.
class LinuxStoppedChild final {
 public:
  LinuxStoppedChild(LinuxStoppedChild&& other) noexcept;
  LinuxStoppedChild& operator=(LinuxStoppedChild&& other) noexcept;
  ~LinuxStoppedChild();

  LinuxStoppedChild(const LinuxStoppedChild&) = delete;
  LinuxStoppedChild& operator=(const LinuxStoppedChild&) = delete;

  [[nodiscard]] const LinuxStoppedChildIdentity& identity() const;
  [[nodiscard]] bool released() const;
  [[nodiscard]] bool exited() const;
  void release_to_exec();
  [[nodiscard]] LinuxChildExitObservation wait_and_reap();
  [[nodiscard]] LinuxChildExitObservation terminate_and_observe();
  void terminate_and_reap() noexcept;

 private:
  friend class LinuxStoppedLauncherKernel;
  LinuxStoppedChild(LinuxStoppedChildIdentity identity, int pidfd,
                    int gate_fd) noexcept;
  [[nodiscard]] std::optional<LinuxChildExitObservation> reap(
      bool terminate, bool fail_on_error);

  LinuxStoppedChildIdentity identity_;
  int pidfd_{-1};
  int gate_fd_{-1};
  bool released_{};
  bool reaped_{};
  std::optional<LinuxChildExitObservation> exit_observation_;
};

// Linux production primitive. It performs clone3 with CLONE_INTO_CGROUP and
// CLONE_PIDFD, leaves the child blocked on a private pipe, and double-attests
// PID starttime and unified cgroup membership before returning authority.
class LinuxStoppedLauncherKernel final {
 public:
  [[nodiscard]] LinuxStoppedChild spawn_stopped(
      const LinuxStoppedLaunchSpec& spec) const;
};

namespace hostd_linux_stopped_launcher_test_seam {

[[nodiscard]] std::uint64_t parse_proc_starttime(std::string_view stat);
[[nodiscard]] std::int32_t parse_proc_nice(std::string_view stat);
[[nodiscard]] std::string parse_unified_cgroup(std::string_view cgroup);
[[nodiscard]] bool install_inherited_worker_descriptors(
    std::optional<int> code_fd, int worker_bootstrap_fd) noexcept;
[[nodiscard]] bool worker_status_has_credentials(
    std::string_view status, const LinuxWorkerCredentialSpec& expected);

}  // namespace hostd_linux_stopped_launcher_test_seam

}  // namespace trainvm
