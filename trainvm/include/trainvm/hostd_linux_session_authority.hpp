#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include <sys/types.h>
#include <time.h>

#include "trainvm/hostd_session_challenge.hpp"

namespace trainvm {

inline constexpr std::string_view kHostdLinuxSessionAuthorityApiVersion =
    "trainvm.hostd-linux-session-authority/v1";

struct HostdLinuxRandomRead final {
  ssize_t count{-1};
  int error_number{};
};

struct HostdLinuxClockRead final {
  bool success{};
  timespec value{};
  int error_number{};
};

struct HostdLinuxBootIdRead final {
  bool success{};
  std::string value;
  int error_number{};
};

struct HostdLinuxPeerKernelObservation final {
  pid_t pid{};
  uid_t effective_uid_before{};
  gid_t effective_gid_before{};
  uid_t effective_uid_after{};
  gid_t effective_gid_after{};
  std::uint64_t process_starttime_ticks_before{};
  std::uint64_t process_starttime_ticks_after{};
  std::uint64_t process_directory_device_before{};
  std::uint64_t process_directory_inode_before{};
  std::uint64_t process_directory_device_after{};
  std::uint64_t process_directory_inode_after{};
  bool pidfd_available{};
  bool pidfd_alive_before{};
  bool pidfd_alive_after{};
  bool complete{};
  int error_number{};

  bool operator==(const HostdLinuxPeerKernelObservation &) const = default;
};

struct HostdLinuxNamespaceIdentity final {
  std::uint64_t device{};
  std::uint64_t inode{};

  bool operator==(const HostdLinuxNamespaceIdentity &) const = default;
};

struct HostdLinuxHostNamespacePolicy final {
  HostdLinuxNamespaceIdentity mount_namespace;
  HostdLinuxNamespaceIdentity pid_namespace;
  HostdLinuxNamespaceIdentity time_namespace;
  HostdLinuxNamespaceIdentity time_for_children_namespace;

  bool operator==(const HostdLinuxHostNamespacePolicy &) const = default;
};

enum class HostdLinuxSessionEnforcementGrade {
  // Pins and continuously reattests the current proc mount plus mount/PID/time
  // namespace identities, but does not claim that an external host authority
  // selected those namespace identities or that SO_PEERPIDFD is available.
  cooperative_namespace_observation,
  // Requires exact externally configured host namespace identities and a
  // socket-derived SO_PEERPIDFD for every bound peer.
  strict_host_namespaces_and_socket_pidfd,
};

// Injectable syscall/kernel seam. Production construction below pins a procfs
// root and implements only read-only observation; it never signals, mutates, or
// launches a process.
class IHostdLinuxSessionKernel {
public:
  virtual ~IHostdLinuxSessionKernel() = default;
  [[nodiscard]] virtual HostdLinuxSessionEnforcementGrade
  enforcement_grade() const = 0;
  [[nodiscard]] virtual HostdLinuxRandomRead
  getrandom_bytes(void *buffer, std::size_t count) = 0;
  [[nodiscard]] virtual HostdLinuxClockRead clock_boottime() = 0;
  [[nodiscard]] virtual HostdLinuxBootIdRead read_boot_id() = 0;
  [[nodiscard]] virtual HostdLinuxPeerKernelObservation
  observe_process(pid_t pid) = 0;
};

struct HostdLinuxSessionKernelConfig final {
  std::string api_version{std::string(kHostdLinuxSessionAuthorityApiVersion)};
  HostdLinuxSessionEnforcementGrade enforcement_grade{
      HostdLinuxSessionEnforcementGrade::cooperative_namespace_observation};
  // Mandatory at the strict grade. These identities must come from guarded
  // host startup policy, not from an untrusted request or this factory's own
  // observation. Both current and child time namespaces are bound explicitly.
  std::optional<HostdLinuxHostNamespacePolicy> expected_host_namespaces;

  bool operator==(const HostdLinuxSessionKernelConfig &) const = default;
};

[[nodiscard]] std::shared_ptr<IHostdLinuxSessionKernel>
make_hostd_linux_session_kernel(HostdLinuxSessionKernelConfig config);

class HostdLinuxCSPRNGNonceSource final
    : public IHostdSessionChallengeNonceSource {
public:
  explicit HostdLinuxCSPRNGNonceSource(
      std::shared_ptr<IHostdLinuxSessionKernel> kernel,
      std::uint64_t maximum_tokens = 1'048'576U);
  ~HostdLinuxCSPRNGNonceSource() override;

  HostdLinuxCSPRNGNonceSource(const HostdLinuxCSPRNGNonceSource &) = delete;
  HostdLinuxCSPRNGNonceSource &
  operator=(const HostdLinuxCSPRNGNonceSource &) = delete;

  [[nodiscard]] std::string next_hex_256(std::string_view purpose) override;

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

class HostdLinuxBoottimeSource final : public IHostdSessionChallengeTimeSource {
public:
  HostdLinuxBoottimeSource(std::string expected_boot_id,
                           std::shared_ptr<IHostdLinuxSessionKernel> kernel);
  ~HostdLinuxBoottimeSource() override;

  HostdLinuxBoottimeSource(const HostdLinuxBoottimeSource &) = delete;
  HostdLinuxBoottimeSource &
  operator=(const HostdLinuxBoottimeSource &) = delete;

  [[nodiscard]] HostdSessionChallengeTime now() override;

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

struct HostdLinuxSocketPeerCredentials final {
  // These values must come directly from the accepted Unix socket's kernel
  // credential observation (never from request payload fields).
  uid_t uid{};
  gid_t gid{};
  pid_t pid{};

  bool operator==(const HostdLinuxSocketPeerCredentials &) const = default;
};

class HostdLinuxPeerProcessObserver final {
public:
  explicit HostdLinuxPeerProcessObserver(
      std::shared_ptr<IHostdLinuxSessionKernel> kernel);
  ~HostdLinuxPeerProcessObserver();

  HostdLinuxPeerProcessObserver(const HostdLinuxPeerProcessObserver &) = delete;
  HostdLinuxPeerProcessObserver &
  operator=(const HostdLinuxPeerProcessObserver &) = delete;

  [[nodiscard]] HostdSocketPeerInstance
  observe(const HostdLinuxSocketPeerCredentials &credentials);
  [[nodiscard]] HostdSocketPeerInstance
  reobserve(const HostdLinuxSocketPeerCredentials &credentials,
            const HostdSocketPeerInstance &expected_instance);

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

// Owns duplicates of one accepted Unix socket and, at strict grade, the
// socket-derived peer pidfd. Construction obtains SO_PEERCRED itself; no
// request payload can provide peer UID/GID/PID. Re-observation proves the same
// socket credentials, live pidfd, and procfs process instance. This object is
// status/challenge evidence only and cannot signal or mutate the peer.
class HostdLinuxBoundSocketPeer final {
public:
  struct Implementation;

  ~HostdLinuxBoundSocketPeer();
  HostdLinuxBoundSocketPeer(HostdLinuxBoundSocketPeer &&) noexcept;
  HostdLinuxBoundSocketPeer &
  operator=(HostdLinuxBoundSocketPeer &&) noexcept;
  HostdLinuxBoundSocketPeer(const HostdLinuxBoundSocketPeer &) = delete;
  HostdLinuxBoundSocketPeer &
  operator=(const HostdLinuxBoundSocketPeer &) = delete;

  [[nodiscard]] const HostdSocketPeerInstance &instance() const;
  [[nodiscard]] HostdSocketPeerInstance reobserve();
  [[nodiscard]] HostdLinuxSessionEnforcementGrade enforcement_grade() const;

private:
  explicit HostdLinuxBoundSocketPeer(
      std::unique_ptr<Implementation> implementation) noexcept;
  std::unique_ptr<Implementation> implementation_;

  friend HostdLinuxBoundSocketPeer make_hostd_linux_bound_socket_peer(
      int, std::shared_ptr<IHostdLinuxSessionKernel>,
      HostdLinuxSessionEnforcementGrade);
};

[[nodiscard]] HostdLinuxBoundSocketPeer make_hostd_linux_bound_socket_peer(
    int accepted_socket_fd, std::shared_ptr<IHostdLinuxSessionKernel> kernel,
    HostdLinuxSessionEnforcementGrade enforcement_grade);

namespace hostd_linux_session_test_seam {

[[nodiscard]] std::optional<std::uint64_t>
parse_proc_stat_starttime(std::string_view stat, pid_t expected_pid);
[[nodiscard]] std::optional<std::pair<uid_t, gid_t>>
parse_proc_status_effective_credentials(std::string_view status);

} // namespace hostd_linux_session_test_seam

// All production authority objects are bound to their creator PID and Linux
// TID. Every operation rejects a sibling thread or fork child before touching
// mutable state, a mutex, CLOCK_BOOTTIME, entropy, or procfs. Callers must keep
// each authority on one owner thread and construct a fresh authority after
// transfer or exec; concurrent work uses one authority per thread. This is
// required because mount and time namespaces are task scoped. No
// cgroup-membership, delegation, containment, signal, mutation, or launch
// grade is claimed.

} // namespace trainvm
