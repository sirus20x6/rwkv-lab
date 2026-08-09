#pragma once

#include <cstdint>
#include <string>

#include "trainvm/hostd_client_configuration.hpp"
#include "trainvm/hostd_daemon_configuration.hpp"
#include "trainvm/hostd_linux_session_authority.hpp"

namespace trainvm {

inline constexpr std::uintmax_t kHostdDaemonConfigurationMaximumBytes =
    256U << 10U;
inline constexpr std::uintmax_t kHostdClientConfigurationMaximumBytes =
    64U << 10U;

// Rebinds only the two boot-scoped authority fields. All journal, peer,
// worker, socket, cgroup, and recovery policies remain byte-for-byte equal to
// the validated static template before the resulting strict configuration is
// validated again.
[[nodiscard]] HostdDaemonConfiguration materialize_hostd_daemon_boot(
    const HostdDaemonConfiguration& configuration_template,
    const HostdLinuxBootAuthoritySnapshot& boot_authority);

[[nodiscard]] std::string hostd_daemon_configuration_json(
    const HostdDaemonConfiguration& configuration);

// Builds the controller policy only after hostd has bound and reattested its
// concrete socket inode. A socket replacement therefore necessarily produces
// a different client authority document.
[[nodiscard]] HostdClientConfiguration make_hostd_client_configuration(
    const HostdDaemonConfiguration& daemon_configuration,
    const HostdSocketIdentity& endpoint, std::int64_t request_timeout_ns);

[[nodiscard]] std::string hostd_client_configuration_json(
    const HostdClientConfiguration& configuration);

}  // namespace trainvm
