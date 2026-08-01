#pragma once

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <sys/types.h>

#include "trainvm/hostd_transport.hpp"

namespace trainvm {

inline constexpr std::string_view kHostdLinuxServiceIdentityApiVersion =
    "trainvm.hostd-linux-service-identity/v1";

class HostdLinuxServiceIdentityError final : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

struct HostdLinuxServiceRole final {
  std::string cgroup_path;
  std::string service_identity;
  uid_t expected_uid{};
  gid_t expected_gid{};
  HostdSessionAccess access{};

  bool operator==(const HostdLinuxServiceRole &) const = default;
};

struct HostdLinuxServiceIdentityConfig final {
  std::string api_version{std::string(kHostdLinuxServiceIdentityApiVersion)};
  std::vector<HostdLinuxServiceRole> roles;
  std::size_t maximum_cgroup_file_bytes{4096U};

  bool operator==(const HostdLinuxServiceIdentityConfig &) const = default;
};

// Strict host-side service authority. It pins the real procfs and cgroup-v2
// roots, pins every configured cgroup directory, and double-reads the peer's
// unified cgroup membership around its proc starttime. The caller must pair it
// with HostdMutationTransportEnforcementGrade::strict_service_identity and a
// strict Linux session kernel that pins the host cgroup namespace.
class HostdLinuxServiceIdentityAuthority final
    : public IHostdMutationServiceIdentityAuthority {
public:
  explicit HostdLinuxServiceIdentityAuthority(
      HostdLinuxServiceIdentityConfig config);
  ~HostdLinuxServiceIdentityAuthority() override;

  HostdLinuxServiceIdentityAuthority(
      const HostdLinuxServiceIdentityAuthority &) = delete;
  HostdLinuxServiceIdentityAuthority &operator=(
      const HostdLinuxServiceIdentityAuthority &) = delete;

  [[nodiscard]] HostdMutationServiceAuthorization
  authorize(const HostdSocketPeerInstance &peer) override;

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

namespace hostd_linux_service_identity_test_seam {

[[nodiscard]] std::optional<std::string>
parse_unified_cgroup_path(std::string_view value,
                          std::size_t maximum_bytes = 4096U);

} // namespace hostd_linux_service_identity_test_seam

} // namespace trainvm
