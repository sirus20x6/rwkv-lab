#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "trainvm/hostd_daemon_configuration.hpp"
#include "trainvm/hostd_linux_session_authority.hpp"

namespace trainvm {

inline constexpr std::string_view kHostdGpuAuthorizationApiVersion =
    "trainvm.hostd-gpu-authorization/v1";
inline constexpr std::uintmax_t kHostdGpuAuthorizationMaximumBytes =
    64U << 10U;

enum class HostdDisplayGpuPolicy {
  deny,
  cooperative_allowlist,
};

class HostdGpuAuthorizationError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct HostdGpuAuthorizationDocument final {
  std::string api_version;
  std::string host_id;
  std::string boot_id;
  std::string broker_instance_id;
  bool driver_probe_authorized{};
  std::string display_policy;
  std::vector<std::string> allowed_display_gpu_ids;
  std::string authorization_digest;

  bool operator==(const HostdGpuAuthorizationDocument&) const = default;
};

// One explicit, boot-scoped root policy. Merely enabling hostd or the
// dashboard cannot manufacture this document. The allowlist controls only
// cooperative compute on a device NVML proves is display-active; exclusive
// display-device admission remains impossible.
class HostdGpuAuthorization final {
 public:
  explicit HostdGpuAuthorization(HostdGpuAuthorizationDocument document);
  static HostdGpuAuthorization load_file(const std::filesystem::path& path);

  [[nodiscard]] const HostdGpuAuthorizationDocument& document() const
      noexcept;
  [[nodiscard]] HostdDisplayGpuPolicy display_policy() const noexcept;
  [[nodiscard]] const std::vector<std::string>&
  allowed_display_gpu_ids() const noexcept;
  void require_matches(
      const HostdDaemonConfiguration& daemon_configuration) const;

 private:
  HostdGpuAuthorizationDocument document_;
  HostdDisplayGpuPolicy display_policy_{};
};

[[nodiscard]] HostdGpuAuthorization make_hostd_gpu_authorization(
    const HostdDaemonConfiguration& configuration_template,
    const HostdLinuxBootAuthoritySnapshot& boot_authority,
    HostdDisplayGpuPolicy display_policy,
    std::vector<std::string> allowed_display_gpu_ids = {});

[[nodiscard]] std::string hostd_gpu_authorization_json(
    const HostdGpuAuthorization& authorization);

}  // namespace trainvm
