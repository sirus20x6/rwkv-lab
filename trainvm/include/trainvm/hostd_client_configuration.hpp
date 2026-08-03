#pragma once

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

#include "trainvm/hostd_transport.hpp"

namespace trainvm {

inline constexpr std::string_view kHostdClientConfigurationApiVersion =
    "trainvm.hostd-client/v1";

class HostdClientConfigurationError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// JSON-facing shape. Fixed-width owner fields keep the document portable;
// construction proves they fit the platform uid_t/gid_t before conversion.
struct HostdClientConfigurationDocument final {
  std::string api_version;
  std::string socket_path;
  HostdSocketIdentity expected_endpoint;
  std::uint32_t expected_server_uid{};
  std::uint32_t expected_server_gid{};
  std::int64_t request_timeout_ns{};

  bool operator==(const HostdClientConfigurationDocument&) const = default;
};

// Immutable, validated connection policy. Runtime broker epoch and host/boot
// identity are intentionally absent: startup obtains them from the pinned
// endpoint and compares host/boot with TrainVM's local authority.
class HostdClientConfiguration final {
 public:
  explicit HostdClientConfiguration(
      HostdClientConfigurationDocument document);
  static HostdClientConfiguration load_file(
      const std::filesystem::path& path);

  [[nodiscard]] const HostdMutationClientConfig& mutation() const noexcept;
  [[nodiscard]] HostdStatusClientConfig status() const;
  [[nodiscard]] std::int64_t request_timeout_ns() const noexcept;
  [[nodiscard]] const HostdClientConfigurationDocument& document() const
      noexcept;

 private:
  HostdClientConfigurationDocument document_;
  HostdMutationClientConfig mutation_;
};

}  // namespace trainvm
