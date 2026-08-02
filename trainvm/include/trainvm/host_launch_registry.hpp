#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "trainvm/adapter_registry.hpp"

namespace trainvm {

struct HostLaunchProfile {
  AdapterKey key;
  std::string code_fingerprint;
  // Exact pre-dispatch runtime closure verified by the sealed worker before
  // any third-party import. Cache/runtime probes must report this identity;
  // they cannot select a different closure after launch authorization.
  std::string bootstrap_runtime_closure_fingerprint;
  // Capabilities implemented by these exact sealed code bytes. Requests may
  // require a subset, but cannot cause the worker to advertise new support.
  std::vector<std::string> provided_capabilities{};
  std::string executable_path;
  std::string executable_fingerprint;
  std::optional<std::string> code_path;
  // Index of the public argv literal replaced by hostd with the immutable
  // inherited code descriptor. Python profiles can therefore retain flags
  // such as `-I` before the script/zipapp without trusting a path argument.
  std::uint16_t code_argument_index{};
  // Fixed non-secret argv literals. Dynamic invocation and credentials use
  // typed/sealed descriptors in later protocol phases, never this manifest.
  std::vector<std::string> public_arguments;
  std::string working_directory;

  bool operator==(const HostLaunchProfile&) const = default;
};

struct HostLaunchRegistryDocument {
  std::string api_version;
  std::vector<std::string> trusted_roots;
  std::vector<HostLaunchProfile> profiles;

  bool operator==(const HostLaunchRegistryDocument&) const = default;
};

class HostLaunchResolutionError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Immutable host authority for binding an adapter operation to bounded launch
// metadata. This registry validates only declarative path identities. A later
// resolver must securely open and hash targets beneath trusted_roots before any
// process may be spawned.
class HostLaunchRegistry {
 public:
  explicit HostLaunchRegistry(HostLaunchRegistryDocument document);

  static HostLaunchRegistry load_file(const std::filesystem::path& path);

  [[nodiscard]] const HostLaunchProfile& resolve(
      const AdapterKey& key, std::string_view code_fingerprint) const;
  [[nodiscard]] const std::vector<std::string>& trusted_roots() const;
  [[nodiscard]] const std::string& registry_digest() const;
  [[nodiscard]] std::string profile_digest(
      const AdapterKey& key, std::string_view code_fingerprint) const;

 private:
  std::vector<std::string> trusted_roots_;
  std::map<AdapterKey, HostLaunchProfile> profiles_;
  std::string registry_digest_;
};

}  // namespace trainvm
