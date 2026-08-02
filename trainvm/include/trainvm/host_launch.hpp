#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "trainvm/host_launch_registry.hpp"
#include "trainvm/worker.hpp"

namespace trainvm {

struct HostIdentity {
  // host_id is a SHA-256 digest of the stable machine identity. boot_id is the
  // Linux boot UUID and prevents persisted PID evidence crossing a reboot.
  std::string host_id;
  std::string boot_id;

  bool operator==(const HostIdentity&) const = default;
};

struct VerifiedLaunchArtifact {
  std::string source_path;
  std::uint64_t source_device{};
  std::uint64_t source_inode{};
  std::uint64_t source_size{};
  std::uint32_t source_mode{};
  std::uint32_t source_uid{};
  std::uint32_t source_gid{};
  std::string sealed_sha256;

  bool operator==(const VerifiedLaunchArtifact&) const = default;
};

struct OpenedDirectoryIdentity {
  std::string source_path;
  std::uint64_t device{};
  std::uint64_t inode{};
  std::uint32_t mode{};
  std::uint32_t uid{};
  std::uint32_t gid{};

  bool operator==(const OpenedDirectoryIdentity&) const = default;
};

// Public, persistable launch identity. It deliberately contains no resolved
// secret value and no raw per-process authentication token.
struct ResolvedLaunchIdentity {
  std::string api_version;
  std::string launch_event_id;
  std::string run_id;
  std::string node_id;
  std::string attempt_id;
  std::string launch_nonce;
  AdapterKey adapter_key;
  std::string code_fingerprint;
  std::vector<std::string> required_capabilities;
  std::vector<std::string> provided_capabilities{};
  std::string host_registry_digest;
  std::string host_profile_digest;
  std::string concurrency_key;
  std::string lease_id;
  std::uint64_t fencing_token{};
  std::optional<HostLaunchGrantClaim> host_grant;
  HostIdentity host;
  VerifiedLaunchArtifact executable;
  std::optional<VerifiedLaunchArtifact> code;
  std::uint16_t code_argument_index{};
  std::vector<std::string> public_arguments;
  OpenedDirectoryIdentity working_directory;

  bool operator==(const ResolvedLaunchIdentity&) const = default;
};

struct ResolvedLaunchSpec {
  ResolvedLaunchIdentity identity;
  std::string spec_digest;

  bool operator==(const ResolvedLaunchSpec&) const = default;
};

nlohmann::json resolved_launch_identity_json(
    const ResolvedLaunchIdentity& identity);
nlohmann::json resolved_launch_spec_json(const ResolvedLaunchSpec& spec);
ResolvedLaunchSpec resolved_launch_spec_from_json(
    const nlohmann::json& source);

// Owns immutable sealed copies of executable/code bytes and securely opened
// working-directory evidence. It is an authorization bundle, not a process.
class ResolvedLaunch final {
 public:
  // Adopts descriptor authority delegated over the hostd transport only after
  // reattesting the exact sealed bytes and opened-directory identity recorded
  // by the canonical specification. The supplied descriptors remain owned by
  // the caller; independent close-on-exec duplicates are retained here.
  [[nodiscard]] static ResolvedLaunch adopt_delegated(
      ResolvedLaunchSpec spec, int executable_fd,
      std::optional<int> code_fd, int working_directory_fd);

  ResolvedLaunch(ResolvedLaunch&& other) noexcept;
  ResolvedLaunch& operator=(ResolvedLaunch&& other) noexcept;
  ~ResolvedLaunch();

  ResolvedLaunch(const ResolvedLaunch&) = delete;
  ResolvedLaunch& operator=(const ResolvedLaunch&) = delete;

  [[nodiscard]] const ResolvedLaunchSpec& spec() const;
  // The resolver retains its owned descriptors. Callers receive independent
  // close-on-exec duplicates suitable for transfer to the launcher helper.
  [[nodiscard]] int duplicate_executable_fd() const;
  [[nodiscard]] std::optional<int> duplicate_code_fd() const;
  [[nodiscard]] int duplicate_working_directory_fd() const;

 private:
  friend class HostLaunchResolver;
  ResolvedLaunch(ResolvedLaunchSpec spec, int executable_fd,
                 std::optional<int> code_fd,
                 int working_directory_fd) noexcept;

  void close_descriptors() noexcept;

  ResolvedLaunchSpec spec_;
  int executable_fd_{-1};
  std::optional<int> code_fd_;
  int working_directory_fd_{-1};
};

class HostLaunchResolver final {
 public:
  HostLaunchResolver(const HostLaunchRegistry& registry, HostIdentity host);

  static HostIdentity local_host_identity();

  // Resolves and seals exact host bytes. No fork, exec, signal, cgroup, lease
  // mutation, or journal mutation occurs here.
  [[nodiscard]] ResolvedLaunch resolve(const WorkerLaunchTicket& ticket,
                                       const AdapterKey& key) const;

 private:
  const HostLaunchRegistry& registry_;
  HostIdentity host_;
};

}  // namespace trainvm
