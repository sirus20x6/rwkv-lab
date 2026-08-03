#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "trainvm/host_resources.hpp"

namespace trainvm {

struct WorkerLaunchRequest {
  std::string code_fingerprint;
  std::vector<std::string> required_capabilities;
};

struct HostLaunchGrantClaim final {
  std::string request_id;
  std::string grant_digest;
  std::vector<ResourceFence> fences;

  bool operator==(const HostLaunchGrantClaim&) const = default;
};

struct WorkerLaunchTicket {
  // Protocol launch authorization, not sufficient OS-process identity or exec
  // authority. The process supervisor must bind a resolved launch spec and host.
  std::string run_id;
  std::string node_id;
  std::string attempt_id;
  std::string launch_nonce;
  std::string adapter;
  std::string adapter_version;
  std::string code_fingerprint;
  std::vector<std::string> required_capabilities;
  std::string concurrency_key;
  std::string lease_id;
  std::uint64_t fencing_token{};
  std::optional<HostLaunchGrantClaim> host_grant;

  bool operator==(const WorkerLaunchTicket&) const = default;
};

struct WorkerHelloEvidence {
  std::string run_id;
  std::string node_id;
  std::string attempt_id;
  std::string launch_nonce;
  std::string adapter;
  std::string adapter_version;
  std::string code_fingerprint;
  std::vector<std::string> capabilities;
  std::uint64_t last_acked_controller_sequence{};
  std::string concurrency_key;
  std::string lease_id;
  std::uint64_t fencing_token{};

  bool operator==(const WorkerHelloEvidence&) const = default;
};

struct WorkerSessionIdentity {
  std::string run_id;
  std::string node_id;
  std::string attempt_id;
  std::string launch_nonce;
  std::string concurrency_key;
  std::string lease_id;
  std::uint64_t fencing_token{};

  bool operator==(const WorkerSessionIdentity&) const = default;
};

enum class WorkerReadinessDisposition { accepted, replayed };

struct WorkerReadinessResult {
  WorkerReadinessDisposition disposition{};
  WorkerLaunchTicket launch;

  bool operator==(const WorkerReadinessResult&) const = default;
};

}  // namespace trainvm
