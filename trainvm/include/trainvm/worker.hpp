#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trainvm {

struct WorkerLaunchRequest {
  std::string code_fingerprint;
  std::vector<std::string> required_capabilities;
};

struct WorkerLaunchTicket {
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
