#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "trainvm/host_launch.hpp"
#include "trainvm/host_ledger.hpp"

namespace trainvm {

inline constexpr std::string_view kHostdProcessPrepareApiVersion =
    "trainvm.hostd-process-prepare/v2";
inline constexpr std::string_view kHostdProcessCommitApiVersion =
    "trainvm.hostd-process-commit/v1";
inline constexpr std::string_view kHostdProcessExitApiVersion =
    "trainvm.hostd-process-exit/v1";
inline constexpr std::string_view kHostdProcessPreparedApiVersion =
    "trainvm.hostd-process-prepared/v1";
inline constexpr std::string_view kHostdProcessCommittedApiVersion =
    "trainvm.hostd-process-committed/v1";

struct HostdProcessPrepareRequest final {
  std::string api_version;
  ResolvedLaunchSpec launch;
  ResourceBundleGrant grant;
  std::string worker_bootstrap_digest;
  std::vector<std::string> descriptor_roles;

  bool operator==(const HostdProcessPrepareRequest&) const = default;
};

struct HostdProcessCommitRequest final {
  std::string api_version;
  std::string launch_id;
  std::string allocation_id;
  std::string grant_digest;
  std::string journal_id;
  std::string run_id;
  std::string logical_lease_id;
  std::uint64_t logical_fencing_token{};
  std::string spawn_receipt_digest;

  bool operator==(const HostdProcessCommitRequest&) const = default;
};

struct HostdProcessExitCommand final {
  std::string api_version;
  std::string exit_request_id;
  std::string launch_id;
  std::string allocation_id;
  std::string grant_digest;
  std::string journal_id;
  std::string run_id;
  std::string logical_lease_id;
  std::uint64_t logical_fencing_token{};
  std::string spawn_receipt_digest;
  bool request_termination{};

  bool operator==(const HostdProcessExitCommand&) const = default;
};

struct HostdProcessPreparedResult final {
  std::string api_version;
  HostProcessLaunchIntent intent;
  HostProcessSpawnReceipt spawn;
  bool replayed{};

  bool operator==(const HostdProcessPreparedResult&) const = default;
};

struct HostdProcessCommittedResult final {
  std::string api_version;
  std::string launch_id;
  std::string spawn_receipt_digest;
  bool released_to_exec{};
  bool replayed{};

  bool operator==(const HostdProcessCommittedResult&) const = default;
};

[[nodiscard]] std::string hostd_process_prepare_canonical_json(
    const HostdProcessPrepareRequest& value);
[[nodiscard]] HostdProcessPrepareRequest
hostd_process_prepare_from_canonical_json(std::string_view value);
// Durable host-ledger binding of the immutable resolved launch and the sealed
// per-attempt bootstrap descriptor without widening the ledger schema.
[[nodiscard]] std::string hostd_bound_process_launch_digest(
    const ResolvedLaunchSpec& launch, std::string_view worker_bootstrap_digest);
[[nodiscard]] std::string hostd_process_commit_canonical_json(
    const HostdProcessCommitRequest& value);
[[nodiscard]] HostdProcessCommitRequest
hostd_process_commit_from_canonical_json(std::string_view value);
[[nodiscard]] std::string hostd_process_exit_canonical_json(
    const HostdProcessExitCommand& value);
[[nodiscard]] HostdProcessExitCommand
hostd_process_exit_from_canonical_json(std::string_view value);
[[nodiscard]] std::string hostd_process_prepared_canonical_json(
    const HostdProcessPreparedResult& value);
[[nodiscard]] HostdProcessPreparedResult
hostd_process_prepared_from_canonical_json(std::string_view value);
[[nodiscard]] std::string hostd_process_committed_canonical_json(
    const HostdProcessCommittedResult& value);
[[nodiscard]] HostdProcessCommittedResult
hostd_process_committed_from_canonical_json(std::string_view value);

}  // namespace trainvm
