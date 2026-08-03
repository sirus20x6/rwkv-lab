#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include "trainvm/host_ledger.hpp"
#include "trainvm/hostd_process_protocol.hpp"
#include "trainvm/hostd_session_challenge.hpp"

namespace trainvm {

inline constexpr std::string_view kHostdMutationProtocolApiVersion =
    "trainvm.hostd-mutation-protocol/v2";
inline constexpr std::string_view kHostdMutationCommandDigestDomain =
    "trainvm.hostd-mutation-command-digest/v1";

class HostdMutationProtocolError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct HostdMutationOpen final {
  std::string api_version;
  HostdSessionChallengeClaim claim;

  bool operator==(const HostdMutationOpen&) const = default;
};

enum class HostdMutationKind {
  request_bundle,
  reconcile_bundle_outcome,
  release_bundle,
  prepare_process,
  commit_process,
  finalize_process,
};

struct HostdMutationCommand final {
  std::string api_version;
  HostdSessionChallengeResponse challenge_response;
  HostdMutationKind mutation{};
  std::optional<ResourceBundleRequest> bundle_request{};
  std::optional<ResourceReleaseRequest> release_request{};
  std::optional<HostdProcessPrepareRequest> process_prepare{};
  std::optional<HostdProcessCommitRequest> process_commit{};
  std::optional<HostdProcessExitCommand> process_exit{};
  std::string command_digest;

  bool operator==(const HostdMutationCommand&) const = default;
};

enum class HostdMutationReplyKind {
  bundle_outcome,
  reconciliation_missing,
  release_outcome,
  process_prepared,
  process_committed,
  process_exited,
};

struct HostdMutationReply final {
  std::string api_version;
  HostdMutationReplyKind kind{};
  std::string challenge_id;
  std::string command_digest;
  std::optional<BundleRequestResult> bundle_result{};
  std::optional<BundleReleaseResult> release_result{};
  std::optional<HostdProcessPreparedResult> process_prepared{};
  std::optional<HostdProcessCommittedResult> process_committed{};
  std::optional<HostProcessExitResult> process_exit{};

  bool operator==(const HostdMutationReply&) const = default;
};

[[nodiscard]] HostdMutationCommand
seal_hostd_mutation_command(HostdMutationCommand command);

[[nodiscard]] std::string
hostd_mutation_open_canonical_json(const HostdMutationOpen& value);
[[nodiscard]] HostdMutationOpen
hostd_mutation_open_from_canonical_json(std::string_view value);
[[nodiscard]] std::string
hostd_mutation_command_canonical_json(const HostdMutationCommand& value);
[[nodiscard]] HostdMutationCommand
hostd_mutation_command_from_canonical_json(std::string_view value);
[[nodiscard]] std::string
hostd_mutation_reply_canonical_json(const HostdMutationReply& value);
[[nodiscard]] HostdMutationReply
hostd_mutation_reply_from_canonical_json(std::string_view value);

void validate_hostd_mutation_exchange(const HostdMutationOpen& open,
                                      const HostdSessionChallenge& challenge,
                                      const HostdMutationCommand& command);
void validate_hostd_mutation_reply(const HostdMutationCommand& command,
                                   const HostdMutationReply& reply);

}  // namespace trainvm
