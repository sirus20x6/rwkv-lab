#pragma once

#include <cstdint>
#include <functional>
#include <string_view>

#include "trainvm/host_process_saga.hpp"
#include "trainvm/hostd_mutation_client.hpp"

namespace trainvm {

using HostdProcessOpenFactory =
    std::function<HostdMutationOpen(std::string_view launch_id)>;
struct HostdProcessClientOptions final {
  HostdMutationClientChannelOptions channel;
  HostdProcessOpenFactory open_for_launch;
};

// Strict synchronous SCM_RIGHTS client used by HostProcessSagaReconciler.
// Authority claims come from a journal-backed startup component rather than
// being synthesized from the process request itself.
class HostdProcessClient final : public IHostProcessClient {
 public:
  explicit HostdProcessClient(HostdProcessClientOptions options);

  [[nodiscard]] HostdProcessPreparedResult prepare_process(
      const HostdProcessPrepareRequest& request,
      const ResolvedLaunch& resolved,
      const SealedWorkerBootstrap& bootstrap) override;
  [[nodiscard]] HostdProcessCommittedResult commit_process(
      const HostdProcessCommitRequest& request) override;
  [[nodiscard]] HostProcessExitResult finalize_process(
      const HostdProcessExitCommand& request) override;

 private:
  [[nodiscard]] HostdMutationOpen open_for(
      std::string_view launch_id, std::string_view journal_id,
      std::string_view run_id, std::string_view logical_lease_id,
      std::uint64_t logical_fencing_token,
      std::string_view concurrency_key = {}) const;

  HostdProcessOpenFactory open_for_launch_;
  HostdMutationClientChannel channel_;
};

}  // namespace trainvm
