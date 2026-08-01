#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string_view>

#include "trainvm/host_process_saga.hpp"
#include "trainvm/hostd_transport.hpp"

namespace trainvm {

using HostdProcessOpenFactory =
    std::function<HostdMutationOpen(std::string_view launch_id)>;
using HostdMutationExchange = std::function<HostdMutationReply(
    const HostdMutationClientConfig&, const HostdMutationRequest&,
    std::uint64_t correlation_id,
    std::int64_t absolute_monotonic_deadline_ns)>;

struct HostdProcessClientOptions final {
  HostdMutationClientConfig transport;
  HostdProcessOpenFactory open_for_launch;
  std::function<std::int64_t()> monotonic_now;
  std::int64_t request_timeout_ns{5'000'000'000LL};
  // Production leaves this empty and uses hostd_request_mutation. Tests may
  // inject an in-process exchange; typed process-result semantics are still
  // checked after it returns.
  HostdMutationExchange exchange;
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

 private:
  [[nodiscard]] HostdMutationReply exchange(
      HostdMutationRequest request);
  [[nodiscard]] HostdMutationOpen open_for(
      std::string_view launch_id, std::string_view journal_id,
      std::string_view run_id, std::string_view logical_lease_id,
      std::uint64_t logical_fencing_token,
      std::string_view concurrency_key = {}) const;

  HostdProcessClientOptions options_;
  std::atomic<std::uint64_t> next_correlation_{1U};
};

}  // namespace trainvm
