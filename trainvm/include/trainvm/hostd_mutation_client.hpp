#pragma once

#include <atomic>
#include <cstdint>
#include <functional>

#include "trainvm/hostd_transport.hpp"

namespace trainvm {

using HostdMutationExchange = std::function<HostdMutationReply(
    const HostdMutationClientConfig&, const HostdMutationRequest&,
    std::uint64_t correlation_id,
    std::int64_t absolute_monotonic_deadline_ns)>;

struct HostdMutationClientChannelOptions final {
  HostdMutationClientConfig transport;
  std::function<std::int64_t()> monotonic_now;
  std::int64_t request_timeout_ns{5'000'000'000LL};
  // Production leaves this empty. Tests may inject an exchange at this seam;
  // each typed client still validates its semantic result.
  HostdMutationExchange exchange;
};

// Shared bounded transport/correlation machinery. Resource and process clients
// own their distinct attribution and response semantics above this layer.
class HostdMutationClientChannel final {
 public:
  explicit HostdMutationClientChannel(
      HostdMutationClientChannelOptions options);

  [[nodiscard]] HostdMutationReply request(HostdMutationRequest request);

 private:
  HostdMutationClientChannelOptions options_;
  std::atomic<std::uint64_t> next_correlation_{1U};
};

}  // namespace trainvm
