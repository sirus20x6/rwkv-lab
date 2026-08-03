#include "trainvm/hostd_mutation_client.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace trainvm {
namespace {

constexpr std::int64_t kMaximumRequestTimeoutNs = 30'000'000'000LL;

}  // namespace

HostdMutationClientChannel::HostdMutationClientChannel(
    HostdMutationClientChannelOptions options)
    : options_(std::move(options)) {
  if (!options_.monotonic_now || options_.request_timeout_ns <= 0 ||
      options_.request_timeout_ns > kMaximumRequestTimeoutNs) {
    throw std::invalid_argument(
        "hostd mutation client channel options are incomplete or unbounded");
  }
  if (!options_.exchange) {
    options_.exchange = [](const HostdMutationClientConfig& config,
                           const HostdMutationRequest& request,
                           std::uint64_t correlation,
                           std::int64_t deadline) {
      return hostd_request_mutation(config, request, correlation, deadline);
    };
  }
}

HostdMutationReply HostdMutationClientChannel::request(
    HostdMutationRequest request) {
  const std::int64_t now = options_.monotonic_now();
  if (now < 0 || options_.request_timeout_ns >
                     std::numeric_limits<std::int64_t>::max() - now) {
    throw HostdTransportError("hostd mutation client deadline overflowed");
  }
  const std::uint64_t correlation =
      next_correlation_.fetch_add(1U, std::memory_order_relaxed);
  if (correlation == 0U ||
      correlation == std::numeric_limits<std::uint64_t>::max()) {
    throw HostdTransportError("hostd mutation client correlation exhausted");
  }
  return options_.exchange(options_.transport, request, correlation,
                           now + options_.request_timeout_ns);
}

}  // namespace trainvm
