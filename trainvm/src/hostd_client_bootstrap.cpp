#include "trainvm/hostd_client_bootstrap.hpp"

#include <cctype>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <utility>

namespace trainvm {
namespace {

bool valid_identifier(std::string_view value) {
  return !value.empty() && value.size() <= 192U &&
         std::ranges::all_of(value, [](char character) {
           const auto byte = static_cast<unsigned char>(character);
           return std::isalnum(byte) != 0 || character == '.' ||
                  character == '_' || character == ':' || character == '/' ||
                  character == '-' || character == '@';
         });
}

}  // namespace

HostdClientBundle bootstrap_hostd_clients(
    Journal& journal, const HostIdentity& local_host,
    const HostdClientConfiguration& configuration,
    AuthorityClock::Source authority_clock,
    HostdStartupStatusExchange status_exchange) {
  if (!authority_clock || !valid_identifier(local_host.host_id) ||
      !valid_identifier(local_host.boot_id)) {
    throw HostdClientConfigurationError(
        "hostd bootstrap requires a canonical local host authority and clock");
  }
  const JournalAuthoritySnapshot journal_authority =
      journal.journal_authority_snapshot();
  if (journal_authority.host != local_host) {
    throw HostdClientConfigurationError(
        "hostd bootstrap local host disagrees with retained journal authority");
  }
  if (!status_exchange) {
    status_exchange = hostd_request_status;
  }
  const std::int64_t now = hostd_monotonic_now_ns();
  if (now < 0 || configuration.request_timeout_ns() >
                     std::numeric_limits<std::int64_t>::max() - now) {
    throw HostdClientConfigurationError(
        "hostd startup status deadline overflowed");
  }
  const HostdStatusReply reply = status_exchange(
      configuration.status(), 1U,
      now + configuration.request_timeout_ns());
  if (reply.kind != HostdStatusReplyKind::status || !reply.status ||
      reply.error || reply.correlation_id != 1U ||
      reply.status->host_id != local_host.host_id ||
      reply.status->boot_id != local_host.boot_id ||
      !valid_identifier(reply.status->broker_epoch) ||
      reply.status->lifecycle == HostdLifecycle::poisoned) {
    throw HostdClientConfigurationError(
        "hostd startup status disagrees with the local authority or is poisoned");
  }

  auto provider = std::make_shared<JournalHostdMutationClaimProvider>(
      journal,
      HostdMutationClaimProviderConfig{
          .api_version =
              std::string(kHostdMutationClaimProviderApiVersion),
          .broker_epoch = reply.status->broker_epoch,
          .authority_clock = std::move(authority_clock),
          .controller_id_source = {},
      });
  HostdMutationClientChannelOptions channel{
      .transport = configuration.mutation(),
      .monotonic_now = hostd_monotonic_now_ns,
      .request_timeout_ns = configuration.request_timeout_ns(),
      .exchange = {},
  };
  auto resource = std::make_shared<HostdResourceClient>(
      HostdResourceClientOptions{
          .channel = channel,
          .open_for_request = [provider](std::string_view request_id) {
            return provider->open_for_resource(request_id);
          },
      });
  auto process = std::make_shared<HostdProcessClient>(
      HostdProcessClientOptions{
          .channel = std::move(channel),
          .open_for_launch = [provider](std::string_view launch_id) {
            return provider->open_for_process(launch_id);
          },
      });
  return {.broker_epoch = reply.status->broker_epoch,
          .claim_provider = std::move(provider),
          .resource_client = std::move(resource),
          .process_client = std::move(process)};
}

}  // namespace trainvm
