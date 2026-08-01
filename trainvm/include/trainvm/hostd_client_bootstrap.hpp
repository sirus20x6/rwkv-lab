#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "trainvm/authority_time.hpp"
#include "trainvm/hostd_client_configuration.hpp"
#include "trainvm/hostd_mutation_claim_provider.hpp"
#include "trainvm/hostd_process_client.hpp"
#include "trainvm/hostd_resource_client.hpp"
#include "trainvm/journal.hpp"

namespace trainvm {

using HostdStartupStatusExchange = std::function<HostdStatusReply(
    const HostdStatusClientConfig&, std::uint64_t correlation_id,
    std::int64_t absolute_monotonic_deadline_ns)>;

struct HostdClientBundle final {
  std::string broker_epoch;
  std::shared_ptr<JournalHostdMutationClaimProvider> claim_provider;
  std::shared_ptr<HostdResourceClient> resource_client;
  std::shared_ptr<HostdProcessClient> process_client;
};

// Connects only to the exact configured socket inode, obtains the runtime
// broker epoch through the read-only status protocol, and refuses a host/boot
// mismatch before constructing any mutation client. No controller generation
// is registered until the first journal-derived mutation is requested.
[[nodiscard]] HostdClientBundle bootstrap_hostd_clients(
    Journal& journal, const HostIdentity& local_host,
    const HostdClientConfiguration& configuration,
    AuthorityClock::Source authority_clock,
    HostdStartupStatusExchange status_exchange = {});

}  // namespace trainvm
